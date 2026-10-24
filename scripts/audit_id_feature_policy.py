"""Audit identifier roles without training a repayment model or writing a submission."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from id_feature_contract import model_feature_columns


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "interim" / "loan_repayment.duckdb"
FOLD_PATH = ROOT / "data" / "interim" / "validation_folds.parquet"
OUTPUT = ROOT / "reports" / "id_role"


def quoted_file(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def columns(connection: duckdb.DuckDBPyConnection, source: str) -> list[str]:
    return [row[0] for row in connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]


def id_stats(connection: duckdb.DuckDBPyConnection, source: str) -> dict[str, int | None]:
    row = connection.execute(
        f"SELECT COUNT(*), COUNT(id), COUNT(DISTINCT id), MIN(id), MAX(id) FROM {source}"
    ).fetchone()
    return dict(zip(("rows", "non_null_ids", "unique_ids", "min_id", "max_id"), row))


def count_query(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    return int(connection.execute(sql).fetchone()[0])


def main() -> None:
    if not DATABASE.is_file() or not FOLD_PATH.is_file():
        raise FileNotFoundError("Initialize DuckDB and build validation folds first")

    fold_source = f"read_parquet({quoted_file(FOLD_PATH)})"
    sources = {
        "train": "processed_train",
        "test": "processed_test",
        "template": "processed_sample_submission",
        "folds": fold_source,
    }
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        schemas = {name: columns(connection, source) for name, source in sources.items()}
        features = model_feature_columns(schemas["train"], schemas["test"])
        if schemas["template"] != ["id", "prediction"]:
            raise ValueError("Submission template must have exactly id,prediction")
        if set(schemas["folds"]) != {
            "id", "fold", "stress_feature_492_upper_tail", "stress_feature_1039_observed"
        }:
            raise ValueError("Unexpected validation-fold schema")

        stats = {name: id_stats(connection, source) for name, source in sources.items()}
        for name, item in stats.items():
            if item["rows"] != item["non_null_ids"] or item["rows"] != item["unique_ids"]:
                raise ValueError(f"{name}: id must be non-null and unique")

        checks = {
            "train_test_shared_ids": count_query(connection,
                "SELECT COUNT(*) FROM processed_test AS te "
                "JOIN processed_train AS tr USING (id)"),
            "test_not_in_template": count_query(connection,
                "SELECT COUNT(*) FROM processed_test AS te "
                "ANTI JOIN processed_sample_submission AS s USING (id)"),
            "template_not_in_test": count_query(connection,
                "SELECT COUNT(*) FROM processed_sample_submission AS s "
                "ANTI JOIN processed_test AS te USING (id)"),
            "train_not_in_folds": count_query(connection,
                f"SELECT COUNT(*) FROM processed_train AS tr ANTI JOIN {fold_source} AS f USING (id)"),
            "folds_not_in_train": count_query(connection,
                f"SELECT COUNT(*) FROM {fold_source} AS f ANTI JOIN processed_train AS tr USING (id)"),
            "invalid_fold_values": count_query(connection,
                f"SELECT COUNT(*) FROM {fold_source} WHERE fold IS NULL OR fold NOT BETWEEN 0 AND 4"),
        }
        fold_counts = {
            int(fold): int(n)
            for fold, n in connection.execute(
                f"SELECT fold, COUNT(*) FROM {fold_source} GROUP BY fold ORDER BY fold"
            ).fetchall()
        }

    if checks["test_not_in_template"] or checks["template_not_in_test"]:
        raise ValueError("Test and submission template IDs do not match")
    if checks["train_not_in_folds"] or checks["folds_not_in_train"]:
        raise ValueError("Fold IDs do not match training IDs")
    if checks["invalid_fold_values"] or set(fold_counts) != set(range(5)):
        raise ValueError("Fold numbers must be exactly 0 through 4")

    result = {
        "feature_count": len(features),
        "feature_first": features[0],
        "feature_last": features[-1],
        "id_stats": stats,
        "checks": checks,
        "fold_counts": fold_counts,
        "policy": "id is a key for within-dataset alignment and submission, never a model predictor",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "id_role_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# `id` 的角色与真实数据核查

## 结论

`id` 是**各文件内部的记录键**，不是贷款的业务特征，也不是时间。模型输入严格为
`feature_1` 到 `feature_1105`（共 {len(features):,} 列）。`label` 是训练目标；
`fold` 和压力组标记是验证元数据，也不能进入模型。

| 数据 | 行数 | 非空且唯一的 id | id 范围 |
|---|---:|---|---|
| 训练集 | {stats['train']['rows']:,} | 是 | {stats['train']['min_id']}–{stats['train']['max_id']} |
| 测试集 | {stats['test']['rows']:,} | 是 | {stats['test']['min_id']}–{stats['test']['max_id']} |
| 官方提交模板 | {stats['template']['rows']:,} | 是 | {stats['template']['min_id']}–{stats['template']['max_id']} |
| 五折表 | {stats['folds']['rows']:,} | 是 | {stats['folds']['min_id']}–{stats['folds']['max_id']} |

训练和测试共有 {checks['train_test_shared_ids']:,} 个相同的 **id 数字**。这是因为测试集
也从 0 编号；**不能**把两个文件里的同号记录视为同一笔贷款，或通过 id 把
训练标签接到测试集。这里的 `id` 只有在明确指定数据集的语境下才有意义。

测试集对模板的反连接剩余 {checks['test_not_in_template']} 行；模板对测试集剩余
{checks['template_not_in_test']} 行，说明两边的 id 集合严格一致。模板的正式列名
是 `id,prediction`，其中 `prediction` 应填模型预测的 `label=1` 还款概率；
模板中的 0.5 只是占位值。

训练集对五折表的反连接剩余 {checks['train_not_in_folds']} 行；五折表对训练集剩余
{checks['folds_not_in_train']} 行。折号仅有 0–4，计数为 {fold_counts}。
使用五折表时应该用训练集 `id` 做一对一关联，再从关联结果中取 `fold`；
不能默认两张表第 N 行属于同一条记录。

本次仅检查数据契约，**没有训练还款模型，也没有生成真实 Kaggle 提交或还款 AUC**。
可复现命令：`.\\.venv\\Scripts\\python.exe .\\scripts\\audit_id_feature_policy.py`。
"""
    (OUTPUT / "id_role_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Report: {OUTPUT / 'id_role_report.md'}")


if __name__ == "__main__":
    main()

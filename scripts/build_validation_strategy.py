"""Build reproducible stratified folds and audit their train-test drift limits.

The row-order study found no defensible time axis. This script therefore builds
five stratified folds for repayment modeling, checks class balance, and compares
within-CV feature differences with the much larger official train-test shift.
It does not train a repayment model or claim an estimated Kaggle score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "interim" / "loan_repayment.duckdb"
FOLD_PATH = ROOT / "data" / "interim" / "validation_folds.parquet"
OUTPUT = ROOT / "reports" / "validation_strategy"
SEED = 42
N_SPLITS = 5
N_PSI_FEATURES = 8
N_MISSING_FEATURES = 5


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def selected_drift_features(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[list[str], list[str]]:
    """Select unsupervised drift signals, never using repayment labels."""
    psi = [
        row[0]
        for row in connection.execute(
            "SELECT feature FROM dq_train_test_drift "
            "WHERE isfinite(psi_quartile) "
            "ORDER BY psi_quartile DESC LIMIT ?",
            [N_PSI_FEATURES],
        ).fetchall()
    ]
    missing = [
        row[0]
        for row in connection.execute(
            "SELECT feature FROM dq_feature_quality "
            "ORDER BY abs(test_missing_rate - train_missing_rate) DESC LIMIT ?",
            [N_MISSING_FEATURES],
        ).fetchall()
    ]
    all_null = [
        row[0]
        for row in connection.execute(
            "SELECT feature FROM dq_feature_quality "
            "WHERE test_all_null ORDER BY feature"
        ).fetchall()
    ]
    if len(all_null) != 7:
        raise RuntimeError("Expected seven test-all-null features")
    # Keep one all-null example in the distribution audit; list all seven in the report.
    names = list(dict.fromkeys(psi + missing + [all_null[0]]))
    return names, all_null


def load_diagnostic_data(
    connection: duckdb.DuckDBPyConnection, names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    projection = ", ".join(
        f"CAST({quote_identifier(name)} AS FLOAT) AS {quote_identifier(name)}"
        for name in names
    )
    train = connection.execute(
        f"SELECT id, label, {projection} FROM processed_train ORDER BY id"
    ).fetchdf()
    test = connection.execute(
        f"SELECT {projection} FROM processed_test ORDER BY id"
    ).fetchdf()
    if train["id"].isna().any() or train["id"].duplicated().any():
        raise RuntimeError("Training IDs must be present and unique")
    if not train["label"].isin([0, 1]).all():
        raise RuntimeError("Training label must be binary 0/1")
    return train, test


def make_folds(labels: np.ndarray, splits: int, seed: int) -> np.ndarray:
    fold_id = np.full(len(labels), -1, dtype=np.int8)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    for fold, (_, valid_index) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        if np.any(fold_id[valid_index] != -1):
            raise RuntimeError("A row was assigned to more than one validation fold")
        fold_id[valid_index] = fold
    if np.any(fold_id < 0):
        raise RuntimeError("Some rows were not assigned a validation fold")
    return fold_id


def make_stress_flags(
    train: pd.DataFrame, fold_id: np.ndarray, splits: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_492 = train["feature_492"].to_numpy(dtype=np.float32)
    feature_1039 = train["feature_1039"].to_numpy(dtype=np.float32)
    upper_tail = np.zeros(len(train), dtype=bool)
    observed_1039 = np.isfinite(feature_1039)
    thresholds = []
    for fold in range(splits):
        fit = fold_id != fold
        valid = ~fit
        cutoff = float(np.nanquantile(feature_492[fit], 0.90))
        upper_tail[valid] = np.isfinite(feature_492[valid]) & (
            feature_492[valid] >= cutoff
        )
        thresholds.append({"fold": fold, "fit_feature_492_p90": cutoff})
    return upper_tail, observed_1039, pd.DataFrame(thresholds)


def fold_balance(
    labels: np.ndarray,
    fold_id: np.ndarray,
    upper_tail: np.ndarray,
    observed_1039: np.ndarray,
    splits: int,
) -> pd.DataFrame:
    rows = []
    for fold in range(splits):
        valid = fold_id == fold
        repay = int(np.sum(labels[valid] == 1))
        rows.append({
            "fold": fold,
            "validation_rows": int(np.sum(valid)),
            "repayment_rows": repay,
            "default_rows": int(np.sum(valid)) - repay,
            "repayment_rate": float(np.mean(labels[valid])),
            "feature_492_upper_tail_rows": int(np.sum(upper_tail & valid)),
            "feature_492_upper_tail_defaults": int(np.sum(upper_tail & valid & (labels == 0))),
            "feature_1039_observed_rows": int(np.sum(observed_1039 & valid)),
            "feature_1039_observed_defaults": int(np.sum(observed_1039 & valid & (labels == 0))),
        })
    return pd.DataFrame(rows)


def drift_audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold_id: np.ndarray,
    names: list[str],
    splits: int,
) -> pd.DataFrame:
    records = []
    for name in names:
        train_values = train[name].to_numpy(dtype=np.float32)
        test_values = test[name].to_numpy(dtype=np.float32)
        train_observed = train_values[np.isfinite(train_values)]
        test_observed = test_values[np.isfinite(test_values)]
        if len(train_observed) == 0:
            raise RuntimeError(f"No observed training values for {name}")
        test_ks = (
            float(ks_2samp(train_observed, test_observed, method="asymp").statistic)
            if len(test_observed) else None
        )
        fold_ks = []
        fold_missing_gaps = []
        for fold in range(splits):
            fit = train_values[fold_id != fold]
            valid = train_values[fold_id == fold]
            fit_observed = fit[np.isfinite(fit)]
            valid_observed = valid[np.isfinite(valid)]
            if len(fit_observed) and len(valid_observed):
                fold_ks.append(
                    float(ks_2samp(fit_observed, valid_observed, method="asymp").statistic)
                )
            fold_missing_gaps.append(
                abs(float(np.mean(~np.isfinite(fit))) - float(np.mean(~np.isfinite(valid))))
            )
        outside = (
            float(np.mean((test_observed < train_observed.min()) |
                          (test_observed > train_observed.max())))
            if len(test_observed) else None
        )
        records.append({
            "feature": name,
            "train_observed_rows": len(train_observed),
            "test_observed_rows": len(test_observed),
            "train_missing_rate": float(np.mean(~np.isfinite(train_values))),
            "test_missing_rate": float(np.mean(~np.isfinite(test_values))),
            "train_test_missing_gap": abs(
                float(np.mean(~np.isfinite(train_values))) -
                float(np.mean(~np.isfinite(test_values)))
            ),
            "max_fold_missing_gap": max(fold_missing_gaps),
            "train_test_ks_observed": test_ks,
            "max_fold_ks_observed": max(fold_ks) if fold_ks else None,
            "test_observed_outside_train_range": outside,
        })
    return pd.DataFrame(records)


def save_folds_safely(table: pd.DataFrame, path: Path) -> str:
    """Never silently replace a different validation assignment."""
    if path.exists():
        prior = pd.read_parquet(path)
        if not prior.equals(table):
            raise RuntimeError(
                f"Existing fold file differs: {path}. Inspect it before changing validation."
            )
        return "reused_identical"
    temporary = path.with_name(path.stem + ".tmp.parquet")
    if temporary.exists():
        raise RuntimeError(f"Temporary fold file already exists: {temporary}")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)
    return "created"


def plot_results(balance: pd.DataFrame, audit: pd.DataFrame, output: Path) -> None:
    numeric = audit.dropna(subset=["train_test_ks_observed"])
    numeric = numeric.nlargest(7, "train_test_ks_observed").iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    axes[0].plot(balance["fold"], 100 * balance["repayment_rate"], marker="o")
    axes[0].axhline(
        100 * balance["repayment_rows"].sum() / balance["validation_rows"].sum(),
        color="black", linestyle="--", linewidth=1, label="Overall training rate",
    )
    axes[0].set_xticks(balance["fold"])
    axes[0].set_ylim(90, 93)
    axes[0].set_xlabel("Validation fold (0-based)")
    axes[0].set_ylabel("Repayment rate (%)")
    axes[0].set_title("Stratified fold balance")
    axes[0].legend()

    y = np.arange(len(numeric))
    axes[1].scatter(numeric["train_test_ks_observed"], y,
                    label="All train vs official test", color="#EE7733")
    axes[1].scatter(numeric["max_fold_ks_observed"], y,
                    label="Largest fit-vs-valid difference", color="#4477AA")
    axes[1].set_yticks(y, numeric["feature"])
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_xlabel("KS distance on non-missing values (0–1)")
    axes[1].set_title("Random folds do not reproduce test drift")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(output / "validation_diagnostics.png", dpi=170)
    plt.close(fig)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def report(
    balance: pd.DataFrame,
    audit: pd.DataFrame,
    thresholds: pd.DataFrame,
    all_null: list[str],
    output: Path,
    elapsed: float,
    fold_status: str,
    seed: int,
) -> None:
    overall_rate = balance.repayment_rows.sum() / balance.validation_rows.sum()
    balance_rows = [
        [str(r.fold), f"{r.validation_rows:,}", f"{r.default_rows:,}",
         f"{r.repayment_rate:.4%}", f"{r.feature_492_upper_tail_rows:,}",
         f"{r.feature_1039_observed_rows:,}"]
        for r in balance.itertuples(index=False)
    ]
    examples = ["feature_492", "feature_903", "feature_586",
                "feature_1039", all_null[0]]
    audit_by_name = audit.set_index("feature")
    audit_rows = []
    for name in examples:
        r = audit_by_name.loc[name]
        audit_rows.append([
            name,
            "n.a." if pd.isna(r.max_fold_ks_observed) else f"{r.max_fold_ks_observed:.3f}",
            "n.a." if pd.isna(r.train_test_ks_observed) else f"{r.train_test_ks_observed:.3f}",
            f"{r.max_fold_missing_gap:.2%}",
            f"{r.train_test_missing_gap:.2%}",
            "n.a." if pd.isna(r.test_observed_outside_train_range)
            else f"{r.test_observed_outside_train_range:.1%}",
        ])
    feature492 = audit_by_name.loc["feature_492"]
    feature1039 = audit_by_name.loc["feature_1039"]
    text = f"""# 验证策略：分层五折与漂移稳健性检查

## 决策

此前逐行及前后段分类检查没有验证出可信的训练集时间顺序。因此本阶段采用
**5 折 StratifiedKFold（shuffle=True，random_state={seed}）**，而不是把
CSV 末段冒充“未来”留出集。每条训练记录恰好属于一个验证折；其余四折用于拟合。
折号保存在 `data/interim/validation_folds.parquet`，状态：`{fold_status}`。

这一步建立并核查验证协议，**尚未训练还款模型，也没有计算还款模型 AUC**。
训练/测试漂移依旧很强；分层只平衡标签比例，不能让验证折自动变得像官方测试集。

## 1. 分层折的基本质量

全量训练集还款率为 **{overall_rate:.4%}**，违约率为 **{1-overall_rate:.4%}**。
每折的验证集规模和类别如下。最后两列是留给以后模型做分组评估的压力样本数。

{md_table(['折', '验证行数', '违约行数', '还款率', 'feature_492 高尾行数', 'feature_1039 非缺失行数'], balance_rows)}

`feature_492 高尾` 的阈值由**该折拟合数据**的第 90 百分位计算，不能用验证折
或测试集拟合阈值。五折阈值范围为
**{thresholds.fit_feature_492_p90.min():.4f}–{thresholds.fit_feature_492_p90.max():.4f}**。
`feature_1039 非缺失` 是另一个测试集更常见的状态。保存的折表含这两个布尔标记，
以后有折外还款概率时，就可以分别计算两组的还款 AUC、违约 PR-AUC 等指标。
每折的高尾组至少有 **{balance.feature_492_upper_tail_defaults.min():,}** 条违约记录，
非缺失组至少有 **{balance.feature_1039_observed_defaults.min():,}** 条违约记录；
这些分组人数足以计算诊断指标，但仍不代表它们覆盖测试集的全部新取值。

## 2. 分层折与官方测试集的分布差距

下表的 KS 距离比较**非缺失数值**的分布：0 表示很接近，越大差异越大。
“最大折内 KS”是五次“拟合四折 vs 验证一折”中的最大值；缺失率差单独计算。
审计特征由此前无监督的 PSI 和缺失率差选出，不使用还款标签。

{md_table(['特征', '最大折内 KS', '训练/测试 KS', '最大折内缺失差', '训练/测试缺失差', '测试非缺失值超训练范围'], audit_rows)}

例如，`feature_492` 的最大折内 KS 为 **{feature492.max_fold_ks_observed:.3f}**，
训练/测试 KS 却为 **{feature492.train_test_ks_observed:.3f}**。
测试集在该列有 **{feature492.test_observed_outside_train_range:.1%}** 的非缺失值
超出全量训练集的观测范围。`feature_1039` 的训练/测试缺失率差为
**{feature1039.train_test_missing_gap:.1%}**，但折内最大差仅
**{feature1039.max_fold_missing_gap:.2%}**。这说明随机折能代表训练集内部，却
不能模拟官方测试集的主要变化。测试集全空的 7 列为：
{', '.join(f'`{name}`' for name in all_null)}。

## 3. 漂移稳健性检查如何落地

本次已完成：五折标签平衡、重点特征的折内/训练-测试分布比较、测试超出训练
取值范围比例，以及两类压力样本的折内标记和人数检查。

等下一步训练还款模型并得到**折外预测概率**后，按相同折号报告：
整体还款 ROC-AUC、违约 PR-AUC，以及 `feature_492 高尾`、`feature_1039 非缺失`
两组各自的指标和样本数。不要把同一折用于早停又当作唯一的最终评价；
需要早停时，从拟合四折内部再切一份。压力组用于诊断，不要反复调参直至它们
的验证分数变高。

**边界：**训练集中没有 `feature_492` 超出自身最大值的带标签样本，因此任何
训练内验证都无法直接验证那部分测试记录的还款表现。不能从折内高分推断
这 {feature492.test_observed_outside_train_range:.1%} 测试样本也会表现良好，
也不能靠简单重加权创造不存在的标签。若未来取得可信日期或批次，重新设计
时间留出或滚动验证，不要把当前 `id` 当作日期。

## 复现

```powershell
.\\.venv\\Scripts\\python.exe .\\scripts\\build_validation_strategy.py
```

运行时间：{elapsed:.1f} 秒。
"""
    (output / "validation_strategy_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fold-path", type=Path, default=FOLD_PATH)
    args = parser.parse_args()
    if args.folds != N_SPLITS:
        raise ValueError("This project protocol currently fixes five folds")
    if not DATABASE.is_file():
        raise FileNotFoundError(DATABASE)
    if not (ROOT / "reports" / "row_order" / "row_order_report.md").is_file():
        raise FileNotFoundError("Complete the row-order check before choosing CV")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    args.fold_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        connection.execute("SET threads = 4")
        names, all_null = selected_drift_features(connection)
        if "feature_492" not in names or "feature_1039" not in names:
            raise RuntimeError("Required stress-check features were not selected")
        train, test = load_diagnostic_data(connection, names)

    ids = train["id"].to_numpy(dtype=np.int64)
    labels = train["label"].to_numpy(dtype=np.uint8)
    if len(ids) != 688_998 or len(test) != 36_324:
        raise RuntimeError("Dataset dimensions changed since the row-order check")
    fold_id = make_folds(labels, args.folds, args.seed)
    upper_tail, observed_1039, thresholds = make_stress_flags(train, fold_id, args.folds)
    balance = fold_balance(labels, fold_id, upper_tail, observed_1039, args.folds)
    audit = drift_audit(train, test, fold_id, names, args.folds)

    fold_table = pd.DataFrame({
        "id": ids,
        "fold": fold_id,
        "stress_feature_492_upper_tail": upper_tail,
        "stress_feature_1039_observed": observed_1039,
    })
    fold_status = save_folds_safely(fold_table, args.fold_path)
    balance.to_parquet(args.output / "fold_balance.parquet", index=False)
    audit.to_parquet(args.output / "drift_audit.parquet", index=False)
    thresholds.to_parquet(args.output / "stress_thresholds.parquet", index=False)
    plot_results(balance, audit, args.output)
    elapsed = time.perf_counter() - started
    report(balance, audit, thresholds, all_null, args.output, elapsed, fold_status,
           args.seed)
    (args.output / "run_config.json").write_text(
        json.dumps({
            "strategy": "stratified_kfold",
            "n_splits": args.folds,
            "shuffle": True,
            "random_state": args.seed,
            "row_order_status": "not_verified_as_time",
            "fold_path": str(args.fold_path),
            "selected_drift_features": names,
            "test_all_null_features": all_null,
            "elapsed_seconds": elapsed,
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Folds: {args.fold_path} ({fold_status})", flush=True)
    print(f"Report: {args.output / 'validation_strategy_report.md'}", flush=True)
    print(f"Runtime: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

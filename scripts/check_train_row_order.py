"""Check whether the physical order of training rows is a credible time proxy.

This is a diagnostic, not proof of transaction dates. It verifies original CSV
order, looks for position trends, and asks whether a model can distinguish the
first and last equally sized blocks of training rows using only loan features.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_adversarial_validation import (
    DATABASE,
    SEED,
    load_metadata,
    quote_identifier,
    run_variant,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "row_order"
SAMPLE_PER_SIDE = 36_324
FOLDS = 3


def verify_csv_ids(path: Path) -> dict[str, int | str]:
    """Stream the CSV once; compare each physical line number with its id."""
    count = 0
    mismatches = 0
    first_bad_line = -1
    with path.open("rb", buffering=4 * 1024 * 1024) as source:
        header = source.readline()
        if not header.startswith(b"id,"):
            raise ValueError(f"Expected id as first CSV column: {path}")
        for count, line in enumerate(source, start=1):
            comma = line.find(b",")
            if comma < 0 or line[:comma] != str(count - 1).encode("ascii"):
                mismatches += 1
                if first_bad_line < 0:
                    first_bad_line = count
    return {
        "file": path.name,
        "rows": count,
        "id_vs_zero_based_physical_row_mismatches": mismatches,
        "first_bad_data_line": first_bad_line,
    }


def summarize_deciles(connection: duckdb.DuckDBPyConnection, rows: int) -> pd.DataFrame:
    return connection.execute(
        f"""
        SELECT CAST(FLOOR(id * 10.0 / {rows}) AS INTEGER) + 1 AS row_decile,
               COUNT(*) AS rows,
               AVG(CAST(label AS DOUBLE)) AS repayment_rate,
               MEDIAN(feature_492) AS feature_492_median,
               MIN(feature_492) AS feature_492_min,
               MAX(feature_492) AS feature_492_max,
               AVG(CAST(feature_1039 IS NULL AS DOUBLE)) AS feature_1039_missing_rate,
               AVG(CAST(feature_903 IS NULL AS DOUBLE)) AS feature_903_missing_rate
        FROM processed_train
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()


def load_edge_features(
    connection: duckdb.DuckDBPyConnection,
    names: list[str],
    total_rows: int,
    sample_per_side: int,
) -> np.ndarray:
    projection = ", ".join(
        f"CAST({quote_identifier(name)} AS FLOAT) AS {quote_identifier(name)}"
        for name in names
    )
    frame = connection.execute(
        f"SELECT {projection} FROM processed_train "
        f"WHERE id < {sample_per_side} OR id >= {total_rows - sample_per_side} "
        "ORDER BY id"
    ).fetchdf()
    if frame.shape != (2 * sample_per_side, len(names)):
        raise RuntimeError("Unexpected edge-sample dimensions")
    values = frame.to_numpy(dtype=np.float32, copy=True)
    del frame
    gc.collect()
    if np.isinf(values).any():
        raise RuntimeError("Infinite feature values require cleaning")
    return values


def plot_diagnostics(
    deciles: pd.DataFrame,
    metrics: pd.DataFrame,
    test_feature_492_median: float,
    output: Path,
) -> None:
    summary = metrics.groupby("variant")["auc"].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(deciles["row_decile"], 100 * deciles["repayment_rate"], marker="o")
    axes[0].set_xlabel("Training row decile (CSV order)")
    axes[0].set_ylabel("Repayment rate (%)")
    axes[0].set_title("Target rate across row order")
    axes[0].set_xticks(range(1, 11))
    axes[0].set_ylim(90, 93)

    axes[1].plot(
        deciles["row_decile"], deciles["feature_492_median"], marker="o",
        label="Train decile median",
    )
    axes[1].axhline(
        test_feature_492_median, color="#EE7733", linestyle="--",
        label="Test median",
    )
    axes[1].set_xlabel("Training row decile (CSV order)")
    axes[1].set_ylabel("feature_492 median")
    axes[1].set_title("Leading drift feature across row order")
    axes[1].set_xticks(range(1, 11))
    axes[1].legend()

    variants = ["first_vs_last", "shuffled_label_control"]
    axes[2].errorbar(
        [0, 1], [summary.loc[name, "mean"] for name in variants],
        yerr=[summary.loc[name, "std"] for name in variants],
        fmt="o", capsize=4, markersize=8, color="#4477AA",
    )
    axes[2].set_xticks([0, 1], ["First vs last", "Shuffled labels"])
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    means = [float(summary.loc[name, "mean"]) for name in variants]
    axes[2].set_ylim(max(0, min(0.5, *means) - 0.07),
                     min(1.02, max(0.5, *means) + 0.07))
    axes[2].set_ylabel("Held-out ROC-AUC")
    axes[2].set_title("Can features recognize row position?")
    fig.tight_layout()
    fig.savefig(output / "row_order_diagnostics.png", dpi=170)
    plt.close(fig)


def write_report(
    output: Path,
    csv_checks: list[dict[str, int | str]],
    deciles: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: dict[str, float | int],
    elapsed: float,
) -> None:
    mean_auc = metrics.groupby("variant")["auc"].mean()
    edge_auc = float(mean_auc["first_vs_last"])
    control_auc = float(mean_auc["shuffled_label_control"])
    first = deciles.iloc[0]
    last = deciles.iloc[-1]
    if edge_auc < 0.55:
        conclusion = (
            "前后位置分类器在未见过的样本上接近随机水平；结合分段统计，"
            "**没有证据支持把训练行号直接当作时间顺序**。"
        )
    else:
        conclusion = (
            "前后位置分类器能在一定程度上识别位置，说明存在行位置相关的分布变化。"
            "但没有真实日期，**仍不能认定它就是时间顺序**。"
        )
    metric_rows = "\n".join(
        f"| {name} | "
        + ", ".join(
            f"{x:.5f}" for x in metrics.loc[metrics.variant == name, "auc"]
        )
        + f" | {mean_auc[name]:.5f} |"
        for name in ["first_vs_last", "shuffled_label_control"]
    )
    text = f"""# 训练集行顺序是否有时间意义？

## 结论

{conclusion}官方训练/测试之间存在极强漂移，并不意味着训练集的 CSV 行顺序也按时间排列。
本次检验只能排查“行号可否作为时间代理”，不能证明真实放款日期。

## 方法与结果

1. **确认物理行号。** 逐行读取原始 CSV 的第一列，不经过排序或 Parquet 重排。
   `train.csv` 检查 {csv_checks[0]['rows']:,} 行，行号与 `id` 不一致数：
   **{csv_checks[0]['id_vs_zero_based_physical_row_mismatches']:,}**；
   `test.csv` 检查 {csv_checks[1]['rows']:,} 行，不一致数：
   **{csv_checks[1]['id_vs_zero_based_physical_row_mismatches']:,}**。
   两份数据各自从 0 编号；`id` 等于各自文件的行号，不是跨数据集共享的时间戳。
2. **按训练行号切成连续十段。** 第一段还款率 **{first.repayment_rate:.3%}**，
   最后一段 **{last.repayment_rate:.3%}**；`feature_492` 中位数分别为
   **{first.feature_492_median:.4f}** 和 **{last.feature_492_median:.4f}**。
   十段中位数范围为 **{deciles.feature_492_median.min():.4f}–{deciles.feature_492_median.max():.4f}**；
   测试集整体中位数为 **{diagnostics['test_feature_492_median']:.4f}**。
   训练内部没有出现向测试集数值范围逐段靠近的趋势。
3. **检查连续关系。** 训练集行号与 `feature_492` 的 Spearman 等级相关系数为
   **{diagnostics['id_feature_492_spearman']:.6f}**。这里主要看相关程度，
   不以大样本下容易显著的 p 值代替实际效应大小。
4. **全特征前后段对抗分类。** 取原训练集最前和最后各
   **{diagnostics['sample_per_side']:,}** 行，构造临时标签 `0=前段、1=后段`。
   `id` 和还款 `label` 都不输入模型。用相同的 1,105 个匿名特征和三折分层验证；
   每折再从拟合部分留 15% 早停，外层验证折只评价 AUC。
   再把位置标签随机打乱，作为“无位置信号”的对照。

| 实验 | 三折 ROC-AUC | 平均 AUC |
| --- | --- | --- |
{metric_rows}

这里的 AUC 只评价能否认出训练集内部的行位置，**不是还款预测 AUC**。
随机标签对照用于检查分类流程在没有来源信号时会得到什么分数；
接近 0.5 是预期现象。前后段实验只检测选定的两端样本，不能排除
局部时间群组、隐藏但未保留的真实日期，或表内存在更弱的顺序信号。

## 验证策略决定

- **不**把 `id` 或 CSV 行号直接用作时间切分依据，也不把它们解释为放款日期。
- 随机分层验证仍是第一版还款模型的基线。官方测试集更晚这一事实，
  仍要求我们警惕随机验证对测试集泛化能力的估计偏乐观。
- 若以后找到可核验的真实日期、批次或其他顺序字段，再重新设计按时间留出的验证；
  不能从匿名特征的漂移直接反推出可靠时间轴。

## 复现

```powershell
.\\.venv\\Scripts\\python.exe .\\scripts\\check_train_row_order.py
```

运行时间：{elapsed:.1f} 秒。
"""
    (output / "row_order_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-side", type=int, default=SAMPLE_PER_SIDE)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.sample_per_side < args.folds * 2 or args.folds < 2:
        raise ValueError("Need enough rows for at least two folds")
    if not DATABASE.is_file() or not TRAIN_CSV.is_file() or not TEST_CSV.is_file():
        raise FileNotFoundError("Processed database and original CSV files are required")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    csv_checks = []
    for path in [TRAIN_CSV, TEST_CSV]:
        result = verify_csv_ids(path)
        csv_checks.append(result)
        print(
            f"{path.name}: {result['rows']:,} physical rows, "
            f"{result['id_vs_zero_based_physical_row_mismatches']:,} id mismatches",
            flush=True,
        )
        if result["id_vs_zero_based_physical_row_mismatches"]:
            raise RuntimeError(f"CSV physical row order does not equal id: {path}")

    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        connection.execute("SET threads = 4")
        total_rows = connection.execute("SELECT COUNT(*) FROM processed_train").fetchone()[0]
        test_rows = connection.execute("SELECT COUNT(*) FROM processed_test").fetchone()[0]
        if (total_rows, test_rows) != (csv_checks[0]["rows"], csv_checks[1]["rows"]):
            raise RuntimeError("Processed counts disagree with original CSV counts")
        if 2 * args.sample_per_side > total_rows:
            raise ValueError("Edge samples would overlap")
        parquet_order_mismatches = connection.execute(
            "SELECT COUNT(*) FILTER (WHERE id <> row_number - 1) "
            "FROM (SELECT id, ROW_NUMBER() OVER () AS row_number "
            "FROM processed_train)"
        ).fetchone()[0]
        if parquet_order_mismatches:
            raise RuntimeError("Processed Parquet order differs from CSV id order")
        deciles = summarize_deciles(connection, total_rows)
        feature_492 = connection.execute(
            "SELECT id, feature_492 FROM processed_train ORDER BY id"
        ).fetchnumpy()
        observed = np.isfinite(feature_492["feature_492"])
        rank_correlation = float(
            spearmanr(
                feature_492["id"][observed],
                feature_492["feature_492"][observed],
            ).statistic
        )
        test_median = float(
            connection.execute("SELECT MEDIAN(feature_492) FROM processed_test").fetchone()[0]
        )
        metadata = load_metadata(connection)
        names = metadata["feature"].tolist()
        values = load_edge_features(connection, names, total_rows, args.sample_per_side)

    edge_label = np.r_[np.zeros(args.sample_per_side, dtype=np.uint8),
                       np.ones(args.sample_per_side, dtype=np.uint8)]
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.zeros(len(edge_label)), edge_label))
    edge_metrics, edge_gains, _ = run_variant(
        values, edge_label, names, "first_vs_last", folds, args.threads
    )
    shuffled = np.random.default_rng(SEED).permutation(edge_label)
    shuffled_folds = list(cv.split(np.zeros(len(shuffled)), shuffled))
    control_metrics, _, _ = run_variant(
        values, shuffled, names, "shuffled_label_control", shuffled_folds,
        args.threads,
    )
    del values
    gc.collect()
    metrics = pd.concat([edge_metrics, control_metrics], ignore_index=True)
    importance = (
        edge_gains.groupby("feature", as_index=False)["gain_share"]
        .agg(mean_gain_share="mean", sd_gain_share="std")
        .sort_values("mean_gain_share", ascending=False)
    )
    diagnostics = {
        "train_rows": total_rows,
        "test_rows": test_rows,
        "sample_per_side": args.sample_per_side,
        "id_feature_492_spearman": rank_correlation,
        "test_feature_492_median": test_median,
        "parquet_order_mismatches": parquet_order_mismatches,
        "csv_checks": csv_checks,
        "seed": SEED,
        "folds": args.folds,
    }
    deciles.to_csv(args.output / "train_deciles.csv", index=False)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    importance.to_csv(args.output / "edge_feature_importance.csv", index=False)
    (args.output / "row_order_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_diagnostics(deciles, metrics, test_median, args.output)
    elapsed = time.perf_counter() - started
    write_report(args.output, csv_checks, deciles, metrics, diagnostics, elapsed)
    print(f"Report: {args.output / 'row_order_report.md'}", flush=True)
    print(f"Runtime: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

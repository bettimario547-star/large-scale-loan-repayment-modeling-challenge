"""Measure train-test drift with an adversarial classifier.

The target here is dataset membership: train=0, test=1. It is never the loan
repayment label. The script samples the same number of training rows as the
test set, excludes id and label from predictors, and compares three feature
sets under the same stratified folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "interim" / "loan_repayment.duckdb"
DEFAULT_OUTPUT = ROOT / "reports" / "adversarial_validation"
SEED = 20261018
FOLDS = 3
THREADS = 4


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def feature_number(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def load_metadata(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    quality = connection.execute(
        "SELECT feature, train_missing_rate, test_missing_rate, "
        "test_all_null FROM dq_feature_quality"
    ).fetchdf()
    drift = connection.execute(
        "SELECT feature, psi_quartile, abs_standardized_mean_difference, "
        "missing_rate_difference FROM dq_train_test_drift"
    ).fetchdf()
    metadata = quality.merge(drift, on="feature", validate="one_to_one")
    metadata = metadata.sort_values(
        "feature", key=lambda series: series.map(feature_number)
    ).reset_index(drop=True)
    if len(metadata) != 1105 or metadata["feature"].nunique() != 1105:
        raise RuntimeError("Data quality tables must contain 1,105 unique features")
    return metadata


def load_balanced_arrays(
    connection: duckdb.DuckDBPyConnection,
    features: list[str],
    seed: int,
    sample_per_source: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use id only for deterministic sampling, never as a model feature."""
    test_count = connection.execute("SELECT COUNT(*) FROM processed_test").fetchone()[0]
    train_count = connection.execute("SELECT COUNT(*) FROM processed_train").fetchone()[0]
    if sample_per_source > min(train_count, test_count):
        raise ValueError("sample_per_source exceeds the smaller dataset")

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE av_sample_ids AS
        SELECT id
        FROM processed_train
        ORDER BY hash(id, {seed})
        LIMIT {sample_per_source}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE av_test_ids AS
        SELECT id
        FROM processed_test
        ORDER BY hash(id, {seed + 1})
        LIMIT {sample_per_source}
        """
    )
    projection = ", ".join(
        f"CAST({quote_identifier(feature)} AS FLOAT) AS "
        f"{quote_identifier(feature)}"
        for feature in features
    )
    train_frame = connection.execute(
        "SELECT " + projection + " FROM processed_train AS t "
        "INNER JOIN av_sample_ids AS s USING (id) ORDER BY t.id"
    ).fetchdf()
    test_frame = connection.execute(
        "SELECT " + projection + " FROM processed_test AS t "
        "INNER JOIN av_test_ids AS s USING (id) ORDER BY t.id"
    ).fetchdf()
    if train_frame.shape != test_frame.shape or train_frame.shape[1] != len(features):
        raise RuntimeError("Train and test sample shapes or feature order differ")
    if train_frame.shape[0] != sample_per_source:
        raise RuntimeError("The expected number of sampled rows was not loaded")

    train_values = train_frame.to_numpy(dtype=np.float32, copy=True)
    test_values = test_frame.to_numpy(dtype=np.float32, copy=True)
    del train_frame, test_frame
    gc.collect()
    values = np.concatenate([train_values, test_values], axis=0)
    domain_label = np.concatenate(
        [
            np.zeros(sample_per_source, dtype=np.uint8),
            np.ones(sample_per_source, dtype=np.uint8),
        ]
    )
    del train_values, test_values
    gc.collect()
    if np.isinf(values).any():
        raise RuntimeError("Infinite feature values require cleaning before modeling")
    return values, domain_label


def run_variant(
    values: np.ndarray,
    domain_label: np.ndarray,
    column_names: list[str],
    variant: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Fit LightGBM on three held-out folds and collect gain importance."""
    fold_records: list[dict[str, object]] = []
    importance_records: list[dict[str, object]] = []
    out_of_fold_score = np.full(len(domain_label), np.nan, dtype=np.float32)

    for fold_number, (fit_index, valid_index) in enumerate(folds, start=1):
        inner_fit_index, early_stop_index = train_test_split(
            fit_index,
            test_size=0.15,
            stratify=domain_label[fit_index],
            random_state=SEED + fold_number,
        )
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=450,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=100,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            max_bin=63,
            random_state=SEED + fold_number,
            n_jobs=threads,
            verbosity=-1,
        )
        model.fit(
            values[inner_fit_index],
            domain_label[inner_fit_index],
            eval_X=values[early_stop_index],
            eval_y=domain_label[early_stop_index],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        score = model.predict_proba(
            values[valid_index], num_iteration=model.best_iteration_
        )[:, 1]
        auc = roc_auc_score(domain_label[valid_index], score)
        out_of_fold_score[valid_index] = score.astype(np.float32)
        gain = model.booster_.feature_importance(importance_type="gain")
        gain_share = gain / gain.sum() if gain.sum() else np.zeros_like(gain)

        fold_records.append(
            {
                "variant": variant,
                "fold": fold_number,
                "fit_rows": len(inner_fit_index),
                "early_stop_rows": len(early_stop_index),
                "validation_rows": len(valid_index),
                "validation_train_rows": int((domain_label[valid_index] == 0).sum()),
                "validation_test_rows": int((domain_label[valid_index] == 1).sum()),
                "auc": auc,
                "best_iteration": int(model.best_iteration_),
            }
        )
        importance_records.extend(
            {
                "variant": variant,
                "fold": fold_number,
                "feature": feature,
                "gain_share": float(share),
            }
            for feature, share in zip(column_names, gain_share, strict=True)
        )
        print(
            f"{variant} fold {fold_number}/{len(folds)}: "
            f"AUC={auc:.8f}, trees={model.best_iteration_}",
            flush=True,
        )
        del model
        gc.collect()

    if np.isnan(out_of_fold_score).any():
        raise RuntimeError(f"Incomplete out-of-fold predictions for {variant}")
    return (
        pd.DataFrame(fold_records),
        pd.DataFrame(importance_records),
        out_of_fold_score,
    )


def check_leading_signal(
    values: np.ndarray,
    domain_label: np.ndarray,
    feature_names: list[str],
    leading_feature: str,
) -> dict[str, float | str]:
    """Check whether one raw feature separates the two sources on its own."""
    column = values[:, feature_names.index(leading_feature)]
    observed = np.isfinite(column)
    train = column[(domain_label == 0) & observed]
    test = column[(domain_label == 1) & observed]
    train_max = float(np.max(train))
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        quoted = quote_identifier(leading_feature)
        full_train_max = float(
            connection.execute(
                f"SELECT MAX({quoted}) FROM processed_train"
            ).fetchone()[0]
        )
        full_test_above = float(
            connection.execute(
                f"SELECT AVG(CAST({quoted} > ? AS DOUBLE)) "
                f"FROM processed_test WHERE {quoted} IS NOT NULL",
                [full_train_max],
            ).fetchone()[0]
        )
    return {
        "feature": leading_feature,
        "observed_only_auc": float(roc_auc_score(domain_label[observed], column[observed])),
        "sample_train_median": float(np.median(train)),
        "sample_test_median": float(np.median(test)),
        "sample_train_max": train_max,
        "test_observed_fraction_above_train_max": float(np.mean(test > train_max)),
        "full_train_max": full_train_max,
        "full_test_observed_fraction_above_train_max": full_test_above,
        "sample_train_observed_rows": int(len(train)),
        "sample_test_observed_rows": int(len(test)),
    }


def plot_results(
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    values: np.ndarray,
    domain_label: np.ndarray,
    feature_names: list[str],
    signal_check: dict[str, float | str],
    output: Path,
) -> None:
    names = {
        "all_values": "All numeric values",
        "exclude_test_all_null": "Exclude 7 test-all-null",
        "missing_mask_only": "Missing mask only",
    }
    summary = metrics.groupby("variant")["auc"].agg(["mean", "std"])
    variants = list(names)
    figure, axes = plt.subplots(1, 3, figsize=(19, 5))
    axes[0].bar(
        range(len(variants)),
        [summary.loc[name, "mean"] for name in variants],
        yerr=[summary.loc[name, "std"] for name in variants],
        capsize=4,
        color=["#4477AA", "#228833", "#CCBB44"],
    )
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(range(len(variants)), [names[name] for name in variants], rotation=15)
    axes[0].set_ylim(0.45, 1.02)
    axes[0].set_ylabel("Adversarial ROC-AUC")
    axes[0].set_title("Train vs test separability (3 folds)")

    top = importance[
        importance["variant"] == "exclude_test_all_null"
    ].nlargest(15, "mean_gain_share").sort_values("mean_gain_share")
    axes[1].barh(top["feature"], top["mean_gain_share"], color="#228833")
    axes[1].set_xlabel("Mean share of LightGBM split gain")
    axes[1].set_title("Leading signals after all-null ablation")

    feature = str(signal_check["feature"])
    column = values[:, feature_names.index(feature)]
    for label, name, color in [(0, "Train", "#4477AA"), (1, "Test", "#EE7733")]:
        observed = column[(domain_label == label) & np.isfinite(column)]
        axes[2].hist(
            observed, bins=70, density=True, histtype="step", linewidth=1.8,
            color=color, label=name,
        )
    axes[2].axvline(
        float(signal_check["sample_train_max"]), color="#4477AA",
        linestyle="--", linewidth=1, label="Train maximum",
    )
    axes[2].set_xlabel(feature + " (observed values)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Leading feature's source shift")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output / "adversarial_validation.png", dpi=170)
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def write_report(
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    metadata: pd.DataFrame,
    signal_check: dict[str, float | str],
    sample_per_source: int,
    output: Path,
    elapsed_seconds: float,
) -> None:
    summary = metrics.groupby("variant")["auc"].agg(["mean", "std"])
    full_auc = float(summary.loc["all_values", "mean"])
    ablated_auc = float(summary.loc["exclude_test_all_null", "mean"])
    mask_auc = float(summary.loc["missing_mask_only", "mean"])
    all_null = metadata.loc[metadata["test_all_null"], "feature"].tolist()
    top = importance[
        importance["variant"] == "exclude_test_all_null"
    ].nlargest(15, "mean_gain_share")
    top_rows = []
    for row in top.itertuples(index=False):
        top_rows.append(
            [
                row.feature,
                f"{row.mean_gain_share:.2%}",
                f"{row.train_missing_rate:.1%}",
                f"{row.test_missing_rate:.1%}",
                "n.a." if pd.isna(row.psi_quartile) else f"{row.psi_quartile:.3f}",
            ]
        )
    metrics_rows = [
        [
            name,
            ", ".join(f"{value:.8f}" for value in metrics.loc[metrics["variant"] == name, "auc"]),
            f"{summary.loc[name, 'mean']:.8f}",
            f"{summary.loc[name, 'std']:.8f}",
        ]
        for name in ["all_values", "exclude_test_all_null", "missing_mask_only"]
    ]
    interpretation = (
        "去掉测试集全空的 7 列后仍几乎能完全区分来源，说明还存在其他强烈的分布差异。"
        if ablated_auc > 0.70
        else "去掉测试集全空的 7 列后区分能力大幅下降，这些列解释了相当一部分来源信号。"
    )
    report = f"""# 对抗验证：识别训练集与测试集漂移

## 目标与方法

把数据来源作为一个新的二分类目标：`0 = 原训练集`，`1 = 原测试集`。
它与贷款还款标签 `label` 是两个不同的目标。若两组数据无法区分，ROC-AUC 接近
0.5；若 AUC 明显高于 0.5，说明模型能利用特征识别数据来源，存在分布差异。

用 `hash(id, {SEED})` 从 688,998 条原训练样本中固定抽取 {sample_per_source:,} 条，
与 {sample_per_source:,} 条测试样本组成平衡数据集。`id` 只用于抽样，不输入模型；
`label` 也不输入模型。所有特征在 DuckDB 中转为 `float32`，缺失值保持为 NaN。
采用 {len(metrics['fold'].unique())} 折分层交叉验证，各方案使用相同的折划分。每折再从拟合部分切出
15% 用于 LightGBM 早停，外层验证折只用于最终 AUC 评价。

## 结果

{markdown_table(['方案', '各折 ROC-AUC', '均值', '折间标准差'], metrics_rows)}

- 全特征 AUC：**{full_auc:.8f}**。
- 去除测试集全空 7 列后 AUC：**{ablated_auc:.8f}**。
- 仅用“是否缺失”指示器的 AUC：**{mask_auc:.8f}**。
- 全特征与去除 7 列的 AUC 差：**{full_auc - ablated_auc:+.8f}**。

{interpretation}缺失指示器模型说明仅凭“哪些位置为空”也能很好地区分来源；
它的 AUC **不是贷款还款预测的 AUC**。这些结果证明存在分布差异，不能仅凭对抗验证
断言差异完全由时间造成，也不能推算其对还款模型成绩的影响大小。

测试集全空的 7 列是：{', '.join(f'`{name}`' for name in all_null)}。

### 去除 7 列后的主要识别信号

这里的 gain share 是 LightGBM 在三折中的平均分裂增益占比，只表示对“区分数据来源”
的贡献，**不是还款预测的重要性**。相关特征可能互相替代，排名也会随采样和参数变化。

{markdown_table(['特征', '平均 gain 占比', '训练缺失率', '测试缺失率', '非缺失值 PSI'], top_rows)}

### 单特征核查

最重要的 `{signal_check['feature']}` 不经模型训练、只按原始值大小排序（排除该列缺失行），
区分来源的 AUC 就达到 **{signal_check['observed_only_auc']:.6f}**。
等量样本中，训练/测试的该列中位数分别为
**{signal_check['sample_train_median']:.4f} / {signal_check['sample_test_median']:.4f}**；
全量训练集最大值为 **{signal_check['full_train_max']:.4f}**，而测试非缺失样本中
**{signal_check['full_test_observed_fraction_above_train_max']:.1%}** 超过它。
这是直接在原始特征分布上看到的差异，支持对抗模型的高 AUC 并非单纯的模型评价错误。
但特征匿名，不能仅凭数值给它命名为“时间”。

## 对后续建模的含义

1. Kaggle 页面说明测试集晚于训练集；本次结果量化了训练/测试分布变化。随机分层
   划分仍可作为初始基线，但它可能高估对未来贷款的泛化能力。
2. 下一步检查训练集原始行序是否有时间信号。如果有，再设计靠后的留出集或滚动验证，
   并把这种验证结果与随机分层结果并排比较。
3. 将高漂移特征作为消融实验的候选，不按对抗验证 gain 排名直接删除；它们可能同时
   含有真实的违约预测信息。
4. 测试集全空的 7 列无法在最终预测时提供数值，可在还款模型中比较“保留”与“删除”
   两种方案。
5. 本项目不把测试集原始特征用于估计填充值、监督特征筛选或还款模型训练。

## 代码怎么复现

在项目根目录执行：

```powershell
.\\.venv\\Scripts\\python.exe .\\scripts\\run_adversarial_validation.py
```

脚本会在 `reports/adversarial_validation/` 生成折指标、特征重要性、图和本报告。

## 面试解释

> 我把训练集与测试集的来源当作分类目标，使用等量抽样和分层交叉验证训练一个
> LightGBM 对抗分类器。全特征 AUC 接近 {full_auc:.6f}，去除测试集全空的 7 列后
> 接近 {ablated_auc:.6f}，仅使用缺失模式为 {mask_auc:.4f}。我还核对了
> `{signal_check['feature']}` 的原始分布，单特征 AUC 为 {signal_check['observed_only_auc']:.4f}。这说明训练和测试数据
> 存在可识别的分布变化。我因此将随机验证作为基线，同时计划评估更接近时间外推的
> 留出方案，并通过消融实验检查高漂移特征，而不直接删除它们。

运行时间：{elapsed_seconds:.1f} 秒。
"""
    (output / "adversarial_validation_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-source", type=int, default=36_324)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.folds < 2 or args.sample_per_source < args.folds * 2:
        raise ValueError("Need at least two folds and enough samples per source")
    if not DATABASE.is_file():
        raise FileNotFoundError(DATABASE)

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        connection.execute("SET threads = 4")
        metadata = load_metadata(connection)
        feature_names = metadata["feature"].tolist()
        values, domain_label = load_balanced_arrays(
            connection,
            feature_names,
            SEED,
            args.sample_per_source,
        )

    print(
        f"Loaded {len(domain_label):,} rows x {values.shape[1]:,} features",
        flush=True,
    )
    keep_index = np.flatnonzero(~metadata["test_all_null"].to_numpy(dtype=bool))
    keep_names = [feature_names[index] for index in keep_index]
    if len(keep_names) != 1098:
        raise RuntimeError("Expected 1,098 features after excluding seven test-all-null")
    folds = list(
        StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=SEED
        ).split(np.zeros(len(domain_label)), domain_label)
    )

    metric_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    variants = [
        ("all_values", values, feature_names),
        ("exclude_test_all_null", values[:, keep_index].copy(), keep_names),
        (
            "missing_mask_only",
            np.isnan(values[:, keep_index]).astype(np.uint8),
            keep_names,
        ),
    ]
    for variant, variant_values, column_names in variants:
        metrics, gains, score = run_variant(
            variant_values,
            domain_label,
            column_names,
            variant,
            folds,
            args.threads,
        )
        metric_frames.append(metrics)
        importance_frames.append(gains)
        oof_frames.append(
            pd.DataFrame({
                "variant": variant,
                "source": np.where(domain_label == 0, "train", "test"),
                "domain_label": domain_label,
                "oof_test_probability": score,
            })
        )
        del variant_values
        gc.collect()
    metrics = pd.concat(metric_frames, ignore_index=True)
    gains = pd.concat(importance_frames, ignore_index=True)
    importance = (
        gains.groupby(["variant", "feature"], as_index=False)["gain_share"]
        .agg(mean_gain_share="mean", sd_gain_share="std")
        .merge(metadata, on="feature", how="left", validate="many_to_one")
        .sort_values(["variant", "mean_gain_share"], ascending=[True, False])
    )
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    importance.to_csv(args.output / "feature_importance.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(
        args.output / "oof_domain_predictions.csv", index=False
    )
    leading_feature = importance.loc[
        importance["variant"] == "exclude_test_all_null", "feature"
    ].iloc[0]
    signal_check = check_leading_signal(
        values, domain_label, feature_names, leading_feature
    )
    (args.output / "signal_check.json").write_text(
        json.dumps(signal_check, indent=2), encoding="utf-8"
    )
    plot_results(
        metrics, importance, values, domain_label, feature_names,
        signal_check, args.output,
    )
    del values, variants
    gc.collect()
    elapsed = time.perf_counter() - started
    write_report(
        metrics,
        importance,
        metadata,
        signal_check,
        args.sample_per_source,
        args.output,
        elapsed,
    )
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "folds": args.folds,
                "threads": args.threads,
                "sample_per_source": args.sample_per_source,
                "feature_count_all": len(feature_names),
                "feature_count_excluding_test_all_null": len(keep_names),
                "lightgbm_version": lgb.__version__,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Report: {args.output / 'adversarial_validation_report.md'}")
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

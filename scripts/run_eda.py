"""Analyze label balance, missingness patterns, and feature distributions.

The expensive statistics are computed from the processed Parquet views in
DuckDB. Charts use deterministic samples so the workflow remains practical on
the 688,998 x 1,105 training matrix.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "interim" / "loan_repayment.duckdb"
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "eda"
FIGURE_DIRECTORY = OUTPUT_DIRECTORY / "figures"

EXPECTED_TRAIN_ROWS = 688_998
EXPECTED_TEST_ROWS = 36_324
EXPECTED_FEATURES = 1_105
MISSING_BATCH_SIZE = 80
SAMPLE_PER_LABEL = 20_000
TEST_SAMPLE_SIZE = 30_000
RANDOM_SEED = 20260922


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def format_percent(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n.a."
    return f"{float(value):.2%}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def feature_sort_key(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def load_feature_quality(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    quality = connection.execute(
        "SELECT * FROM dq_feature_quality ORDER BY "
        "TRY_CAST(REGEXP_EXTRACT(feature, '[0-9]+$') AS INTEGER)"
    ).fetchdf()
    if len(quality) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURES:,} features, found {len(quality):,}"
        )
    return quality


def label_distribution(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    result = connection.execute(
        """
        WITH counts AS (
            SELECT label, COUNT(*)::BIGINT AS row_count
            FROM processed_train
            GROUP BY label
        )
        SELECT
            label,
            row_count,
            row_count::DOUBLE / SUM(row_count) OVER () AS row_share
        FROM counts
        ORDER BY label
        """
    ).fetchdf()
    if result["row_count"].sum() != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("Label counts do not reconcile to the training rows")
    if set(result["label"].tolist()) != {0, 1}:
        raise RuntimeError("Training label must contain exactly 0 and 1")
    return result


def missingness_by_label(
    connection: duckdb.DuckDBPyConnection,
    features: list[str],
    quality: pd.DataFrame,
    class_counts: dict[int, int],
) -> pd.DataFrame:
    print("Computing feature missingness by label...", flush=True)
    records: list[dict[str, object]] = []
    for start in range(0, len(features), MISSING_BATCH_SIZE):
        batch = features[start : start + MISSING_BATCH_SIZE]
        expressions: list[str] = []
        positions: list[tuple[str, int]] = []
        for feature in batch:
            column = quote_identifier(feature)
            for label in (0, 1):
                expressions.append(
                    "SUM(CASE WHEN label = "
                    f"{label} AND {column} IS NULL THEN 1 ELSE 0 END)::BIGINT"
                )
                positions.append((feature, label))

        values = connection.execute(
            "SELECT " + ", ".join(expressions) + " FROM processed_train"
        ).fetchone()
        batch_counts: dict[str, dict[int, int]] = {
            feature: {} for feature in batch
        }
        for (feature, label), value in zip(positions, values, strict=True):
            batch_counts[feature][label] = int(value or 0)

        for feature in batch:
            missing_0 = batch_counts[feature][0]
            missing_1 = batch_counts[feature][1]
            rate_0 = missing_0 / class_counts[0]
            rate_1 = missing_1 / class_counts[1]
            records.append(
                {
                    "feature": feature,
                    "label_0_missing_count": missing_0,
                    "label_0_missing_rate": rate_0,
                    "label_1_missing_count": missing_1,
                    "label_1_missing_rate": rate_1,
                    "missing_rate_gap_label_1_minus_0": rate_1 - rate_0,
                    "abs_missing_rate_gap": abs(rate_1 - rate_0),
                }
            )
        print(
            f"  {min(start + MISSING_BATCH_SIZE, len(features)):,}/"
            f"{len(features):,} features",
            flush=True,
        )

    by_label = pd.DataFrame(records)
    columns = [
        "feature",
        "train_missing_count",
        "train_missing_rate",
        "test_missing_count",
        "test_missing_rate",
    ]
    result = quality[columns].merge(by_label, on="feature", validate="one_to_one")
    if not np.all(
        result["train_missing_count"].to_numpy()
        == (
            result["label_0_missing_count"]
            + result["label_1_missing_count"]
        ).to_numpy()
    ):
        raise RuntimeError("Per-label missing counts do not reconcile")
    return result.sort_values("feature", key=lambda s: s.map(feature_sort_key))


def null_count_expression(features: list[str]) -> str:
    chunk_expressions: list[str] = []
    for start in range(0, len(features), 40):
        batch = features[start : start + 40]
        terms = [
            f"CAST({quote_identifier(feature)} IS NULL AS INTEGER)"
            for feature in batch
        ]
        chunk_expressions.append("(" + " + ".join(terms) + ")")
    return " + ".join(chunk_expressions)


def row_missingness_distribution(
    connection: duckdb.DuckDBPyConnection,
    features: list[str],
) -> pd.DataFrame:
    print("Computing row-level missing-count distributions...", flush=True)
    missing_expression = null_count_expression(features)
    train = connection.execute(
        f"""
        SELECT
            'train' AS dataset,
            label,
            missing_count,
            COUNT(*)::BIGINT AS row_count
        FROM (
            SELECT label, {missing_expression} AS missing_count
            FROM processed_train
        )
        GROUP BY label, missing_count
        ORDER BY label, missing_count
        """
    ).fetchdf()
    test = connection.execute(
        f"""
        SELECT
            'test' AS dataset,
            NULL::INTEGER AS label,
            missing_count,
            COUNT(*)::BIGINT AS row_count
        FROM (
            SELECT {missing_expression} AS missing_count
            FROM processed_test
        )
        GROUP BY missing_count
        ORDER BY missing_count
        """
    ).fetchdf()
    result = pd.concat([train, test], ignore_index=True)
    train_total = int(train["row_count"].sum())
    test_total = int(test["row_count"].sum())
    if train_total != EXPECTED_TRAIN_ROWS or test_total != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Row-missingness histograms do not reconcile to dataset rows"
        )
    if not result["missing_count"].between(0, len(features)).all():
        raise RuntimeError("Invalid per-row missing count")
    return result


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    threshold = probability * weights.sum()
    index = int(np.searchsorted(np.cumsum(weights), threshold, side="left"))
    return float(values[min(index, len(values) - 1)])


def summarize_row_missingness(histogram: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, label), group in histogram.groupby(
        ["dataset", "label"], dropna=False
    ):
        values = group["missing_count"].to_numpy(dtype=float)
        weights = group["row_count"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "label": "all" if pd.isna(label) else int(label),
                "row_count": int(weights.sum()),
                "mean_missing_features": float(np.average(values, weights=weights)),
                "median_missing_features": weighted_quantile(values, weights, 0.50),
                "p90_missing_features": weighted_quantile(values, weights, 0.90),
                "p99_missing_features": weighted_quantile(values, weights, 0.99),
                "max_missing_features": int(values.max()),
                "complete_row_rate": float(
                    weights[values == 0].sum() / weights.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_distribution_summary(
    quality: pd.DataFrame,
    missing_by_label: pd.DataFrame,
) -> pd.DataFrame:
    result = quality.copy()
    result["train_iqr"] = result["train_q75"] - result["train_q25"]
    result["test_iqr"] = result["test_q75"] - result["test_q25"]
    valid_iqr = result["train_iqr"].abs() > 1e-15
    result["train_bowley_skewness"] = np.nan
    result.loc[valid_iqr, "train_bowley_skewness"] = (
        result.loc[valid_iqr, "train_q75"]
        + result.loc[valid_iqr, "train_q25"]
        - 2 * result.loc[valid_iqr, "train_median"]
    ) / result.loc[valid_iqr, "train_iqr"]
    result["abs_train_bowley_skewness"] = result[
        "train_bowley_skewness"
    ].abs()
    result["cardinality_group"] = pd.cut(
        result["train_approx_unique"],
        bins=[-np.inf, 2, 10, 100, 1000, np.inf],
        labels=["<=2", "3-10", "11-100", "101-1000", ">1000"],
    ).astype(str)
    result["missingness_group"] = pd.cut(
        result["train_missing_rate"],
        bins=[-1e-12, 0, 0.01, 0.10, 0.50, 0.80, 1.0],
        labels=["0%", "(0,1%]", "(1,10%]", "(10,50%]", "(50,80%]", "(80,100%]"],
        include_lowest=True,
    ).astype(str)
    result["skewness_group"] = pd.cut(
        result["abs_train_bowley_skewness"],
        bins=[-np.inf, 0.15, 0.35, 0.60, np.inf],
        labels=["roughly symmetric", "mild", "strong", "very strong"],
    ).astype(str)
    result.loc[~valid_iqr, "skewness_group"] = "zero IQR / unavailable"
    result = result.merge(
        missing_by_label[
            [
                "feature",
                "label_0_missing_rate",
                "label_1_missing_rate",
                "missing_rate_gap_label_1_minus_0",
                "abs_missing_rate_gap",
            ]
        ],
        on="feature",
        validate="one_to_one",
    )
    return result.sort_values("feature", key=lambda s: s.map(feature_sort_key))


def select_representative_features(
    distribution: pd.DataFrame,
) -> pd.DataFrame:
    candidates: list[tuple[str, str]] = []

    def add(reason: str, frame: pd.DataFrame, sort_column: str, ascending: bool) -> None:
        remaining = frame[~frame["feature"].isin([item[1] for item in candidates])]
        if remaining.empty:
            return
        feature = str(
            remaining.sort_values(sort_column, ascending=ascending).iloc[0]["feature"]
        )
        candidates.append((reason, feature))

    usable = distribution[
        ~distribution["train_all_null"] & ~distribution["test_all_null"]
    ]
    add("highest train missing rate", usable, "train_missing_rate", False)
    add("largest label missing-rate gap", usable, "abs_missing_rate_gap", False)
    add("highest 5xIQR extreme-value rate", usable, "train_extreme_rate", False)
    skewed = usable[
        (usable["train_missing_rate"] < 0.50)
        & (usable["train_approx_unique"] > 100)
    ]
    add("strong robust skewness", skewed, "abs_train_bowley_skewness", False)
    dense = usable[
        (usable["train_missing_rate"] <= 0.01)
        & (usable["train_approx_unique"] > 1000)
    ].copy()
    dense["dense_score"] = (
        dense["abs_train_bowley_skewness"].fillna(1.0)
        + dense["train_extreme_rate"].fillna(1.0)
    )
    add("dense and comparatively regular", dense, "dense_score", True)
    discrete = usable[usable["train_approx_unique"].between(3, 20)]
    add("low-cardinality numeric feature", discrete, "train_approx_unique", True)

    selected = pd.DataFrame(candidates, columns=["selection_reason", "feature"])
    selected = selected.merge(distribution, on="feature", how="left")
    return selected


def deterministic_samples(
    connection: duckdb.DuckDBPyConnection,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ", ".join(quote_identifier(name) for name in selected_features)
    train = connection.execute(
        f"""
        SELECT label, {columns}
        FROM (
            SELECT id, label, {columns}
            FROM processed_train
            WHERE label = 0
            ORDER BY hash(id, {RANDOM_SEED})
            LIMIT {SAMPLE_PER_LABEL}
        )
        UNION ALL
        SELECT label, {columns}
        FROM (
            SELECT id, label, {columns}
            FROM processed_train
            WHERE label = 1
            ORDER BY hash(id, {RANDOM_SEED})
            LIMIT {SAMPLE_PER_LABEL}
        )
        """
    ).fetchdf()
    test = connection.execute(
        f"""
        SELECT {columns}
        FROM processed_test
        ORDER BY hash(id, {RANDOM_SEED})
        LIMIT {TEST_SAMPLE_SIZE}
        """
    ).fetchdf()
    return train, test


def standardized_clipped(
    series: pd.Series, median: float, iqr: float
) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    scale = iqr if math.isfinite(iqr) and abs(iqr) > 1e-15 else 1.0
    values = (values - median) / scale
    return np.clip(values[np.isfinite(values)], -5, 5)


def plot_overview(
    labels: pd.DataFrame,
    missing: pd.DataFrame,
    row_histogram: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    label_names = ["Default (0)", "Repaid (1)"]
    colors = ["#C44E52", "#4C72B0"]
    axes[0, 0].bar(label_names, labels["row_count"], color=colors)
    axes[0, 0].set_title("Training label distribution")
    axes[0, 0].set_ylabel("Rows")
    for index, row in labels.reset_index(drop=True).iterrows():
        axes[0, 0].text(
            index,
            row["row_count"],
            f"{int(row['row_count']):,}\n{row['row_share']:.2%}",
            ha="center",
            va="bottom",
        )

    bins = np.linspace(0, 1, 21)
    axes[0, 1].hist(
        missing["train_missing_rate"], bins=bins, alpha=0.65, label="train"
    )
    axes[0, 1].hist(
        missing["test_missing_rate"], bins=bins, alpha=0.65, label="test"
    )
    axes[0, 1].set_title("Feature missing-rate distribution")
    axes[0, 1].set_xlabel("Missing rate")
    axes[0, 1].set_ylabel("Feature count")
    axes[0, 1].legend()

    top = missing.nlargest(15, "train_missing_rate").sort_values(
        "train_missing_rate"
    )
    y = np.arange(len(top))
    axes[1, 0].barh(y - 0.18, top["train_missing_rate"], 0.36, label="train")
    axes[1, 0].barh(y + 0.18, top["test_missing_rate"], 0.36, label="test")
    axes[1, 0].set_yticks(y, top["feature"])
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_title("Features with highest train missingness")
    axes[1, 0].set_xlabel("Missing rate")
    axes[1, 0].legend()

    for (dataset, label), group in row_histogram.groupby(
        ["dataset", "label"], dropna=False
    ):
        group = group.sort_values("missing_count")
        total = group["row_count"].sum()
        if dataset == "test":
            name = "test"
        else:
            name = f"train label={int(label)}"
        axes[1, 1].plot(
            group["missing_count"],
            group["row_count"] / total,
            label=name,
            linewidth=1.8,
        )
    axes[1, 1].set_title("Missing features per row")
    axes[1, 1].set_xlabel("Number of missing features")
    axes[1, 1].set_ylabel("Share of rows")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "eda_overview.png", dpi=160)
    plt.close(figure)


def plot_selected_distributions(
    train_sample: pd.DataFrame,
    test_sample: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    feature_count = len(selected)
    figure, axes = plt.subplots(feature_count, 2, figsize=(14, 3.2 * feature_count))
    if feature_count == 1:
        axes = np.array([axes])

    for row_index, row in selected.reset_index(drop=True).iterrows():
        feature = row["feature"]
        median = float(row["train_median"])
        iqr = float(row["train_iqr"])
        label_0 = standardized_clipped(
            train_sample.loc[train_sample["label"] == 0, feature], median, iqr
        )
        label_1 = standardized_clipped(
            train_sample.loc[train_sample["label"] == 1, feature], median, iqr
        )
        test = standardized_clipped(test_sample[feature], median, iqr)
        all_train = np.concatenate([label_0, label_1])
        bins = np.linspace(-5, 5, 51)

        axes[row_index, 0].hist(
            label_0,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label="default (0)",
            color="#C44E52",
        )
        axes[row_index, 0].hist(
            label_1,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label="repaid (1)",
            color="#4C72B0",
        )
        axes[row_index, 0].set_title(
            f"{feature}: {row['selection_reason']} — by label"
        )
        axes[row_index, 0].set_xlabel("Robust standardized value, clipped to [-5, 5]")
        axes[row_index, 0].set_ylabel("Density")
        axes[row_index, 0].legend()

        axes[row_index, 1].hist(
            all_train,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label="train sample",
        )
        axes[row_index, 1].hist(
            test,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label="test sample",
        )
        axes[row_index, 1].set_title(f"{feature} — train vs test")
        axes[row_index, 1].set_xlabel("Robust standardized value, clipped to [-5, 5]")
        axes[row_index, 1].set_ylabel("Density")
        axes[row_index, 1].legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIRECTORY / "selected_feature_distributions.png", dpi=160
    )
    plt.close(figure)


def write_report(
    labels: pd.DataFrame,
    missing: pd.DataFrame,
    row_summary: pd.DataFrame,
    distribution: pd.DataFrame,
    selected: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    total_train_cells = EXPECTED_TRAIN_ROWS * EXPECTED_FEATURES
    total_test_cells = EXPECTED_TEST_ROWS * EXPECTED_FEATURES
    train_cell_missing_rate = missing["train_missing_count"].sum() / total_train_cells
    test_cell_missing_rate = missing["test_missing_count"].sum() / total_test_cells
    default_row = labels.loc[labels["label"] == 0].iloc[0]
    repaid_row = labels.loc[labels["label"] == 1].iloc[0]
    imbalance_ratio = repaid_row["row_count"] / default_row["row_count"]

    label_rows = [
        [
            int(row.label),
            "default" if int(row.label) == 0 else "repaid",
            f"{int(row.row_count):,}",
            format_percent(row.row_share),
        ]
        for row in labels.itertuples(index=False)
    ]
    row_missing_rows = [
        [
            row.dataset,
            row.label,
            f"{int(row.row_count):,}",
            f"{row.mean_missing_features:.1f}",
            f"{row.median_missing_features:.0f}",
            f"{row.p90_missing_features:.0f}",
            f"{row.p99_missing_features:.0f}",
            f"{row.max_missing_features:,}",
            format_percent(row.complete_row_rate),
        ]
        for row in row_summary.itertuples(index=False)
    ]
    top_missing = missing.nlargest(10, "train_missing_rate")
    top_missing_rows = [
        [
            row.feature,
            format_percent(row.train_missing_rate),
            format_percent(row.test_missing_rate),
        ]
        for row in top_missing.itertuples(index=False)
    ]
    top_label_gap = missing.nlargest(10, "abs_missing_rate_gap")
    top_label_gap_rows = [
        [
            row.feature,
            format_percent(row.label_0_missing_rate),
            format_percent(row.label_1_missing_rate),
            f"{row.missing_rate_gap_label_1_minus_0:+.2%}",
        ]
        for row in top_label_gap.itertuples(index=False)
    ]

    missing_bucket_order = [
        "0%",
        "(0,1%]",
        "(1,10%]",
        "(10,50%]",
        "(50,80%]",
        "(80,100%]",
    ]
    missing_bucket_counts = (
        distribution["missingness_group"].value_counts().reindex(missing_bucket_order, fill_value=0)
    )
    cardinality_order = ["<=2", "3-10", "11-100", "101-1000", ">1000"]
    cardinality_counts = (
        distribution["cardinality_group"].value_counts().reindex(cardinality_order, fill_value=0)
    )
    skew_order = [
        "roughly symmetric",
        "mild",
        "strong",
        "very strong",
        "zero IQR / unavailable",
    ]
    skew_counts = distribution["skewness_group"].value_counts().reindex(
        skew_order, fill_value=0
    )

    selected_rows = [
        [
            row.feature,
            row.selection_reason,
            format_percent(row.train_missing_rate),
            f"{row.train_approx_unique:,}",
            f"{row.train_bowley_skewness:.3f}"
            if math.isfinite(row.train_bowley_skewness)
            else "n.a.",
            format_percent(row.train_extreme_rate),
        ]
        for row in selected.itertuples(index=False)
    ]

    report = f"""# EDA: label balance, missingness, and feature distributions

This analysis uses the full processed Parquet datasets for counts and summary
statistics. Distribution charts use deterministic samples only for rendering.

## 1. Label balance

{markdown_table(['Label', 'Meaning', 'Rows', 'Share'], label_rows)}

- The positive class is repayment (`label = 1`), not default.
- Default represents {default_row['row_share']:.2%} of training rows.
- Repaid-to-default imbalance is approximately {imbalance_ratio:.2f}:1.
- Accuracy would be misleading: an all-repaid classifier would already achieve
  {repaid_row['row_share']:.2%} accuracy. ROC-AUC remains appropriate for the
  competition, while PR-AUC and recall at fixed false-positive rates should be
  retained as diagnostic metrics.

## 2. Missingness patterns

- Overall missing-cell rate: train {train_cell_missing_rate:.2%}, test
  {test_cell_missing_rate:.2%}.
- Features with any training missingness:
  {int((missing['train_missing_rate'] > 0).sum()):,} of {EXPECTED_FEATURES:,}.
- Features with at least 50% training missingness:
  {int((missing['train_missing_rate'] >= 0.50).sum()):,}.
- Features entirely missing in test:
  {int((missing['test_missing_rate'] == 1).sum()):,}.
- Features whose missing rate differs by at least 10 percentage points between
  default and repaid rows:
  {int((missing['abs_missing_rate_gap'] >= 0.10).sum()):,}.

### Missing features per row

{markdown_table(['Dataset', 'Label', 'Rows', 'Mean', 'Median', 'P90', 'P99', 'Max', 'Complete-row rate'], row_missing_rows)}

### Highest training missing rates

{markdown_table(['Feature', 'Train missing', 'Test missing'], top_missing_rows)}

### Missingness most associated with the label

The gap is `repaid missing rate - default missing rate`; it describes
association, not causation.

{markdown_table(['Feature', 'Default missing', 'Repaid missing', 'Gap'], top_label_gap_rows)}

Recommended handling for the first baseline:

- Keep missing values for LightGBM's native missing-value routing.
- Add missing indicators first for features with a material label-specific gap.
- Remove or separately ablate the seven features that are entirely missing in
  test, because they cannot provide test-time values.
- Fit imputers inside each validation fold. Never compute fill values from the
  combined train and test data.

## 3. Feature distributions

### Training missingness groups

{markdown_table(['Missing-rate group', 'Features'], [[key, f'{int(value):,}'] for key, value in missing_bucket_counts.items()])}

### Approximate cardinality groups

{markdown_table(['Approximate unique values', 'Features'], [[key, f'{int(value):,}'] for key, value in cardinality_counts.items()])}

### Robust skewness groups

Bowley skewness is calculated from Q25, median, and Q75, so it is less sensitive
to extreme values than moment skewness.

{markdown_table(['Absolute Bowley skewness', 'Features'], [[key, f'{int(value):,}'] for key, value in skew_counts.items()])}

- Features with at least one 5xIQR extreme-value flag:
  {int((distribution['train_extreme_count'] > 0).sum()):,}.
- Features with at least a 5% extreme-value rate:
  {int((distribution['train_extreme_rate'] >= 0.05).sum()):,}.
- Constant training features: {int(distribution['train_constant'].sum()):,}.
- Near-constant training features: {int(distribution['train_near_constant'].sum()):,}.

The features are numerically scaled by unknown random factors, so comparing raw
ranges across columns is not meaningful. Tree models are suitable for the first
baseline because they do not require global standardization and can handle
nonlinear, skewed distributions. Do not apply blanket winsorization or log
transforms; test those choices through cross-validation.

### Representative features used in charts

{markdown_table(['Feature', 'Selection reason', 'Train missing', 'Approx. unique', 'Bowley skew', '5xIQR extreme rate'], selected_rows)}

## 4. Validation implications

1. Use stratification so every fold preserves the roughly {default_row['row_share']:.2%}
   default rate.
2. Report ROC-AUC as the competition metric, plus PR-AUC and KS for diagnostic
   interpretation.
3. Treat missing indicators as candidate features because missingness itself is
   associated with the target for some columns.
4. Compare three feature sets: all usable features, removal of test-all-null and
   very-high-missing features, and removal of strongly shifted features.
5. Because no explicit timestamp is available, do not call a random split
   "time validation." The next step should test whether row order contains a
   temporal signal and run adversarial validation against the test set.

## Outputs

- `label_distribution.csv`
- `missingness_by_feature.csv`
- `row_missingness_distribution.csv`
- `row_missingness_summary.csv`
- `feature_distribution_summary.csv`
- `selected_feature_statistics.csv`
- `figures/eda_overview.png`
- `figures/selected_feature_distributions.png`

Total runtime: {elapsed_seconds:.1f} seconds.
"""
    (OUTPUT_DIRECTORY / "eda_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing DuckDB database: {DATABASE_PATH}. Run init_duckdb.py first."
        )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DATABASE_PATH), read_only=True) as connection:
        connection.execute("SET max_expression_depth = 10000")
        connection.execute("SET preserve_insertion_order = false")
        quality = load_feature_quality(connection)
        features = sorted(quality["feature"].tolist(), key=feature_sort_key)
        labels = label_distribution(connection)
        class_counts = {
            int(row.label): int(row.row_count)
            for row in labels.itertuples(index=False)
        }
        missing = missingness_by_label(
            connection, features, quality, class_counts
        )
        row_histogram = row_missingness_distribution(connection, features)
        distribution = build_distribution_summary(quality, missing)
        selected = select_representative_features(distribution)
        train_sample, test_sample = deterministic_samples(
            connection, selected["feature"].tolist()
        )

    row_summary = summarize_row_missingness(row_histogram)

    labels.to_csv(OUTPUT_DIRECTORY / "label_distribution.csv", index=False)
    missing.to_csv(OUTPUT_DIRECTORY / "missingness_by_feature.csv", index=False)
    row_histogram.to_csv(
        OUTPUT_DIRECTORY / "row_missingness_distribution.csv", index=False
    )
    row_summary.to_csv(
        OUTPUT_DIRECTORY / "row_missingness_summary.csv", index=False
    )
    distribution.to_csv(
        OUTPUT_DIRECTORY / "feature_distribution_summary.csv", index=False
    )
    selected.to_csv(
        OUTPUT_DIRECTORY / "selected_feature_statistics.csv", index=False
    )

    plot_overview(labels, missing, row_histogram)
    plot_selected_distributions(train_sample, test_sample, selected)
    elapsed = time.perf_counter() - started
    write_report(
        labels, missing, row_summary, distribution, selected, elapsed
    )

    print(f"EDA report: {OUTPUT_DIRECTORY / 'eda_report.md'}")
    print(f"EDA tables: {OUTPUT_DIRECTORY}")
    print(f"EDA figures: {FIGURE_DIRECTORY}")
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

"""Profile the processed loan data and quantify train-test feature drift."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "interim" / "loan_repayment.duckdb"
TRAIN_PARQUET = PROJECT_ROOT / "data" / "processed" / "train.parquet"
TEST_PARQUET = PROJECT_ROOT / "data" / "processed" / "test.parquet"
DATASET_SQL_PATH = (
    PROJECT_ROOT / "sql" / "02_data_quality" / "01_dataset_summary.sql"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "tables" / "data_quality"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality_report.md"

NEAR_CONSTANT_SHARE = 0.995
EXTREME_IQR_MULTIPLIER = 5.0
PSI_MODERATE = 0.10
PSI_HIGH = 0.25
SMD_MODERATE = 0.20
SMD_HIGH = 0.50
MISSING_DIFF_MODERATE = 0.05
MISSING_DIFF_HIGH = 0.10
BATCH_SIZE = 20


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite SQL threshold: {value}")
    return repr(float(value))


def numeric(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_view(
    connection: duckdb.DuckDBPyConnection,
    view_name: str,
    features: list[str],
) -> pd.DataFrame:
    started = time.perf_counter()
    print(f"Profiling {view_name}...", flush=True)
    numeric_projection = ", ".join(
        f"TRY_CAST({quote_identifier(feature)} AS DOUBLE) "
        f"AS {quote_identifier(feature)}"
        for feature in features
    )
    frame = connection.execute(
        f"SUMMARIZE SELECT {numeric_projection} FROM {view_name}"
    ).fetchdf()
    for column in [
        "approx_unique",
        "avg",
        "std",
        "q25",
        "q50",
        "q75",
        "count",
        "null_percentage",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["min_numeric"] = pd.to_numeric(frame["min"], errors="coerce")
    frame["max_numeric"] = pd.to_numeric(frame["max"], errors="coerce")
    print(
        f"  {len(frame):,} columns profiled in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return frame


def parquet_null_counts(
    connection: duckdb.DuckDBPyConnection, parquet_path: Path
) -> dict[str, int]:
    escaped = parquet_path.resolve().as_posix().replace("'", "''")
    rows = connection.execute(
        f"""
        SELECT path_in_schema, COALESCE(SUM(stats_null_count), 0)::BIGINT
        FROM parquet_metadata('{escaped}')
        GROUP BY path_in_schema
        """
    ).fetchall()
    return {column: int(count) for column, count in rows}


def profile_map(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        str(row["column_name"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def feature_names(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [
            str(name)
            for name in frame["column_name"]
            if str(name).startswith("feature_")
        ],
        key=lambda name: int(name.split("_")[1]),
    )


def is_same_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def compute_feature_counts(
    connection: duckdb.DuckDBPyConnection,
    view_name: str,
    features: list[str],
    own_profiles: dict[str, dict[str, object]],
    train_profiles: dict[str, dict[str, object]],
) -> dict[str, dict[str, int]]:
    results = {feature: {} for feature in features}
    print(f"Computing quality and PSI bins for {view_name}...", flush=True)
    started = time.perf_counter()

    for start in range(0, len(features), BATCH_SIZE):
        batch = features[start : start + BATCH_SIZE]
        expressions: list[str] = []
        positions: list[tuple[str, str]] = []

        for index, feature in enumerate(batch):
            column = quote_identifier(feature)
            numeric_column = f"TRY_CAST({column} AS DOUBLE)"
            own = own_profiles[feature]
            train = train_profiles[feature]

            expressions.append(
                f"SUM(CASE WHEN {column} IS NOT NULL AND "
                f"{numeric_column} IS NULL THEN 1 ELSE 0 END)"
            )
            positions.append((feature, "cast_failure_count"))

            expressions.append(
                f"SUM(CASE WHEN {numeric_column} IS NOT NULL "
                f"AND NOT isfinite({numeric_column}) THEN 1 ELSE 0 END)"
            )
            positions.append((feature, "non_finite_count"))

            q25 = numeric(own.get("q25"))
            q75 = numeric(own.get("q75"))
            if q25 is not None and q75 is not None and q75 > q25:
                lower = q25 - EXTREME_IQR_MULTIPLIER * (q75 - q25)
                upper = q75 + EXTREME_IQR_MULTIPLIER * (q75 - q25)
                expressions.append(
                    f"SUM(CASE WHEN isfinite({numeric_column}) AND "
                    f"({numeric_column} < {sql_number(lower)} OR "
                    f"{numeric_column} > {sql_number(upper)}) "
                    f"THEN 1 ELSE 0 END)"
                )
            else:
                expressions.append("0::BIGINT")
            positions.append((feature, "extreme_count"))

            q50 = numeric(own.get("q50"))
            if q25 is not None and q75 is not None and q50 is not None and is_same_number(q25, q75):
                expressions.append(
                    f"SUM(CASE WHEN {numeric_column} = {sql_number(q50)} "
                    f"THEN 1 ELSE 0 END)"
                )
            else:
                expressions.append("0::BIGINT")
            positions.append((feature, "median_value_count"))

            train_q25 = numeric(train.get("q25"))
            train_q50 = numeric(train.get("q50"))
            train_q75 = numeric(train.get("q75"))
            if None not in (train_q25, train_q50, train_q75):
                bins = [
                    f"{numeric_column} <= {sql_number(train_q25)}",
                    f"{numeric_column} > {sql_number(train_q25)} AND "
                    f"{numeric_column} <= {sql_number(train_q50)}",
                    f"{numeric_column} > {sql_number(train_q50)} AND "
                    f"{numeric_column} <= {sql_number(train_q75)}",
                    f"{numeric_column} > {sql_number(train_q75)}",
                ]
                for bin_index, condition in enumerate(bins):
                    expressions.append(
                        f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END)"
                    )
                    positions.append((feature, f"psi_bin_{bin_index}"))
            else:
                for bin_index in range(4):
                    expressions.append("0::BIGINT")
                    positions.append((feature, f"psi_bin_{bin_index}"))

        values = connection.execute(
            f"SELECT {', '.join(expressions)} FROM {view_name}"
        ).fetchone()
        for (feature, metric), value in zip(positions, values, strict=True):
            results[feature][metric] = int(value or 0)

        completed = min(start + BATCH_SIZE, len(features))
        print(f"  {completed:,}/{len(features):,} features", flush=True)

    print(
        f"  Finished {view_name} counts in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return results


def psi_from_counts(train_counts: list[int], test_counts: list[int]) -> float:
    train_total = sum(train_counts)
    test_total = sum(test_counts)
    if train_total == 0 or test_total == 0:
        return math.nan
    epsilon = 1e-6
    psi = 0.0
    for train_count, test_count in zip(train_counts, test_counts, strict=True):
        train_share = max(train_count / train_total, epsilon)
        test_share = max(test_count / test_total, epsilon)
        psi += (test_share - train_share) * math.log(test_share / train_share)
    return psi


def drift_level(psi: float, abs_smd: float, missing_diff: float) -> str:
    if (
        (math.isfinite(psi) and psi >= PSI_HIGH)
        or (math.isfinite(abs_smd) and abs_smd >= SMD_HIGH)
        or missing_diff >= MISSING_DIFF_HIGH
    ):
        return "high"
    if (
        (math.isfinite(psi) and psi >= PSI_MODERATE)
        or (math.isfinite(abs_smd) and abs_smd >= SMD_MODERATE)
        or missing_diff >= MISSING_DIFF_MODERATE
    ):
        return "moderate"
    return "low"


def build_feature_tables(
    features: list[str],
    train_profiles: dict[str, dict[str, object]],
    test_profiles: dict[str, dict[str, object]],
    train_nulls: dict[str, int],
    test_nulls: dict[str, int],
    train_counts: dict[str, dict[str, int]],
    test_counts: dict[str, dict[str, int]],
    train_types: dict[str, str],
    test_types: dict[str, str],
    train_rows: int,
    test_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []

    for feature in features:
        train = train_profiles[feature]
        test = test_profiles[feature]
        train_null_count = train_nulls.get(feature, 0)
        test_null_count = test_nulls.get(feature, 0)
        train_non_null = train_rows - train_null_count
        test_non_null = test_rows - test_null_count

        train_min = numeric(train.get("min_numeric"))
        train_max = numeric(train.get("max_numeric"))
        test_min = numeric(test.get("min_numeric"))
        test_max = numeric(test.get("max_numeric"))
        train_constant = (
            train_non_null > 0 and is_same_number(train_min, train_max)
        )
        test_constant = test_non_null > 0 and is_same_number(test_min, test_max)

        train_dominant_share = (
            train_counts[feature]["median_value_count"] / train_non_null
            if train_non_null
            else math.nan
        )
        test_dominant_share = (
            test_counts[feature]["median_value_count"] / test_non_null
            if test_non_null
            else math.nan
        )
        train_near_constant = (
            not train_constant
            and math.isfinite(train_dominant_share)
            and train_dominant_share >= NEAR_CONSTANT_SHARE
        )
        test_near_constant = (
            not test_constant
            and math.isfinite(test_dominant_share)
            and test_dominant_share >= NEAR_CONSTANT_SHARE
        )

        train_extreme_count = train_counts[feature]["extreme_count"]
        test_extreme_count = test_counts[feature]["extreme_count"]
        train_q25 = numeric(train.get("q25"))
        train_q75 = numeric(train.get("q75"))
        test_q25 = numeric(test.get("q25"))
        test_q75 = numeric(test.get("q75"))
        train_iqr = (
            train_q75 - train_q25
            if train_q25 is not None and train_q75 is not None
            else math.nan
        )
        test_iqr = (
            test_q75 - test_q25
            if test_q25 is not None and test_q75 is not None
            else math.nan
        )
        quality_rows.append(
            {
                "feature": feature,
                "train_type": train_types[feature],
                "test_type": test_types[feature],
                "type_match": train_types[feature] == test_types[feature],
                "type_mismatch_due_to_all_null_test": (
                    train_types[feature] != test_types[feature]
                    and test_non_null == 0
                ),
                "train_missing_count": train_null_count,
                "train_missing_rate": train_null_count / train_rows,
                "test_missing_count": test_null_count,
                "test_missing_rate": test_null_count / test_rows,
                "train_approx_unique": int(train["approx_unique"]),
                "test_approx_unique": int(test["approx_unique"]),
                "train_min": train_min,
                "train_q25": numeric(train.get("q25")),
                "train_median": numeric(train.get("q50")),
                "train_q75": numeric(train.get("q75")),
                "train_max": train_max,
                "train_mean": numeric(train.get("avg")),
                "train_std": numeric(train.get("std")),
                "test_min": test_min,
                "test_q25": numeric(test.get("q25")),
                "test_median": numeric(test.get("q50")),
                "test_q75": numeric(test.get("q75")),
                "test_max": test_max,
                "test_mean": numeric(test.get("avg")),
                "test_std": numeric(test.get("std")),
                "train_all_null": train_non_null == 0,
                "test_all_null": test_non_null == 0,
                "train_constant": train_constant,
                "test_constant": test_constant,
                "train_dominant_share": train_dominant_share,
                "test_dominant_share": test_dominant_share,
                "train_near_constant": train_near_constant,
                "test_near_constant": test_near_constant,
                "train_non_finite_count": train_counts[feature][
                    "non_finite_count"
                ],
                "test_non_finite_count": test_counts[feature][
                    "non_finite_count"
                ],
                "train_numeric_cast_failure_count": train_counts[feature][
                    "cast_failure_count"
                ],
                "test_numeric_cast_failure_count": test_counts[feature][
                    "cast_failure_count"
                ],
                "train_extreme_count": train_extreme_count,
                "train_extreme_lower_bound": (
                    train_q25 - EXTREME_IQR_MULTIPLIER * train_iqr
                    if math.isfinite(train_iqr) and train_iqr > 0
                    else math.nan
                ),
                "train_extreme_upper_bound": (
                    train_q75 + EXTREME_IQR_MULTIPLIER * train_iqr
                    if math.isfinite(train_iqr) and train_iqr > 0
                    else math.nan
                ),
                "train_extreme_rate": (
                    train_extreme_count / train_non_null
                    if train_non_null
                    else math.nan
                ),
                "test_extreme_count": test_extreme_count,
                "test_extreme_lower_bound": (
                    test_q25 - EXTREME_IQR_MULTIPLIER * test_iqr
                    if math.isfinite(test_iqr) and test_iqr > 0
                    else math.nan
                ),
                "test_extreme_upper_bound": (
                    test_q75 + EXTREME_IQR_MULTIPLIER * test_iqr
                    if math.isfinite(test_iqr) and test_iqr > 0
                    else math.nan
                ),
                "test_extreme_rate": (
                    test_extreme_count / test_non_null
                    if test_non_null
                    else math.nan
                ),
            }
        )

        train_bins = [
            train_counts[feature][f"psi_bin_{index}"] for index in range(4)
        ]
        test_bins = [
            test_counts[feature][f"psi_bin_{index}"] for index in range(4)
        ]
        psi = psi_from_counts(train_bins, test_bins)
        train_mean = numeric(train.get("avg"))
        test_mean = numeric(test.get("avg"))
        train_std = numeric(train.get("std"))
        test_std = numeric(test.get("std"))
        if None in (train_mean, test_mean, train_std, test_std):
            smd = math.nan
        else:
            pooled_std = math.sqrt((train_std**2 + test_std**2) / 2)
            if pooled_std == 0:
                smd = 0.0 if is_same_number(train_mean, test_mean) else math.inf
            else:
                smd = (test_mean - train_mean) / pooled_std
        missing_diff = abs(test_null_count / test_rows - train_null_count / train_rows)
        drift_rows.append(
            {
                "feature": feature,
                "psi_quartile": psi,
                "standardized_mean_difference": smd,
                "abs_standardized_mean_difference": abs(smd),
                "missing_rate_difference": missing_diff,
                "train_missing_rate": train_null_count / train_rows,
                "test_missing_rate": test_null_count / test_rows,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "drift_level": drift_level(psi, abs(smd), missing_diff),
            }
        )

    return pd.DataFrame(quality_rows), pd.DataFrame(drift_rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    dataset_summary: pd.DataFrame,
    quality: pd.DataFrame,
    drift: pd.DataFrame,
    elapsed_seconds: float,
) -> None:
    high_drift = drift[drift["drift_level"] == "high"].sort_values(
        "psi_quartile", ascending=False
    )
    moderate_drift = drift[drift["drift_level"] == "moderate"]
    top_missing = quality.sort_values("train_missing_rate", ascending=False).head(10)
    top_extreme = quality.sort_values("train_extreme_rate", ascending=False).head(10)
    type_mismatch_count = int((~quality["type_match"]).sum())
    all_null_type_mismatch_count = int(
        quality["type_mismatch_due_to_all_null_test"].sum()
    )

    def format_percent(value: float) -> str:
        return f"{value:.2%}" if math.isfinite(value) else "n.a."

    dataset_rows = [
        [
            row.dataset,
            f"{int(row.row_count):,}",
            f"{int(row.column_count):,}",
            f"{int(row.duplicate_id_rows):,}",
            f"{int(row.null_id_count):,}",
        ]
        for row in dataset_summary.itertuples(index=False)
    ]
    drift_rows = [
        [
            row.feature,
            f"{row.psi_quartile:.4f}",
            f"{row.abs_standardized_mean_difference:.4f}",
            f"{row.missing_rate_difference:.2%}",
        ]
        for row in high_drift.head(15).itertuples(index=False)
    ]
    missing_rows = [
        [row.feature, f"{row.train_missing_rate:.2%}", f"{row.test_missing_rate:.2%}"]
        for row in top_missing.itertuples(index=False)
    ]
    extreme_rows = [
        [
            row.feature,
            format_percent(row.train_extreme_rate),
            format_percent(row.test_extreme_rate),
        ]
        for row in top_extreme.itertuples(index=False)
    ]

    report = f"""# Data quality report

Generated from the validated Parquet datasets using DuckDB.

## Dataset checks

{markdown_table(['Dataset', 'Rows', 'Columns', 'Duplicate ID rows', 'Null IDs'], dataset_rows)}

## Feature issue summary

- Feature count: {len(quality):,}
- Train all-null features: {int(quality['train_all_null'].sum()):,}
- Test all-null features: {int(quality['test_all_null'].sum()):,}
- Train constant features: {int(quality['train_constant'].sum()):,}
- Train near-constant features: {int(quality['train_near_constant'].sum()):,}
- Raw type mismatches: {type_mismatch_count:,} ({all_null_type_mismatch_count:,} caused by all-null test columns)
- Features with numeric cast failures: {int(((quality['train_numeric_cast_failure_count'] > 0) | (quality['test_numeric_cast_failure_count'] > 0)).sum()):,}
- Features with non-finite values: {int(((quality['train_non_finite_count'] > 0) | (quality['test_non_finite_count'] > 0)).sum()):,}
- Features with at least one train extreme-value flag: {int((quality['train_extreme_count'] > 0).sum()):,}
- Features with at least 50% train missingness: {int((quality['train_missing_rate'] >= 0.50).sum()):,}
- Features with at least 50% test missingness: {int((quality['test_missing_rate'] >= 0.50).sum()):,}
- High-drift features: {len(high_drift):,}
- Moderate-drift features: {len(moderate_drift):,}

Near-constant means at least {NEAR_CONSTANT_SHARE:.1%} of non-null values equal the median. Extreme values use the {EXTREME_IQR_MULTIPLIER:g}×IQR rule. These are investigation flags, not automatic deletion rules.

## Highest train missing rates

{markdown_table(['Feature', 'Train missing', 'Test missing'], missing_rows)}

## Highest extreme-value rates

{markdown_table(['Feature', 'Train extreme', 'Test extreme'], extreme_rows)}

## Highest drift features

{markdown_table(['Feature', 'Quartile PSI', '|SMD|', 'Missing-rate difference'], drift_rows) if drift_rows else 'No high-drift features were detected.'}

Drift levels use PSI thresholds of {PSI_MODERATE:.2f}/{PSI_HIGH:.2f}, absolute standardized mean difference thresholds of {SMD_MODERATE:.2f}/{SMD_HIGH:.2f}, and missing-rate differences of {MISSING_DIFF_MODERATE:.0%}/{MISSING_DIFF_HIGH:.0%}.

Total runtime: {elapsed_seconds:.1f} seconds.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def persist_outputs(
    connection: duckdb.DuckDBPyConnection,
    schema: pd.DataFrame,
    quality: pd.DataFrame,
    drift: pd.DataFrame,
) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    schema.to_csv(OUTPUT_DIRECTORY / "schema.csv", index=False)
    quality.to_csv(OUTPUT_DIRECTORY / "feature_quality.csv", index=False)
    drift.to_csv(OUTPUT_DIRECTORY / "train_test_drift.csv", index=False)

    for table_name, frame in {
        "dq_schema": schema,
        "dq_feature_quality": quality,
        "dq_train_test_drift": drift,
    }.items():
        registration = f"_{table_name}_frame"
        connection.register(registration, frame)
        connection.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {registration}"
        )
        connection.unregister(registration)


def main() -> None:
    started = time.perf_counter()
    for path in [DATABASE_PATH, TRAIN_PARQUET, TEST_PARQUET, DATASET_SQL_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(DATASET_SQL_PATH.read_text(encoding="utf-8"))
        dataset_summary = connection.table("dq_dataset_summary").fetchdf()
        train_rows = int(
            dataset_summary.loc[
                dataset_summary["dataset"] == "train", "row_count"
            ].iloc[0]
        )
        test_rows = int(
            dataset_summary.loc[
                dataset_summary["dataset"] == "test", "row_count"
            ].iloc[0]
        )

        train_schema = connection.execute("DESCRIBE processed_train").fetchdf()
        train_schema.insert(0, "dataset", "train")
        test_schema = connection.execute("DESCRIBE processed_test").fetchdf()
        test_schema.insert(0, "dataset", "test")
        schema = pd.concat([train_schema, test_schema], ignore_index=True)
        features = feature_names(train_schema)
        train_types = dict(
            zip(train_schema["column_name"], train_schema["column_type"], strict=True)
        )
        test_types = dict(
            zip(test_schema["column_name"], test_schema["column_type"], strict=True)
        )

        train_summary = summarize_view(connection, "processed_train", features)
        test_summary = summarize_view(connection, "processed_test", features)
        train_profiles = profile_map(train_summary)
        test_profiles = profile_map(test_summary)

        train_nulls = parquet_null_counts(connection, TRAIN_PARQUET)
        test_nulls = parquet_null_counts(connection, TEST_PARQUET)
        train_counts = compute_feature_counts(
            connection,
            "processed_train",
            features,
            train_profiles,
            train_profiles,
        )
        test_counts = compute_feature_counts(
            connection,
            "processed_test",
            features,
            test_profiles,
            train_profiles,
        )
        quality, drift = build_feature_tables(
            features,
            train_profiles,
            test_profiles,
            train_nulls,
            test_nulls,
            train_counts,
            test_counts,
            train_types,
            test_types,
            train_rows,
            test_rows,
        )

        persist_outputs(connection, schema, quality, drift)
        dataset_summary.to_csv(
            OUTPUT_DIRECTORY / "dataset_summary.csv", index=False
        )

    elapsed_seconds = time.perf_counter() - started
    write_report(dataset_summary, quality, drift, elapsed_seconds)
    summary = {
        "runtime_seconds": round(elapsed_seconds, 2),
        "features": len(quality),
        "train_constant": int(quality["train_constant"].sum()),
        "train_near_constant": int(quality["train_near_constant"].sum()),
        "high_drift": int((drift["drift_level"] == "high").sum()),
        "moderate_drift": int((drift["drift_level"] == "moderate").sum()),
    }
    (OUTPUT_DIRECTORY / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()

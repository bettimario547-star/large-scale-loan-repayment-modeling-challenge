"""Generate SQL-based data-quality summaries and feature statistics exports."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "interim" / "loan_repayment.duckdb"
SQL_DIRECTORY = PROJECT_ROOT / "sql" / "02_data_quality"
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "tables" / "data_quality"
REPORT_PATH = PROJECT_ROOT / "reports" / "sql_data_quality_report.md"

FEATURE_SQL_PATH = SQL_DIRECTORY / "02_create_feature_statistics.sql"
REPORT_SQL_PATH = SQL_DIRECTORY / "03_create_quality_report_tables.sql"


def markdown_table(frame: pd.DataFrame) -> str:
    """Convert a small DataFrame to a Markdown table without extra packages."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("n.a.")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Format selected feature metrics for the human-readable report."""
    formatted = frame.copy()
    rate_columns = [
        "train_missing_rate",
        "test_missing_rate",
        "missing_rate_difference",
        "train_extreme_rate",
        "test_extreme_rate",
    ]
    for column in rate_columns:
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: (
                    f"{value:.2%}"
                    if value is not None and math.isfinite(value)
                    else "n.a."
                )
            )
    for column in ["psi_quartile", "abs_standardized_mean_difference"]:
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: (
                    f"{value:.4f}"
                    if value is not None and math.isfinite(value)
                    else "n.a."
                )
            )
    return formatted


def write_report(
    overview: pd.DataFrame,
    issues: pd.DataFrame,
    top_priority: pd.DataFrame,
) -> None:
    """Write the SQL query results as a concise portfolio-ready report."""
    overview_display = overview[["section", "metric", "metric_value", "definition"]].copy()
    overview_display["metric_value"] = overview_display["metric_value"].map(
        lambda value: f"{int(value):,}"
    )

    issue_display = issues[["issue", "feature_count", "rule", "next_step"]]
    priority_columns = [
        "feature",
        "review_priority",
        "issue_count",
        "train_missing_rate",
        "test_missing_rate",
        "psi_quartile",
        "abs_standardized_mean_difference",
        "drift_level",
        "recommended_action",
    ]
    priority_display = format_feature_table(top_priority[priority_columns])

    report = f"""# SQL data quality report

This report is generated from DuckDB tables created by the data-quality pipeline. SQL joins the feature-quality and train/test-drift results into one auditable feature-level table.

## Dataset and quality overview

{markdown_table(overview_display)}

## Issue summary

{markdown_table(issue_display)}

The issue counts are investigation flags. A feature may appear in several issue categories, so the counts should not be added together.

## Highest-priority feature review

{markdown_table(priority_display)}

## Rules used

- Near constant: one value represents at least 99.5% of non-null observations.
- Extreme value: outside Q25 - 5 x IQR or Q75 + 5 x IQR.
- High missingness: at least 50% missing in train or test.
- Large missingness shift: at least 10 percentage points between train and test.
- PSI: moderate at 0.10 and high at 0.25.
- Absolute SMD: moderate at 0.20 and high at 0.50.

These rules identify features for investigation. They do not automatically remove rows or columns. Feature handling decisions must be validated through modeling experiments.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_report_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """Fail early if SQL report tables are incomplete or internally inconsistent."""
    validation = connection.execute(
        """
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT feature) AS distinct_feature_count,
            COUNT(*) FILTER (WHERE feature IS NULL) AS null_feature_count,
            COUNT(*) FILTER (
                WHERE review_priority NOT IN ('high', 'medium', 'low')
                   OR review_priority IS NULL
            ) AS invalid_priority_count,
            COUNT(*) FILTER (WHERE recommended_action IS NULL)
                AS null_action_count
        FROM dq_feature_statistics
        """
    ).fetchone()
    (
        row_count,
        distinct_feature_count,
        null_feature_count,
        invalid_priority_count,
        null_action_count,
    ) = validation

    expected_feature_count = connection.execute(
        "SELECT COUNT(*) FROM dq_feature_quality"
    ).fetchone()[0]
    if row_count != expected_feature_count:
        raise RuntimeError(
            "Feature-statistics row count does not match dq_feature_quality: "
            f"{row_count:,} != {expected_feature_count:,}"
        )
    if distinct_feature_count != row_count:
        raise RuntimeError("dq_feature_statistics contains duplicate feature names")
    if any(
        count != 0
        for count in (
            null_feature_count,
            invalid_priority_count,
            null_action_count,
        )
    ):
        raise RuntimeError(
            "SQL report validation failed: null feature, invalid priority, "
            "or missing recommended action detected"
        )


def main() -> None:
    required_paths = [DATABASE_PATH, FEATURE_SQL_PATH, REPORT_SQL_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Required files are missing:\n" + "\n".join(missing_paths)
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        required_tables = {
            "dq_dataset_summary",
            "dq_feature_quality",
            "dq_train_test_drift",
        }
        available_tables = {
            row[0] for row in connection.execute("SHOW TABLES").fetchall()
        }
        missing_tables = sorted(required_tables - available_tables)
        if missing_tables:
            raise RuntimeError(
                "Run scripts/run_data_quality.py first. Missing DuckDB tables: "
                + ", ".join(missing_tables)
            )

        connection.execute(FEATURE_SQL_PATH.read_text(encoding="utf-8"))
        connection.execute(REPORT_SQL_PATH.read_text(encoding="utf-8"))
        validate_report_tables(connection)

        exports = {
            "sql_feature_statistics.csv": "dq_feature_statistics",
            "sql_quality_overview.csv": "dq_quality_overview",
            "sql_quality_issue_summary.csv": "dq_quality_issue_summary",
            "sql_priority_features.csv": "dq_priority_features",
        }
        for filename, table_name in exports.items():
            output_path = OUTPUT_DIRECTORY / filename
            connection.execute(
                f"COPY {table_name} TO ? (HEADER, DELIMITER ',')",
                [str(output_path)],
            )

        overview = connection.execute(
            "SELECT * FROM dq_quality_overview ORDER BY display_order"
        ).fetchdf()
        issues = connection.execute(
            "SELECT * FROM dq_quality_issue_summary ORDER BY display_order"
        ).fetchdf()
        top_priority = connection.execute(
            """
            SELECT *
            FROM dq_priority_features
            ORDER BY
                CASE review_priority WHEN 'high' THEN 1 ELSE 2 END,
                issue_count DESC,
                psi_quartile DESC NULLS LAST
            LIMIT 20
            """
        ).fetchdf()

    write_report(overview, issues, top_priority)
    print(f"Feature statistics: {OUTPUT_DIRECTORY / 'sql_feature_statistics.csv'}")
    print(f"Issue summary: {OUTPUT_DIRECTORY / 'sql_quality_issue_summary.csv'}")
    print(f"Priority features: {OUTPUT_DIRECTORY / 'sql_priority_features.csv'}")
    print(f"Markdown report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

"""Create DuckDB views that query the Kaggle CSV files in place."""

from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "interim" / "loan_repayment.duckdb"
SQL_PATH = PROJECT_ROOT / "sql" / "01_ingestion" / "01_create_raw_views.sql"

CSV_PATHS = {
    "__TRAIN_CSV__": PROJECT_ROOT / "train.csv",
    "__TEST_CSV__": PROJECT_ROOT / "test.csv",
    "__SAMPLE_SUBMISSION_CSV__": PROJECT_ROOT / "sample_submission.csv",
}


def sql_path(path: Path) -> str:
    """Return an absolute path safely escaped for a DuckDB SQL string."""
    return path.resolve().as_posix().replace("'", "''")


def main() -> None:
    missing = [str(path) for path in CSV_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required CSV files: {missing}")

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_directory = PROJECT_ROOT / "data" / "interim" / "duckdb_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)

    sql = SQL_PATH.read_text(encoding="utf-8")
    for placeholder, path in CSV_PATHS.items():
        sql = sql.replace(placeholder, sql_path(path))

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            f"SET temp_directory = '{sql_path(temp_directory)}'"
        )
        connection.execute(sql)

        views = connection.execute(
            """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'main'
              AND table_name LIKE 'raw_%'
            ORDER BY table_name
            """
        ).fetchall()

        column_counts = {
            view_name: connection.execute(
                f"SELECT COUNT(*) FROM pragma_table_info('{view_name}')"
            ).fetchone()[0]
            for (view_name,) in views
        }

        preview = connection.execute(
            """
            SELECT id, feature_1, feature_2, feature_3, label
            FROM raw_train
            LIMIT 3
            """
        ).fetchall()

    print(f"DuckDB database: {DATABASE_PATH}")
    print("Created views:")
    for view_name, column_count in column_counts.items():
        print(f"  - {view_name}: {column_count} columns")
    print("Training data preview:")
    for row in preview:
        print(f"  {row}")


if __name__ == "__main__":
    main()

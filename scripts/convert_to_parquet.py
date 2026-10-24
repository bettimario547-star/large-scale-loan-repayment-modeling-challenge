"""Convert the Kaggle CSV files to validated ZSTD-compressed Parquet files."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "interim" / "loan_repayment.duckdb"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = OUTPUT_DIRECTORY / "parquet_manifest.json"


@dataclass(frozen=True)
class Dataset:
    name: str
    source_view: str
    source_csv: Path
    target_parquet: Path
    expected_rows: int
    expected_columns: int


DATASETS = {
    "train": Dataset(
        name="train",
        source_view="raw_train",
        source_csv=PROJECT_ROOT / "train.csv",
        target_parquet=OUTPUT_DIRECTORY / "train.parquet",
        expected_rows=688_998,
        expected_columns=1_107,
    ),
    "test": Dataset(
        name="test",
        source_view="raw_test",
        source_csv=PROJECT_ROOT / "test.csv",
        target_parquet=OUTPUT_DIRECTORY / "test.parquet",
        expected_rows=36_324,
        expected_columns=1_106,
    ),
    "sample_submission": Dataset(
        name="sample_submission",
        source_view="raw_sample_submission",
        source_csv=PROJECT_ROOT / "sample_submission.csv",
        target_parquet=OUTPUT_DIRECTORY / "sample_submission.parquet",
        expected_rows=36_324,
        expected_columns=2,
    ),
}


def sql_path(path: Path) -> str:
    """Return an absolute path safely escaped for a DuckDB SQL string."""
    return path.resolve().as_posix().replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to convert. Defaults to all datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing validated Parquet file.",
    )
    return parser.parse_args()


def require_inputs(datasets: list[Dataset]) -> None:
    missing = [str(item.source_csv) for item in datasets if not item.source_csv.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required CSV files: {missing}")
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            "DuckDB database not found. Run scripts/init_duckdb.py first."
        )


def parquet_dimensions(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> tuple[int, int]:
    escaped_path = sql_path(path)
    row_count = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet('{escaped_path}')"
    ).fetchone()[0]
    column_count = len(
        connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{escaped_path}')"
        ).fetchall()
    )
    return row_count, column_count


def validate_parquet(
    connection: duckdb.DuckDBPyConnection, dataset: Dataset, path: Path
) -> tuple[int, int]:
    row_count, column_count = parquet_dimensions(connection, path)
    if row_count != dataset.expected_rows:
        raise ValueError(
            f"{dataset.name}: expected {dataset.expected_rows:,} rows, "
            f"found {row_count:,}."
        )
    if column_count != dataset.expected_columns:
        raise ValueError(
            f"{dataset.name}: expected {dataset.expected_columns:,} columns, "
            f"found {column_count:,}."
        )
    return row_count, column_count


def convert_dataset(
    connection: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    force: bool,
) -> dict[str, object]:
    target = dataset.target_parquet
    temporary = target.with_suffix(".parquet.tmp")

    if target.exists() and not force:
        print(f"Validating existing {target.name}...", flush=True)
        rows, columns = validate_parquet(connection, dataset, target)
        elapsed_seconds = 0.0
        status = "existing"
    else:
        if temporary.exists():
            temporary.unlink()

        print(
            f"Converting {dataset.source_csv.name} -> {target.name}...",
            flush=True,
        )
        started = time.perf_counter()
        connection.execute(
            f"""
            COPY (
                SELECT * FROM {dataset.source_view}
            )
            TO '{sql_path(temporary)}' (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                COMPRESSION_LEVEL 1,
                ROW_GROUP_SIZE 20000
            )
            """
        )
        rows, columns = validate_parquet(connection, dataset, temporary)
        temporary.replace(target)
        elapsed_seconds = time.perf_counter() - started
        status = "converted"

    csv_bytes = dataset.source_csv.stat().st_size
    parquet_bytes = target.stat().st_size
    result = {
        **asdict(dataset),
        "source_csv": str(dataset.source_csv),
        "target_parquet": str(target),
        "rows": rows,
        "columns": columns,
        "csv_bytes": csv_bytes,
        "parquet_bytes": parquet_bytes,
        "size_ratio": parquet_bytes / csv_bytes,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "status": status,
    }
    print(
        f"  {rows:,} rows, {columns:,} columns, "
        f"{parquet_bytes / 1024**3:.2f} GB, "
        f"{result['size_ratio']:.1%} of CSV size, "
        f"{elapsed_seconds:.1f}s",
        flush=True,
    )
    return result


def create_processed_views(
    connection: duckdb.DuckDBPyConnection, datasets: list[Dataset]
) -> None:
    for dataset in datasets:
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW processed_{dataset.name} AS
            SELECT * FROM read_parquet('{sql_path(dataset.target_parquet)}')
            """
        )


def main() -> None:
    args = parse_args()
    datasets = [DATASETS[name] for name in args.datasets]
    require_inputs(datasets)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        connection.execute("SET preserve_insertion_order = true")
        connection.execute("PRAGMA disable_progress_bar")

        results = [
            convert_dataset(connection, dataset, args.force) for dataset in datasets
        ]
        create_processed_views(connection, datasets)

    manifest = {
        "format": "parquet",
        "compression": "zstd",
        "compression_level": 1,
        "row_group_size": 20_000,
        "datasets": results,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()

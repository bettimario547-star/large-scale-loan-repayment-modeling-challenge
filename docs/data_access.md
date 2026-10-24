# DuckDB data access

The project uses DuckDB views to query the original Kaggle CSV files without
loading the entire dataset into Python memory.

## Initialize the database

Run this command from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\scripts\init_duckdb.py
```

This creates the ignored local database:

```text
data/interim/loan_repayment.duckdb
```

The database contains three views:

- `raw_train`
- `raw_test`
- `raw_sample_submission`

The views point to the original CSV files. They do not copy the full data into
the DuckDB database.

## Query from Python

```python
import duckdb

connection = duckdb.connect("data/interim/loan_repayment.duckdb")

preview = connection.sql(
    """
    SELECT id, feature_1, feature_2, feature_3, label
    FROM raw_train
    LIMIT 10
    """
).df()

print(preview)
```

Example SQL is available in `sql/01_ingestion/02_example_queries.sql`.
Queries such as `COUNT(*)`, full missing-value summaries, and label counts scan
the complete 11.9 GB training file and will take longer than preview queries.

## Convert CSV files to Parquet

Run this command from the project root:

```powershell
.\.venv\Scripts\python.exe .\scripts\convert_to_parquet.py
```

The script writes temporary files first, verifies their row and column counts,
and only then replaces the final files in `data/processed/`. It uses ZSTD level
1 compression and 20,000-row row groups to keep memory use manageable for this
wide dataset.

After conversion, the DuckDB database contains these additional views:

- `processed_train`
- `processed_test`
- `processed_sample_submission`

Use `processed_*` views for repeated analysis because Parquet supports column
pruning and avoids repeatedly parsing CSV text.

To rebuild selected files:

```powershell
.\.venv\Scripts\python.exe .\scripts\convert_to_parquet.py --datasets train test --force
```

## Run data quality checks

```powershell
.\.venv\Scripts\python.exe .\scripts\run_data_quality.py
```

The check writes detailed tables to `reports/tables/data_quality/`, persists
queryable `dq_*` tables in DuckDB, and creates `reports/data_quality_report.md`.
It checks dimensions, duplicate IDs, missing values, feature types, constant and
near-constant columns, non-finite and extreme values, and train-test drift.

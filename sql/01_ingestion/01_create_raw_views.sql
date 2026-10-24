-- Raw views read the original Kaggle CSV files directly.
-- Paths are injected by scripts/init_duckdb.py.

CREATE OR REPLACE VIEW raw_train AS
SELECT *
FROM read_csv_auto(
    '__TRAIN_CSV__',
    header = true,
    sample_size = 20480,
    parallel = true,
    ignore_errors = false
);

CREATE OR REPLACE VIEW raw_test AS
SELECT *
FROM read_csv_auto(
    '__TEST_CSV__',
    header = true,
    sample_size = 20480,
    parallel = true,
    ignore_errors = false
);

CREATE OR REPLACE VIEW raw_sample_submission AS
SELECT *
FROM read_csv_auto(
    '__SAMPLE_SUBMISSION_CSV__',
    header = true,
    sample_size = 20480,
    parallel = true,
    ignore_errors = false
);

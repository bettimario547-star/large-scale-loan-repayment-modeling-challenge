-- Reference SQL used by scripts/convert_to_parquet.py.
-- The Python script writes to temporary paths first and validates each file.

COPY (
    SELECT * FROM raw_train
)
TO '__TRAIN_PARQUET__' (
    FORMAT PARQUET,
    COMPRESSION ZSTD,
    COMPRESSION_LEVEL 1,
    ROW_GROUP_SIZE 20000
);

COPY (
    SELECT * FROM raw_test
)
TO '__TEST_PARQUET__' (
    FORMAT PARQUET,
    COMPRESSION ZSTD,
    COMPRESSION_LEVEL 1,
    ROW_GROUP_SIZE 20000
);

COPY (
    SELECT * FROM raw_sample_submission
)
TO '__SAMPLE_SUBMISSION_PARQUET__' (
    FORMAT PARQUET,
    COMPRESSION ZSTD,
    COMPRESSION_LEVEL 1,
    ROW_GROUP_SIZE 20000
);

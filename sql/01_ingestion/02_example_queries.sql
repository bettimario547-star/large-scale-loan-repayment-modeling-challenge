-- Run the initialization script before using these queries.

-- Preview selected columns without loading the full training set into memory.
SELECT
    id,
    feature_1,
    feature_2,
    feature_3,
    label
FROM raw_train
LIMIT 10;

-- Inspect the detected schema.
DESCRIBE raw_train;

-- These queries scan the full training CSV and therefore take longer.
SELECT label, COUNT(*) AS loan_count
FROM raw_train
GROUP BY label
ORDER BY label;

SELECT COUNT(*) AS train_rows
FROM raw_train;

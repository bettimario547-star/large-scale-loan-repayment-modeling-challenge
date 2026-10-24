CREATE OR REPLACE TABLE dq_dataset_summary AS
WITH summary AS (
    SELECT
        'train' AS dataset,
        COUNT(*) AS row_count,
        COUNT(DISTINCT id) AS distinct_id_count,
        COUNT(*) FILTER (WHERE id IS NULL) AS null_id_count
    FROM processed_train

    UNION ALL

    SELECT
        'test' AS dataset,
        COUNT(*) AS row_count,
        COUNT(DISTINCT id) AS distinct_id_count,
        COUNT(*) FILTER (WHERE id IS NULL) AS null_id_count
    FROM processed_test

    UNION ALL

    SELECT
        'sample_submission' AS dataset,
        COUNT(*) AS row_count,
        COUNT(DISTINCT id) AS distinct_id_count,
        COUNT(*) FILTER (WHERE id IS NULL) AS null_id_count
    FROM processed_sample_submission
),
duplicate_groups AS (
    SELECT 'train' AS dataset, COUNT(*) AS duplicate_id_groups
    FROM (
        SELECT id FROM processed_train
        WHERE id IS NOT NULL
        GROUP BY id HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT 'test' AS dataset, COUNT(*) AS duplicate_id_groups
    FROM (
        SELECT id FROM processed_test
        WHERE id IS NOT NULL
        GROUP BY id HAVING COUNT(*) > 1
    )

    UNION ALL

    SELECT 'sample_submission' AS dataset, COUNT(*) AS duplicate_id_groups
    FROM (
        SELECT id FROM processed_sample_submission
        WHERE id IS NOT NULL
        GROUP BY id HAVING COUNT(*) > 1
    )
),
column_counts AS (
    SELECT 'train' AS dataset, COUNT(*) AS column_count
    FROM pragma_table_info('processed_train')

    UNION ALL

    SELECT 'test' AS dataset, COUNT(*) AS column_count
    FROM pragma_table_info('processed_test')

    UNION ALL

    SELECT 'sample_submission' AS dataset, COUNT(*) AS column_count
    FROM pragma_table_info('processed_sample_submission')
)
SELECT
    summary.dataset,
    summary.row_count,
    column_counts.column_count,
    summary.distinct_id_count,
    summary.null_id_count,
    summary.row_count - summary.null_id_count - summary.distinct_id_count
        AS duplicate_id_rows,
    duplicate_groups.duplicate_id_groups
FROM summary
JOIN duplicate_groups USING (dataset)
JOIN column_counts USING (dataset)
ORDER BY
    CASE summary.dataset
        WHEN 'train' THEN 1
        WHEN 'test' THEN 2
        ELSE 3
    END;

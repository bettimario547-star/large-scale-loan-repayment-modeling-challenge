-- Exact label balance for the binary repayment target.
-- label = 1: fully repaid; label = 0: default.

WITH label_counts AS (
    SELECT
        label,
        COUNT(*)::BIGINT AS row_count
    FROM processed_train
    GROUP BY label
)
SELECT
    label,
    row_count,
    row_count::DOUBLE / SUM(row_count) OVER () AS row_share
FROM label_counts
ORDER BY label;

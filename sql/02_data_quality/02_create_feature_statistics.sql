-- Build one feature-level table that combines data-quality statistics and
-- train/test drift metrics. The upstream dq_* tables are produced by
-- scripts/run_data_quality.py; this SQL layer is intended for analysis,
-- filtering, reporting, and portfolio demonstrations.

CREATE OR REPLACE TABLE dq_feature_statistics AS
WITH combined AS (
    SELECT
        quality.*,
        drift.psi_quartile,
        drift.standardized_mean_difference,
        drift.abs_standardized_mean_difference,
        drift.missing_rate_difference,
        drift.drift_level,
        GREATEST(quality.train_missing_rate, quality.test_missing_rate)
            AS max_missing_rate,
        GREATEST(
            COALESCE(quality.train_extreme_rate, 0.0),
            COALESCE(quality.test_extreme_rate, 0.0)
        ) AS max_extreme_rate
    FROM dq_feature_quality AS quality
    INNER JOIN dq_train_test_drift AS drift USING (feature)
),
flags AS (
    SELECT
        *,
        train_all_null OR test_all_null AS has_all_null_dataset,
        train_constant OR test_constant AS has_constant_distribution,
        train_near_constant OR test_near_constant
            AS has_near_constant_distribution,
        NOT type_match AS has_type_mismatch,
        train_numeric_cast_failure_count > 0
            OR test_numeric_cast_failure_count > 0 AS has_cast_failure,
        train_non_finite_count > 0 OR test_non_finite_count > 0
            AS has_non_finite_value,
        train_extreme_count > 0 OR test_extreme_count > 0
            AS has_extreme_value,
        max_extreme_rate >= 0.05 AS has_high_extreme_rate,
        max_missing_rate >= 0.50 AS has_high_missing_rate,
        missing_rate_difference >= 0.10 AS has_large_missing_rate_shift,
        drift_level IN ('moderate', 'high') AS has_distribution_drift
    FROM combined
)
SELECT
    *,
    CAST(has_all_null_dataset AS INTEGER)
        + CAST(has_constant_distribution AS INTEGER)
        + CAST(has_near_constant_distribution AS INTEGER)
        + CAST(has_type_mismatch AS INTEGER)
        + CAST(has_cast_failure AS INTEGER)
        + CAST(has_non_finite_value AS INTEGER)
        + CAST(has_high_extreme_rate AS INTEGER)
        + CAST(has_high_missing_rate AS INTEGER)
        + CAST(has_large_missing_rate_shift AS INTEGER)
        + CAST(has_distribution_drift AS INTEGER) AS issue_count,
    CASE
        WHEN train_all_null
            OR train_constant
            OR has_cast_failure
            OR has_non_finite_value
            OR drift_level = 'high'
            OR has_large_missing_rate_shift
            THEN 'high'
        WHEN test_all_null
            OR has_near_constant_distribution
            OR has_type_mismatch
            OR has_high_missing_rate
            OR has_high_extreme_rate
            OR drift_level = 'moderate'
            THEN 'medium'
        ELSE 'low'
    END AS review_priority,
    COALESCE(
        NULLIF(
            CONCAT_WS(
                '; ',
                CASE WHEN train_all_null THEN 'exclude: train is entirely null' END,
                CASE WHEN test_all_null THEN 'review: test is entirely null' END,
                CASE WHEN train_constant THEN 'exclude candidate: train is constant' END,
                CASE
                    WHEN train_near_constant
                        THEN 'review: train is near constant'
                END,
                CASE WHEN has_type_mismatch THEN 'review data type mismatch' END,
                CASE WHEN has_cast_failure THEN 'repair numeric cast failures' END,
                CASE WHEN has_non_finite_value THEN 'repair NaN or infinity values' END,
                CASE WHEN has_high_missing_rate THEN 'test missing-value strategy' END,
                CASE
                    WHEN has_large_missing_rate_shift
                        THEN 'review missingness drift'
                END,
                CASE
                    WHEN has_high_extreme_rate
                        THEN 'compare original and capped values'
                END,
                CASE
                    WHEN drift_level = 'high'
                        THEN 'run high-drift feature ablation'
                    WHEN drift_level = 'moderate'
                        THEN 'monitor feature drift'
                END
            ),
            ''
        ),
        'retain for baseline modeling'
    ) AS recommended_action
FROM flags
ORDER BY
    CASE review_priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
    issue_count DESC,
    TRY_CAST(REGEXP_EXTRACT(feature, '[0-9]+$') AS INTEGER),
    feature;

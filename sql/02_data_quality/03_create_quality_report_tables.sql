-- Create compact SQL tables used by the Markdown report and downstream EDA.
-- All counts come from dq_feature_statistics, so every headline number can be
-- traced back to a feature-level row.

CREATE OR REPLACE TABLE dq_quality_overview AS
SELECT 1 AS display_order, 'dataset' AS section,
       'train_rows' AS metric, CAST(row_count AS DOUBLE) AS metric_value,
       'Rows in processed training data' AS definition
FROM dq_dataset_summary WHERE dataset = 'train'
UNION ALL
SELECT 2, 'dataset', 'test_rows', CAST(row_count AS DOUBLE),
       'Rows in processed test data'
FROM dq_dataset_summary WHERE dataset = 'test'
UNION ALL
SELECT 3, 'dataset', 'feature_count', CAST(COUNT(*) AS DOUBLE),
       'Predictor columns shared by train and test'
FROM dq_feature_statistics
UNION ALL
SELECT 4, 'identity', 'train_duplicate_id_rows',
       CAST(duplicate_id_rows AS DOUBLE), 'Repeated non-null training IDs'
FROM dq_dataset_summary WHERE dataset = 'train'
UNION ALL
SELECT 5, 'identity', 'test_duplicate_id_rows',
       CAST(duplicate_id_rows AS DOUBLE), 'Repeated non-null test IDs'
FROM dq_dataset_summary WHERE dataset = 'test'
UNION ALL
SELECT 6, 'missingness', 'train_high_missing_features', CAST(COUNT(*) AS DOUBLE),
       'Features with at least 50% missing values in train'
FROM dq_feature_statistics WHERE train_missing_rate >= 0.50
UNION ALL
SELECT 7, 'missingness', 'test_high_missing_features', CAST(COUNT(*) AS DOUBLE),
       'Features with at least 50% missing values in test'
FROM dq_feature_statistics WHERE test_missing_rate >= 0.50
UNION ALL
SELECT 8, 'distribution', 'train_constant_features', CAST(COUNT(*) AS DOUBLE),
       'Features with one distinct non-null value in train'
FROM dq_feature_statistics WHERE train_constant
UNION ALL
SELECT 9, 'distribution', 'train_near_constant_features', CAST(COUNT(*) AS DOUBLE),
       'Features whose dominant train value share is at least 99.5%'
FROM dq_feature_statistics WHERE train_near_constant
UNION ALL
SELECT 10, 'validity', 'features_with_cast_failures', CAST(COUNT(*) AS DOUBLE),
       'Features containing non-null values that cannot convert to DOUBLE'
FROM dq_feature_statistics WHERE has_cast_failure
UNION ALL
SELECT 11, 'validity', 'features_with_non_finite_values', CAST(COUNT(*) AS DOUBLE),
       'Features containing NaN or infinity'
FROM dq_feature_statistics WHERE has_non_finite_value
UNION ALL
SELECT 12, 'outliers', 'features_with_extreme_values', CAST(COUNT(*) AS DOUBLE),
       'Features with at least one value outside the 5 x IQR limits'
FROM dq_feature_statistics WHERE has_extreme_value
UNION ALL
SELECT 13, 'drift', 'high_drift_features', CAST(COUNT(*) AS DOUBLE),
       'Features classified as high drift by PSI, SMD, or missingness shift'
FROM dq_feature_statistics WHERE drift_level = 'high'
UNION ALL
SELECT 14, 'drift', 'moderate_drift_features', CAST(COUNT(*) AS DOUBLE),
       'Features classified as moderate drift by PSI, SMD, or missingness shift'
FROM dq_feature_statistics WHERE drift_level = 'moderate'
UNION ALL
SELECT 15, 'review', 'high_priority_features', CAST(COUNT(*) AS DOUBLE),
       'Features requiring high-priority review before modeling'
FROM dq_feature_statistics WHERE review_priority = 'high'
UNION ALL
SELECT 16, 'review', 'medium_priority_features', CAST(COUNT(*) AS DOUBLE),
       'Features requiring medium-priority review before modeling'
FROM dq_feature_statistics WHERE review_priority = 'medium'
ORDER BY display_order;

CREATE OR REPLACE TABLE dq_quality_issue_summary AS
SELECT 1 AS display_order, 'All-null in either dataset' AS issue,
       COUNT(*) AS feature_count,
       'train_all_null OR test_all_null' AS rule,
       'Remove train-all-null fields; define a test-all-null handling rule' AS next_step
FROM dq_feature_statistics WHERE has_all_null_dataset
UNION ALL
SELECT 2, 'Constant in either dataset', COUNT(*),
       'one distinct non-null value',
       'Exclude train constants; inspect test-only constants'
FROM dq_feature_statistics WHERE has_constant_distribution
UNION ALL
SELECT 3, 'Near constant in either dataset', COUNT(*),
       'dominant non-null value share >= 99.5%',
       'Retain initially and test removal with cross-validation'
FROM dq_feature_statistics WHERE has_near_constant_distribution
UNION ALL
SELECT 4, 'Raw data-type mismatch', COUNT(*),
       'train_type <> test_type',
       'Confirm whether the mismatch is caused by an all-null column'
FROM dq_feature_statistics WHERE has_type_mismatch
UNION ALL
SELECT 5, 'Numeric cast failure', COUNT(*),
       'non-null value cannot convert to DOUBLE',
       'Locate and clean invalid text before modeling'
FROM dq_feature_statistics WHERE has_cast_failure
UNION ALL
SELECT 6, 'NaN or infinity', COUNT(*),
       'numeric value is not finite',
       'Replace or impute non-finite values'
FROM dq_feature_statistics WHERE has_non_finite_value
UNION ALL
SELECT 7, 'High missingness', COUNT(*),
       'maximum train/test missing rate >= 50%',
       'Compare native missing handling, imputation, and removal'
FROM dq_feature_statistics WHERE has_high_missing_rate
UNION ALL
SELECT 8, 'High extreme-value rate', COUNT(*),
       'maximum train/test 5 x IQR extreme rate >= 5%',
       'Compare original values with winsorized values'
FROM dq_feature_statistics WHERE has_high_extreme_rate
UNION ALL
SELECT 9, 'Large missingness shift', COUNT(*),
       'absolute train/test missing-rate difference >= 10%',
       'Investigate collection changes and test missing indicators'
FROM dq_feature_statistics WHERE has_large_missing_rate_shift
UNION ALL
SELECT 10, 'Moderate or high distribution drift', COUNT(*),
       'combined drift level is moderate or high',
       'Run feature ablation under a time-aware validation scheme'
FROM dq_feature_statistics WHERE has_distribution_drift
ORDER BY display_order;

CREATE OR REPLACE TABLE dq_priority_features AS
SELECT
    feature,
    review_priority,
    issue_count,
    recommended_action,
    train_type,
    test_type,
    train_missing_rate,
    test_missing_rate,
    missing_rate_difference,
    train_approx_unique,
    train_extreme_rate,
    test_extreme_rate,
    psi_quartile,
    abs_standardized_mean_difference,
    drift_level,
    train_all_null,
    test_all_null,
    train_constant,
    train_near_constant,
    has_cast_failure,
    has_non_finite_value
FROM dq_feature_statistics
WHERE review_priority IN ('high', 'medium')
ORDER BY
    CASE review_priority WHEN 'high' THEN 1 ELSE 2 END,
    issue_count DESC,
    psi_quartile DESC NULLS LAST,
    TRY_CAST(REGEXP_EXTRACT(feature, '[0-9]+$') AS INTEGER);

# SQL data quality report

This report is generated from DuckDB tables created by the data-quality pipeline. SQL joins the feature-quality and train/test-drift results into one auditable feature-level table.

## Dataset and quality overview

| section | metric | metric_value | definition |
| --- | --- | --- | --- |
| dataset | train_rows | 688,998 | Rows in processed training data |
| dataset | test_rows | 36,324 | Rows in processed test data |
| dataset | feature_count | 1,105 | Predictor columns shared by train and test |
| identity | train_duplicate_id_rows | 0 | Repeated non-null training IDs |
| identity | test_duplicate_id_rows | 0 | Repeated non-null test IDs |
| missingness | train_high_missing_features | 87 | Features with at least 50% missing values in train |
| missingness | test_high_missing_features | 77 | Features with at least 50% missing values in test |
| distribution | train_constant_features | 0 | Features with one distinct non-null value in train |
| distribution | train_near_constant_features | 0 | Features whose dominant train value share is at least 99.5% |
| validity | features_with_cast_failures | 0 | Features containing non-null values that cannot convert to DOUBLE |
| validity | features_with_non_finite_values | 0 | Features containing NaN or infinity |
| outliers | features_with_extreme_values | 991 | Features with at least one value outside the 5 x IQR limits |
| drift | high_drift_features | 541 | Features classified as high drift by PSI, SMD, or missingness shift |
| drift | moderate_drift_features | 271 | Features classified as moderate drift by PSI, SMD, or missingness shift |
| review | high_priority_features | 541 | Features requiring high-priority review before modeling |
| review | medium_priority_features | 328 | Features requiring medium-priority review before modeling |

## Issue summary

| issue | feature_count | rule | next_step |
| --- | --- | --- | --- |
| All-null in either dataset | 7 | train_all_null OR test_all_null | Remove train-all-null fields; define a test-all-null handling rule |
| Constant in either dataset | 0 | one distinct non-null value | Exclude train constants; inspect test-only constants |
| Near constant in either dataset | 1 | dominant non-null value share >= 99.5% | Retain initially and test removal with cross-validation |
| Raw data-type mismatch | 7 | train_type <> test_type | Confirm whether the mismatch is caused by an all-null column |
| Numeric cast failure | 0 | non-null value cannot convert to DOUBLE | Locate and clean invalid text before modeling |
| NaN or infinity | 0 | numeric value is not finite | Replace or impute non-finite values |
| High missingness | 120 | maximum train/test missing rate >= 50% | Compare native missing handling, imputation, and removal |
| High extreme-value rate | 251 | maximum train/test 5 x IQR extreme rate >= 5% | Compare original values with winsorized values |
| Large missingness shift | 196 | absolute train/test missing-rate difference >= 10% | Investigate collection changes and test missing indicators |
| Moderate or high distribution drift | 812 | combined drift level is moderate or high | Run feature ablation under a time-aware validation scheme |

The issue counts are investigation flags. A feature may appear in several issue categories, so the counts should not be added together.

## Highest-priority feature review

| feature | review_priority | issue_count | train_missing_rate | test_missing_rate | psi_quartile | abs_standardized_mean_difference | drift_level | recommended_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_614 | high | 6 | 66.93% | 100.00% | n.a. | n.a. | high | review: test is entirely null; review data type mismatch; test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_648 | high | 6 | 56.56% | 100.00% | n.a. | n.a. | high | review: test is entirely null; review data type mismatch; test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_660 | high | 6 | 66.93% | 100.00% | n.a. | n.a. | high | review: test is entirely null; review data type mismatch; test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_805 | high | 6 | 66.93% | 100.00% | n.a. | n.a. | high | review: test is entirely null; review data type mismatch; test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_1009 | high | 6 | 56.56% | 100.00% | n.a. | n.a. | high | review: test is entirely null; review data type mismatch; test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_1031 | high | 6 | 61.79% | 100.00% | n.a. | n.a. | high | review: test is entirely null; review data type mismatch; test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_364 | high | 5 | 56.56% | 100.00% | n.a. | n.a. | high | review: test is entirely null; review data type mismatch; test missing-value strategy; review missingness drift; run high-drift feature ablation |
| feature_1045 | high | 4 | 18.24% | 68.27% | 2.5802 | 1.5702 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_175 | high | 4 | 40.82% | 68.87% | 0.5233 | 0.0193 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_493 | high | 4 | 57.48% | 33.00% | 0.4921 | 0.3756 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_984 | high | 4 | 40.30% | 60.40% | 0.2710 | 0.0019 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_1040 | high | 4 | 75.19% | 63.96% | 0.2611 | 0.1390 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_297 | high | 4 | 57.48% | 33.00% | 0.2571 | 0.1302 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_715 | high | 4 | 37.84% | 58.32% | 0.2532 | 0.0152 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_773 | high | 4 | 52.35% | 62.82% | 0.2030 | 0.0084 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_219 | high | 4 | 55.30% | 36.72% | 0.2014 | 0.0086 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_947 | high | 4 | 55.27% | 66.74% | 0.1881 | 0.0080 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_414 | high | 4 | 66.40% | 42.77% | 0.1832 | 0.2428 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_907 | high | 4 | 29.40% | 63.42% | 0.1793 | 0.4169 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |
| feature_1039 | high | 4 | 82.12% | 1.76% | 0.1678 | 0.3342 | high | test missing-value strategy; review missingness drift; compare original and capped values; run high-drift feature ablation |

## Rules used

- Near constant: one value represents at least 99.5% of non-null observations.
- Extreme value: outside Q25 - 5 x IQR or Q75 + 5 x IQR.
- High missingness: at least 50% missing in train or test.
- Large missingness shift: at least 10 percentage points between train and test.
- PSI: moderate at 0.10 and high at 0.25.
- Absolute SMD: moderate at 0.20 and high at 0.50.

These rules identify features for investigation. They do not automatically remove rows or columns. Feature handling decisions must be validated through modeling experiments.

# Data quality report

Generated from the validated Parquet datasets using DuckDB.

## Dataset checks

| Dataset | Rows | Columns | Duplicate ID rows | Null IDs |
| --- | --- | --- | --- | --- |
| train | 688,998 | 1,107 | 0 | 0 |
| test | 36,324 | 1,106 | 0 | 0 |
| sample_submission | 36,324 | 2 | 0 | 0 |

## Feature issue summary

- Feature count: 1,105
- Train all-null features: 0
- Test all-null features: 7
- Train constant features: 0
- Train near-constant features: 0
- Raw type mismatches: 7 (7 caused by all-null test columns)
- Features with numeric cast failures: 0
- Features with non-finite values: 0
- Features with at least one train extreme-value flag: 962
- Features with at least 50% train missingness: 87
- Features with at least 50% test missingness: 77
- High-drift features: 541
- Moderate-drift features: 271

Near-constant means at least 99.5% of non-null values equal the median. Extreme values use the 5×IQR rule. These are investigation flags, not automatic deletion rules.

## Highest train missing rates

| Feature | Train missing | Test missing |
| --- | --- | --- |
| feature_522 | 94.76% | 99.17% |
| feature_1044 | 86.09% | 93.98% |
| feature_730 | 84.44% | 84.67% |
| feature_389 | 82.89% | 91.87% |
| feature_909 | 82.71% | 92.75% |
| feature_1039 | 82.12% | 1.76% |
| feature_1025 | 81.85% | 91.06% |
| feature_1056 | 81.73% | 90.97% |
| feature_27 | 81.39% | 70.99% |
| feature_983 | 81.06% | 73.98% |

## Highest extreme-value rates

| Feature | Train extreme | Test extreme |
| --- | --- | --- |
| feature_660 | 23.78% | n.a. |
| feature_566 | 23.76% | 11.52% |
| feature_733 | 23.53% | 0.00% |
| feature_473 | 23.22% | 13.70% |
| feature_474 | 22.94% | 0.00% |
| feature_58 | 20.68% | 11.38% |
| feature_1031 | 20.37% | n.a. |
| feature_456 | 19.83% | 23.40% |
| feature_1069 | 18.38% | 13.77% |
| feature_28 | 18.36% | 13.74% |

## Highest drift features

| Feature | Quartile PSI | |SMD| | Missing-rate difference |
| --- | --- | --- | --- |
| feature_492 | 7.5941 | 1.8243 | 0.14% |
| feature_903 | 3.7735 | 1.7578 | 2.95% |
| feature_586 | 3.6986 | 1.7512 | 3.15% |
| feature_96 | 2.9351 | 1.6481 | 50.08% |
| feature_1018 | 2.8621 | 1.6305 | 50.17% |
| feature_134 | 2.7413 | 1.6225 | 6.63% |
| feature_1045 | 2.5802 | 1.5702 | 50.04% |
| feature_313 | 2.3866 | 1.6130 | 6.40% |
| feature_1097 | 2.3666 | 1.7123 | 26.58% |
| feature_386 | 2.2812 | 1.4932 | 9.04% |
| feature_1029 | 2.2278 | 1.4794 | 21.82% |
| feature_693 | 1.9593 | 1.4367 | 6.16% |
| feature_565 | 1.8523 | 0.9323 | 2.41% |
| feature_561 | 1.7839 | 1.2978 | 52.07% |
| feature_218 | 1.7364 | 1.3174 | 52.64% |

Drift levels use PSI thresholds of 0.10/0.25, absolute standardized mean difference thresholds of 0.20/0.50, and missing-rate differences of 5%/10%.

Total runtime: 156.1 seconds.

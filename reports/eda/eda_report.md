# EDA: label balance, missingness, and feature distributions

This analysis uses the full processed Parquet datasets for counts and summary
statistics. Distribution charts use deterministic samples only for rendering.

## 1. Label balance

| Label | Meaning | Rows | Share |
| --- | --- | --- | --- |
| 0 | default | 57,399 | 8.33% |
| 1 | repaid | 631,599 | 91.67% |

- The positive class is repayment (`label = 1`), not default.
- Default represents 8.33% of training rows.
- Repaid-to-default imbalance is approximately 11.00:1.
- Accuracy would be misleading: an all-repaid classifier would already achieve
  91.67% accuracy. ROC-AUC remains appropriate for the
  competition, while PR-AUC and recall at fixed false-positive rates should be
  retained as diagnostic metrics.

## 2. Missingness patterns

- Overall missing-cell rate: train 10.61%, test
  11.02%.
- Features with any training missingness:
  1,105 of 1,105.
- Features with at least 50% training missingness:
  87.
- Features entirely missing in test:
  7.
- Features whose missing rate differs by at least 10 percentage points between
  default and repaid rows:
  112.

### Missing features per row

| Dataset | Label | Rows | Mean | Median | P90 | P99 | Max | Complete-row rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | all | 36,324 | 121.7 | 113 | 173 | 452 | 883 | 0.00% |
| train | 0 | 57,399 | 138.2 | 115 | 241 | 502 | 771 | 0.00% |
| train | 1 | 631,599 | 115.3 | 106 | 170 | 336 | 909 | 0.00% |

### Highest training missing rates

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
| feature_479 | 81.06% | 73.98% |

### Missingness most associated with the label

The gap is `repaid missing rate - default missing rate`; it describes
association, not causation.

| Feature | Default missing | Repaid missing | Gap |
| --- | --- | --- | --- |
| feature_1045 | 42.20% | 16.06% | -26.13% |
| feature_1018 | 42.13% | 16.04% | -26.09% |
| feature_96 | 42.09% | 16.02% | -26.08% |
| feature_561 | 43.62% | 17.76% | -25.86% |
| feature_218 | 43.40% | 18.12% | -25.27% |
| feature_121 | 48.49% | 27.67% | -20.82% |
| feature_907 | 48.49% | 27.67% | -20.82% |
| feature_583 | 46.29% | 26.39% | -19.90% |
| feature_1029 | 50.41% | 31.58% | -18.83% |
| feature_34 | 32.75% | 14.20% | -18.55% |

Recommended handling for the first baseline:

- Keep missing values for LightGBM's native missing-value routing.
- Add missing indicators first for features with a material label-specific gap.
- Remove or separately ablate the seven features that are entirely missing in
  test, because they cannot provide test-time values.
- Fit imputers inside each validation fold. Never compute fill values from the
  combined train and test data.

## 3. Feature distributions

### Training missingness groups

| Missing-rate group | Features |
| --- | --- |
| 0% | 0 |
| (0,1%] | 470 |
| (1,10%] | 382 |
| (10,50%] | 166 |
| (50,80%] | 76 |
| (80,100%] | 11 |

### Approximate cardinality groups

| Approximate unique values | Features |
| --- | --- |
| <=2 | 4 |
| 3-10 | 5 |
| 11-100 | 48 |
| 101-1000 | 146 |
| >1000 | 902 |

### Robust skewness groups

Bowley skewness is calculated from Q25, median, and Q75, so it is less sensitive
to extreme values than moment skewness.

| Absolute Bowley skewness | Features |
| --- | --- |
| roughly symmetric | 207 |
| mild | 260 |
| strong | 419 |
| very strong | 209 |
| zero IQR / unavailable | 10 |

- Features with at least one 5xIQR extreme-value flag:
  962.
- Features with at least a 5% extreme-value rate:
  183.
- Constant training features: 0.
- Near-constant training features: 0.

The features are numerically scaled by unknown random factors, so comparing raw
ranges across columns is not meaningful. Tree models are suitable for the first
baseline because they do not require global standardization and can handle
nonlinear, skewed distributions. Do not apply blanket winsorization or log
transforms; test those choices through cross-validation.

### Representative features used in charts

| Feature | Selection reason | Train missing | Approx. unique | Bowley skew | 5xIQR extreme rate |
| --- | --- | --- | --- | --- | --- |
| feature_522 | highest train missing rate | 94.76% | 989 | 0.305 | 4.76% |
| feature_1045 | largest label missing-rate gap | 18.24% | 1,427 | -0.119 | 0.00% |
| feature_566 | highest 5xIQR extreme-value rate | 1.64% | 62,136 | 1.000 | 23.76% |
| feature_21 | strong robust skewness | 1.64% | 36,978 | 1.000 | 12.39% |
| feature_689 | dense and comparatively regular | 0.00% | 130,501 | 0.001 | 0.00% |
| feature_423 | low-cardinality numeric feature | 0.00% | 3 | n.a. | 0.00% |

## 4. Validation implications

1. Use stratification so every fold preserves the roughly 8.33%
   default rate.
2. Report ROC-AUC as the competition metric, plus PR-AUC and KS for diagnostic
   interpretation.
3. Treat missing indicators as candidate features because missingness itself is
   associated with the target for some columns.
4. Compare three feature sets: all usable features, removal of test-all-null and
   very-high-missing features, and removal of strongly shifted features.
5. Because no explicit timestamp is available, do not call a random split
   "time validation." The next step should test whether row order contains a
   temporal signal and run adversarial validation against the test set.

## Outputs

- `label_distribution.csv`
- `missingness_by_feature.csv`
- `row_missingness_distribution.csv`
- `row_missingness_summary.csv`
- `feature_distribution_summary.csv`
- `selected_feature_statistics.csv`
- `figures/eda_overview.png`
- `figures/selected_feature_distributions.png`

Total runtime: 26.0 seconds.

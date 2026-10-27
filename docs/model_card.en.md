# Model card: full loan repayment prediction

[中文](model_card.md) · [Project](../README.en.md) · [Experiment card](experiment_card.en.md)

Date: 2026-10-25. Intended use is Kaggle scoring, historical risk analysis and offline policy comparison. **Champion is retained; neither model is deployed as a production approval service.** This card covers the two frozen business models, not every exploratory competition candidate.

## Model registry

| Item | Champion | Challenger |
| --- | --- | --- |
| Version | `lightgbm_baseline` | `lightgbm_tuned_seed_ensemble` |
| Method | Binary LightGBM baseline | Four predeclared joint configurations, inner selection, three-seed averaging |
| Outer folds | 5 | 5 |
| Model seeds | 42, with fold offsets | 42 / 2026 / 3407, with fold offsets |
| Final refit models | 5 | 15 |
| Training OOF | Held-out outer-fold prediction | Three-seed mean within the held-out fold |
| Test prediction | Mean of five refit models | Mean of fifteen refit models |
| Frozen repayment threshold | 0.9247938423619432 | 0.9256811726550405 |
| Governance | Retained reference | Development candidate, not promoted |

See [configuration](../configs/sql_business_analysis.json), [model freeze manifest](../artifacts/sql_business_analysis/model_freeze_manifest.json) and [registry](../reports/sql_business_analysis/tables/model_registry.csv). Challenger training additionally produced 20 inner candidate models; the business freeze uses 20 final refit models across both roles, excluding those search models.

## Data, target and preprocessing

688,998 training applications, 36,324 unlabeled test applications and 1,105 anonymous numerical features. `label=1` is repayment, `0` is default. The output is `p_repay`; default score is `1-p_repay`.

Only `feature_*` columns enter the model. IDs, labels, folds and stress flags are excluded. Inputs are raw float32; LightGBM learns native NaN branches and preserves real zero. No mean imputation, standardization, manual missing indicators or probability calibration is applied. A mean predicted default score is not a validated future default probability.

Trustworthy loan dates, terms/outcome windows, persistent borrower IDs and interpretable demographic attributes are unavailable. Anonymous columns are not assigned invented income, age or geographic meanings.

## Training and evaluation

The fixed protocol is shuffled stratified five-fold CV with split seed 42 and saved folds. Row position was not established as time. Within each outer fitting set, a stratified 15% split selects early stopping; models are then refit on all four fitting folds for the selected rounds. The outer validation fold is not used for that round's early stopping.

Challenger compares four configurations on common inner splits/bins, using inner AUC and a predeclared tie priority. Three refit seeds share the selected rounds; no lucky seed is selected. This is a finite manual candidate search, not an Optuna study. The baseline uses up to 600 rounds/patience 50; Challenger uses 900/patience 60. Baseline settings include learning rate .05, 31 leaves, minimum leaf 200, L2=1, row/column sampling .8 and 63 bins. Per-fold settings and rounds are in the [baseline record](../reports/lightgbm_baseline/run_config.json) and [tuning record](../reports/hyperparameter_seed_optimization/run_config.json).

**OOF was reused through development.** Fold-local early stopping does not turn the full development history into a new independent confirmatory test. A later policy development/evaluation split does not restore previously consumed independence.

## Historical performance

| Metric | Champion | Challenger |
| --- | ---: | ---: |
| Repayment ROC-AUC | 0.816569 | 0.820395 |
| Default PR-AUC (AP) | 0.388912 | 0.395367 |
| KS | 0.472156 | 0.479461 |
| Log loss | 0.225850 | 0.224226 |
| Brier score | 0.062500 | 0.062114 |

The absolute AUC gain is +0.003825 on 688,998 reused OOF rows. No test labels are available; these metrics are not Kaggle test scores, realized profit or causal uplift. [Summary](../reports/sql_business_analysis/summary.json) and [independent checks](../reports/sql_business_analysis/artifact_verification.json) include SQL/sklearn agreement and tie-aware metrics.

Train/test distribution shift is substantial, including test values outside training support for `feature_492`. Frozen-policy test approval is about 17%, versus about 70% in policy evaluation. Those diagnostics do not establish test default rates. Training stress groups cannot validate unlabeled test regions outside training support.

## Policy and limitations

Policies were selected on 413,398 development applications under assumed economics and constraints, then described on 275,600 evaluation applications, all using historical OOF. Both have a 70% development approval budget and the rule `p_repay >= frozen_threshold`. [Experiment governance](experiment_card.en.md) separately controls eligibility and promotion.

Discrimination gain is not a promotion criterion. V2 simulated profit is +CNY 3.672 with 95% CI [0.119, 7.226]; the lower bound does not exceed CNY 5. The artifacts do not authorize live lending decisions.

Known gaps include unverifiable customer clustering, weak time extrapolation, uncalibrated scores, unavailable real cash flows/terms, unknown approval-selection and label-coverage mechanisms, and no interpretable data for meaningful fairness subgroup analysis. Feature importance is diagnostic, not causal or a semantic mapping of anonymous fields.

## Reproduction and operational boundary

```powershell
python scripts/project.py reproduce --target models --dry-run
.\.venv\Scripts\python.exe scripts/project.py verify --scope business
```

Model execution needs the data/fold stage and linear baseline; only omission of `--dry-run` starts training. Business audits reuse frozen predictions and do not rerun inference on the full feature matrix. Historical model reload/inference checks remain distinct evidence.

Historical training used LightGBM 4.7.0 / NumPy 2.5.3; current review uses NumPy 2.5.4. Seeds and signatures support reproducibility but do not guarantee byte-identical retraining across environments. Preserve signed outputs and use a clean working copy for a changed training protocol. See [reproduction](reproducibility.md).

Production would need credible time cohorts, borrower deduplication, independently evaluated score calibration, real cash flows/costs, shift/local safety monitoring, serving/version logs and mature experimental outcomes. That live system is not delivered.

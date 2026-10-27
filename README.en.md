# Large-scale loan repayment modeling and risk experiment decisions

[中文](README.md) · [Delivery index](docs/INDEX.md) · [Reproduction guide](docs/reproducibility.md)

A local portfolio project built around Kaggle's [Large-Scale Loan Repayment Modeling Challenge](https://www.kaggle.com/competitions/large-scale-loan-repayment-modeling-challenge): Python modeling, DuckDB/SQL engineering, approval economics, rigorous offline A/A validation, feasible A/B design, and a native Power BI report.

**Business decision: retain Champion; do not promote Challenger.** Better development OOF discrimination did not establish sufficient business value. The V2 simulated profit interval does not clear the CNY 5 lower-bound requirement, the fixed sample is incomplete, and mature real-world outcomes and a live randomized experiment are unavailable.

## Results and denominators

| Metric | Champion | Challenger / V2 | Evidence scope |
| --- | ---: | ---: | --- |
| Repayment ROC-AUC | 0.816569 | 0.820395 | Reused development OOF, 688,998 training applications |
| Default PR-AUC (AP) | 0.388912 | 0.395367 | Same historical OOF; default is the positive class |
| KS | 0.472156 | 0.479461 | Same historical OOF |
| Frozen-policy approval rate | 70.0599% | 70.0399% | Full policy evaluation cohort, 275,600 applications |
| Simulated profit/application | CNY 276.357 | CNY 279.856 | Historical labels and assumed cash flows |
| V2 profit difference, 95% CI | Policy A reference | +3.672; [0.119, 7.226] | Stratified independent-arm replay standardized to the full common-eligible cohort |
| Randomized disagreements | — | 12,614 / 17,141 planned | Historical offline sample, below the fixed target |
| Assumed duration | — | 289 enrollment / 684 final-analysis days | 5,000 daily applications, 20% eligible, all disagreements enrolled |

The +3.499 full-cohort paired policy difference and the +3.672 V2 replay estimate use different populations and estimators. No local development AUC is presented as an unlabeled Kaggle test AUC or a verified current leaderboard result. [Machine-readable facts and source hashes](docs/project_facts.json) accompany the claims.

## What the project demonstrates

- **Python and ML:** fixed outer folds, inner early stopping, LightGBM and seed ensembling, native missing values, distribution-shift checks, disk-backed features and artifact verification. Separate XGBoost/CatBoost exploration records are retained.
- **SQL and data engineering:** CSV views, Parquet conversion, quality/drift summaries, tie-aware AUC/AP/KS, risk bands, approval and experiment tables.
- **Business analysis:** frozen policies, constrained approval, rejection costs, assumed interest/LGD/funding costs, and profit at the full-population ITT denominator.
- **Experiment design:** label-blind assignment, SRM, A/A false-positive calibration, fault injection, business/NI boundary calibration, stratified ITT, power, duration and fixed-endpoint governance.
- **Delivery:** source hashes, bilingual documentation, model/experiment cards, a one-page decision memo and six Power BI pages.

## Data and frozen models

Training: 688,998 rows and 1,107 columns. Test: 36,324 rows and 1,106 columns. Models use 1,105 anonymous numerical features. `label=1` means repayment; `label=0` means default. Submit `p_repay`; `p_default=1-p_repay`. IDs are join/submission keys only. Train and test IDs may overlap, so prediction identity is `(dataset, id, model_version)`.

The competition uses a time-separated train/test split, but training row order was not established as a trustworthy time axis. A fixed shuffled five-fold stratified split (seed 42) is used, with inner fitting/early-stopping splits and outer OOF evaluation. Those OOF results were reused during development and are not a new independent confirmatory holdout. LightGBM uses all raw float32 features, preserves NaN and real zero, and applies no score calibration.

The business **Champion** is `lightgbm_baseline` (five refit models). The **Challenger** is `lightgbm_tuned_seed_ensemble` (five folds × seeds 42/2026/3407). Kaggle exploration and these frozen business roles are separate records. See the [model card](docs/model_card.en.md) and [experiment card](docs/experiment_card.en.md).

## Architecture

![Project architecture](docs/figures/architecture.svg)

[Architecture and Mermaid source](docs/architecture.md) document the dependency chain, freeze boundaries and proposed live-data path.

## Quick review and execution entry

Run from the project root. Documentation review needs only Python 3.11+; the current review environment is Windows 11 / Python 3.13.11.

```powershell
python scripts/project.py status
python scripts/project.py doctor --profile docs
python scripts/project.py verify --scope docs
python scripts/project.py reproduce --target business --dry-run
```

Install execution dependencies and check the local workspace:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/project.py doctor --profile analysis
.\.venv\Scripts\python.exe scripts/project.py test --scope delivery
.\.venv\Scripts\python.exe scripts/project.py verify --scope dashboard
```

`requirements-lock.txt` and the [environment snapshot](docs/environment_snapshot.json) record the current installed environment. Historical model runs used NumPy 2.5.3; current review uses 2.5.4. Retraining across environments is not guaranteed to reproduce model bytes. The PDF generator additionally needs reportlab/pypdf; the existing PDF was produced with a separate document runtime.

`project.py reproduce --target data|models|business|dashboard` executes **one selected phase only**. Prerequisites are not run implicitly. `--dry-run` prints a plan; omitting it executes it. For a fresh rebuild, obtain the competition files and place `train.csv`, `test.csv`, and `sample_submission.csv` at the root. The current local workspace has Parquet/model/business artifacts but no train/test CSVs. See [the reproduction guide](docs/reproducibility.md) for dependencies, frozen-run behavior and resource needs.

## Power BI and delivery files

Open [LoanRisk.pbip](dashboard/powerbi/LoanRisk.pbip) and use **Home → Refresh**. Pages cover the decision, historical models, approval economics, A/A, A/B feasibility, and profit/data gaps. If the folder moves, set Power Query's `DataFolder` parameter to the new absolute `dashboard/powerbi/data` path.

The user confirmed that the decision page refreshed and displayed normally in Desktop 2.159.883.0. The other five pages were not individually confirmed. Traffic, eligibility and exposure controls change duration assumptions only; frozen profit evidence, the CNY 5 threshold and promotion status remain fixed. [Power BI instructions](docs/powerbi_dashboard.md) include the validation scope.

| Deliverable | Entry |
| --- | --- |
| Model card | [English](docs/model_card.en.md) · [中文](docs/model_card.md) |
| Experiment card | [English](docs/experiment_card.en.md) · [中文](docs/experiment_card.md) |
| Architecture / reproduction | [Architecture](docs/architecture.md) · [Guide](docs/reproducibility.md) |
| SQL / dictionary | [SQL](sql) · [Business dictionary](docs/sql_business_data_dictionary.md) |
| One-page decision memo | [Markdown](docs/experiment_closeout_decision_memo.md) · [PDF](output/pdf/experiment_closeout_decision_memo.pdf) |
| Evidence / delivery audit | [Facts](docs/project_facts.json) · [Delivery checks](docs/delivery_verification.json) |

## Evidence limits and live implementation gaps

**Historical evidence** comes from reused OOF and labels. **Simulation** assumes fixed loan terms/cash flows, IID or finite-population sampling, stable composition, and traffic. **Unverified live inputs** include persistent borrower IDs, first-application deduplication, safety eligibility, execution logs, full-cohort weights, actual costs/cash flows and mature DPD30/60/90. Unknown test defaults/profit and live outcomes remain NULL. Anonymous columns are not assigned invented business meanings.

The original A/A retains six uncertified stress checks. All 20 V2 matched A/A/boundary checks passed within their declared scope. Naive five-look stopping inflated false positives to about 14.20%, versus 4.934% at the fixed endpoint. Planned historical simulation joint success was about 0.55%; a synthetic +CNY 10 alternative reached about 90.53%. These are conditional simulation frequencies, not posterior probabilities or observed Challenger effects.

The proposed live sequence is shadow data/safety validation, an independent pilot, then a locked main experiment and a mature fixed-sample decision. A serving system, live randomized trial, realized profit, score calibration, production safety approval and external publication have not been delivered. This local directory has not been initialized as a Git repository; no remote repository or data publication was performed.

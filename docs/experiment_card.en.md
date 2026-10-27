# Experiment card: Champion–Challenger V2

[中文](experiment_card.md) · [Model card](model_card.en.md) · [Decision memo](experiment_closeout_decision_memo.md)

Date: 2026-10-25. ID: `loan_policy_disagreement_v2`. **Offline design/validation/simulation closed; retain Champion and do not promote Challenger.** No live randomized experiment, external preregistration, realized profit or production rollout was completed.

## Business question and target

Can policy B increase mean net profit by more than CNY 5 per **full common-eligible application**, with all risk, sample, maturity and quality requirements satisfied? The primary estimand is full-cohort ITT profit B−A, including rejected-application costs, not profit among approved borrowers alone.

A is `lightgbm_baseline` at .9247938423619432; B is `lightgbm_tuned_seed_ensemble` at .9256811726550405. Proxy eligibility requires the minimum of the two repayment scores to be at least .90. This is not a validated real safety rule; loan terms and fees are assumed identical.

Policy development/evaluation contains 413,398/275,600 applications. Evaluation has 210,856 common-eligible applications and 12,614 randomized disagreements. These development OOF/evaluation results were previously used; V2 is not a fresh independent holdout.

## Why the design changed

The original full-eligible independent-arm design required 3,721,531 randomized applications and 7,444 assumed enrollment days. Only about 5.94% of development-eligible applications had different approval decisions. V2 randomizes disagreements 1:1 with Bernoulli assignment, estimates independent-arm effects in three risk-by-disagreement-direction cells and standardizes them to the full common-eligible population:

```text
Delta_full = sum_h(weight_h * (mean_profit_B,h - mean_profit_A,h))
```

Agreement effects are zero only under a per-application identity contract for amount, pricing, term, fees and cash flow. If actual policies change those terms or behavior, redesign the estimand or use full randomization. Disagreement effects must not be reported as full-cohort effects. Paired zero variance must not size an independent-arm trial.

## Assignment, locking and inference

- SHA-256 assignment uses domain-separated application keys and a fixed salt. Eligibility/strata use only pretreatment scores. Persistent borrower IDs and first-application deduplication are still needed for live randomization.
- Development data determine variance, 50%/100% exposure, sample size and the preanalysis lock before matched calibration and evaluation replay. No outcome-driven salt, eligibility or label changes are allowed.
- Evaluation standardization uses the full contemporaneous pretreatment eligible census; planning uses development weights. Agreement/rejected applications are required for the target weights.
- Independent-arm cell variances, Welch/Satterthwaite inference and 2,000 stratified bootstrap replicates are used. Monte Carlo separately estimates joint success; marginal planning power is not joint power.
- One final efficacy analysis requires the fixed sample, at least 100 mature valid applications per cell/arm, quality clearance and complete outcome maturity. A safety pause is not early efficacy success.

## Economics, sample and duration

Assume principal CNY 10,000; 12-month interest 12%, LGD 80%, funding 4%; approved servicing CNY 100 and per-application cost CNY 10. Net cash flow is +690 for approved repayment, −8,510 for approved default and −10 for rejection. Returned principal is not revenue. Historical labels are assumed unchanged by the decision/fixed terms, not observed real counterfactuals.

The business boundary is +5, planning alternative +10 and gap 5. Two-sided alpha is 5%, marginal target power 80%, variance safety factor 1.25; 17,141 randomized disagreements are planned. This does not guarantee success or represent completed live enrollment.

At 5,000 applications/day, 20% eligible and all disagreements enrolled, duration is 289 enrollment +365 assumed maturity +30 grace =684 days. Half-disagreement enrollment takes 577 days, exceeding the 365-day enrollment cap. All-disagreement enrollment roughly doubles changed-decision exposure versus the original design and still needs local safety approval. Actual future traffic/composition/maturity are unverified.

## A/A and calibration

Original rigorous A/A has 63 endpoint/alpha records. The three required cases passed; six 10,000-application/B10% stress checks remain uncertified. The approximately 5.536% profit FPR interval did not fit completely inside the acceptance interval. Certification is not universal.

V2 includes 18 same-Champion A/A checks plus synthetic boundaries at true profit CNY 5 and ITT default incidence difference .002. All 20 met the predeclared Clopper–Pearson/Bonferroni simultaneous-interval standard. Thirty-two actual hash checks and 100,000 statistical repetitions per case are different evidence. Synthetic calibration does not alter historical labels. Fixed-endpoint FPR is 4.934%; naive any-of-five-look stopping is 14.20%.

## Replay and conjunctive promotion rules

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Profit 95% lower bound >CNY 5 | +3.672; 95% CI [0.119, 7.226] | Not met |
| ITT default one-sided upper bound <+.2000 percentage points | Difference −.0430; upper −.0044 pp | Offline pass |
| Approval difference >−2.0000 pp | −.0408 pp, complete policy-decision census | Offline pass |
| Full B approved-portfolio default upper bound <=5% | 3.2222% | Offline pass |
| Matched calibration/assignment quality | 20/20 checks, current quality passed | Limited offline pass |
| Fixed sample and minimum cells | 12,614/17,141; current minimum-cell check passed | Total sample incomplete |
| Real outcome maturity | No mature cash-flow/DPD dates | Unverified |
| Real safety and deduplication | Proxy eligibility, no borrower_id | Unverified |

Zero-effect p≈.0428 is not the business success criterion. Planned historical joint success is about .55%; an explicitly synthetic +CNY 10 alternative gives about 90.53%. They are conditional simulation frequencies, not posterior probabilities or real Challenger causal effects.

The fixed cash flow implies `Delta_profit = 700*Delta_approval - 9200*Delta_default_incidence`; profit and relative-risk outcomes are strongly coupled. Equal simulated business and joint success does not extend to heterogeneous real amounts/fees/recoveries.

## Live data and restart

Required inputs include persistent borrower/first-application identity; enrollment times, probabilities, pilot/main stage and model/policy/execution logs; a full eligible pretreatment score/safety/weight census; actual amounts, pricing, terms, interest, write-offs, recoveries and costs; maturity/completion dates, defaults, mature DPD30/60/90 and outcome completeness. Unknown outcomes remain NULL.

Start with shadow data/execution/safety validation, then an independent pilot, then a locked main experiment. Changed eligibility, randomization unit, composition, terms, estimator or stopping rules require a new protocol, matched calibration and version. Reconsider promotion only after every requirement passes. No production promotion is recorded.

## Evidence and reproduction

[Config](../configs/ab_feasibility_v2.json) · [Summary](../reports/ab_feasibility_v2/summary.json) · [Preanalysis lock](../artifacts/ab_feasibility_v2/preanalysis_design_lock.json) · [Freeze hashes](../artifacts/ab_feasibility_v2/feasibility_freeze_manifest.json) · [Independent audit](../reports/ab_feasibility_v2/artifact_verification.json) · [Full design](ab_feasibility_v2.md)

```powershell
python scripts/project.py reproduce --target business --dry-run
.\.venv\Scripts\python.exe scripts/project.py verify --scope business
```

The unified entry audits existing frozen SQL/policy/A/B/A/A/V2 stages read-only; it does not acquire new random samples. Only missing, unoccupied stages without frozen descendants are built. Changed designs require a clean work copy with new version evidence and parent configuration, preserving old results. See [the reproduction guide](reproducibility.md).

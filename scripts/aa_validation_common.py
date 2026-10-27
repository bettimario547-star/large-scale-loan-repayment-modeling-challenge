"""Champion-versus-Champion calibration of the existing experimental pipeline."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import experiment_design_common as experiment
except ImportError:
    import experiment_design_common as experiment

business = experiment.business
ROOT = business.ROOT
CONFIG = ROOT / "configs/aa_validation.json"
SQL_FILE = ROOT / "sql/05_experiment/01_analysis_tables.sql"
EXTRA_TABLES = ("aa_covariate_balance", "aa_null_cells", "aa_hash_replays", "aa_calibration_results",
                "aa_pvalue_histogram", "aa_srm_calibration", "aa_optional_stopping", "aa_negative_controls")
SNAPSHOT_TABLES = ("eligibility_summary", "experiment_metrics", "risk_score_concentration", "assignment_quality")


def paths(run_name="aa_validation", root=ROOT):
    return business.Paths(root=root, run_name=run_name)


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["scope"] != "strict_offline_aa_validation_only" or not config["current_champion_retained"] or config["challenger_promoted"]:
        raise ValueError("A/A validation retains Champion and cannot promote Challenger")
    if not config["randomization_salt"] or config["hash_repetitions"] < 2:
        raise ValueError("Need fixed salt and repeated real hashes")
    if config["calibration_repetitions"] < 10000 or config["peeking_repetitions"] < 1000:
        raise ValueError("Insufficient calibration precision")
    if config["alpha_grid"] != sorted(set(config["alpha_grid"])) or .05 not in config["alpha_grid"]:
        raise ValueError("Require an ordered alpha grid including 0.05")
    for alpha in config["alpha_grid"]:
        tolerance = config["fpr_absolute_tolerances"][str(alpha)]
        if not 0 < alpha < .5 or not 0 < tolerance < alpha: raise ValueError("Invalid calibration envelope")
    if not 0 < config["calibration_familywise_error"] < .5 or not 0 < config["srm_fpr_upper_limit"] < 1:
        raise ValueError("Invalid calibration confidence/SRM criterion")
    looks = config["peeking_look_fractions"]
    if looks != sorted(set(looks)) or looks[-1] != 1 or looks[0] <= 0: raise ValueError("Invalid peeking schedule")
    if min(config["stress_sample_sizes"]) < 10 or any(not 0 < r < 1 for r in config["stress_b_fractions"]):
        raise ValueError("Invalid stress regimes")
    if not 0 < config["max_absolute_covariate_smd"] < 1 or config["injected_profit_bias"] <= 0:
        raise ValueError("Invalid balance/negative-control criteria")
    return config


def pipeline_config(parent_config, config, salt=None):
    derived = copy.deepcopy(parent_config)
    derived["experiment_id"] = config["experiment_id"]
    derived["randomization"]["salt"] = salt or config["randomization_salt"]
    return derived


def make_aa_population(source, champion):
    """Keep the frozen A/B eligibility covariates, execute Champion on BOTH arms."""
    experiment.validate_population(source)
    result = source.copy()
    result["approved_b"] = result.approved_a
    result["potential_profit_b"] = result.potential_profit_a
    result["decision_disagreement"] = False
    result["treatment_score_a"] = result.score_a
    result["treatment_score_b"] = result.score_a
    result["treatment_policy_a"] = champion["model_version"]
    result["treatment_policy_b"] = champion["model_version"]
    result["treatment_threshold_a"] = champion["threshold"]
    result["treatment_threshold_b"] = champion["threshold"]
    # score_b remains a FROZEN ELIGIBILITY covariate, never arm-B treatment.
    return result


def load_inputs(output, config):
    parent = paths(config["parent_run_name"], output.root)
    parent_config = experiment.load_config(output.root / "configs/ab_test_design.json")
    protocol = json.loads((parent.artifacts / "experiment_protocol.json").read_text(encoding="utf-8"))
    champion = next(row for row in protocol["selected_policies"] if row["model_role"] == "champion")
    source = pd.read_parquet(parent.artifacts / "experiment_population.parquet")
    return source, make_aa_population(source, champion), parent_config, champion, int(protocol["required_randomized_total"])


def validated_expected_population(source, population, parent_config, champion):
    """Check immutable identity once and return the ordered eligible reference."""
    experiment.validate_population(population)
    keys = ["dataset", "id"]
    ordered = population.sort_values(keys).reset_index(drop=True)
    baseline = source.sort_values(keys).reset_index(drop=True)
    for field in ("dataset", "id", "label", "score_a", "score_b", "common_score", "eligible_proxy", "risk_stratum", "policy_cohort"):
        pd.testing.assert_series_equal(ordered[field], baseline[field], check_dtype=False, check_exact=True)
    threshold = parent_config["eligibility"]["common_min_raw_repayment_score"]
    gate = ordered[["score_a", "score_b"]].min(axis=1).ge(threshold)
    np.testing.assert_array_equal(ordered.eligible_proxy, gate)
    for field in ("approved_b", "approved_a"):
        np.testing.assert_array_equal(ordered[field], baseline.approved_a)
    for field in ("potential_profit_a", "potential_profit_b"):
        np.testing.assert_allclose(ordered[field], baseline.potential_profit_a, rtol=0, atol=0, equal_nan=True)
    for field in ("treatment_score_a", "treatment_score_b"):
        np.testing.assert_array_equal(ordered[field], baseline.score_a)
    for field in ("treatment_policy_a", "treatment_policy_b"):
        if not ordered[field].eq(champion["model_version"]).all(): raise AssertionError("Treatment is not Champion")
    for field in ("treatment_threshold_a", "treatment_threshold_b"):
        if not ordered[field].eq(champion["threshold"]).all(): raise AssertionError("Champion threshold changed")
    if ordered.decision_disagreement.any(): raise AssertionError("A/A decisions differ")
    if not ordered.simulation_only.all(): raise AssertionError("False live population scope")
    expected = ordered.loc[ordered.dataset.eq("train") & ordered.policy_cohort.eq("policy_evaluation") & gate].sort_values(keys).reset_index(drop=True)
    return expected


def validate_assignments(source, population, assignments, parent_config, config, champion, salt=None, expected=None):
    """Check every assigned row; cached references are internal immutable inputs."""
    if expected is None: expected = validated_expected_population(source,population,parent_config,champion)
    keys = ["dataset", "id"]
    actual = assignments.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[keys], expected[keys], check_dtype=False)
    if actual.duplicated(keys).any() or actual.observed_dpd30.notna().any(): raise AssertionError("Duplicate key or invented DPD")
    r = parent_config["randomization"]["challenger_probability"]
    if not actual.assignment_probability_b.eq(r).all(): raise AssertionError("Declared allocation probability changed")
    if not actual.experiment_id.eq(config["experiment_id"]).all(): raise AssertionError("Wrong experiment ID")
    if not actual.simulation_only.all(): raise AssertionError("False live assignment scope")
    uniforms = np.array([experiment.stable_uniform(d, i, salt or config["randomization_salt"]) for d, i in zip(actual.dataset, actual.id)])
    np.testing.assert_array_equal(actual.assignment_uniform, uniforms)
    np.testing.assert_array_equal(actual.arm, np.where(uniforms < r, "B", "A"))
    np.testing.assert_array_equal(actual.assigned_approved, expected.approved_a)
    np.testing.assert_allclose(actual.simulated_observed_net_profit, expected.potential_profit_a, rtol=0, atol=0)
    np.testing.assert_array_equal(actual.simulated_observed_default, expected.approved_a & expected.label.eq(0))
    np.testing.assert_array_equal(actual.simulated_observed_repaid, expected.approved_a & expected.label.eq(1))
    for field in ("label","score_a","score_b","common_score","eligible_proxy","risk_stratum","policy_cohort",
                  "approved_a","approved_b","potential_profit_a","potential_profit_b","decision_disagreement",
                  "treatment_policy_a", "treatment_policy_b", "treatment_score_a", "treatment_score_b", "treatment_threshold_a", "treatment_threshold_b"):
        pd.testing.assert_series_equal(actual[field], expected[field], check_dtype=False, check_exact=True)


def rng_for(config, domain):
    token = int.from_bytes(hashlib.sha256(domain.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([config["random_seed"], token]))


def null_cells(population):
    rows = population.loc[population.dataset.eq("train") & population.policy_cohort.eq("policy_evaluation") & population.eligible_proxy]
    cell_rows = pd.DataFrame({"profit": rows.potential_profit_a, "approved": rows.approved_a,
                             "default_event": rows.approved_a & rows.label.eq(0)})
    return cell_rows.groupby(["profit", "approved", "default_event"], dropna=False).size().rename("rows").reset_index().sort_values(["profit", "approved", "default_event"]).reset_index(drop=True)


def moments(counts, cells):
    n = counts.sum(axis=1).astype(float)
    support = cells.profit.to_numpy(dtype=float)
    mean = counts @ support / n
    variance = np.maximum(0, (counts @ (support * support) - n * mean * mean) / (n - 1))
    return {"n": n, "mean": mean, "variance": variance,
            "approval": counts @ cells.approved.to_numpy(dtype=float) / n,
            "default": counts @ cells.default_event.to_numpy(dtype=float) / n}


def infer_moments(a, b):
    output = {}
    va, vb = a["variance"] / a["n"], b["variance"] / b["n"]
    se = np.sqrt(va + vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (va + vb)**2 / (va**2 / (a["n"]-1) + vb**2 / (b["n"]-1))
        delta = b["mean"] - a["mean"]
        p = 2 * stats.t.sf(np.abs(delta / se), df)
    output["profit_itt"] = {"delta": delta, "se": se, "df": df, "p": p, "usable": np.isfinite(p) & (se > 0)}
    for endpoint, key in (("approval_rate", "approval"), ("default_incidence_itt", "default")):
        qa, qb = a[key], b[key]
        se = np.sqrt(qa*(1-qa)/a["n"] + qb*(1-qb)/b["n"])
        with np.errstate(divide="ignore", invalid="ignore"):
            p = 2 * stats.norm.sf(np.abs((qb-qa)/se))
        output[endpoint] = {"delta": qb-qa, "se": se, "p": p, "usable": np.isfinite(p) & (se > 0)}
    return output


def clopper_pearson(events, total, error=.05):
    if total <= 0: return None, None
    low = float(stats.beta.ppf(error/2, events, total-events+1)) if events else 0.
    high = float(stats.beta.ppf(1-error/2, events+1, total-events)) if events < total else 1.
    return low, high


def cases(cells, planned_n, parent_config, config):
    n = int(cells.rows.sum())
    r = parent_config["randomization"]["challenger_probability"]
    output = [dict(case_id="finite_full_current", sampling="finite_bernoulli", n_total=n, fraction_b=r, required=True),
              dict(case_id="iid_current", sampling="iid_same_champion", n_total=n, fraction_b=r, required=True),
              dict(case_id="iid_previous_planned", sampling="iid_same_champion", n_total=planned_n, fraction_b=r, required=True)]
    for size in config["stress_sample_sizes"]:
        for fraction in config["stress_b_fractions"]:
            output.append(dict(case_id=f"stress_n{size}_r{fraction:g}", sampling="iid_same_champion", n_total=size, fraction_b=fraction, required=False))
    return output


def calibration(cells, planned_n, parent_config, config, progress=None):
    regimes = cases(cells, planned_n, parent_config, config)
    # Simultaneous exact binomial intervals: FPR endpoints/alphas plus SRM checks.
    comparisons = len(regimes) * 3 * len(config["alpha_grid"]) + sum(row["sampling"] == "finite_bernoulli" for row in regimes)
    error = config["calibration_familywise_error"] / comparisons
    repetitions = config["calibration_repetitions"]
    weights = cells.rows.to_numpy(dtype=float) / cells.rows.sum()
    rows, histograms, srm_rows, draw_frames = [], [], [], []
    for case in regimes:
        rng = rng_for(config, "calibration:" + case["case_id"])
        if case["sampling"] == "finite_bernoulli":
            count_b = rng.binomial(cells.rows.to_numpy(), case["fraction_b"], size=(repetitions, len(cells)))
            count_a = cells.rows.to_numpy() - count_b
        else:
            n_b = int(np.floor(case["n_total"] * case["fraction_b"]))
            n_a = case["n_total"] - n_b
            count_a = rng.multinomial(n_a, weights, size=repetitions)
            count_b = rng.multinomial(n_b, weights, size=repetitions)
        a, b = moments(count_a, cells), moments(count_b, cells)
        inferred = infer_moments(a, b)
        for endpoint, values in inferred.items():
            usable = values["usable"]
            valid = int(usable.sum())
            for alpha in config["alpha_grid"]:
                rejected = int((values["p"][usable] < alpha).sum())
                low, high = clopper_pearson(rejected, valid, error)
                tolerance = config["fpr_absolute_tolerances"][str(alpha)]
                calibrated = valid == repetitions and low is not None and low >= alpha-tolerance and high <= alpha+tolerance
                rows.append({**case, "endpoint": endpoint, "alpha": alpha, "repetitions": repetitions,
                    "valid_repetitions": valid, "rejections": rejected, "false_positive_rate": rejected/valid if valid else None,
                    "fpr_simultaneous_ci_low": low, "fpr_simultaneous_ci_high": high,
                    "simultaneous_ci_error_per_comparison": error, "fpr_tolerance": tolerance,
                    "zero_effect_interval_coverage": 1-rejected/valid if valid else None,
                    "coverage_simultaneous_ci_low": 1-high if high is not None else None,
                    "coverage_simultaneous_ci_high": 1-low if low is not None else None,
                    "mean_simulated_delta": float(np.mean(values["delta"])),
                    "mean_delta_mc_se": float(np.std(values["delta"], ddof=1)/np.sqrt(repetitions)),
                    "calibration_passed": bool(calibrated), "simulation_only": True})
            bins, _ = np.histogram(values["p"][usable], bins=np.linspace(0, 1, 11))
            for i, count in enumerate(bins):
                histograms.append({"case_id": case["case_id"], "endpoint": endpoint, "p_bin_lower": i/10,
                    "p_bin_upper": (i+1)/10, "count": int(count), "fraction": int(count)/valid if valid else None,
                    "expected_uniform_fraction": .1, "diagnostic_only": True})
        if case["sampling"] == "finite_bernoulli":
            # Valid equal-tail exact p-values. At r=.5 this equals binomtest.
            counts_b = count_b.sum(axis=1)
            p_srm = np.minimum(1, 2*np.minimum(stats.binom.cdf(counts_b,case["n_total"],case["fraction_b"]),stats.binom.sf(counts_b-1,case["n_total"],case["fraction_b"])))
            cutoff = parent_config["randomization"]["srm_alpha"]
            rejected = int((p_srm < cutoff).sum())
            low, high = clopper_pearson(rejected, repetitions, error)
            srm_rows.append({"case_id":case["case_id"],"repetitions":repetitions,"nominal_srm_alpha":cutoff,
                "false_srm_flags":rejected,"false_srm_flag_rate":rejected/repetitions,
                "simultaneous_ci_low":low,"simultaneous_ci_high":high,"upper_limit":config["srm_fpr_upper_limit"],
                "calibration_passed":bool(high <= config["srm_fpr_upper_limit"])})
        draw_frames.append(pd.DataFrame({"case_id":case["case_id"],"replicate":np.arange(repetitions),
            "profit_delta":inferred["profit_itt"]["delta"],"profit_se":inferred["profit_itt"]["se"],
            "profit_p_value":inferred["profit_itt"]["p"],"n_a":a["n"].astype(int),"n_b":b["n"].astype(int)}))
        if progress:
            primary = next(row for row in rows if row['case_id'] == case['case_id'] and row['endpoint'] == 'profit_itt' and row['alpha'] == .05)
            rate = f"{primary['false_positive_rate']:.3%}" if primary['false_positive_rate'] is not None else 'unusable'
            certified = all(row['calibration_passed'] for row in rows if row['case_id'] == case['case_id'])
            progress(f"Calibrated {case['case_id']} ({repetitions:,} replications): profit FPR={rate}; all endpoints certified={certified}")
    return pd.DataFrame(rows), pd.DataFrame(histograms), pd.DataFrame(srm_rows), pd.concat(draw_frames,ignore_index=True)


def balance(assignments):
    rows = []
    for field in ("score_a", "common_score"):
        a, b = [assignments.loc[assignments.arm.eq(arm),field].to_numpy() for arm in ("A","B")]
        pooled = np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        difference = b.mean()-a.mean()
        rows.append({"covariate":field,"mean_a":float(a.mean()),"mean_b":float(b.mean()),
                     "standardized_mean_difference":float(difference/pooled) if pooled else (0. if difference == 0 else float(np.copysign(np.inf,difference)))})
    return pd.DataFrame(rows)


def peek_simulation(cells, config):
    repetitions, total = config["peeking_repetitions"], config["peeking_sample_size"]
    rng = rng_for(config,"unadjusted_optional_stopping")
    weights = cells.rows.to_numpy(dtype=float)/cells.rows.sum()
    ca = np.zeros((repetitions,len(cells)),dtype=np.int64)
    cb = np.zeros_like(ca)
    previous_a = previous_b = 0
    any_two = np.zeros(repetitions,dtype=bool)
    any_positive = np.zeros_like(any_two)
    rows = []
    for fraction in config["peeking_look_fractions"]:
        n_b = int(np.floor(total*fraction/2))
        n_a = int(np.floor(total*fraction))-n_b
        ca += rng.multinomial(n_a-previous_a,weights,size=repetitions)
        cb += rng.multinomial(n_b-previous_b,weights,size=repetitions)
        previous_a,previous_b = n_a,n_b
        values = infer_moments(moments(ca,cells),moments(cb,cells))["profit_itt"]
        reject = values["p"] < .05
        positive = reject & (values["delta"] > 0)
        any_two |= reject
        any_positive |= positive
        rows.append({"look_fraction":fraction,"n_total":n_a+n_b,"repetitions":repetitions,
            "single_look_two_sided_fpr":float(reject.mean()),"naive_any_look_two_sided_fpr":float(any_two.mean()),
            "single_look_positive_ci_rate":float(positive.mean()),"naive_any_look_positive_ci_rate":float(any_positive.mean()),
            "nominal_two_sided_alpha":.05,"nominal_positive_direction_alpha":.025,
            "allowed_efficacy_look":bool(fraction == 1.),"simulation_only":True})
    return pd.DataFrame(rows)


def analysis_readiness(n_current, n_required, outcomes_complete, quality_passed, is_final_look):
    reasons = []
    if n_current < n_required: reasons.append("fixed_sample_not_reached")
    if not outcomes_complete: reasons.append("outcomes_not_complete")
    if not quality_passed: reasons.append("unresolved_data_or_srm_issue")
    if not is_final_look: reasons.append("early_efficacy_look_not_allowed")
    return {"efficacy_analysis_allowed":not reasons,"blocking_reasons":reasons,
            "champion_retained":True,"challenger_promoted":False,"live_experiment_completed":False}


def snapshot_bootstrap(assignments, config):
    rng = rng_for(config,"snapshot_independent_bootstrap")
    a,b = [assignments.loc[assignments.arm.eq(arm),"simulated_observed_net_profit"].to_numpy() for arm in ("A","B")]
    draws = experiment.bootstrap_mean(b,config["bootstrap_repetitions"],rng)-experiment.bootstrap_mean(a,config["bootstrap_repetitions"],rng)
    low,high = np.quantile(draws,[.025,.975])
    return pd.DataFrame([{"method":"independent_arm_aa_bootstrap","estimate":float(b.mean()-a.mean()),
        "ci_low":float(low),"ci_high":float(high),"repetitions":len(draws),
        "paired_zero_not_used_for_inference":True}]),pd.DataFrame({"replicate":np.arange(len(draws)),"profit_delta":draws})


def negative_controls(source,population,assignments,parent_config,config,champion,metrics,metric_validator):
    records = []
    def mutation_check(name, changed_population=None, changed_assignment=None):
        detected = False
        try:
            validate_assignments(source,changed_population if changed_population is not None else population,
                changed_assignment if changed_assignment is not None else assignments,parent_config,config,champion)
        except (AssertionError,ValueError): detected = True
        records.append({"control":name,"detector":"independent_identity_and_assignment_validation","detected_as_expected":detected})
    changed = population.copy()
    mask = changed.dataset.eq("train")
    changed.loc[mask,"potential_profit_b"] += config["injected_profit_bias"]
    mutation_check("changed_arm_b_cashflow",changed_population=changed)
    changed = assignments.copy()
    changed["arm"] = np.where(changed.label.eq(0),"B","A")
    mutation_check("label_dependent_allocation",changed_assignment=changed)
    mutation_check("duplicate_application_key",changed_assignment=pd.concat([assignments,assignments.iloc[[0]]],ignore_index=True))
    changed = population.copy()
    changed["eligible_proxy"] = changed.label.eq(1)
    mutation_check("label_dependent_eligibility",changed_population=changed)
    changed = assignments.copy()
    changed["observed_dpd30"] = False
    mutation_check("fabricated_zero_dpd",changed_assignment=changed)
    for field,value,name in (
        ("assignment_probability_b",.99,"wrong_declared_assignment_probability"),
        ("experiment_id","wrong-experiment","wrong_experiment_identity"),
        ("label",1,"corrupted_stored_outcome_label"),
        ("risk_stratum","wrong-stratum","corrupted_preassignment_stratum")):
        changed = assignments.copy()
        changed[field] = value
        mutation_check(name,changed_assignment=changed)
    changed = assignments.copy()
    changed["arm"] = np.where(changed.assignment_uniform < config["injected_allocation_b_fraction"],"B","A")
    mutation_check("biased_allocation_vs_declared_probability",changed_assignment=changed)
    n_b = int(changed.arm.eq("B").sum())
    p_srm = stats.binomtest(n_b,len(changed),parent_config["randomization"]["challenger_probability"]).pvalue
    records.append({"control":"srm_detects_injected_bias","detector":"exact_binomial_srm",
        "detected_as_expected":bool(p_srm < parent_config["randomization"]["srm_alpha"])})
    corrupted = metrics.copy()
    corrupted["simulated_profit_per_randomized_application_itt"] = corrupted.simulated_net_profit / corrupted.approved_applications
    detected = False
    try: metric_validator(corrupted,assignments)
    except AssertionError: detected = True
    records.append({"control":"approved_only_denominator_bug","detector":"independent_itt_metric_validation","detected_as_expected":detected})
    a,b = [assignments.loc[assignments.arm.eq(arm),"simulated_observed_net_profit"].to_numpy() for arm in ("A","B")]
    effect = experiment.mean_comparison(a,b+config["injected_profit_bias"])
    records.append({"control":"profit_test_detects_injected_effect","detector":"welch_profit_test",
        "detected_as_expected":bool(effect["ci_low"]>0)})
    for name,readiness,reason in (
        ("early_efficacy_blocked",analysis_readiness(100,100,True,True,False),"early_efficacy_look_not_allowed"),
        ("immature_outcomes_blocked",analysis_readiness(100,100,False,True,True),"outcomes_not_complete"),
        ("insufficient_sample_blocked",analysis_readiness(99,100,True,True,True),"fixed_sample_not_reached"),
        ("unresolved_srm_blocked",analysis_readiness(100,100,True,False,True),"unresolved_data_or_srm_issue")):
        records.append({"control":name,"detector":"readiness_gate","detected_as_expected":bool(not readiness["efficacy_analysis_allowed"] and reason in readiness["blocking_reasons"])})
    return pd.DataFrame(records)


def hash_replays(population, source, parent_config, config, champion, progress=None):
    output = []
    expected = validated_expected_population(source,population,parent_config,champion)
    for replicate in range(config["hash_repetitions"]):
        salt = config["randomization_salt"] if replicate == 0 else f"{config['randomization_salt']}:replicate:{replicate}"
        derived = pipeline_config(parent_config,config,salt)
        assigned = experiment.assign(population,derived)
        validate_assignments(source,population,assigned,parent_config,config,champion,salt,expected)
        a,b = [assigned.loc[assigned.arm.eq(arm),"simulated_observed_net_profit"].to_numpy() for arm in ("A","B")]
        comparison = experiment.mean_comparison(a,b)
        quality = experiment.assignment_quality(assigned,derived)
        srm_p = float(quality.loc[quality.risk_stratum.eq("all"),"srm_p_value"].iloc[0])
        output.append({"replicate":replicate,"salt":salt,"n_a":len(a),"n_b":len(b),
            "delta":comparison["delta"],"se":comparison["se"],"p_value":comparison["simulation_p_value"],
            "ci_low":comparison["ci_low"],"ci_high":comparison["ci_high"],
            "reject_at_005":comparison["simulation_p_value"] < .05,"srm_p_value":srm_p,
            "srm_flag":srm_p < parent_config["randomization"]["srm_alpha"]})
        if progress and (replicate+1)%8 == 0: progress(f"Verified real hash replays {replicate+1}/{config['hash_repetitions']}")
    return pd.DataFrame(output)


def input_hashes(output, config):
    parent = paths(config["parent_run_name"],output.root)
    files = [CONFIG, output.root/"configs/ab_test_design.json",parent.artifacts/"experiment_freeze_manifest.json",
             parent.artifacts/"experiment_population.parquet",parent.artifacts/"experiment_protocol.json",
             parent.reports/"run_config.json",parent.reports/"artifact_verification.json"]
    return {business.relative(path,output.root):business.sha256(path) for path in files}


def protocol_hashes(output):
    files = [Path(__file__),ROOT/"scripts/run_aa_validation.py",ROOT/"scripts/verify_aa_validation.py",
             ROOT/"scripts/experiment_design_common.py",SQL_FILE]
    return {business.relative(path,output.root):business.sha256(path) for path in files}


def check_freeze(output,config):
    path = output.artifacts/"aa_freeze_manifest.json"
    if not path.exists(): return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != input_hashes(output,config) or manifest["protocol_sha256"] != protocol_hashes(output):
        raise ValueError("Frozen A/A protocol/input changed; use a new --run-name")
    for file,checksum in manifest["output_sha256"].items():
        if business.sha256(output.root/file) != checksum: raise AssertionError("Frozen A/A artifact changed")
    return manifest

"""Frozen-policy experimental design; all generated outcomes are simulations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import business_analysis_common as business
except ImportError:
    import business_analysis_common as business

ROOT = business.ROOT
CONFIG = ROOT / "configs/ab_test_design.json"
SQL_FILE = ROOT / "sql/05_experiment/01_analysis_tables.sql"
TABLES = ("experiment_population", "eligibility_summary", "experiment_assignments", "experiment_metrics",
          "risk_score_concentration", "assignment_quality", "power_analysis", "experiment_duration", "bootstrap_results", "monte_carlo_results")
CSV_TABLES = tuple(name for name in TABLES if name not in ("experiment_population", "experiment_assignments"))


def paths(run_name="ab_test_design", root=ROOT):
    return business.Paths(root=root, run_name=run_name)


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    inference, assignment, eligibility = config["inference"], config["randomization"], config["eligibility"]
    if config["scope"] != "offline_design_and_simulation_only" or eligibility["production_safety_validated"]:
        raise ValueError("This runner supports offline simulations only")
    if not 0 < assignment["challenger_probability"] < 1 or not 0 < assignment["srm_alpha"] < 1:
        raise ValueError("Invalid allocation or SRM level")
    if not assignment["salt"] or not 0 < inference["alpha_two_sided"] < .5 or not .5 < inference["target_power"] < 1:
        raise ValueError("Invalid inference parameters")
    if not np.isfinite(inference["variance_safety_factor"]) or inference["variance_safety_factor"] < 1:
        raise ValueError("Variance safety factor must be at least one")
    if inference["profit_mde_per_application"] not in inference["profit_mde_grid"] or any(not np.isfinite(v) or v <= 0 for v in inference["profit_mde_grid"]):
        raise ValueError("Require a positive declared MDE in the grid")
    if any(not 0 < value < 1 for value in inference["planning_challenger_probabilities"]):
        raise ValueError("Invalid planning allocation")
    if assignment["challenger_probability"] not in inference["planning_challenger_probabilities"]:
        raise ValueError("Actual allocation must be represented in planning")
    for field in ("default_incidence_ni_margin", "approval_rate_ni_margin", "absolute_default_rate_limit_approved"):
        if not 0 < inference[field] < 1: raise ValueError("Invalid guardrail margin")
    if inference["bootstrap_repetitions"] < 200 or inference["monte_carlo_repetitions"] < 200:
        raise ValueError("Need at least 200 simulation repetitions")
    edges = eligibility["risk_stratum_edges"]
    if edges != sorted(set(edges)) or edges[0] != eligibility["common_min_raw_repayment_score"] or edges[-1] <= 1:
        raise ValueError("Strata must cover the common gate, including a score of one")
    if not 0 <= edges[0] < 1: raise ValueError("Invalid common score gate")
    traffic = config["traffic_assumptions"]
    if any(v <= 0 for v in traffic["daily_all_applications_grid"]) or any(not 0 < v <= 1 for v in traffic["eligible_fraction_grid"]):
        raise ValueError("Invalid traffic assumptions")
    if not 0 < traffic["main_experiment_fraction_of_eligible"] <= 1:
        raise ValueError("Invalid experiment exposure")
    if traffic["outcome_maturity_days_assumed"] <= 0 or traffic["data_grace_days"] < 0:
        raise ValueError("Invalid maturity horizon")
    if traffic["primary_daily_all_applications"] not in traffic["daily_all_applications_grid"] or traffic["primary_eligible_fraction"] not in traffic["eligible_fraction_grid"]:
        raise ValueError("Primary traffic scenario must be in the grid")
    return config


def stable_uniform(dataset, application_id, salt, domain="arm"):
    """Stable row-level demo; production needs an actual persistent borrower ID."""
    key = json.dumps([salt, domain, str(dataset), str(application_id)], ensure_ascii=False, separators=(",", ":"))
    # Use 53 bits: conversion cannot round a value up to 1.0.
    bits = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") >> 11
    return bits / 2**53


def make_population(parent_database, config):
    with duckdb.connect(str(parent_database), read_only=True) as connection:
        population = connection.execute("""SELECT a.dataset, a.policy_cohort, a.id, a.label,
            a.p_repay AS score_a, b.p_repay AS score_b,
            a.approved AS approved_a, b.approved AS approved_b,
            a.historical_simulated_net_profit AS potential_profit_a,
            b.historical_simulated_net_profit AS potential_profit_b
            FROM selected_policy_application_outcomes a JOIN selected_policy_application_outcomes b
            ON a.dataset=b.dataset AND a.id=b.id AND b.prospective_arm='B'
            WHERE a.prospective_arm='A' ORDER BY a.dataset,a.id""").fetchdf()
    if population.duplicated(["dataset", "id"]).any() or population.empty:
        raise ValueError("Invalid paired policy coverage")
    population["common_score"] = population[["score_a", "score_b"]].min(axis=1)
    population["eligible_proxy"] = population.common_score.ge(config["eligibility"]["common_min_raw_repayment_score"])
    population["decision_disagreement"] = population.approved_a.ne(population.approved_b)
    edges = np.asarray(config["eligibility"]["risk_stratum_edges"])
    codes = np.searchsorted(edges, population.common_score.to_numpy(), side="right") - 1
    population["risk_stratum"] = [f"S{code + 1}" if allowed else "excluded" for code, allowed in zip(codes, population.eligible_proxy)]
    population["simulation_only"] = True
    validate_population(population)
    return population


def validate_population(population):
    if population.duplicated(["dataset", "id"]).any() or population.id.isna().any():
        raise ValueError("Duplicate/missing application key")
    if not population.dataset.isin(["train", "test"]).all(): raise ValueError("Invalid dataset")
    training = population.dataset.eq("train")
    if not population.loc[training, "label"].isin([0,1]).all() or population.loc[~training,"label"].notna().any():
        raise ValueError("Training needs labels; test labels must be unavailable")
    for column in ("score_a", "score_b"):
        values = population[column].to_numpy()
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any(): raise ValueError("Invalid policy scores")
    for column in ("potential_profit_a", "potential_profit_b"):
        if not np.isfinite(population.loc[training,column]).all() or population.loc[~training,column].notna().any():
            raise ValueError("Historical profits require labels and remain NULL on test")


def assign(population, config):
    # No outcome/label-based gate, stratification, allocation or seed search.
    rows = population.loc[population.dataset.eq("train") & population.policy_cohort.eq("policy_evaluation") & population.eligible_proxy].copy()
    r = config["randomization"]["challenger_probability"]
    rows["assignment_uniform"] = [stable_uniform(d, i, config["randomization"]["salt"]) for d, i in zip(rows.dataset, rows.id)]
    rows["experiment_id"] = config["experiment_id"]
    rows["assignment_probability_b"] = r
    rows["arm"] = np.where(rows.assignment_uniform.lt(r), "B", "A")
    rows["assigned_approved"] = np.where(rows.arm.eq("B"), rows.approved_b, rows.approved_a)
    rows["simulated_observed_net_profit"] = np.where(rows.arm.eq("B"), rows.potential_profit_b, rows.potential_profit_a)
    rows["simulated_observed_default"] = rows.assigned_approved & rows.label.eq(0)
    rows["simulated_observed_repaid"] = rows.assigned_approved & rows.label.eq(1)
    rows["observed_dpd30"] = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    return rows.reset_index(drop=True)


def normal_power(n_total, variance_a, variance_b, fraction_b, effect, alpha=.05, one_sided=False):
    if n_total < 4 or min(variance_a, variance_b) < 0 or not 0 < fraction_b < 1:
        raise ValueError("Invalid power inputs")
    se = np.sqrt((variance_a / (1 - fraction_b) + variance_b / fraction_b) / n_total)
    critical = stats.norm.ppf(1 - alpha if one_sided else 1 - alpha / 2)
    if se == 0: return float(effect > 0)
    shift = effect / se
    return float(stats.norm.cdf(shift - critical) if one_sided else stats.norm.cdf(shift - critical) + stats.norm.cdf(-shift - critical))


def required_sample(variance_a, variance_b, fraction_b, effect, alpha, power, one_sided=False):
    if not np.isfinite(effect) or effect <= 0: raise ValueError("MDE must be positive")
    if min(variance_a,variance_b)<0 or not np.isfinite(variance_a+variance_b) or not 0<fraction_b<1 or not 0<alpha<1 or not 0<power<1:
        raise ValueError("Invalid planning inputs")
    critical = stats.norm.ppf(1 - alpha if one_sided else 1 - alpha / 2)
    approximate = (critical + stats.norm.ppf(power))**2 * (variance_a / (1 - fraction_b) + variance_b / fraction_b) / effect**2
    low, high = 4, max(4, int(np.ceil(approximate)) + 1)
    while normal_power(high, variance_a, variance_b, fraction_b, effect, alpha, one_sided) < power: high *= 2
    while low < high:
        mid = (low + high) // 2
        if normal_power(mid, variance_a, variance_b, fraction_b, effect, alpha, one_sided) >= power: high = mid
        else: low = mid + 1
    return int(low)


def planning(population, config):
    development = population.loc[population.policy_cohort.eq("policy_development") & population.eligible_proxy]
    if len(development) < 10 or development.label.isna().any(): raise ValueError("Insufficient planning-only cohort")
    inference = config["inference"]
    v_a, v_b = [float(development[f"potential_profit_{arm}"].var(ddof=1)) * inference["variance_safety_factor"] for arm in ("a", "b")]
    q_default = float((development.approved_a & development.label.eq(0)).mean())
    q_approval = float(development.approved_a.mean())
    rows = []
    for fraction in inference["planning_challenger_probabilities"]:
        endpoints = [("profit_superiority", mde, v_a, v_b, inference["alpha_two_sided"], False) for mde in inference["profit_mde_grid"]]
        endpoints += [("default_incidence_noninferiority", inference["default_incidence_ni_margin"], q_default * (1-q_default), q_default * (1-q_default), inference["alpha_two_sided"] / 2, True),
                      ("approval_rate_noninferiority", inference["approval_rate_ni_margin"], q_approval * (1-q_approval), q_approval * (1-q_approval), inference["alpha_two_sided"] / 2, True)]
        for endpoint, effect, va, vb, alpha, one in endpoints:
            n = required_sample(va, vb, fraction, effect, alpha, inference["target_power"], one)
            rows.append({"endpoint": endpoint, "fraction_b": fraction, "effect_or_margin": effect,
                "planning_variance_a": va, "planning_variance_b": vb, "alpha": alpha, "one_sided": one,
                "target_power": inference["target_power"], "n_total": n,
                "n_a_expected": (1-fraction)*n, "n_b_expected": fraction*n,
                "achieved_normal_power": normal_power(n, va, vb, fraction, effect, alpha, one),
                "planning_cohort": "policy_development", "baseline_default_incidence": q_default,
                "baseline_approval_rate": q_approval, "simulation_only": True})
    frame = pd.DataFrame(rows)
    r = config["randomization"]["challenger_probability"]
    main = frame.loc[frame.fraction_b.eq(r) & (frame.endpoint.ne("profit_superiority") | frame.effect_or_margin.eq(inference["profit_mde_per_application"]))]
    return frame, int(main.n_total.max())


def duration_table(n_required, config):
    traffic = config["traffic_assumptions"]
    rows = []
    for daily in traffic["daily_all_applications_grid"]:
        for coverage in traffic["eligible_fraction_grid"]:
            effective = daily * coverage * traffic["main_experiment_fraction_of_eligible"]
            days = int(np.ceil(n_required / effective))
            rows.append({"daily_all_applications_assumed": daily, "eligible_fraction_assumed": coverage,
                "experiment_fraction_of_eligible": traffic["main_experiment_fraction_of_eligible"],
                "daily_randomized_expected": effective, "required_randomized_total": n_required,
                "enrollment_days": days, "maturity_days_assumed": traffic["outcome_maturity_days_assumed"],
                "data_grace_days": traffic["data_grace_days"],
                "days_until_final_analysis": days + traffic["outcome_maturity_days_assumed"] + traffic["data_grace_days"],
                "enrollment_feasible_under_365_day_assumption": days <= traffic["max_enrollment_days_for_feasibility"]})
    return pd.DataFrame(rows)


def assignment_quality(assignments, config):
    r = config["randomization"]["challenger_probability"]
    rows = []
    for stratum, group in [("all", assignments), *list(assignments.groupby("risk_stratum", sort=True))]:
        n, n_b = len(group), int(group.arm.eq("B").sum())
        p = float(stats.binomtest(n_b, n, r).pvalue)
        rows.append({"risk_stratum": stratum, "n_total": n, "n_a": n - n_b, "n_b": n_b,
            "expected_b_fraction": r, "actual_b_fraction": n_b / n, "srm_p_value": p,
            "srm_alpha": config["randomization"]["srm_alpha"], "srm_flag": p < config["randomization"]["srm_alpha"]})
    return pd.DataFrame(rows)


def mean_comparison(a, b, alpha=.05):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if min(len(a),len(b)) < 2: raise ValueError("Need at least two values per arm")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va+vb)
    delta = b.mean()-a.mean()
    if se == 0:
        return {"delta": float(delta), "se": 0., "df": None, "ci_low": float(delta), "ci_high": float(delta), "simulation_p_value": float(delta == 0)}
    degrees = (va+vb)**2 / (va**2/(len(a)-1)+vb**2/(len(b)-1))
    critical = stats.t.ppf(1-alpha/2, degrees)
    return {"delta": float(delta), "se": float(se), "df": float(degrees), "ci_low": float(delta-critical*se),
        "ci_high": float(delta+critical*se), "simulation_p_value": float(2*stats.t.sf(abs(delta/se), degrees))}


def wilson_upper(events, total, alpha=.025):
    if total <= 0: return None
    z, q = stats.norm.ppf(1-alpha), events/total
    return float((q+z*z/(2*total)+z*np.sqrt(q*(1-q)/total+z*z/(4*total**2))) / (1+z*z/total))


def bootstrap_mean(values, repetitions, rng):
    """Multinomial cell resampling is exactly the IID empirical row bootstrap."""
    values = np.asarray(values, dtype=float)
    support, counts = np.unique(values, return_counts=True)
    return rng.multinomial(len(values), counts / len(values), size=repetitions) @ support / len(values)


def bootstrap(population, assignments, config):
    settings = config["inference"]
    evaluation = population.loc[population.policy_cohort.eq("policy_evaluation") & population.eligible_proxy]
    rng = np.random.default_rng(settings["random_seed"])
    paired = bootstrap_mean(evaluation.potential_profit_b - evaluation.potential_profit_a, settings["bootstrap_repetitions"], rng)
    a = assignments.loc[assignments.arm.eq("A"), "simulated_observed_net_profit"].to_numpy()
    b = assignments.loc[assignments.arm.eq("B"), "simulated_observed_net_profit"].to_numpy()
    unpaired = bootstrap_mean(b, settings["bootstrap_repetitions"], rng) - bootstrap_mean(a, settings["bootstrap_repetitions"], rng)
    rows = []
    for name, samples, estimate in (("paired_historical_policy_difference", paired, float((evaluation.potential_profit_b-evaluation.potential_profit_a).mean())),
                                    ("independent_arm_offline_replay", unpaired, float(b.mean()-a.mean()))):
        low, high = np.quantile(samples, [settings["alpha_two_sided"]/2, 1-settings["alpha_two_sided"]/2])
        rows.append({"method": name, "estimate_profit_delta": estimate, "ci_low": low, "ci_high": high,
            "bootstrap_se": float(samples.std(ddof=1)), "repetitions": len(samples),
            "scope": "conditional_historical_iid_simulation_not_live_causal_evidence"})
    return pd.DataFrame(rows), pd.DataFrame({"replicate": np.arange(len(paired)), "paired_historical_delta": paired, "independent_replay_delta": unpaired})


def simulated_means_variances(values, n, repetitions, rng):
    values = np.asarray(values, dtype=float)
    support, counts = np.unique(values, return_counts=True)
    sampled = rng.multinomial(n, counts/len(values), size=repetitions)
    mean = sampled @ support / n
    variance = np.maximum(0, (sampled @ (support*support) - n*mean*mean) / (n-1))
    return mean, variance


def monte_carlo(population, n_required, config):
    settings = config["inference"]
    evaluation = population.loc[population.policy_cohort.eq("policy_evaluation") & population.eligible_proxy]
    a, b = evaluation.potential_profit_a.to_numpy(), evaluation.potential_profit_b.to_numpy()
    fraction = config["randomization"]["challenger_probability"]
    rng = np.random.default_rng(settings["random_seed"]+1)
    rows, planned_draws = [], None
    for total in sorted({10000, 50000, 100000, len(evaluation), n_required}):
        na, nb = total-int(np.floor(total*fraction)), int(np.floor(total*fraction))
        ma, va = simulated_means_variances(a, na, settings["monte_carlo_repetitions"], rng)
        mb, vb = simulated_means_variances(b, nb, settings["monte_carlo_repetitions"], rng)
        se = np.sqrt(va/na+vb/nb)
        df = (va/na+vb/nb)**2 / ((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
        critical = stats.t.ppf(1-settings["alpha_two_sided"]/2, df)
        delta = mb-ma
        underlying = float(b.mean()-a.mean())
        reject = np.abs(delta/se)>critical
        null_reject = np.abs((delta-underlying)/se)>critical
        rate, null_rate = float(reject.mean()), float(null_reject.mean())
        rows.append({"n_total": total, "n_a": na, "n_b": nb, "empirical_historical_delta": underlying,
            "simulated_power_at_empirical_delta": rate, "power_monte_carlo_se": float(np.sqrt(rate*(1-rate)/len(delta))),
            "null_rejection_rate_centered": null_rate, "null_rate_monte_carlo_se": float(np.sqrt(null_rate*(1-null_rate)/len(delta))),
            "repetitions": len(delta), "sampling": "independent_with_replacement_from_evaluation_policy_marginals",
            "simulation_only": True})
        if total == n_required: planned_draws = pd.DataFrame({"replicate": np.arange(len(delta)), "independent_future_delta_simulation": delta})
    return pd.DataFrame(rows), planned_draws


def decision_diagnostics(assignments, quality, config):
    settings = config["inference"]
    groups = {arm: assignments.loc[assignments.arm.eq(arm)] for arm in ("A","B")}
    result = mean_comparison(*(groups[arm].simulated_observed_net_profit for arm in ("A","B")), settings["alpha_two_sided"])
    alpha = settings["alpha_two_sided"]/2
    z = stats.norm.ppf(1-alpha)
    guards = {}
    for column, key, margin, upper in (("simulated_observed_default", "default_incidence_ni", settings["default_incidence_ni_margin"], True),
                                        ("assigned_approved", "approval_rate_ni", settings["approval_rate_ni_margin"], False)):
        qa,qb = [float(groups[arm][column].mean()) for arm in ("A","B")]
        se = np.sqrt(sum(q*(1-q)/len(groups[arm]) for q,arm in ((qa,"A"),(qb,"B"))))
        bound = qb-qa+(z*se if upper else -z*se)
        guards[key] = {"delta_b_minus_a": qb-qa, "one_sided_bound": float(bound), "margin": margin,
                       "passed_in_simulation": bool(bound < margin if upper else bound > -margin)}
    bad = int(groups["B"].simulated_observed_default.sum())
    approved = int(groups["B"].assigned_approved.sum())
    bound = wilson_upper(bad, approved, alpha)
    guards["approved_default_absolute"] = {"wilson_upper_bound": bound, "limit": settings["absolute_default_rate_limit_approved"],
        "passed_in_simulation": bool(bound is not None and bound <= settings["absolute_default_rate_limit_approved"])}
    return {"welch_offline_replay": result, "guardrails": guards,
        "srm_overall_passed": bool(not quality.loc[quality.risk_stratum.eq("all"), "srm_flag"].iloc[0]),
        "positive_primary_ci_in_simulation": bool(result["ci_low"] > 0),
        "practical_gain_point_estimate_screen": bool(result["delta"] >= settings["minimum_practical_profit_gain"]),
        "live_experiment_completed": False, "production_rollout_ready": False}


def input_hashes(output, config):
    parent = business.Paths(output.root, config["parent_run_name"])
    files = [CONFIG, parent.database, parent.artifacts/"policy_freeze_manifest.json", parent.artifacts/"selected_policies.parquet",
             parent.artifacts/"selected_policy_application_outcomes.parquet", parent.artifacts/"experiment_policy_handoff.json",
             parent.reports/"run_config.json", parent.reports/"artifact_verification.json"]
    return {business.relative(path,output.root): business.sha256(path) for path in files}


def protocol_hashes(output):
    files = [Path(__file__), SQL_FILE, ROOT/"scripts/run_ab_test_design.py"]
    return {business.relative(path,output.root): business.sha256(path) for path in files}


def check_freeze(output, config):
    path = output.artifacts/"experiment_freeze_manifest.json"
    if not path.exists(): return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != input_hashes(output,config) or manifest["protocol_sha256"] != protocol_hashes(output):
        raise ValueError("Frozen experiment protocol/input changed; use a new --run-name")
    for name, checksum in manifest["output_sha256"].items():
        if business.sha256(output.root/name) != checksum: raise AssertionError("Frozen experiment output changed")
    return manifest

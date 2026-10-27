"""Policy-only development split, frozen thresholds and explicit loan economics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from . import business_analysis_common as business
except ImportError:
    import business_analysis_common as business

ROOT = business.ROOT
CONFIG = ROOT / "configs/approval_policy_simulation.json"
SQL_FILES = [ROOT / "sql/04_policy_simulation" / name for name in
             ("01_economic_results.sql", "02_selected_policies.sql")]
ECONOMIC_FIELDS = ("principal", "interest_rate_for_horizon", "loss_given_default",
                   "funding_rate_for_horizon", "servicing_cost_per_approved", "application_cost_per_application")
EXPORT_TABLES = ("economic_assumptions", "policy_candidates", "policy_constraints",
                 "approval_economic_results", "matched_approval_comparison",
                 "selected_policy_definitions", "selected_policy_results")


def load_config(path=CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    rates = config["target_approval_rates"]
    if rates != sorted(set(rates)) or not rates or rates[0] != 0 or rates[-1] != 1:
        raise ValueError("Approval-rate grid must be ordered, unique and include 0 and 1")
    if not all(np.isfinite(rate) and 0 <= rate <= 1 for rate in rates):
        raise ValueError("Invalid approval-rate target")
    if not 0 < config["policy_split"]["evaluation_fraction"] < 1 or not config["policy_split"]["stratify_label"]:
        raise ValueError("Require a stratified policy development/evaluation split")
    limits = config["constraints"]
    if not 0 <= limits["max_development_default_rate"] <= 1 or not 0 < limits["min_development_approval_rate"] <= 1:
        raise ValueError("Invalid policy constraints")
    economics_frame(config)  # Validate every economic scenario, not just base.
    return config


def economics_frame(config):
    base, scenarios = config["economics"], config["sensitivity_scenarios"]
    ids = [s["scenario_id"] for s in scenarios]
    if len(set(ids)) != len(ids) or "base" not in ids or base["horizon_months"] <= 0:
        raise ValueError("Need unique economic scenarios and a positive common horizon")
    rows = []
    for scenario in scenarios:
        overrides = scenario["overrides"]
        if set(overrides) - set(ECONOMIC_FIELDS) or len(overrides) > 1:
            raise ValueError("Sensitivity scenarios must change at most one economic input")
        if scenario["scenario_id"] == "base" and overrides:
            raise ValueError("Base economics cannot have overrides")
        values = {**base, **overrides}
        if not all(np.isfinite(values[field]) and values[field] >= 0 for field in ECONOMIC_FIELDS):
            raise ValueError("Economic assumptions must be finite and nonnegative")
        if values["principal"] <= 0 or not 0 <= values["loss_given_default"] <= 1:
            raise ValueError("Require positive principal and LGD between 0 and 1")
        income = values["principal"] * values["interest_rate_for_horizon"]
        loss = values["principal"] * values["loss_given_default"]
        funding = values["principal"] * values["funding_rate_for_horizon"]
        denominator = income + loss
        rows.append({"scenario_id": scenario["scenario_id"], **values,
            "incremental_break_even_p_repay": (loss + funding + values["servicing_cost_per_approved"]) / denominator
              if denominator > 0 else None})
    return pd.DataFrame(rows)


def policy_split(predictions, config):
    training = predictions.query("dataset == 'train' and model_role == 'champion'").sort_values("id")
    if training.id.duplicated().any() or not training.label.isin([0, 1]).all():
        raise ValueError("Invalid policy-split training rows")
    development, evaluation = train_test_split(np.arange(len(training)),
        test_size=config["policy_split"]["evaluation_fraction"], stratify=training.label.to_numpy(),
        random_state=config["policy_split"]["random_seed"])
    cohort = np.full(len(training), "policy_development", dtype=object)
    cohort[evaluation] = "policy_evaluation"
    result = pd.DataFrame({"id": training.id.to_numpy(), "policy_cohort": cohort})
    if len(development) + len(evaluation) != len(result): raise AssertionError("Invalid split coverage")
    return result


def budget_threshold(scores, target_rate):
    """Keep whole score ties; never exceed floor(n*target) to force a quota.

    Approval is p_repay >= threshold. If a boundary tie would exceed the budget,
    exclude the entire tie using the next representable higher float.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Require finite repayment scores")
    if not np.isfinite(target_rate) or not 0 <= target_rate <= 1: raise ValueError("Invalid approval budget")
    budget = int(np.floor(len(scores) * target_rate))
    if budget == 0: return "reject_all", None, 0
    if budget == len(scores): return "approve_all", 0.0, len(scores)
    boundary = float(np.sort(scores)[::-1][budget - 1])
    count = int(np.sum(scores >= boundary))
    if count > budget:
        count = int(np.sum(scores > boundary))
        if count == 0: return "reject_all", None, 0
        boundary = float(np.nextafter(boundary, np.inf))
    return "score_threshold", boundary, count


def approval_mask(scores, rule, threshold):
    scores = np.asarray(scores)
    if rule == "reject_all": return np.zeros(len(scores), dtype=bool)
    if rule == "approve_all": return np.ones(len(scores), dtype=bool)
    if rule != "score_threshold" or threshold is None or pd.isna(threshold): raise ValueError("Invalid decision rule")
    return scores >= threshold


def fit_candidates(predictions, split, config):
    development_ids = set(split.loc[split.policy_cohort.eq("policy_development"), "id"])
    rows = []
    training = predictions.loc[predictions.dataset.eq("train") & predictions.id.isin(development_ids)]
    for (version, role), group in training.groupby(["model_version", "model_role"], sort=True):
        for rate in config["target_approval_rates"]:
            rule, threshold, approved = budget_threshold(group.p_repay.to_numpy(), rate)
            rows.append({"policy_id": f"{role}_target_{rate:.4f}", "model_version": version, "model_role": role,
                "target_approval_rate": rate, "threshold": threshold, "decision_rule": rule,
                "development_approved_rows": approved, "development_total_rows": len(group)})
    return pd.DataFrame(rows)


def select_policies(results, candidates, config):
    """Current selection uses base/development labels ONLY; evaluation never enters."""
    limits = config["constraints"]
    eligible = results.loc[results.scenario_id.eq("base") & results.policy_cohort.eq("policy_development")
        & results.observed_default_rate_approved.le(limits["max_development_default_rate"])
        & results.approval_rate.ge(limits["min_development_approval_rate"])]
    selected = []
    for role, arm in (("champion", "A"), ("challenger", "B")):
        rows = eligible.loc[eligible.model_role.eq(role)].copy()
        if rows.empty: raise ValueError(f"No feasible {role} policy under the predeclared development constraints")
        rows["profit_sort"] = rows.historical_profit_per_application.round(6)
        winner = rows.sort_values(["profit_sort", "observed_default_rate_approved", "approval_rate", "policy_id"],
                                  ascending=[False, True, True, True]).iloc[0]
        definition = candidates.loc[candidates.policy_id.eq(winner.policy_id)].iloc[0].to_dict()
        selected.append({"prospective_arm": arm, **definition})
    for rate, arm in ((1.0, "approve_all_control"), (0.0, "reject_all_control")):
        definition = candidates.loc[candidates.model_role.eq("champion") & candidates.target_approval_rate.eq(rate)].iloc[0].to_dict()
        selected.append({"prospective_arm": arm, **definition})
    return pd.DataFrame(selected)


def economic_metrics(labels, scores, approved, economics):
    """Independent row-level cash flows; returned principal is never profit."""
    labels, scores, approved = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float), np.asarray(approved, dtype=bool)
    if not (len(labels) == len(scores) == len(approved)) or not len(labels): raise ValueError("Invalid economic cohort")
    available = np.isfinite(labels)
    if available.any() and not available.all(): raise ValueError("A cohort must be fully labeled or unlabeled")
    if available.all() and not np.isin(labels, [0, 1]).all(): raise ValueError("Invalid repayment label")
    p, i, lgd, f, service, app = [economics[field] for field in ECONOMIC_FIELDS]
    income, loss, funding = p * i, p * lgd, p * f
    implied = approved * (scores * income - (1 - scores) * loss - funding - service) - app
    realized = approved * (labels * income - (1 - labels) * loss - funding - service) - app if available.all() else np.full(len(labels), np.nan)
    bad = np.sum(1 - labels[approved]) if available.all() else np.nan
    good = np.sum(labels[approved]) if available.all() else np.nan
    count = int(approved.sum())
    return {"total_applications": len(labels), "labeled_applications": int(available.sum()),
        "approved_applications": count, "rejected_applications": len(labels) - count,
        "approved_default_rows": bad,
        "approved_repaid_rows": good,
        "approval_rate": count / len(labels), "observed_default_rate_approved": bad / count if count else np.nan,
        "historical_simulated_interest_income": good * income,
        "historical_simulated_principal_loss": bad * loss,
        "assumed_funding_cost": count * funding, "assumed_servicing_cost": count * service,
        "assumed_application_cost": len(labels) * app,
        "historical_simulated_net_profit": realized.sum() if available.all() else np.nan,
        "historical_profit_per_application": realized.mean() if available.all() else np.nan,
        "historical_profit_per_approved_loan": realized.sum() / count if available.all() and count else np.nan,
        "raw_score_implied_net_profit": implied.sum(), "raw_score_implied_profit_per_application": implied.mean(),
        "mean_p_repay_approved": scores[approved].mean() if count else np.nan}, realized, implied


def paths(run_name="approval_policy_simulation", root=ROOT):
    return business.Paths(root=root, run_name=run_name)


def input_hashes(output_paths, config, config_path=CONFIG):
    parent = business.Paths(output_paths.root, config["parent_run_name"])
    files = [Path(config_path), parent.manifest, parent.predictions, parent.registry,
             parent.reports / "run_config.json", parent.reports / "artifact_verification.json"]
    return {business.relative(p, output_paths.root): business.sha256(p) for p in files}


def protocol_hashes(output_paths):
    return {business.relative(p, output_paths.root): business.sha256(p)
            for p in [Path(__file__), *SQL_FILES]}


def check_manifest(output_paths, config, config_path=CONFIG):
    manifest_path = output_paths.artifacts / "policy_freeze_manifest.json"
    if not manifest_path.exists(): return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != input_hashes(output_paths, config, config_path) or manifest["protocol_sha256"] != protocol_hashes(output_paths):
        raise ValueError("Frozen policy inputs/protocol changed; use a new --run-name")
    for file, checksum in manifest["output_sha256"].items():
        if business.sha256(output_paths.root / file) != checksum: raise ValueError("Frozen policy output changed")
    return manifest


def prepare_inputs(output_paths, config):
    parent = business.Paths(output_paths.root, config["parent_run_name"])
    predictions = pd.read_parquet(parent.predictions)
    business.validate_predictions(predictions)
    output_paths.artifacts.mkdir(parents=True, exist_ok=True)
    existing = check_manifest(output_paths, config)
    if existing:
        print("Reusing frozen policy split and thresholds", flush=True)
        return predictions, pd.read_parquet(output_paths.artifacts / "policy_cohorts.parquet"), pd.read_parquet(output_paths.artifacts / "policy_candidates.parquet")
    split = policy_split(predictions, config)
    candidates = fit_candidates(predictions, split, config)
    split.to_parquet(output_paths.artifacts / "policy_cohorts.parquet", index=False)
    candidates.to_parquet(output_paths.artifacts / "policy_candidates.parquet", index=False)
    return predictions, split, candidates


def freeze_definitions(output_paths, config, selected):
    # Write definitions once. Repeated runs keep the same cutoffs and policy IDs.
    path = output_paths.artifacts / "selected_policies.parquet"
    existing = check_manifest(output_paths, config)
    if existing:
        pd.testing.assert_frame_equal(selected.reset_index(drop=True), pd.read_parquet(path), check_dtype=False)
        return existing
    selected.to_parquet(path, index=False)
    outputs = [output_paths.artifacts / name for name in ("policy_cohorts.parquet", "policy_candidates.parquet", "selected_policies.parquet")]
    manifest = {"protocol_version": 1, "input_sha256": input_hashes(output_paths, config),
        "protocol_sha256": protocol_hashes(output_paths),
        "output_sha256": {business.relative(p, output_paths.root): business.sha256(p) for p in outputs},
        "selection_cohort": "policy_development", "selection_scenario": "base",
        "evaluation_scope": "new_policy_split_of_reused_development_oof_not_independent_model_test",
        "threshold_ties": "whole_ties_kept_within_development_approval_budget",
        "probability_calibrated": False, "experiment_randomization_completed": False}
    business.write_json(output_paths.artifacts / "policy_freeze_manifest.json", manifest)
    return manifest


def execute_sql(connection, output_paths, config, candidates):
    connection.register("_policy_candidates", candidates)
    connection.register("_economic_assumptions", economics_frame(config))
    connection.register("_policy_constraints", pd.DataFrame([config["constraints"]]))
    parent = business.Paths(output_paths.root, config["parent_run_name"])
    sql = SQL_FILES[0].read_text(encoding="utf-8")
    for token, path in (("__SPLIT_PARQUET__", output_paths.artifacts / "policy_cohorts.parquet"),
                        ("__PREDICTIONS_PARQUET__", parent.predictions)):
        sql = sql.replace(token, path.resolve().as_posix().replace("'", "''"))
    connection.execute(sql).fetchall()
    for name in ("_policy_candidates", "_economic_assumptions", "_policy_constraints"): connection.unregister(name)
    results = connection.table("approval_economic_results").fetchdf()
    existing = check_manifest(output_paths, config)
    selected = pd.read_parquet(output_paths.artifacts / "selected_policies.parquet") if existing else select_policies(results, candidates, config)
    connection.register("_selected_policies", selected)
    connection.execute(SQL_FILES[1].read_text(encoding="utf-8")).fetchall()
    connection.unregister("_selected_policies")
    return selected

"""Audit offline eligibility, allocation, cash flows, power and frozen artifacts."""
from __future__ import annotations

import argparse
import json

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import experiment_design_common as experiment
except ImportError:
    import experiment_design_common as experiment


def compare_frames(actual, expected, keys, exact=True):
    actual=actual.sort_values(keys).reset_index(drop=True)
    expected=expected.sort_values(keys).reset_index(drop=True)
    # DuckDB Parquet exports have no pandas extension-dtype metadata. A NULL
    # BOOLEAN can return as Python None; normalize it without changing values.
    nullable_booleans={name:dtype for name,dtype in expected.dtypes.items() if isinstance(dtype,pd.BooleanDtype)}
    if nullable_booleans: actual=actual.astype(nullable_booleans)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=exact,rtol=1e-12,atol=1e-9)


def verify_connection(connection,population,assignments,config):
    experiment.validate_population(population)
    compare_frames(connection.table("experiment_population").fetchdf(),population,["dataset","id"])
    compare_frames(connection.table("experiment_assignments").fetchdf(),assignments,["dataset","id"])
    gate=population[["score_a","score_b"]].min(axis=1).ge(config["eligibility"]["common_min_raw_repayment_score"])
    np.testing.assert_array_equal(gate,population.eligible_proxy)
    expected_keys=population.loc[gate & population.policy_cohort.eq("policy_evaluation") & population.dataset.eq("train"),["dataset","id"]]
    compare_frames(assignments[["dataset","id"]],expected_keys,["dataset","id"])
    if assignments.label.isna().any() or assignments.observed_dpd30.notna().any():
        raise AssertionError("Only labeled replay rows permitted; DPD is unavailable")
    if not assignments.simulation_only.all(): raise AssertionError("Missing simulation scope")
    r=config["randomization"]["challenger_probability"]
    hashed=np.asarray([experiment.stable_uniform(d,i,config["randomization"]["salt"]) for d,i in zip(assignments.dataset,assignments.id)])
    np.testing.assert_array_equal(assignments.assignment_uniform,hashed)
    np.testing.assert_array_equal(assignments.arm,np.where(hashed<r,"B","A"))
    np.testing.assert_array_equal(assignments.assigned_approved,np.where(assignments.arm.eq("B"),assignments.approved_b,assignments.approved_a))
    np.testing.assert_allclose(assignments.simulated_observed_net_profit,
        np.where(assignments.arm.eq("B"),assignments.potential_profit_b,assignments.potential_profit_a),atol=0,rtol=0)
    metrics=connection.table("experiment_metrics").fetchdf()
    if set(metrics.arm)!={"A","B"}: raise AssertionError("Missing arm")
    for row in metrics.itertuples(index=False):
        group=assignments.loc[assignments.arm.eq(row.arm)]
        n,approved,bad,repaid=len(group),int(group.assigned_approved.sum()),int((group.assigned_approved & group.label.eq(0)).sum()),int((group.assigned_approved & group.label.eq(1)).sum())
        for field,value in {"randomized_applications":n,"approved_applications":approved,"simulated_default_events":bad}.items():
            if getattr(row,field)!=value: raise AssertionError("SQL count mismatch")
        for field,value in {"approval_rate":approved/n,"simulated_default_incidence_itt":bad/n,
            "simulated_default_rate_approved":bad/approved if approved else np.nan,
            "simulated_repayment_rate_approved":repaid/approved if approved else np.nan,
            "simulated_net_profit":group.simulated_observed_net_profit.sum(),
            "simulated_profit_per_randomized_application_itt":group.simulated_observed_net_profit.mean(),
            "simulated_profit_variance":group.simulated_observed_net_profit.var(ddof=1)}.items():
            np.testing.assert_allclose(getattr(row,field),value,atol=1e-8,rtol=1e-11,equal_nan=True,err_msg=field)
        if not pd.isna(row.actual_dpd30_rate): raise AssertionError("Invented DPD outcome")
    coverage=connection.table("eligibility_summary").fetchdf()
    if len(coverage)!=3: raise AssertionError("Missing population scope")
    for row in coverage.itertuples(index=False):
        group=population.loc[population.policy_cohort.eq(row.policy_cohort)]
        eligible=group.loc[group.eligible_proxy]
        if row.applications!=len(group) or row.eligible_proxy_applications!=len(eligible) or row.eligible_disagreements!=eligible.decision_disagreement.sum():
            raise AssertionError("Eligibility coverage mismatch")
        expected=(eligible.potential_profit_b-eligible.potential_profit_a).mean() if row.dataset=="train" else np.nan
        np.testing.assert_allclose(row.paired_historical_profit_delta,expected,atol=1e-10,rtol=0,equal_nan=True)
    concentration=connection.table("risk_score_concentration").fetchdf()
    for row in concentration.itertuples(index=False):
        group=assignments.loc[assignments.arm.eq(row.arm) & assignments.risk_stratum.eq(row.risk_stratum)]
        arm_total=int(assignments.loc[assignments.arm.eq(row.arm),"assigned_approved"].sum())
        if row.randomized_applications!=len(group) or row.approved_applications!=group.assigned_approved.sum(): raise AssertionError("Concentration counts mismatch")
        np.testing.assert_allclose(row.share_of_arm_approved,row.approved_applications/arm_total if arm_total else np.nan,rtol=0,atol=1e-12,equal_nan=True)
    expected_power,n_required=experiment.planning(population,config)
    saved_power=connection.table("power_analysis").fetchdf()
    compare_frames(saved_power,expected_power,["endpoint","fraction_b","effect_or_margin"])
    for row in saved_power.itertuples(index=False):
        variance=row.planning_variance_a/(1-row.fraction_b)+row.planning_variance_b/row.fraction_b
        z=stats.norm.ppf(1-row.alpha if row.one_sided else 1-row.alpha/2)
        def independent_power(n):
            if variance==0: return 1.
            shift=row.effect_or_margin/np.sqrt(variance/n)
            return stats.norm.cdf(shift-z) if row.one_sided else stats.norm.cdf(shift-z)+stats.norm.cdf(-shift-z)
        if independent_power(row.n_total)<row.target_power-1e-12 or (row.n_total>4 and independent_power(row.n_total-1)>=row.target_power):
            raise AssertionError("Sample-size boundary mismatch")
    compare_frames(connection.table("experiment_duration").fetchdf(),experiment.duration_table(n_required,config),["daily_all_applications_assumed","eligible_fraction_assumed"])
    compare_frames(connection.table("assignment_quality").fetchdf(),experiment.assignment_quality(assignments,config),["risk_stratum"])
    boot,draws=experiment.bootstrap(population,assignments,config)
    compare_frames(connection.table("bootstrap_results").fetchdf(),boot,["method"])
    mc,planned_draws=experiment.monte_carlo(population,n_required,config)
    compare_frames(connection.table("monte_carlo_results").fetchdf(),mc,["n_total"])
    return {"status":"passed","offline_randomized_rows":len(assignments),"required_randomized_total":n_required,
        "common_gate_and_shared_evaluation_scope_verified":True,"stable_hash_allocation_verified":True,
        "sql_metrics_match_independent_row_calculations":True,"planning_uses_development_marginal_variances_only":True,
        "normal_power_minimum_integer_boundary_verified":True,"bootstrap_and_monte_carlo_reproducible":True,
        "unavailable_test_and_dpd_outcomes_not_invented":True,"live_experiment_completed":False},draws,planned_draws


def verify(output):
    config=experiment.load_config()
    manifest=experiment.check_freeze(output,config)
    if manifest is None: raise AssertionError("Experiment freeze not complete")
    parent=experiment.paths(config["parent_run_name"],output.root)
    population=experiment.make_population(parent.database,config)
    assignments=experiment.assign(population,config)
    with duckdb.connect(str(output.database),read_only=True) as connection:
        audit,draws,future_draws=verify_connection(connection,population,assignments,config)
        keys={"eligibility_summary":["dataset","policy_cohort"],"experiment_metrics":["arm"],
              "risk_score_concentration":["arm","risk_stratum"],"assignment_quality":["risk_stratum"],
              "power_analysis":["endpoint","fraction_b","effect_or_margin"],
              "experiment_duration":["daily_all_applications_assumed","eligible_fraction_assumed"],
              "bootstrap_results":["method"],"monte_carlo_results":["n_total"]}
        for table in experiment.CSV_TABLES:
            expected=connection.table(table).fetchdf()
            floats=[name for name,dtype in expected.dtypes.items() if pd.api.types.is_float_dtype(dtype)]
            actual=pd.read_csv(output.reports/"tables"/f"{table}.csv",
                dtype={name:dtype for name,dtype in expected.dtypes.items() if name not in floats},
                converters={name:lambda value:float(value) if value else np.nan for name in floats}).astype(expected.dtypes.to_dict())
            compare_frames(actual,expected,keys[table])
    for table,expected in (("experiment_population",population),("experiment_assignments",assignments)):
        compare_frames(pd.read_parquet(output.artifacts/f"{table}.parquet"),expected,["dataset","id"])
    compare_frames(pd.read_parquet(output.artifacts/"bootstrap_draws.parquet"),draws,["replicate"])
    compare_frames(pd.read_parquet(output.artifacts/"monte_carlo_planned_draws.parquet"),future_draws,["replicate"])
    summary=json.loads((output.reports/"summary.json").read_text(encoding="utf-8"))
    protocol=json.loads((output.artifacts/"experiment_protocol.json").read_text(encoding="utf-8"))
    handoff=json.loads((parent.artifacts/"experiment_policy_handoff.json").read_text(encoding="utf-8"))
    for document in (summary,protocol):
        if document["live_experiment_completed"] or document["production_rollout_ready"] or document["scope"]!="offline_design_and_simulation_only":
            raise AssertionError("False live experiment claim")
        if document["selected_policies"]!=handoff["selected_policies"]: raise AssertionError("Frozen policy definition changed")
        if document["required_randomized_total"]!=audit["required_randomized_total"]: raise AssertionError("Summary sample size mismatch")
    quality=experiment.assignment_quality(assignments,config)
    if summary["offline_replay_diagnostics"]!=experiment.decision_diagnostics(assignments,quality,config): raise AssertionError("Decision diagnostic mismatch")
    run_path=output.reports/"run_config.json"
    if run_path.exists():
        run=json.loads(run_path.read_text(encoding="utf-8"))
        if run["manifest_sha256"]!=experiment.business.sha256(output.artifacts/"experiment_freeze_manifest.json"):
            raise AssertionError("Run manifest checksum differs")
    audit.update(frozen_input_protocol_and_output_hashes_verified=True,csv_and_parquet_exports_verified=True,
                 policy_threshold_full_precision_preserved=True,production_rollout_ready=False)
    return audit


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="ab_test_design")
    args=parser.parse_args()
    output=experiment.paths(args.run_name)
    audit=verify(output)
    experiment.business.write_json(output.reports/"artifact_verification.json",audit)
    run_path=output.reports/"run_config.json"
    if not run_path.exists():
        experiment.business.write_json(run_path,{"status":"completed","elapsed_seconds":None,
            "config":experiment.load_config(),"duckdb_version":duckdb.__version__,
            "manifest_sha256":experiment.business.sha256(output.artifacts/"experiment_freeze_manifest.json"),
            "finalized_by":"standalone_artifact_verification"})
    print(json.dumps(audit,indent=2),flush=True)

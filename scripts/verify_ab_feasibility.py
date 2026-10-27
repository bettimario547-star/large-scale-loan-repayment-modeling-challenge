"""Independent row/SQL audit and reproducibility of the redesigned experiment."""
from __future__ import annotations
import argparse
import json

import duckdb
import numpy as np
import pandas as pd

try:
    from . import ab_feasibility_common as f
    from .verify_ab_test_design import compare_frames
    from .verify_aa_validation import read_csv_typed
except ImportError:
    import ab_feasibility_common as f
    from verify_ab_test_design import compare_frames
    from verify_aa_validation import read_csv_typed


def audit_sql(con,assignments,target,est,cells):
    observed = con.table("observed_stratum_metrics").fetchdf()
    # Independent DuckDB/NumPy variance reductions differ in final float bits.
    compare_frames(observed,cells.drop(columns="weight"),["stratum","arm"],exact=False)
    summary = con.table("sql_primary_estimate").fetchdf().iloc[0]
    np.testing.assert_allclose([summary.full_eligible_profit_delta,summary.independent_stratified_se,summary.full_eligible_default_delta],
        [est["delta"],est["se"],est["default_delta"]],atol=1e-8,rtol=1e-12)
    expected = target.loc[target.decision_disagreement].groupby("analysis_stratum").size()
    actual = con.table("target_stratum_weights").fetchdf().sort_values("stratum")
    np.testing.assert_array_equal(actual.applications,expected.to_numpy())
    np.testing.assert_allclose(actual.weight_of_full_eligible,expected.to_numpy()/len(target),rtol=0,atol=1e-12)
    for row in con.table("experiment_metrics").fetchdf().itertuples(index=False):
        g = assignments.loc[assignments.arm.eq(row.arm)]
        if row.randomized_disagreements!=len(g) or row.approved!=g.assigned_approved.sum() or row.defaults!=g.simulated_observed_default.sum():
            raise AssertionError("SQL numerator/ITT denominator mismatch")
        np.testing.assert_allclose(row.simulated_profit,g.simulated_observed_net_profit.sum(),rtol=0,atol=1e-8)
        if not pd.isna(row.actual_dpd30_rate) or not row.simulation_only: raise AssertionError("Fabricated DPD/live results")


def audit_calibration(cal,config):
    if len(cal)!=20 or cal.duplicated(["case_id","endpoint","alpha"]).any(): raise AssertionError("Missing/duplicate calibration checks")
    for row in cal.itertuples(index=False):
        error = config["calibration_familywise_error"]/len(cal)
        low,high = f.aa.clopper_pearson(row.rejections,row.valid_repetitions,error)
        tolerance = config["calibration_tolerances"][str(row.alpha)]
        np.testing.assert_allclose([row.false_positive_rate,row.simultaneous_ci_low,row.simultaneous_ci_high],
            [row.rejections/row.valid_repetitions,low,high],atol=1e-12,rtol=0)
        if row.repetitions!=config["calibration_repetitions"] or row.passed!=(row.valid_repetitions==row.repetitions and low>=row.alpha-tolerance and high<=row.alpha+tolerance):
            raise AssertionError("Post-hoc calibration rule or invalid result")
    if not cal.passed.all(): raise AssertionError("Matching calibration not certified")


def verify(output,require_freeze=True,deep=True,progress=None):
    config = f.load_config()
    if require_freeze and not f.check_manifest(output,config): raise AssertionError("Incomplete freeze")
    lock = json.loads((output.artifacts/"preanalysis_design_lock.json").read_text(encoding="utf-8"))
    if lock["evaluation_outcomes_used_to_select_design"] or lock["input_sha256"]!=f.input_hashes(config) or lock["protocol_sha256"]!=f.protocol_hashes():
        raise AssertionError("Design lock/provenance mismatch")
    for path,digest in lock["design_file_sha256"].items():
        if f.business.sha256(f.ROOT/path)!=digest: raise AssertionError("Design changed after lock")
    source = pd.read_parquet(f.ROOT/f"artifacts/{config['parent_run_name']}/experiment_population.parquet")
    population = f.enrich(source)
    strata,meta,candidates,protocol,old_power = f.plan(population,config,f.legacy.load_config())
    protocol["selected_policies"] = json.loads((f.ROOT/f"artifacts/{config['parent_run_name']}/experiment_protocol.json").read_text(encoding="utf-8"))["selected_policies"]
    if json.loads((output.artifacts/"design_protocol.json").read_text(encoding="utf-8"))!=protocol: raise AssertionError("Planning differs")
    assigned = f.assign(population,config)
    stored = pd.read_parquet(output.artifacts/"assignments.parquet")
    f.validate_assignments(population,stored,config)
    target = f.cohort(population,"policy_evaluation")
    est,cells = f.estimate(stored,target,config)
    with duckdb.connect(str(output.database),read_only=True) as con:
        compare_frames(con.table("experiment_population").fetchdf(),population,["dataset","id"])
        compare_frames(con.table("experiment_assignments").fetchdf(),assigned,["dataset","id"])
        audit_sql(con,stored,target,est,cells)
        tables = {"planning_strata":(strata,["stratum"]),"design_candidates":(candidates,["design","disagreement_exposure"]),
            "primary_stratum_estimates":(cells,["stratum","arm"]),"assignment_quality":(f.quality(stored,config),["stratum"]),
            "experiment_duration":(f.duration(population,protocol,config),["coverage_source","daily_applications","eligible_fraction"]),
            "original_power_reference":(old_power,["endpoint","fraction_b","effect_or_margin"])}
        for name,(expected,keys) in tables.items(): compare_frames(con.table(name).fetchdf(),expected,keys)
        cal = con.table("matched_calibration").fetchdf()
        audit_calibration(cal,config)
        boot,bootdraws = f.bootstrap(stored,target,config)
        compare_frames(con.table("bootstrap_results").fetchdf(),boot,["method"])
        compare_frames(pd.read_parquet(output.artifacts/"bootstrap_draws.parquet"),bootdraws,["replicate"])
        for path in sorted((output.reports/"tables").glob("*.csv")):
            expected = con.table(path.stem).fetchdf()
            actual = read_csv_typed(path,expected)
            # SQL totals may be unordered; compare sorted numeric/string records.
            order = [name for name in expected.columns if not pd.api.types.is_float_dtype(expected[name].dtype)] or list(expected.columns)
            compare_frames(actual,expected,order)
        compare_frames(read_csv_typed(output.reports/"planning_strata.csv",strata),strata,["stratum"])
        compare_frames(read_csv_typed(output.reports/"design_candidates.csv",candidates),candidates,["design","disagreement_exposure"])
        if deep:
            expected_cal,draws = f.calibrate(population,protocol,config,progress)
            compare_frames(cal,expected_cal,["case_id","endpoint","alpha"])
            compare_frames(pd.read_parquet(output.artifacts/"calibration_draws.parquet"),draws,["case_id","replicate"])
            compare_frames(con.table("matched_aa_hash_replays").fetchdf(),f.hash_replays(population,config,progress),["replicate"])
            expected_sim,simdraws = f.simulations(population,protocol,config,progress)
            compare_frames(con.table("ab_simulation_results").fetchdf(),expected_sim,["case_id"])
            compare_frames(pd.read_parquet(output.artifacts/"ab_draws.parquet"),simdraws,["case_id","replicate"])
    summary = json.loads((output.reports/"summary.json").read_text(encoding="utf-8"))
    if summary["replay"]!=est or summary["protocol"]!=protocol: raise AssertionError("Summary differs from row audit")
    if not summary["champion_retained"] or summary["challenger_promoted"] or summary["production_rollout_ready"] or summary["live_experiment_completed"]:
        raise AssertionError("Unsupported live promotion claim")
    return {"status":"passed","randomized_disagreements":len(stored),"full_eligible_target_rows":len(target),
        "locked_development_only_plan_verified":True,"unchanged_frozen_models_thresholds_and_risk_margins":True,
        "label_blind_gate_strata_and_stable_hash_verified":True,"full_population_weights_itt_and_sql_verified":True,
        "same_champion_aa_and_profit_ni_boundary_calibration_verified":True,
        "independent_stratified_standard_errors_verified":True,"all_repeated_samples_reproduced":deep,
        "frozen_checksums_verified":require_freeze,"champion_retained":True,"challenger_promoted":False,
        "live_experiment_completed":False}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="ab_feasibility_v2")
    args = parser.parse_args()
    result = verify(f.paths(args.run_name),progress=lambda s:print(s,flush=True))
    f.business.write_json(f.paths(args.run_name).reports/"artifact_verification.json",result)
    print(json.dumps(result,indent=2),flush=True)

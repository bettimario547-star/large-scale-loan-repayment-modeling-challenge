"""Independent audit of Champion/Champion identity, SQL and null calibration."""
from __future__ import annotations

import argparse
import json

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import aa_validation_common as aa
    from .verify_ab_test_design import compare_frames
except ImportError:
    import aa_validation_common as aa
    from verify_ab_test_design import compare_frames


def validate_metrics(metrics, assignments):
    if set(metrics.arm) != {"A","B"} or len(metrics) != 2: raise AssertionError("Missing/duplicate arm metrics")
    for row in metrics.itertuples(index=False):
        rows = assignments.loc[assignments.arm.eq(row.arm)]
        n,approved,bad,repaid = len(rows),int(rows.assigned_approved.sum()),int(rows.simulated_observed_default.sum()),int(rows.simulated_observed_repaid.sum())
        for field,value in {"randomized_applications":n,"approved_applications":approved,"simulated_default_events":bad}.items():
            if getattr(row,field) != value: raise AssertionError("Metric denominator/count changed")
        expected = {"approval_rate":approved/n,"simulated_default_incidence_itt":bad/n,
            "simulated_default_rate_approved":bad/approved if approved else np.nan,
            "simulated_repayment_rate_approved":repaid/approved if approved else np.nan,
            "simulated_net_profit":rows.simulated_observed_net_profit.sum(),
            "simulated_profit_per_randomized_application_itt":rows.simulated_observed_net_profit.mean(),
            "simulated_profit_variance":rows.simulated_observed_net_profit.var(ddof=1)}
        for field,value in expected.items():
            np.testing.assert_allclose(getattr(row,field),value,atol=1e-8,rtol=1e-11,equal_nan=True,err_msg=field)
        if not pd.isna(row.actual_dpd30_rate) or not row.simulation_only: raise AssertionError("False observed outcome/scope")


def validate_calibration(calibration, srm, config):
    keys = ["case_id","endpoint","alpha"]
    if calibration.duplicated(keys).any(): raise AssertionError("Duplicate calibration result")
    comparisons = len(calibration) + len(srm)
    error = config["calibration_familywise_error"] / comparisons
    for row in calibration.itertuples(index=False):
        if row.repetitions != config["calibration_repetitions"] or row.valid_repetitions > row.repetitions:
            raise AssertionError("Replication count changed")
        low, high = aa.clopper_pearson(row.rejections,row.valid_repetitions,error)
        if low is None:
            if row.calibration_passed: raise AssertionError("Degenerate data certified")
            continue
        np.testing.assert_allclose([row.fpr_simultaneous_ci_low,row.fpr_simultaneous_ci_high,row.false_positive_rate],
            [low,high,row.rejections/row.valid_repetitions],atol=1e-12,rtol=0)
        np.testing.assert_allclose(row.zero_effect_interval_coverage,1-row.false_positive_rate,atol=1e-12,rtol=0)
        tolerance = config["fpr_absolute_tolerances"][str(row.alpha)]
        passed = row.valid_repetitions == row.repetitions and low >= row.alpha-tolerance and high <= row.alpha+tolerance
        if row.calibration_passed != passed or row.fpr_tolerance != tolerance:
            raise AssertionError("Post-hoc calibration envelope/result")
    for row in srm.itertuples(index=False):
        _,upper = aa.clopper_pearson(row.false_srm_flags,row.repetitions,error)
        np.testing.assert_allclose(row.simultaneous_ci_high,upper,atol=1e-12,rtol=0)
        if row.calibration_passed != (upper <= config["srm_fpr_upper_limit"]): raise AssertionError("SRM envelope changed")


def read_csv_typed(path, expected):
    floats = [name for name,dtype in expected.dtypes.items() if pd.api.types.is_float_dtype(dtype)]
    actual = pd.read_csv(path,dtype={name:dtype for name,dtype in expected.dtypes.items() if name not in floats},
        converters={name:lambda value:float(value) if value else np.nan for name in floats})
    return actual.astype(expected.dtypes.to_dict())


def verify(output, require_freeze=True, deep=True, progress=None):
    config = aa.load_config()
    manifest = aa.check_freeze(output,config) if require_freeze else None
    if require_freeze and manifest is None: raise AssertionError("A/A freeze is incomplete")
    source,population,parent_config,champion,planned_n = aa.load_inputs(output,config)
    derived = aa.pipeline_config(parent_config,config)
    assigned = aa.experiment.assign(population,derived)
    aa.validate_assignments(source,population,assigned,parent_config,config,champion)
    cells = aa.null_cells(population)
    expected_keys = {
        "eligibility_summary":["dataset","policy_cohort"],"experiment_metrics":["arm"],
        "risk_score_concentration":["arm","risk_stratum"],"assignment_quality":["risk_stratum"],
        "aa_covariate_balance":["covariate"],"aa_null_cells":["profit","approved","default_event"],
        "aa_hash_replays":["replicate"],"aa_calibration_results":["case_id","endpoint","alpha"],
        "aa_pvalue_histogram":["case_id","endpoint","p_bin_lower"],"aa_srm_calibration":["case_id"],
        "aa_optional_stopping":["look_fraction"],"aa_negative_controls":["control"],"aa_snapshot_bootstrap":["method"]}
    with duckdb.connect(str(output.database),read_only=True) as connection:
        compare_frames(connection.table("experiment_population").fetchdf(),population,["dataset","id"])
        compare_frames(connection.table("experiment_assignments").fetchdf(),assigned,["dataset","id"])
        validate_metrics(connection.table("experiment_metrics").fetchdf(),assigned)
        quality = aa.experiment.assignment_quality(assigned,derived)
        compare_frames(connection.table("assignment_quality").fetchdf(),quality,["risk_stratum"])
        compare_frames(connection.table("aa_covariate_balance").fetchdf(),aa.balance(assigned),["covariate"])
        compare_frames(connection.table("aa_null_cells").fetchdf(),cells,["profit","approved","default_event"])
        calibration = connection.table("aa_calibration_results").fetchdf()
        srm = connection.table("aa_srm_calibration").fetchdf()
        validate_calibration(calibration,srm,config)
        for table,keys in expected_keys.items():
            expected = connection.table(table).fetchdf()
            compare_frames(read_csv_typed(output.reports/"tables"/f"{table}.csv",expected),expected,keys)
        negative = connection.table("aa_negative_controls").fetchdf()
        if negative.empty or not negative.detected_as_expected.all(): raise AssertionError("Negative controls failed")
        expected_negative = aa.negative_controls(source,population,assigned,parent_config,config,champion,
            connection.table("experiment_metrics").fetchdf(),validate_metrics)
        compare_frames(negative,expected_negative,["control"])
        expected_bootstrap,bootstrap_draws = aa.snapshot_bootstrap(assigned,config)
        compare_frames(connection.table("aa_snapshot_bootstrap").fetchdf(),expected_bootstrap,["method"])
        compare_frames(pd.read_parquet(output.artifacts/"aa_bootstrap_draws.parquet"),bootstrap_draws,["replicate"])
        if deep:
            cal,histograms,srm_recomputed,draws = aa.calibration(cells,planned_n,parent_config,config,progress)
            compare_frames(calibration,cal,["case_id","endpoint","alpha"])
            compare_frames(connection.table("aa_pvalue_histogram").fetchdf(),histograms,["case_id","endpoint","p_bin_lower"])
            compare_frames(srm,srm_recomputed,["case_id"])
            compare_frames(pd.read_parquet(output.artifacts/"aa_calibration_draws.parquet"),draws,["case_id","replicate"])
            replays = aa.hash_replays(population,source,parent_config,config,champion,progress)
            compare_frames(connection.table("aa_hash_replays").fetchdf(),replays,["replicate"])
            compare_frames(connection.table("aa_optional_stopping").fetchdf(),aa.peek_simulation(cells,config),["look_fraction"])
    compare_frames(pd.read_parquet(output.artifacts/"aa_assignments.parquet"),assigned,["dataset","id"])
    summary = json.loads((output.reports/"summary.json").read_text(encoding="utf-8"))
    if not summary["champion_retained"] or summary["challenger_promoted"] or summary["live_experiment_completed"] or summary["production_rollout_ready"]:
        raise AssertionError("False policy promotion/live experiment claim")
    if summary["champion_definition"] != champion: raise AssertionError("Champion definition changed")
    if summary["required_regimes_calibrated"] != bool(calibration.loc[calibration.required,"calibration_passed"].all()):
        raise AssertionError("Calibration status does not match tables")
    with duckdb.connect(str(output.database),read_only=True) as connection:
        replays = connection.table("aa_hash_replays").fetchdf()
        rejections = int(replays.reject_at_005.sum())
        hash_p = float(stats.binomtest(rejections,len(replays),.05).pvalue)
        balance = connection.table("aa_covariate_balance").fetchdf()
    quality_passed = bool(quality.srm_p_value.ge(parent_config["randomization"]["srm_alpha"]/len(quality)).all())
    balance_passed = bool(balance.standardized_mean_difference.abs().le(config["max_absolute_covariate_smd"]).all())
    expected_pass = bool(calibration.loc[calibration.required,"calibration_passed"].all() and srm.calibration_passed.all()
        and quality_passed and balance_passed and negative.detected_as_expected.all() and hash_p >= .001)
    if summary["offline_aa_validation_passed"] != expected_pass:
        raise AssertionError("Overall certification does not match predeclared rules")
    # Recompute readiness blockers; a small, immature replay never authorizes efficacy.
    readiness = aa.analysis_readiness(len(assigned),planned_n,False,True,False)
    if summary["live_analysis_readiness"] != readiness: raise AssertionError("Early/immature efficacy was authorized")
    run_path = output.reports/"run_config.json"
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["input_sha256"] != aa.input_hashes(output,config) or run["protocol_sha256"] != aa.protocol_hashes(output):
            raise AssertionError("Run inputs/protocol changed")
        if run["status"] == "completed" and run["manifest_sha256"] != aa.business.sha256(output.artifacts/"aa_freeze_manifest.json"):
            raise AssertionError("Run freeze checksum mismatch")
    return {"status":"passed","rows":len(assigned),"champion_same_scores_decisions_and_cashflows_verified":True,
        "frozen_eligibility_and_stable_hash_allocation_verified":True,"sql_metrics_match_itt_rows":True,
        "simultaneous_mc_intervals_and_predeclared_acceptance_verified":True,
        "deep_hash_and_calibration_reproduction":deep,"negative_controls_detected":True,
        "csv_and_parquet_verified":True,"frozen_checksums_verified":require_freeze,
        "champion_retained":True,"challenger_promoted":False,"live_experiment_completed":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="aa_validation")
    args = parser.parse_args()
    audit = verify(aa.paths(args.run_name),progress=lambda message:print(message,flush=True))
    aa.business.write_json(aa.paths(args.run_name).reports/"artifact_verification.json",audit)
    print(json.dumps(audit,indent=2),flush=True)

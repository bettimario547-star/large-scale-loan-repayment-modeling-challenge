"""Verify policy cash flows and selection against independent row-level calculations."""
from __future__ import annotations

import argparse
import json

import duckdb
import numpy as np
import pandas as pd

try:
    from . import policy_simulation_common as policy
except ImportError:
    import policy_simulation_common as policy


def assert_value(actual, expected, field):
    if pd.isna(expected):
        if not pd.isna(actual): raise AssertionError(f"{field} must be NULL")
    else:
        counts = {"total_applications", "labeled_applications", "approved_applications",
                  "rejected_applications", "approved_default_rows", "approved_repaid_rows",
                  "development approvals", "development total", "champion_approved_rows", "challenger_approved_rows"}
        if field in counts:
            if actual != expected: raise AssertionError(f"{field}: integer counts differ")
        else:
            # Raw-score sum * payoff and sum(per-row payoff) have different
            # floating summation/cancellation error on hundreds of thousands
            # of rows. Bound that total-only difference below one tenth cent.
            tolerance = 1e-3 if field == "raw_score_implied_net_profit" else 1e-8
            if "rate" in field or field == "mean_p_repay_approved": tolerance = 1e-12
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=1e-11, err_msg=field)


def verify_csv_export(connection, path, table):
    """Compare typed values by stable keys, including nullable booleans."""
    keys = {
        "economic_assumptions": ["scenario_id"],
        "policy_candidates": ["policy_id"],
        "policy_constraints": [],
        "approval_economic_results": ["scenario_id", "policy_id", "policy_cohort"],
        "matched_approval_comparison": ["scenario_id", "target_approval_rate", "policy_cohort"],
        "selected_policy_definitions": ["prospective_arm"],
        "selected_policy_results": ["prospective_arm", "scenario_id", "policy_cohort"],
    }[table]
    expected = connection.table(table).fetchdf()
    # Inference represents CSV NULL booleans as float NaN rather than pd.NA.
    # Parse float strings with Python's round-trip conversion, also on pandas 2.
    float_columns = [name for name, dtype in expected.dtypes.items() if pd.api.types.is_float_dtype(dtype)]
    actual = pd.read_csv(path,
        dtype={name: dtype for name, dtype in expected.dtypes.items() if name not in float_columns},
        converters={name: lambda value: float(value) if value else np.nan for name in float_columns})
    actual = actual.astype(expected.dtypes.to_dict())
    if keys:
        actual = actual.sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def verify_connection(connection, predictions, split, candidates, config):
    policy.business.validate_predictions(predictions)
    if split.id.duplicated().any() or split.policy_cohort.isna().any() or set(split.policy_cohort) != {"policy_development", "policy_evaluation"}:
        raise AssertionError("Invalid policy cohort assignment")
    train_ids = set(predictions.loc[predictions.dataset.eq("train"), "id"])
    if set(split.id) != train_ids: raise AssertionError("Policy split coverage differs")
    pd.testing.assert_frame_equal(connection.table("policy_cohorts").fetchdf(), split, check_dtype=False)
    pd.testing.assert_frame_equal(connection.table("policy_candidates").fetchdf(), candidates, check_dtype=False)
    economies = policy.economics_frame(config).set_index("scenario_id")
    pd.testing.assert_frame_equal(connection.table("economic_assumptions").fetchdf(), economies.reset_index(), check_dtype=False)
    sql_results = connection.table("approval_economic_results").fetchdf()
    if sql_results.duplicated(["scenario_id", "policy_id", "policy_cohort"]).any(): raise AssertionError("Duplicate economic result")
    expected_groups = len(economies) * len(candidates) * 3
    if len(sql_results) != expected_groups: raise AssertionError("Missing economic scenarios or cohorts")
    cohorts = split.set_index("id").policy_cohort
    keyed = sql_results.set_index(["scenario_id", "policy_id", "policy_cohort"])
    expected_rows = []
    for candidate in candidates.itertuples(index=False):
        source = predictions.loc[predictions.model_version.eq(candidate.model_version)].copy()
        source["policy_cohort"] = source.id.map(cohorts)
        source.loc[source.dataset.eq("test"), "policy_cohort"] = "unlabeled_test"
        for cohort, group in source.groupby("policy_cohort"):
            scores = group.p_repay.to_numpy()
            labels = group.label.to_numpy(dtype=float, na_value=np.nan)
            approved = policy.approval_mask(scores, candidate.decision_rule, candidate.threshold)
            if cohort == "policy_development":
                if approved.sum() > np.floor(len(group) * candidate.target_approval_rate): raise AssertionError("Tie exceeded development budget")
                assert_value(approved.sum(), candidate.development_approved_rows, "development approvals")
                assert_value(len(group), candidate.development_total_rows, "development total")
            for scenario, economic in economies.iterrows():
                reference, _, _ = policy.economic_metrics(labels, scores, approved, economic)
                saved = keyed.loc[(scenario, candidate.policy_id, cohort)]
                for field, expected in reference.items(): assert_value(saved[field], expected, field)
                reference.update(scenario_id=scenario, policy_cohort=cohort, model_role=candidate.model_role,
                                 policy_id=candidate.policy_id)
                expected_rows.append(reference)
    # Selection is independently based on the recomputed cash-flow results.
    selected = policy.select_policies(pd.DataFrame(expected_rows), candidates, config)
    pd.testing.assert_frame_equal(connection.table("selected_policy_definitions").fetchdf(), selected, check_dtype=False)
    selected_results = connection.table("selected_policy_results").fetchdf()
    if len(selected_results) != len(selected) * len(economies) * 3: raise AssertionError("Selected policy coverage differs")
    for definition in selected.itertuples(index=False):
        order = ["scenario_id", "policy_id", "policy_cohort"]
        saved = selected_results.loc[selected_results.prospective_arm.eq(definition.prospective_arm)].drop(columns="prospective_arm").sort_values(order).reset_index(drop=True)
        expected = sql_results.loc[sql_results.policy_id.eq(definition.policy_id)].sort_values(order).reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
        for cohort in ("policy_development", "policy_evaluation", "unlabeled_test"):
            source = predictions.loc[predictions.model_version.eq(definition.model_version)].copy()
            if cohort == "unlabeled_test": source = source.loc[source.dataset.eq("test")]
            else: source = source.loc[source.dataset.eq("train") & source.id.map(cohorts).eq(cohort)]
            source = source.sort_values("id")
            approved = policy.approval_mask(source.p_repay.to_numpy(), definition.decision_rule, definition.threshold)
            _, realized, implied = policy.economic_metrics(source.label.to_numpy(dtype=float, na_value=np.nan), source.p_repay.to_numpy(), approved, economies.loc["base"])
            actual = connection.execute("SELECT id, approved, historical_simulated_net_profit, raw_score_implied_net_profit FROM selected_policy_application_outcomes WHERE prospective_arm = ? AND policy_cohort = ? ORDER BY id",
                                        [definition.prospective_arm, cohort]).fetchdf()
            np.testing.assert_array_equal(actual.id, source.id)
            np.testing.assert_array_equal(actual.approved, approved)
            np.testing.assert_allclose(actual.historical_simulated_net_profit, realized, atol=1e-8, rtol=0, equal_nan=True)
            np.testing.assert_allclose(actual.raw_score_implied_net_profit, implied, atol=1e-8, rtol=0)
    paired = connection.table("matched_approval_comparison").fetchdf()
    if len(paired) != len(economies) * len(config["target_approval_rates"]) * 3: raise AssertionError("Matched-budget comparison coverage differs")
    for row in paired.itertuples(index=False):
        a, b = [keyed.loc[(row.scenario_id, candidate_id, row.policy_cohort)] for candidate_id in (row.champion_policy_id, row.challenger_policy_id)]
        for name, expected in {"champion_approved_rows": a.approved_applications,
            "challenger_approved_rows": b.approved_applications, "champion_approval_rate": a.approval_rate,
            "challenger_approval_rate": b.approval_rate, "delta_approval_rate": b.approval_rate - a.approval_rate,
            "champion_default_rate": a.observed_default_rate_approved, "challenger_default_rate": b.observed_default_rate_approved,
            "delta_default_rate": b.observed_default_rate_approved - a.observed_default_rate_approved,
            "champion_profit_per_application": a.historical_profit_per_application,
            "challenger_profit_per_application": b.historical_profit_per_application,
            "delta_profit_per_application": b.historical_profit_per_application - a.historical_profit_per_application}.items():
            assert_value(getattr(row, name), expected, name)
        if row.exactly_equal_approved_count != (a.approved_applications == b.approved_applications): raise AssertionError("False matched-count claim")
    return {"status": "passed", "economic_result_rows": len(sql_results),
        "sql_cash_flows_match_row_level_numpy": True, "principal_repayment_excluded_from_revenue": True,
        "rejected_applications_incur_application_cost": True, "selection_uses_base_development_only": True,
        "evaluation_and_test_use_frozen_thresholds": True, "test_realized_profits_and_default_rates_are_null": True,
        "matched_budget_comparisons_and_sensitivity_verified": True, "selected_application_outcomes_verified": True}


def verify(output_paths, config_path=policy.CONFIG, check_exports=True):
    config = policy.load_config(config_path)
    manifest = policy.check_manifest(output_paths, config, config_path)
    if manifest is None: raise AssertionError("Policy freeze has not completed")
    parent = policy.business.Paths(output_paths.root, config["parent_run_name"])
    predictions = pd.read_parquet(parent.predictions)
    split = pd.read_parquet(output_paths.artifacts / "policy_cohorts.parquet")
    candidates = pd.read_parquet(output_paths.artifacts / "policy_candidates.parquet")
    with duckdb.connect(str(output_paths.database), read_only=True) as connection:
        result = verify_connection(connection, predictions, split, candidates, config)
        if check_exports:
            for table in policy.EXPORT_TABLES:
                verify_csv_export(connection, output_paths.reports / "tables" / f"{table}.csv", table)
            path = output_paths.artifacts / "selected_policy_application_outcomes.parquet"
            conditions = " OR ".join(f"p.{name} IS DISTINCT FROM f.{name}" for name in connection.execute("DESCRIBE selected_policy_application_outcomes").fetchdf().column_name)
            mismatches = connection.execute(f"SELECT count(*) FROM selected_policy_application_outcomes p FULL JOIN read_parquet(?) f USING (dataset, id, prospective_arm) WHERE {conditions}", [str(path)]).fetchall()[0][0]
            if mismatches: raise AssertionError("Exported application outcomes differ")
    run_path = output_paths.reports / "run_config.json"
    if run_path.exists():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        for path, checksum in run["analysis_code_sha256"].items():
            if policy.business.sha256(output_paths.root / path) != checksum: raise AssertionError("Analysis code changed")
        if policy.business.sha256(output_paths.artifacts / "policy_freeze_manifest.json") != run["policy_manifest_sha256"]:
            raise AssertionError("Policy manifest changed")
        if run["status"] == "completed" and policy.business.sha256(output_paths.database) != run["database_sha256"]:
            raise AssertionError("Completed database changed")
        if run["status"] == "completed" and policy.business.sha256(output_paths.artifacts / "selected_policy_application_outcomes.parquet") != run["application_outcomes_sha256"]:
            raise AssertionError("Completed application-outcome export changed")
    handoff = output_paths.artifacts / "experiment_policy_handoff.json"
    if handoff.exists():
        recorded = json.loads(handoff.read_text(encoding="utf-8"))["selected_policies"]
        stored = pd.read_parquet(output_paths.artifacts / "selected_policies.parquet").query("prospective_arm in ['A', 'B']")
        for definition in recorded:
            row = stored.loc[stored.prospective_arm.eq(definition["prospective_arm"])].iloc[0]
            if definition["threshold"] != row.threshold: raise AssertionError("Handoff rounded a frozen policy threshold")
    result.update(policy_input_protocol_and_output_checksums_verified=True, csv_and_parquet_exports_verified=check_exports,
        evaluation_scope=manifest["evaluation_scope"], randomized_experiment_completed=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="approval_policy_simulation")
    args = parser.parse_args()
    output = policy.paths(args.run_name)
    audit = verify(output)
    policy.business.write_json(output.reports / "artifact_verification.json", audit)
    print(json.dumps(audit, indent=2), flush=True)

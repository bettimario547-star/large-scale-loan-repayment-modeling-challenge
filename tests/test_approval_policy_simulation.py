"""Check business cash-flow assumptions and policy selection data boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

from scripts import policy_simulation_common as policy
from scripts.verify_approval_policy_simulation import verify_connection, verify_csv_export
from tests.test_sql_business_analysis import fixture_predictions


class ApprovalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = policy.load_config()
        self.base = policy.economics_frame(self.config).query("scenario_id == 'base'").iloc[0]

    def test_csv_audit_preserves_nullable_booleans_and_checks_values_by_key(self):
        with tempfile.TemporaryDirectory() as directory, duckdb.connect() as connection:
            connection.execute("CREATE TABLE approval_economic_results AS SELECT 'base' AS scenario_id, "
                "'b' AS policy_id, 'policy_evaluation' AS policy_cohort, NULL::BOOLEAN AS eligible_for_selection, "
                "NULL::DOUBLE AS threshold UNION ALL SELECT 'base', 'a', 'policy_development', TRUE, "
                "0.9256811726550405::DOUBLE")
            path = Path(directory) / "results.csv"
            connection.execute("COPY (SELECT * FROM approval_economic_results ORDER BY policy_id) "
                "TO ? (FORMAT CSV, HEADER)", [str(path)])
            verify_csv_export(connection, path, "approval_economic_results")
            changed = pd.read_csv(path)
            changed.loc[0, "eligible_for_selection"] = False
            changed.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                verify_csv_export(connection, path, "approval_economic_results")

    def test_principal_is_not_income_lgd_is_partial_loss_and_rejected_rows_pay_cost(self):
        metrics, realized, implied = policy.economic_metrics([1, 0, 1], [.9, .1, .6], [True, True, False], self.base)
        np.testing.assert_allclose(realized, [690, -8510, -10])
        np.testing.assert_allclose(implied, [-230, -7590, -10])
        self.assertAlmostEqual(metrics["historical_profit_per_application"], -2610)
        self.assertEqual(metrics["historical_simulated_interest_income"], 1200)
        self.assertEqual(metrics["historical_simulated_principal_loss"], 8000)
        self.assertEqual(metrics["assumed_application_cost"], 30)

    def test_zero_approval_and_unlabeled_test_do_not_become_zero_risk_or_real_profit(self):
        metrics, realized, _ = policy.economic_metrics([0, 1], [.1, .9], [False, False], self.base)
        self.assertEqual(metrics["approved_default_rows"], 0)
        self.assertTrue(np.isnan(metrics["observed_default_rate_approved"]))
        np.testing.assert_allclose(realized, [-10, -10])
        metrics, realized, implied = policy.economic_metrics([np.nan, np.nan], [.1, .9], [True, False], self.base)
        self.assertTrue(np.isnan(realized).all())
        self.assertTrue(np.isfinite(implied).all())
        self.assertTrue(np.isnan(metrics["historical_profit_per_application"]))

    def test_budget_keeps_ties_and_score_one_endpoints(self):
        scores = np.array([.9, .8, .8, .7])
        rule, cutoff, count = policy.budget_threshold(scores, .5)
        self.assertEqual(count, 1)
        self.assertGreater(cutoff, .8)
        np.testing.assert_array_equal(policy.approval_mask(scores, rule, cutoff), [True, False, False, False])
        self.assertEqual(policy.budget_threshold([1, 1, 1], .5), ("reject_all", None, 0))
        self.assertEqual(policy.budget_threshold([0, 1], 0), ("reject_all", None, 0))
        self.assertEqual(policy.budget_threshold([0, 1], 1), ("approve_all", 0., 2))

    def test_evaluation_and_test_scores_labels_cannot_change_fitted_thresholds(self):
        predictions = fixture_predictions()
        split = pd.DataFrame({"id": np.arange(10), "policy_cohort": ["policy_development"] * 6 + ["policy_evaluation"] * 4})
        expected = policy.fit_candidates(predictions, split, self.config)
        changed = predictions.copy()
        changed.loc[changed.dataset.eq("test") | changed.id.ge(6), "p_repay"] = .999
        changed.loc[changed.dataset.eq("train") & changed.id.ge(6), "label"] = 1 - changed.loc[changed.dataset.eq("train") & changed.id.ge(6), "label"]
        pd.testing.assert_frame_equal(expected, policy.fit_candidates(changed, split, self.config))

    def test_selection_ignores_evaluation_and_refuses_no_feasible_policy(self):
        candidates = pd.DataFrame([{"policy_id": f"{role}_{rate}", "model_role": role, "target_approval_rate": rate,
            "model_version": role, "threshold": None if rate == 0 else .8, "decision_rule": "reject_all" if rate == 0 else "score_threshold",
            "development_approved_rows": int(rate * 10), "development_total_rows": 10}
            for role in ("champion", "challenger") for rate in (0., .5, 1.)])
        rows = [{"scenario_id": "base", "policy_cohort": cohort, "model_role": role,
                 "policy_id": f"{role}_{rate}", "approval_rate": rate,
                 "observed_default_rate_approved": .02 if rate == .5 else .08,
                 "historical_profit_per_application": 100 if cohort == "policy_development" else 100000}
                for role in ("champion", "challenger") for rate in (.5, 1.)
                for cohort in ("policy_development", "policy_evaluation")]
        results = pd.DataFrame(rows)
        chosen = policy.select_policies(results, candidates, self.config)
        self.assertTrue(chosen.loc[chosen.prospective_arm.isin(["A", "B"]), "target_approval_rate"].eq(.5).all())
        results.loc[results.policy_cohort.eq("policy_evaluation"), "historical_profit_per_application"] = -100000
        pd.testing.assert_frame_equal(chosen, policy.select_policies(results, candidates, self.config))
        results.loc[results.policy_cohort.eq("policy_development"), "observed_default_rate_approved"] = .2
        with self.assertRaises(ValueError): policy.select_policies(results, candidates, self.config)

    def test_sensitivity_changes_one_assumption_and_all_sql_cash_flows_replay(self):
        predictions = fixture_predictions()
        config = copy.deepcopy(self.config)
        config["target_approval_rates"] = [0., .5, 1.]
        config["constraints"] = {"max_development_default_rate": .6, "min_development_approval_rate": .2}
        split = pd.DataFrame({"id": np.arange(10), "policy_cohort": ["policy_development"] * 6 + ["policy_evaluation"] * 4})
        candidates = policy.fit_candidates(predictions, split, config)
        with tempfile.TemporaryDirectory() as directory:
            output = policy.paths(root=Path(directory))
            parent = policy.business.Paths(output.root, config["parent_run_name"])
            parent.artifacts.mkdir(parents=True)
            output.artifacts.mkdir(parents=True)
            predictions.to_parquet(parent.predictions, index=False)
            split.to_parquet(output.artifacts / "policy_cohorts.parquet", index=False)
            with duckdb.connect() as connection:
                policy.execute_sql(connection, output, config, candidates)
                result = verify_connection(connection, predictions, split, candidates, config)
                self.assertEqual(result["status"], "passed")
                self.assertTrue(result["test_realized_profits_and_default_rates_are_null"])
        invalid = copy.deepcopy(config)
        invalid["sensitivity_scenarios"][1]["overrides"] = {"principal": 5000, "loss_given_default": .5}
        with self.assertRaises(ValueError): policy.economics_frame(invalid)

    def test_policy_split_is_shared_reproducible_and_has_disjoint_ids(self):
        predictions = fixture_predictions()
        one = policy.policy_split(predictions, self.config)
        two = policy.policy_split(predictions.sample(frac=1, random_state=3), self.config)
        pd.testing.assert_frame_equal(one, two)
        self.assertFalse(one.id.duplicated().any())
        self.assertEqual(len(one), 10)


if __name__ == "__main__": unittest.main()

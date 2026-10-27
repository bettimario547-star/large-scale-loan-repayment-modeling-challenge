"""A/A tests with intentional faults; no assertion that every null p exceeds .05."""
import copy
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from scripts import aa_validation_common as aa
from scripts.verify_aa_validation import validate_metrics,validate_calibration
from tests.test_ab_test_design import fixture


class AAValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = aa.load_config()
        self.config.update(hash_repetitions=2,calibration_repetitions=2000,peeking_repetitions=1000,
                           injected_profit_bias=50000.,injected_allocation_b_fraction=.99)
        self.parent = aa.experiment.load_config()
        self.champion = {"model_version":"lightgbm_baseline","threshold":.9247938423619432,"model_role":"champion"}
        self.source = fixture(self.parent)
        self.population = aa.make_aa_population(self.source,self.champion)
        self.derived = aa.pipeline_config(self.parent,self.config)
        self.assigned = aa.experiment.assign(self.population,self.derived)

    def test_same_treatment_retains_frozen_gate_and_dataset_scope(self):
        aa.validate_assignments(self.source,self.population,self.assigned,self.parent,self.config,self.champion)
        np.testing.assert_array_equal(self.population.eligible_proxy,self.source.eligible_proxy)
        np.testing.assert_array_equal(self.population.risk_stratum,self.source.risk_stratum)
        np.testing.assert_array_equal(self.population.treatment_score_a,self.population.treatment_score_b)
        np.testing.assert_allclose(self.population.potential_profit_a,self.population.potential_profit_b,equal_nan=True)
        self.assertTrue(self.assigned.dataset.eq("train").all())
        self.assertTrue(self.population.loc[self.population.dataset.eq("test"),"potential_profit_b"].isna().all())
        self.assertTrue(self.assigned.observed_dpd30.isna().all())

    def test_identity_checks_reject_challenger_cashflows_and_scores(self):
        for field,value in (("treatment_policy_b","challenger"),("treatment_threshold_b",.91)):
            changed = self.population.copy()
            changed[field] = value
            with self.assertRaises(AssertionError): aa.validate_assignments(self.source,changed,self.assigned,self.parent,self.config,self.champion)
        changed = self.population.copy()
        changed["treatment_score_b"] = self.source.score_b
        with self.assertRaises(AssertionError): aa.validate_assignments(self.source,changed,self.assigned,self.parent,self.config,self.champion)
        changed = self.population.copy()
        changed.loc[changed.dataset.eq("train"),"potential_profit_b"] += 1
        with self.assertRaises(AssertionError): aa.validate_assignments(self.source,changed,self.assigned,self.parent,self.config,self.champion)

    def test_arm_assignment_sticky_label_blind_and_domain_scoped(self):
        changed = self.population.sample(frac=1,random_state=8).copy()
        changed.loc[changed.dataset.eq("train"),"label"] = 1-changed.loc[changed.dataset.eq("train"),"label"]
        actual = aa.experiment.assign(changed,self.derived)
        columns = ["dataset","id","arm","assignment_uniform"]
        pd.testing.assert_frame_equal(actual[columns].sort_values("id").reset_index(drop=True),self.assigned[columns].sort_values("id").reset_index(drop=True))
        salt = self.config["randomization_salt"]
        self.assertNotEqual(aa.experiment.stable_uniform("train",1,salt),aa.experiment.stable_uniform("test",1,salt))

    def test_declared_probability_identity_and_lineage_cannot_be_changed(self):
        for field,value in (("assignment_probability_b",.8),("experiment_id","wrong"),("risk_stratum","wrong"),("label",1)):
            changed = self.assigned.copy()
            changed[field] = value
            with self.assertRaises(AssertionError): aa.validate_assignments(self.source,self.population,changed,self.parent,self.config,self.champion)
        # Same cached invariant reference still validates every assigned row.
        expected = aa.validated_expected_population(self.source,self.population,self.parent,self.champion)
        changed = self.assigned.copy()
        changed["assignment_probability_b"] = .8
        with self.assertRaises(AssertionError): aa.validate_assignments(self.source,self.population,changed,self.parent,self.config,self.champion,expected=expected)

    def test_compressed_moments_match_explicit_rows_and_scipy(self):
        cells = pd.DataFrame({"profit":[-8510.,-10.,690.],"approved":[True,False,True],"default_event":[True,False,False],"rows":[20,40,400]})
        ca,cb = np.array([[4,10,100],[7,11,101]]),np.array([[8,9,110],[6,15,105]])
        ma,mb = aa.moments(ca,cells),aa.moments(cb,cells)
        inferred = aa.infer_moments(ma,mb)
        for i in range(2):
            a,b = np.repeat(cells.profit,ca[i]),np.repeat(cells.profit,cb[i])
            test = stats.ttest_ind(b,a,equal_var=False)
            self.assertAlmostEqual(inferred["profit_itt"]["p"][i],test.pvalue,places=12)
            self.assertAlmostEqual(inferred["profit_itt"]["df"][i],test.df,places=12)
            self.assertAlmostEqual(ma["mean"][i],a.mean(),places=12)
            self.assertAlmostEqual(ma["variance"][i],a.var(ddof=1),places=7)

    def test_zero_paired_difference_still_has_nonzero_independent_standard_error(self):
        self.assertTrue((self.assigned.potential_profit_b-self.assigned.potential_profit_a).eq(0).all())
        a,b = [self.assigned.loc[self.assigned.arm.eq(arm),"simulated_observed_net_profit"] for arm in ("A","B")]
        self.assertGreater(aa.experiment.mean_comparison(a,b)["se"],0)

    def test_clopper_pearson_matches_scipy_and_zero_count_is_not_zero_uncertainty(self):
        for events,total in ((0,1000),(50,1000),(1000,1000)):
            low,high = aa.clopper_pearson(events,total,.01)
            reference = stats.binomtest(events,total).proportion_ci(.99,method="exact")
            self.assertAlmostEqual(low,reference.low,places=10)
            self.assertAlmostEqual(high,reference.high,places=10)
        self.assertGreater(aa.clopper_pearson(0,1000)[1],0)
        self.assertEqual(aa.clopper_pearson(0,0),(None,None))

    def test_calibration_is_reproducible_complementary_and_honestly_certified(self):
        cells = aa.null_cells(self.population)
        cal,hist,srm,draws = aa.calibration(cells,20000,self.parent,self.config)
        again = aa.calibration(cells,20000,self.parent,self.config)
        for expected,actual in zip((cal,hist,srm,draws),again): pd.testing.assert_frame_equal(expected,actual)
        finite = draws.loc[draws.case_id.eq("finite_full_current")]
        self.assertTrue((finite.n_a+finite.n_b).eq(cells.rows.sum()).all())
        validate_calibration(cal,srm,self.config)
        false = cal.copy()
        false["false_positive_rate"] = 0
        with self.assertRaises(AssertionError): validate_calibration(false,srm,self.config)
        self.assertTrue(cal.loc[cal.alpha.eq(.05),"zero_effect_interval_coverage"].equals(1-cal.loc[cal.alpha.eq(.05),"false_positive_rate"]))

    def test_degenerate_outcomes_do_not_get_a_false_calibration_certificate(self):
        cells = pd.DataFrame({"profit":[-10.],"approved":[False],"default_event":[False],"rows":[100]})
        config = copy.deepcopy(self.config)
        config["stress_sample_sizes"] = [1000]
        with np.errstate(divide="ignore",invalid="ignore"):
            result,_,_,_ = aa.calibration(cells,1000,self.parent,config)
        self.assertFalse(result.calibration_passed.any())
        self.assertTrue(result.valid_repetitions.eq(0).all())

    def test_sql_and_all_intentional_mutations_are_detected(self):
        with duckdb.connect() as connection:
            connection.register("_population",self.population)
            connection.register("_assignments",self.assigned)
            connection.execute(aa.SQL_FILE.read_text(encoding="utf-8"))
            metrics = connection.table("experiment_metrics").fetchdf()
            validate_metrics(metrics,self.assigned)
            negative = aa.negative_controls(self.source,self.population,self.assigned,self.parent,self.config,self.champion,metrics,validate_metrics)
            self.assertTrue(negative.detected_as_expected.all(),negative.to_string())
            self.assertEqual(len(negative),17)

    def test_readiness_requires_fixed_sample_maturity_quality_and_final_look(self):
        baseline = aa.analysis_readiness(100,100,True,True,True)
        self.assertTrue(baseline["efficacy_analysis_allowed"])
        for args in ((99,100,True,True,True),(100,100,False,True,True),(100,100,True,False,True),(100,100,True,True,False)):
            result = aa.analysis_readiness(*args)
            self.assertFalse(result["efficacy_analysis_allowed"])
            self.assertTrue(result["champion_retained"])
            self.assertFalse(result["challenger_promoted"])

    def test_optional_look_uses_cumulative_stream_and_retains_final_analysis_only(self):
        config = copy.deepcopy(self.config)
        config["peeking_repetitions"] = 5000
        result = aa.peek_simulation(aa.null_cells(self.population),config)
        self.assertEqual(int(result.iloc[-1].n_total),config["peeking_sample_size"])
        self.assertEqual(int(result.allowed_efficacy_look.sum()),1)
        self.assertTrue(result.naive_any_look_two_sided_fpr.is_monotonic_increasing)
        self.assertGreater(result.iloc[-1].naive_any_look_two_sided_fpr,result.iloc[-1].single_look_two_sided_fpr)

    def test_balance_does_not_hide_complete_separation_of_constant_covariates(self):
        values = pd.DataFrame({"arm":["A","A","B","B"],"score_a":[.9,.9,.95,.95],"common_score":[.9,.9,.95,.95]})
        result = aa.balance(values)
        self.assertTrue(np.isinf(result.standardized_mean_difference).all())

    def test_configuration_refuses_model_promotion_or_insufficient_precision(self):
        for field,value in (("challenger_promoted",True),("current_champion_retained",False),("calibration_repetitions",100)):
            config = aa.load_config()
            config[field] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"config.json"
                aa.business.write_json(path,config)
                with self.assertRaises(ValueError): aa.load_config(path)


if __name__ == "__main__": unittest.main()

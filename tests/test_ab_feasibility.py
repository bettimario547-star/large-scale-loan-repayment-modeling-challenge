"""Substantive statistical/provenance tests for the new business experiment."""
import copy
import unittest
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from scripts import ab_feasibility_common as f
from scripts.verify_ab_feasibility import audit_sql


def fixture():
    records = []
    for cohort in ("policy_development","policy_evaluation","unlabeled_test"):
        for key,a,b,count in (("S1",False,True,800),("S1",True,False,800),("S2",True,False,400),("S3",True,True,5000)):
            for i in range(count):
                label = np.nan if cohort=="unlabeled_test" else int(i%13!=0)
                def gain(approved): return np.nan if np.isnan(label) else (690 if label else -8510) if approved else -10.
                score_a,score_b = .95 if a else .91,.95 if b else (.9253 if key=="S2" else .91)
                records.append({"dataset":"test" if cohort=="unlabeled_test" else "train","policy_cohort":cohort,
                    "id":len(records),"label":label,"score_a":score_a,"score_b":score_b,
                    "approved_a":a,"approved_b":b,"potential_profit_a":gain(a),"potential_profit_b":gain(b),
                    "common_score":min(score_a,score_b),"eligible_proxy":True,"decision_disagreement":a!=b,
                    "risk_stratum":key,"simulation_only":True})
    return f.enrich(pd.DataFrame(records))


class FeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.population = fixture()
    def setUp(self):
        self.c = f.load_config()
        self.c.update(calibration_repetitions=2000,simulation_repetitions=1000,bootstrap_repetitions=300,hash_repetitions=2)
        self.target = f.cohort(self.population,"policy_evaluation")
        self.assigned = f.assign(self.population,self.c)

    def test_gate_weights_and_hash_are_label_blind_order_independent_and_sticky(self):
        changed = self.population.copy()
        changed.loc[changed.dataset.eq("train"),"label"] = 1-changed.loc[changed.dataset.eq("train"),"label"]
        actual = f.assign(changed.sample(frac=1,random_state=1),self.c)
        columns = ["dataset","id","arm","analysis_stratum","assignment_uniform"]
        pd.testing.assert_frame_equal(actual[columns],self.assigned[columns])
        self.assertTrue(actual.trial_eligible.all())
        self.assertTrue(actual.dataset.eq("train").all())
        self.assertTrue(actual.policy_cohort.eq("policy_evaluation").all())
        extra = self.population.iloc[[7100]].copy()
        extra["id"] = 100000
        extra["policy_cohort"] = "policy_evaluation"
        appended = f.assign(pd.concat([self.population,extra],ignore_index=True),self.c)
        pd.testing.assert_frame_equal(appended.loc[appended.id.ne(100000),columns].reset_index(drop=True),self.assigned[columns])

    def test_planning_ignores_evaluation_outcomes_and_keeps_economic_boundary(self):
        expected = f.plan(self.population,self.c,f.legacy.load_config())
        changed = self.population.copy()
        mask = changed.policy_cohort.ne("policy_development")
        changed.loc[mask,["potential_profit_a","potential_profit_b"]] = 1e9
        changed.loc[mask,"label"] = 0
        actual = f.plan(changed,self.c,f.legacy.load_config())
        for i in (0,2,4): pd.testing.assert_frame_equal(expected[i],actual[i])
        self.assertEqual(expected[3],actual[3])
        self.assertEqual(expected[3]["precision_gap"],5.)
        self.assertEqual(expected[3]["business_boundary"],5.)

    def test_independent_weighted_welch_matches_direct_scipy_single_stratum(self):
        a,b = np.array([690,-8510,690,690,690]),np.array([-10.,690.,-10.,-8510.,-10.,690.])
        result = f.infer([len(a)],[len(b)],[a.mean()],[b.mean()],[a.var(ddof=1)],[b.var(ddof=1)],[.06])
        reference = stats.ttest_ind(b,a,equal_var=False)
        self.assertAlmostEqual(float(result["p"]),reference.pvalue,places=12)
        self.assertAlmostEqual(float(result["df"]),reference.df,places=12)
        self.assertAlmostEqual(float(result["se"]),.06*np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)),places=12)

    def test_unscaled_disagreement_difference_is_not_full_population_gain(self):
        result,cells = f.estimate(self.assigned,self.target,self.c)
        a,b = [cells.loc[cells.arm.eq(arm)].sort_values("stratum") for arm in ("A","B")]
        direct = np.sum(a.weight.to_numpy()*(b.profit_mean.to_numpy()-a.profit_mean.to_numpy()))
        self.assertAlmostEqual(result["delta"],direct,places=10)
        d = self.target.decision_disagreement.mean()
        conditional = direct/d
        self.assertAlmostEqual(result["delta"],d*conditional,places=10)
        self.assertGreater(result["se"],0)

    def test_aa_uses_champion_cashflows_with_frozen_original_disagreement_covariates(self):
        aa = f.assign(self.population,self.c,"AA")
        f.validate_assignments(self.population,aa,self.c,"AA")
        np.testing.assert_array_equal(aa.assigned_approved,aa.approved_a)
        np.testing.assert_array_equal(aa.simulated_observed_net_profit,aa.potential_profit_a)
        self.assertTrue(aa.observed_dpd30.isna().all())
        result,_ = f.estimate(aa,self.target,self.c,"AA")
        self.assertGreater(result["se"],0)

    def test_cashflow_contract_duplicate_and_outcome_corruption_are_rejected(self):
        changed = self.population.copy()
        changed.loc[changed.dataset.eq("train"),"potential_profit_a"] += 1
        with self.assertRaises(AssertionError): f.enrich(changed)
        for field,value in (("eligible_proxy",False),("risk_stratum","wrong"),("decision_disagreement",False)):
            changed = self.population.copy()
            changed[field] = value
            with self.assertRaises(AssertionError): f.enrich(changed)
        for field,value in (("arm","B"),("simulated_observed_net_profit",0),("observed_dpd30",False),("analysis_stratum","wrong")):
            bad = self.assigned.copy()
            bad[field] = value
            with self.assertRaises(AssertionError): f.validate_assignments(self.population,bad,self.c)
        with self.assertRaises(AssertionError): f.validate_assignments(self.population,pd.concat([self.assigned,self.assigned.iloc[[0]]]),self.c)

    def test_sql_itt_weights_and_independent_variance_match_rows(self):
        est,cells = f.estimate(self.assigned,self.target,self.c)
        with duckdb.connect() as con:
            con.register("_population",self.population)
            con.register("_assignments",self.assigned)
            con.execute(f.SQL.read_text(encoding="utf-8"))
            audit_sql(con,self.assigned,self.target,est,cells)
            con.execute("UPDATE target_stratum_weights SET weight_of_full_eligible=1")
            with self.assertRaises(AssertionError): audit_sql(con,self.assigned,self.target,est,cells)

    def test_business_boundary_and_ni_boundary_have_correct_binary_population_truth(self):
        frame,meta = f.profile(f.cohort(self.population,"policy_development"))
        sim = f.simulate(frame,meta,10000,1000,self.c,"test_boundary",target_profit=5.)
        np.testing.assert_allclose(sim.true_profit,5,atol=1e-10,rtol=0)
        self.assertLess(sim.business_success.mean(),.1)
        ni = f.simulate(frame,meta,10000,1000,self.c,"test_ni_boundary",target_default=.002)
        np.testing.assert_allclose(ni.true_default,.002,atol=1e-12,rtol=0)
        self.assertLess(ni.default_ni_success.mean(),.1)
        with self.assertRaises(ValueError): f.simulate(frame,meta,10000,1000,self.c,"fake_aa",mode="AA",target_profit=5.)

    def test_exact_risk_bound_keeps_agreement_uncertainty_and_zero_bad_uncertainty(self):
        value = f.absolute_risk_upper([0],[100],[True],[.03],0,1000,.90,.93,.025)
        self.assertGreater(value,0)
        ignored_agreement = .03*f.binomial_upper(0,100,.0125)/.93
        self.assertGreater(value,ignored_agreement)

    def test_compressed_aa_simulation_is_reproducible_and_finite_group_counts_conservative(self):
        frame,meta = f.profile(self.target)
        first = f.simulate(frame,meta,int(frame.n.sum()),3000,self.c,"aa_finite",mode="AA",finite=True)
        second = f.simulate(frame,meta,int(frame.n.sum()),3000,self.c,"aa_finite",mode="AA",finite=True)
        pd.testing.assert_frame_equal(first,second)
        self.assertTrue(first.usable.all())
        self.assertTrue(first.true_profit.eq(0).all())
        self.assertGreater(first.profit_se.min(),0)
        self.assertLess(first.profit_p_zero.lt(.05).mean(),.09)

    def test_minimum_cell_prevents_underpowered_stratum_from_claiming_ready(self):
        too_strict = copy.deepcopy(self.c)
        too_strict["minimum_per_arm_per_stratum"] = 10000
        result,_ = f.estimate(self.assigned,self.target,too_strict)
        self.assertFalse(result["minimum_cell_sample_reached"])

    def test_bootstrap_is_independent_stratified_and_profit_ci_not_claimed_at_zero(self):
        boot,draws = f.bootstrap(self.assigned,self.target,self.c)
        self.assertGreater(draws.profit_delta.var(),0)
        self.assertEqual(boot.iloc[0].repetitions,300)
        est,_ = f.estimate(self.assigned,self.target,self.c)
        self.assertEqual(est["business_value_demonstrated_in_simulation"],est["ci_low"]>5)


if __name__=="__main__": unittest.main()

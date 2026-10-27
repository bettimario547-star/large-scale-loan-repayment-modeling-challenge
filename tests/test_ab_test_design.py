"""Meaningful boundaries for randomization and credit-policy experiment inference."""
import copy
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from scripts import experiment_design_common as experiment
from scripts.verify_ab_test_design import verify_connection, compare_frames


def fixture(config):
    n=120
    score_a=np.r_[np.tile(np.linspace(.87,.995,40),3),np.linspace(.83,.94,10)]
    score_b=np.clip(score_a+np.where(np.arange(n+10)%2==0,.014,-.014),0,1)
    label=np.r_[np.where(np.arange(n)%9==0,0,1),np.full(10,np.nan)]
    approved_a,approved_b=score_a>=.9247938423619432,score_b>=.9256811726550405
    def payoff(approved):
        return np.where(np.isfinite(label),approved*(label*1200-(1-label)*8000-500)-10,np.nan)
    result=pd.DataFrame({"dataset":["train"]*n+["test"]*10,
        "policy_cohort":["policy_development"]*80+["policy_evaluation"]*40+["unlabeled_test"]*10,
        "id":np.r_[np.arange(n),np.arange(10)],"label":label,"score_a":score_a,"score_b":score_b,
        "approved_a":approved_a,"approved_b":approved_b,
        "potential_profit_a":payoff(approved_a),"potential_profit_b":payoff(approved_b)})
    result["common_score"]=result[["score_a","score_b"]].min(axis=1)
    result["eligible_proxy"]=result.common_score.ge(config["eligibility"]["common_min_raw_repayment_score"])
    result["decision_disagreement"]=result.approved_a.ne(result.approved_b)
    codes=np.searchsorted(config["eligibility"]["risk_stratum_edges"],result.common_score,side="right")-1
    result["risk_stratum"]=[f"S{c+1}" if allowed else "excluded" for c,allowed in zip(codes,result.eligible_proxy)]
    result["simulation_only"]=True
    return result


class ExperimentDesignTests(unittest.TestCase):
    def setUp(self):
        self.config=experiment.load_config()
        self.config["inference"]["bootstrap_repetitions"]=300
        self.config["inference"]["monte_carlo_repetitions"]=300
        self.population=fixture(self.config)

    def test_parquet_nullable_boolean_is_semantically_equal_and_false_outcomes_fail(self):
        expected=pd.DataFrame({"id":[1,2],"observed_dpd30":pd.Series([pd.NA,pd.NA],dtype="boolean")})
        actual=pd.DataFrame({"id":[2,1],"observed_dpd30":[None,None]})
        compare_frames(actual,expected,["id"])
        actual.loc[0,"observed_dpd30"]=False
        with self.assertRaises(AssertionError): compare_frames(actual,expected,["id"])

    def test_hash_is_sticky_order_independent_dataset_scoped_and_label_blind(self):
        baseline=experiment.assign(self.population,self.config)
        changed=self.population.copy()
        changed.loc[changed.dataset.eq("train"),"label"]=1-changed.loc[changed.dataset.eq("train"),"label"]
        changed.loc[changed.dataset.eq("train"),["potential_profit_a","potential_profit_b"]]=0
        reordered=experiment.assign(changed.sample(frac=1,random_state=7),self.config)
        columns=["dataset","id","arm","assignment_uniform","risk_stratum"]
        pd.testing.assert_frame_equal(baseline[columns].sort_values("id").reset_index(drop=True),reordered[columns].sort_values("id").reset_index(drop=True))
        salt=self.config["randomization"]["salt"]
        self.assertNotEqual(experiment.stable_uniform("train",1,salt),experiment.stable_uniform("test",1,salt))
        self.assertNotEqual(experiment.stable_uniform("train",1,salt,"arm"),experiment.stable_uniform("train",1,salt,"exposure"))
        extra=self.population.loc[self.population.id.eq(90) & self.population.dataset.eq("train")].copy()
        extra["id"]=1000
        appended=experiment.assign(pd.concat([self.population,extra],ignore_index=True),self.config)
        pd.testing.assert_frame_equal(baseline[columns],appended.loc[appended.id.ne(1000),columns].reset_index(drop=True))
        self.assertTrue(baseline.assignment_uniform.between(0,1,inclusive="left").all())

    def test_common_gate_retains_disagreements_and_never_randomizes_test(self):
        assignments=experiment.assign(self.population,self.config)
        self.assertTrue(assignments.decision_disagreement.any())
        self.assertTrue((assignments.approved_a.ne(assignments.approved_b)).any())
        self.assertTrue(assignments.dataset.eq("train").all())
        self.assertTrue(assignments.policy_cohort.eq("policy_evaluation").all())
        self.assertTrue(assignments.common_score.ge(.9).all())
        self.assertTrue(assignments.observed_dpd30.isna().all())

    def test_power_minimum_integer_and_mde_allocation_tradeoffs(self):
        n=experiment.required_sample(10000,10000,.5,10,.05,.8)
        self.assertGreaterEqual(experiment.normal_power(n,10000,10000,.5,10),.8)
        self.assertLess(experiment.normal_power(n-1,10000,10000,.5,10),.8)
        smaller=experiment.required_sample(10000,10000,.5,5,.05,.8)
        self.assertAlmostEqual(smaller/n,4,delta=.003)
        self.assertGreater(experiment.required_sample(10000,10000,.1,10,.05,.8),n)
        self.assertAlmostEqual(experiment.normal_power(1000,10000,10000,.5,0),.05,places=12)
        with self.assertRaises(ValueError): experiment.required_sample(10000,10000,.5,0,.05,.8)

    def test_planning_cannot_use_evaluation_or_test_outcomes_or_paired_variance(self):
        expected,n=experiment.planning(self.population,self.config)
        changed=self.population.copy()
        changed.loc[changed.policy_cohort.ne("policy_development"),["potential_profit_a","potential_profit_b"]]=1e9
        actual,m=experiment.planning(changed,self.config)
        pd.testing.assert_frame_equal(expected,actual)
        self.assertEqual(n,m)
        # Even identical policy outcomes have nonzero independent-arm variance.
        same=self.population.copy()
        same["potential_profit_b"]=same.potential_profit_a
        planned,_=experiment.planning(same,self.config)
        self.assertTrue(planned.loc[planned.endpoint.eq("profit_superiority"),"planning_variance_b"].gt(0).all())

    def test_welch_matches_scipy_and_wilson_zero_bad_is_not_zero_risk(self):
        a=np.array([690,-8510,-10,690,690])
        b=np.array([690,690,-10,-10,690,-8510])
        result=experiment.mean_comparison(a,b)
        reference=stats.ttest_ind(b,a,equal_var=False)
        self.assertAlmostEqual(result["simulation_p_value"],reference.pvalue,places=12)
        self.assertAlmostEqual(result["df"],reference.df,places=12)
        self.assertLess(result["ci_low"],result["delta"])
        self.assertGreater(result["ci_high"],result["delta"])
        self.assertIsNone(experiment.wilson_upper(0,0))
        self.assertGreater(experiment.wilson_upper(0,100),0)

    def test_compressed_bootstrap_has_correct_mean_variance_and_constant_boundary(self):
        values=np.array([-8510,-10,690,690])
        draws=experiment.bootstrap_mean(values,6000,np.random.default_rng(19))
        expected_var=values.var(ddof=0)/len(values)
        self.assertAlmostEqual(draws.mean(),values.mean(),delta=5*np.sqrt(expected_var/len(draws)))
        self.assertAlmostEqual(draws.var(ddof=1),expected_var,delta=.08*expected_var)
        np.testing.assert_array_equal(experiment.bootstrap_mean([0,0,0],300,np.random.default_rng(2)),np.zeros(300))

    def test_unavailable_test_labels_profits_and_duplicate_keys_are_rejected(self):
        experiment.validate_population(self.population)
        for column in ("label","potential_profit_a"):
            bad=self.population.copy()
            bad.loc[bad.dataset.eq("test"),column]=0
            with self.assertRaises(ValueError): experiment.validate_population(bad)
        with self.assertRaises(ValueError): experiment.validate_population(pd.concat([self.population,self.population.iloc[[0]]]))

    def test_sql_replay_itt_cashflows_guardrails_and_deterministic_simulations(self):
        assignments=experiment.assign(self.population,self.config)
        power,n=experiment.planning(self.population,self.config)
        quality=experiment.assignment_quality(assignments,self.config)
        boot,draws=experiment.bootstrap(self.population,assignments,self.config)
        mc,future=experiment.monte_carlo(self.population,n,self.config)
        self.assertTrue(mc.null_rejection_rate_centered.between(0,1).all())
        self.assertTrue(mc.simulated_power_at_empirical_delta.between(0,1).all())
        diagnostics=experiment.decision_diagnostics(assignments,quality,self.config)
        self.assertFalse(diagnostics["production_rollout_ready"])
        with duckdb.connect() as connection:
            connection.register("_population",self.population)
            connection.register("_assignments",assignments)
            connection.execute(experiment.SQL_FILE.read_text(encoding="utf-8"))
            frames={"power_analysis":power,"assignment_quality":quality,"bootstrap_results":boot,
                    "monte_carlo_results":mc,"experiment_duration":experiment.duration_table(n,self.config)}
            for table,frame in frames.items():
                connection.register("_frame",frame)
                connection.execute(f"CREATE TABLE {table} AS SELECT * FROM _frame")
                connection.unregister("_frame")
            audit,checked_draws,checked_future=verify_connection(connection,self.population,assignments,self.config)
            self.assertEqual(audit["status"],"passed")
            pd.testing.assert_frame_equal(draws,checked_draws)
            pd.testing.assert_frame_equal(future,checked_future)
            rejected=assignments.loc[~assignments.assigned_approved,"simulated_observed_net_profit"]
            self.assertTrue(rejected.eq(-10).all())

    def test_score_one_belongs_to_last_stratum_and_parent_pairing_keeps_dataset_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"parent.duckdb"
            rows=[]
            for dataset,label,profit in (("train",1,690),("test",None,None)):
                for arm in ("A","B"):
                    rows.append({"dataset":dataset,"policy_cohort":"policy_evaluation" if dataset=="train" else "unlabeled_test",
                        "id":0,"label":label,"prospective_arm":arm,"p_repay":1.,"approved":True,"historical_simulated_net_profit":profit})
            with duckdb.connect(str(path)) as connection:
                connection.register("_rows",pd.DataFrame(rows))
                connection.execute("CREATE TABLE selected_policy_application_outcomes AS SELECT * FROM _rows")
            population=experiment.make_population(path,self.config)
            self.assertEqual(len(population),2)
            self.assertTrue(population.risk_stratum.eq("S4").all())
            self.assertTrue(population.loc[population.dataset.eq("test"),"potential_profit_a"].isna().all())

    def test_configuration_rejects_invalid_allocations_and_false_live_scope(self):
        for field,value in (("challenger_probability",0),("challenger_probability",1)):
            config=copy.deepcopy(self.config)
            config["randomization"][field]=value
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"config.json"
                experiment.business.write_json(path,config)
                with self.assertRaises(ValueError): experiment.load_config(path)
        config=copy.deepcopy(self.config)
        config["eligibility"]["production_safety_validated"]=True
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"config.json"
            experiment.business.write_json(path,config)
            with self.assertRaises(ValueError): experiment.load_config(path)


if __name__=="__main__": unittest.main()

"""Dashboard evidence boundaries, filter protection and artifact checks."""
import copy
import unittest
import tempfile
from pathlib import Path
import pandas as pd

from scripts import powerbi_data as data
from scripts import build_powerbi_dashboard as builder
from scripts import verify_powerbi_dashboard as audit


class PowerBIDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables,cls.evidence = data.collect()
        cls.model = audit.read(data.OUT/"LoanRisk.SemanticModel/model.bim")

    def test_frozen_sources_and_complete_project(self):
        result = audit.verify()
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["pages"],6)
        self.assertGreater(result["visual_field_bindings"],80)
        manifest = audit.read(data.OUT/"source_manifest.json")
        self.assertEqual(result["desktop_refresh_verified"],bool(manifest.get("desktop_refresh_verified",False)))
        self.assertEqual(result["dax_engine_execution_verified"],manifest["dax_engine_execution_verified"])
        if result["dax_engine_execution_verified"]:
            self.assertNotEqual(result["dax_engine_execution_scope"],"not_executed")
            self.assertNotEqual(result["desktop_runtime_validation_basis"],"not_confirmed")

    def test_missing_report_version_is_rejected_before_schema_enumeration(self):
        with tempfile.TemporaryDirectory(prefix="powerbi_contract_",dir=data.ROOT/"tmp") as folder:
            root = Path(folder)
            for relative in audit.REQUIRED_PROJECT_FILES:
                if relative.endswith("/version.json"): continue
                path = root/relative
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("{}",encoding="utf-8")
            with self.assertRaisesRegex(AssertionError,"definition/version.json"):
                audit.schema_errors(root)

    def test_measure_column_name_conflict_is_rejected_case_insensitively(self):
        broken = {"model":{"tables":[{"name":"ModelMetrics","columns":[{"name":"KS"}],
            "measures":[{"name":"ks","expression":"MAX(ModelMetrics[KS])"}]}]}}
        with self.assertRaisesRegex(AssertionError,"Measure/column name conflict"):
            audit.verify_model_names(broken)

    def test_legacy_or_missing_source_metadata_is_rejected(self):
        for declared in (None,"powerBI_V1","powerBI_V2"):
            model = {"model":{}}
            if declared is not None: model["model"]["defaultPowerBIDataSourceVersion"] = declared
            with self.assertRaisesRegex(AssertionError,"Enhanced refresh metadata"):
                audit.verify_refresh_metadata(model)
        audit.verify_refresh_metadata({"model":{"defaultPowerBIDataSourceVersion":"powerBI_V3"}})

    def test_unlabeled_results_and_unverified_gates_stay_null(self):
        data.assert_contract(self.tables,self.evidence)
        for name in ("SelectedPolicies","PolicyCurves"):
            unknown = self.tables[name].loc[self.tables[name].Dataset.eq("test")]
            self.assertGreater(len(unknown),0)
            self.assertTrue(unknown.HistoricalSimulatedProfitPerApplication.isna().all())
            self.assertTrue(unknown.HistoricalApprovedDefaultRate.isna().all())
        gates = self.tables["PromotionGates"]
        self.assertEqual(gates.PassFlag.isna().sum(),2)
        self.assertEqual(self.evidence["state"]["ProductionReady"],0)

    def test_null_to_zero_mutation_is_rejected(self):
        broken = copy.deepcopy(self.evidence)
        broken["state"]["LiveProfitDelta"] = 0.
        with self.assertRaises(AssertionError): data.assert_contract(self.tables,broken)
        broken_tables = {k:v.copy() for k,v in self.tables.items()}
        t = broken_tables["SelectedPolicies"]
        t.loc[t.Dataset.eq("test"),"HistoricalApprovedDefaultRate"] = 0.
        with self.assertRaises(AssertionError): data.assert_contract(broken_tables,self.evidence)

    def test_historical_population_denominators_are_distinct(self):
        self.assertTrue(self.tables["ModelMetrics"].Rows.eq(688998).all())
        policies = self.tables["SelectedPolicies"]
        base_eval = policies.loc[policies.Cohort.eq("policy_evaluation") & policies.ScenarioId.eq("base")]
        self.assertTrue(base_eval.Applications.eq(275600).all())
        self.assertEqual(self.evidence["state"]["TargetEligible"],210856)
        self.assertEqual(self.evidence["state"]["CurrentSamples"],12614)
        self.assertTrue(base_eval.PopulationScope.str.contains("ITT").all())

    def test_changing_exposure_only_changes_duration_assumption(self):
        state = copy.deepcopy(self.evidence["state"])
        all_in = data.reference_forecast(state)
        half = data.reference_forecast(state,exposure=.5)
        self.assertEqual((all_in["enrollment_days"],all_in["final_days"]),(289,684))
        self.assertEqual(half["enrollment_days"],577)
        self.assertFalse(half["feasible"])
        self.assertAlmostEqual(half["changed_decision_fraction"]*2,all_in["changed_decision_fraction"])
        self.assertEqual(state,self.evidence["state"])

    def test_decision_measure_cannot_filter_away_failed_gate(self):
        builder.add_measures()
        measures = {m["name"]:m["expression"] for group in builder.MEASURES.values() for m in group}
        for name in ("当前晋级决定","冻结收益差","冻结收益下界","业务收益门槛","所需分歧样本"):
            self.assertIn("REMOVEFILTERS()",measures[name])
            self.assertNotIn("PromotionGates",measures[name])
            self.assertNotIn("Exposure",measures[name])

    def test_mistyped_visual_field_binding_is_rejected(self):
        v = {"field":{"Measure":{"Expression":{"SourceRef":{"Entity":"DashboardState"}},"Property":"missing_measure"}}}
        with self.assertRaisesRegex(AssertionError,"Unknown Measure"):
            audit.verify_bindings(self.model,v,"test visual")

    def test_unapproved_stress_failures_and_synthetic_scenarios_remain_visible(self):
        calibration = self.tables["Calibration"]
        self.assertEqual(int((calibration.Passed==0).sum()),6)
        self.assertEqual(int((calibration.Pipeline.eq("V2 匹配校准") & calibration.Passed.eq(1)).sum()),20)
        sims = self.tables["ABSimulations"]
        synthetic = sims.loc[sims.Synthetic.eq(1)]
        self.assertTrue(synthetic.Basis.str.contains("非 Challenger").all())
        self.assertTrue(synthetic.EvidenceCategory.eq("模拟分析").all())


if __name__=="__main__": unittest.main()

"""Export a small, source-hashed portfolio fact sheet without fitting models."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"docs/project_facts.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def collect(root=ROOT):
    sources = {}
    def source(name):
        path = root/name
        sources[name] = sha(path)
        return path
    def read(name): return json.loads(source(name).read_text(encoding="utf-8-sig"))
    def rows(name):
        with source(name).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
    quality = read("configs/sql_business_analysis.json")
    performance = read("reports/sql_business_analysis/summary.json")
    policy = read("configs/approval_policy_simulation.json")
    design = read("configs/ab_feasibility_v2.json")
    ab = read("reports/ab_feasibility_v2/summary.json")
    for stage in ("sql_business_analysis","approval_policy_simulation","ab_test_design","aa_validation","ab_feasibility_v2"):
        assert read(f"reports/{stage}/artifact_verification.json")["status"]=="passed"
    old_cal = rows("reports/aa_validation/tables/aa_calibration_results.csv")
    matched = rows("reports/ab_feasibility_v2/tables/matched_calibration.csv")
    simulation = rows("reports/ab_feasibility_v2/tables/ab_simulation_results.csv")
    selected = rows("reports/approval_policy_simulation/tables/selected_policy_results.csv")
    baseline = read("reports/lightgbm_baseline/run_config.json")
    tuned = read("reports/hyperparameter_seed_optimization/run_config.json")["experiment"]
    storage = read("data/processed/parquet_manifest.json")
    dashboard = read("dashboard/powerbi/source_manifest.json")
    memo = read("reports/experiment_closeout/closeout_provenance.json")
    by_role = {r["model_role"]:r for r in performance["overall_metrics"]}
    policies = {r["model_role"]:r for r in ab["protocol"]["selected_policies"]}
    models = [{**m, **by_role[m["model_role"]], "feature_count":baseline["feature_count"],
        "refit_models":5 if m["model_role"]=="champion" else 15,
        "threshold":policies[m["model_role"]]["threshold"]} for m in quality["models"]]
    evaluations=[]
    for r in selected:
        if r["scenario_id"]=="base" and r["policy_cohort"]=="policy_evaluation" and r["prospective_arm"] in ("A","B"):
            evaluations.append({"arm":r["prospective_arm"],"model_version":r["model_version"],
                "applications":int(r["total_applications"]),"approval_rate":float(r["approval_rate"]),
                "approved_historical_default_rate":float(r["observed_default_rate_approved"]),
                "simulated_profit_per_application":float(r["historical_profit_per_application"])})
    by_case = {r["case_id"]:r for r in simulation}
    assert ab["champion_retained"] and not ab["challenger_promoted"] and not ab["live_experiment_completed"]
    assert len(matched)==20 and all(r["passed"].lower() in ("true","1") for r in matched)
    facts = {
        "schema_version":1,"release_date":"2026-10-25","timezone":"Asia/Shanghai",
        "scope":"historical_Kaggle_modeling_and_offline_business_experiment_portfolio",
        "dataset":{"train_rows":quality["expected_train_rows"],"test_rows":quality["expected_test_rows"],
            "train_columns":1107,"test_columns":1106,"anonymous_numeric_features":baseline["feature_count"],
            "repayment_label":1,"default_label":0,"test_labels_available":False,
            "id_is_model_feature":False,"verified_training_time_order":False},
        "models":models,"auc_delta":performance["challenger_auc_delta"],
        "model_training_versions":{"champion":{"lightgbm":baseline["lightgbm_version"],"numpy":baseline["numpy_version"]},
            "challenger":{"lightgbm":tuned["lightgbm_version"],"numpy":tuned["numpy_version"]}},
        "validation":{"folds":5,"split_seed":42,"method":"shuffled_stratified_with_inner_early_stopping",
            "reused_development_oof":True,"probability_calibrated":False},
        "policy":{"development_rows":413398,"evaluation_rows":275600,"economics":policy["economics"],
            "selected_evaluation":evaluations,"outcome_assumption":policy["offline_outcome_assumption"]},
        "aa":{"original_checks":len(old_cal),
            "original_uncertified_checks":sum(r["calibration_passed"].lower() in ("false","0") for r in old_cal),
            "v2_matched_checks":len(matched),"v2_passed":True,
            "fixed_look_fpr":0.04934,"naive_any_look_fpr":0.1420,
            "certification_scope":"declared historical finite-population/IID settings only"},
        "experiment":{"experiment_id":design["experiment_id"],"protocol":ab["protocol"],"replay":ab["replay"],
            "full_eligible_evaluation_rows":210856,"randomized_disagreements":12614,
            "primary_assumed_enrollment_days":ab["primary_assumed_enrollment_days"],
            "primary_assumed_final_analysis_days":ab["primary_assumed_final_analysis_days"],
            "historical_planned_joint_success":float(by_case["historical_at_planned"]["joint_success_rate"]),
            "synthetic_10_joint_success":float(by_case["synthetic_true_gain_10"]["joint_success_rate"]),
            "champion_retained":True,"challenger_promoted":False,"live_experiment_completed":False,
            "production_rollout_ready":False,"live_outcome_maturity_validated":False},
        "dashboard":{"pages":6,"tables":23,"aggregate_rows":665,"format":"PBIP/PBIR with local CSV import",
            "desktop_open_verified":dashboard["desktop_open_verified"],
            "desktop_refresh_verified":dashboard.get("desktop_refresh_verified",False),
            "decision_page_render_verified":dashboard.get("decision_page_render_verified",False),
            "all_pages_render_verified":dashboard.get("all_pages_render_verified",False),
            "runtime_basis":dashboard.get("desktop_runtime_validation_basis")},
        "closeout":{"one_page_pdf":memo["page_count"]==1,"visual_review_passed":memo["visual_review_passed"]},
        "storage":{"train_parquet_bytes":storage["datasets"][0]["parquet_bytes"],
            "test_parquet_bytes":storage["datasets"][1]["parquet_bytes"]},
        "unknown_live_results":{"profit":None,"default_rate":None,"dpd30":None},
        "sources_sha256":sources,
    }
    return facts


def main():
    facts=collect()
    OUT.write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Portfolio facts exported: {OUT}")


if __name__=="__main__": main()

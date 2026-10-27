"""Single local entry point for status, preflight, tests and staged reproduction."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

PIPELINES = {
    "data":[["scripts/init_duckdb.py"],["scripts/convert_to_parquet.py"],
        ["scripts/run_data_quality.py"],["scripts/generate_sql_quality_report.py"],
        ["scripts/run_eda.py"],["scripts/check_train_row_order.py"],
        ["scripts/run_adversarial_validation.py"],["scripts/build_validation_strategy.py"]],
    "models":[["scripts/run_linear_baselines.py"],["scripts/verify_linear_baselines.py"],
        ["scripts/run_lightgbm_baseline.py","--resume"],["scripts/verify_lightgbm_baseline.py"],
        ["scripts/run_hyperparameter_seed_optimization.py","--stage","init"],
        ["scripts/run_hyperparameter_seed_optimization.py","--stage","train"],
        ["scripts/run_hyperparameter_seed_optimization.py","--stage","collect"],
        ["scripts/verify_hyperparameter_seed_optimization.py"]],
    "business":[["scripts/run_sql_business_analysis.py"],["scripts/run_approval_policy_simulation.py"],
        ["scripts/run_ab_test_design.py"],["scripts/run_aa_validation.py"],["scripts/run_ab_feasibility.py"]],
    "dashboard":[["scripts/build_powerbi_dashboard.py"]],
}
VERIFY = {
    "dashboard":[["scripts/verify_powerbi_dashboard.py"]],
    "business":[["scripts/replay_frozen_business.py"]],
}
BUSINESS_STAGES = {
    "sql_business_analysis": "model_freeze_manifest.json",
    "approval_policy_simulation": "policy_freeze_manifest.json",
    "ab_test_design": "experiment_freeze_manifest.json",
    "aa_validation": "aa_freeze_manifest.json",
    "ab_feasibility_v2": "feasibility_freeze_manifest.json",
}
TEST_MODULES = ["tests.test_sql_business_analysis","tests.test_approval_policy_simulation",
    "tests.test_ab_test_design","tests.test_aa_validation","tests.test_ab_feasibility","tests.test_powerbi_dashboard"]
REQUIRED_DOCS = ["README.md","README.en.md","docs/INDEX.md","docs/model_card.md","docs/model_card.en.md",
    "docs/experiment_card.md","docs/experiment_card.en.md","docs/architecture.md","docs/reproducibility.md",
    "docs/project_facts.json","docs/figures/architecture.svg","docs/environment_snapshot.json"]
ANALYSIS_PACKAGES = ["duckdb","pandas","numpy","scipy","scikit-learn","pyarrow","jsonschema","referencing","Pillow"]


def reproduction_plan(target, root=ROOT):
    """Reuse frozen stages; never overwrite a frozen parent or an occupied run."""
    freezes = {stage: (root/"artifacts"/stage/name).is_file()
               for stage,name in BUSINESS_STAGES.items()}
    commands = [list(args) for args in PIPELINES[target]]
    blocked = []
    if target in ("data", "models") and any(freezes.values()):
        blocked.append("Frozen business evidence depends on data/models here. Rebuild in a clean work copy; preserve this workspace.")
    if target == "business":
        commands = []
        stages = list(BUSINESS_STAGES)
        for index,stage in enumerate(stages):
            if freezes[stage]:
                commands.append(["scripts/replay_frozen_business.py", "--stage", stage])
                continue
            occupied = any(path.exists() and (path.is_file() or any(path.iterdir())) for path in (
                root/"artifacts"/stage, root/"reports"/stage, root/"data/interim"/f"{stage}.duckdb"))
            if occupied or any(freezes[s] for s in stages[index+1:]):
                blocked.append(f"{stage}: missing freeze with existing output/descendant. Preserve evidence and use a clean work copy.")
            commands.append(list(PIPELINES["business"][index]))
    if target == "dashboard" and (root/"dashboard/powerbi/LoanRisk.pbip").exists():
        commands = [list(args) for args in VERIFY["dashboard"]]
    return {"target":target, "commands":commands, "blocked_reasons":blocked,
            "existing_business_freezes":freezes,
            "note":"Frozen stages are checked read-only. Missing, unoccupied stages are built. No implicit prerequisite rebuild."}


def doctor(profile="docs",root=ROOT):
    paths = list(REQUIRED_DOCS)
    packages=[]
    if profile in ("analysis","full"):
        packages = list(ANALYSIS_PACKAGES)
        paths += ["data/processed/train.parquet","data/processed/test.parquet","data/processed/sample_submission.parquet",
            "data/interim/validation_folds.parquet","data/interim/loan_repayment.duckdb",
            "artifacts/lightgbm_baseline/oof_predictions.parquet",
            "artifacts/hyperparameter_seed_optimization/oof_predictions.parquet"]
    if profile=="full":
        packages += ["lightgbm","xgboost","catboost"]
        paths += ["train.csv","test.csv","sample_submission.csv"]
    missing_paths=[name for name in paths if not (root/name).is_file()]
    installed,missing_packages={},[]
    for name in packages:
        try: installed[name]=metadata.version(name)
        except metadata.PackageNotFoundError: missing_packages.append(name)
    return {"profile":profile,"status":"ready" if not missing_paths and not missing_packages else "missing_prerequisites",
        "python":sys.version.split()[0],"missing_files":missing_paths,"missing_packages":missing_packages,
        "installed_packages":installed,"raw_csv_present":{n:(root/n).is_file() for n in ("train.csv","test.csv")},
        "note":"docs is source-only review; analysis requires private outputs; full starts from original Kaggle CSVs"}


def execute(commands,dry_run=False,runner=subprocess.run):
    for step,args in enumerate(commands,1):
        command=[sys.executable,*args]
        print(json.dumps({"step":step,"command":command,"working_directory":str(ROOT),"dry_run":dry_run},ensure_ascii=False),flush=True)
        if dry_run: continue
        result=runner(command,cwd=ROOT,check=False)
        if result.returncode: return result.returncode
    return 0


def status():
    facts=json.loads((ROOT/"docs/project_facts.json").read_text(encoding="utf-8"))
    return {"scope":facts["scope"],"champion_retained":facts["experiment"]["champion_retained"],
        "challenger_promoted":facts["experiment"]["challenger_promoted"],
        "live_experiment_completed":facts["experiment"]["live_experiment_completed"],
        "profit_delta":facts["experiment"]["replay"]["delta"],
        "profit_95_ci":[facts["experiment"]["replay"]["ci_low"],facts["experiment"]["replay"]["ci_high"]],
        "required_randomized":facts["experiment"]["protocol"]["required_randomized_disagreements"],
        "current_randomized":facts["experiment"]["randomized_disagreements"],
        "dashboard":facts["dashboard"],"entry":"README.md / README.en.md"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("status",help="Print frozen business decision and delivery status")
    p=sub.add_parser("doctor",help="Check prerequisites without changing data")
    p.add_argument("--profile",choices=("docs","analysis","full"),default="docs")
    p=sub.add_parser("verify",help="Verify documentation or existing output; business verification may take time")
    p.add_argument("--scope",choices=("docs","dashboard","business"),default="docs")
    p.add_argument("--write",action="store_true",help="Write docs/delivery_verification.json for docs scope")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--quick",action="store_true",help="Business only: verify rows/SQL/exports/hashes without all Monte Carlo replays")
    p=sub.add_parser("test",help="Run delivery tests or the existing business test suite")
    p.add_argument("--scope",choices=("delivery","business"),default="delivery")
    p=sub.add_parser("reproduce",help="Run one phase only; dependencies must already exist")
    p.add_argument("--target",choices=tuple(PIPELINES),required=True)
    p.add_argument("--dry-run",action="store_true",help="Print the command plan without running stages")
    args=parser.parse_args()
    if args.command=="status": print(json.dumps(status(),ensure_ascii=False,indent=2));return 0
    if args.command=="doctor":
        result=doctor(args.profile);print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result["status"]=="ready" else 2
    if args.command=="verify":
        if args.scope=="docs":
            if args.quick: parser.error("--quick is supported only for business scope")
            if args.dry_run: print("python scripts/verify_project_delivery.py");return 0
            from scripts.verify_project_delivery import verify
            result=verify()
            if args.write: (ROOT/"docs/delivery_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result["status"]=="passed" else 2
        if args.write: parser.error("--write is supported only for docs scope")
        if args.quick and args.scope != "business": parser.error("--quick is supported only for business scope")
        commands = [list(step) for step in VERIFY[args.scope]]
        if args.quick: commands[0].append("--quick")
        return execute(commands,args.dry_run)
    if args.command=="test":
        modules=["tests.test_project_delivery"] if args.scope=="delivery" else ["tests.test_project_delivery",*TEST_MODULES]
        return execute([["-m","unittest",*modules,"-v"]])
    plan = reproduction_plan(args.target)
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if plan["blocked_reasons"] and not args.dry_run: return 2
    return execute(plan["commands"],args.dry_run)


if __name__=="__main__": raise SystemExit(main())

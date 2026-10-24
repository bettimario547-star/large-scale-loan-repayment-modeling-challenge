"""Read completed metadata only; do not load data/libraries or train models."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def progress():
    config_path = ROOT / "reports/tree_model_comparison/run_config.json"
    if not config_path.exists():
        print("Study has not been initialized")
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    signature = config["training_signature"]
    result = {"status": config["status"], "engines": {}, "scope": "completed folds only; not full OOF until collected"}
    for engine in ("xgboost", "catboost"):
        inner, refit = [], []
        for folder, destination in (("inner_models", inner), ("models", refit)):
            for path in sorted((ROOT / "artifacts/tree_model_comparison" / folder).glob(f"{engine}_fold_*.json")):
                try:
                    saved = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if saved["training_signature"] != signature:
                    raise RuntimeError("Unexpected experiment in completed checkpoint")
                record = saved["record"]
                keys = ("fold", "selected_rounds", "inner_auc", "hit_round_cap") if folder == "inner_models" else (
                    "fold", "refit_rounds", "repay_roc_auc", "default_pr_auc_ap", "log_loss")
                destination.append({key: record[key] for key in keys})
        result["engines"][engine] = {"inner_completed": len(inner), "refit_completed": len(refit),
                                    "inner": inner, "refit": refit}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    progress()

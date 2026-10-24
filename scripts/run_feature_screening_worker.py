"""Train one independent family while the main runner trains earlier families.

Writes only this family's complete-fold checkpoints and feature importance.
The main --resume runner will collect them. No shared report/config mutation.
"""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

try:
    from . import run_feature_screening as screening
except ImportError:
    import run_feature_screening as screening


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["low_information_filter", "drift_core_filter"], required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    config = json.loads((screening.REPORT / "run_config.json").read_text(encoding="utf-8"))
    for name, checksum in config["code_sha256"].items():
        if screening.sha256(screening.ROOT / "scripts" / name) != checksum:
            raise RuntimeError("Main training sources changed")
    args.threads, args.batch_rows = config["threads"], config["batch_rows"]
    context = screening.load_context(args.batch_rows)
    _, _, _, _, labels, fold_id, names, values, test_values, _ = context
    plans = json.loads((screening.REPORT / "feature_plans.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=args.threads):
        for fold in args.folds:
            plan = next(item for item in plans if item["variant"] == args.variant and item["fold"] == fold)
            output = screening.ARTIFACTS / args.variant
            checkpoint = output / f"fold_predictions/fold_{fold}.npz"
            metadata = output / f"models/fold_{fold}.json"
            if checkpoint.exists() and metadata.exists():
                saved = json.loads(metadata.read_text(encoding="utf-8"))
                if saved["training_signature"] != config["training_signature"] or saved["plan"] != plan:
                    raise RuntimeError("Worker checkpoint differs from current run")
                if screening.sha256(Path(saved["model_path"])) != saved["model_sha256"]:
                    raise RuntimeError("Worker model checksum mismatch")
                print(f"Worker reuse {args.variant} fold {fold}", flush=True)
                continue
            screening.run_variant_fold(values, test_values, labels, fold_id, names, fold, plan, args,
                                       config["training_signature"])
    print("Independent family completed; main runner will collect it", flush=True)


if __name__ == "__main__":
    main()

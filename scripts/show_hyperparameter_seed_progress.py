"""Read-only progress: completed trials and complete-fold seed ensembles.

Partial fold results are explicitly partial development scores, never a full
OOF score. No feature cache or model training is needed to inspect progress.
"""

import json

import numpy as np
import pandas as pd

try:
    from . import run_hyperparameter_seed_optimization as tuning
except ImportError:
    import run_hyperparameter_seed_optimization as tuning


def show():
    config = tuning.checked_config()
    signature = config["training_signature"]
    meta = pd.read_parquet(tuning.base.TRAIN, columns=["id", "label"])
    folds = tuning.base.align_validation_folds(meta.id, pd.read_parquet(tuning.base.FOLDS))
    labels = meta.label.to_numpy(dtype=np.int8)
    baseline = pd.read_parquet(tuning.base.ARTIFACTS / "oof_predictions.parquet")
    pd.testing.assert_frame_equal(baseline[["id", "label"]], meta, check_dtype=False)
    rows = []
    for fold in range(5):
        selection_path = tuning.REPORT / f"selections/fold_{fold}.json"
        row = {"fold": fold, "candidate": "search_not_completed", "rounds": "pending", "completed_seeds": []}
        if selection_path.exists():
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                selection = None
            if selection is not None:
                assert selection["training_signature"] == signature
                row.update(candidate=selection["chosen"]["candidate"], rounds=selection["chosen"]["selected_rounds"])
        predictions = []
        valid = np.flatnonzero(folds.fold.to_numpy() == fold)
        for seed in tuning.SEEDS:
            tag = f"seed_{seed}_fold_{fold}"
            prediction_path = tuning.ARTIFACTS / f"fold_predictions/{tag}.npz"
            metadata = tuning.ARTIFACTS / f"models/{tag}.json"
            try:
                record = tuning.checkpoint_record(metadata, prediction_path, signature)
            except json.JSONDecodeError:
                # A metadata file may be in its final write during a live run.
                continue
            if record is None:
                continue
            row["completed_seeds"].append(seed)
            with np.load(prediction_path) as saved:
                np.testing.assert_array_equal(saved["outer_valid_index"], valid)
                predictions.append(saved["p_repay_valid"].copy())
        if len(predictions) == len(tuning.SEEDS):
            ensemble = tuning.probability_mean(predictions)
            score = tuning.base.metrics(labels[valid], ensemble)
            original = tuning.base.metrics(labels[valid], baseline.p_repay.to_numpy()[valid])
            row.update(partial_fold_ensemble_auc=score["repay_roc_auc"],
                       delta_auc=score["repay_roc_auc"] - original["repay_roc_auc"],
                       partial_fold_default_ap=score["default_pr_auc_ap"],
                       partial_fold_log_loss=score["log_loss"])
        rows.append(row)
    completed_inner = len(list((tuning.ARTIFACTS / "inner_models").glob("*.json")))
    completed_final = sum(len(item["completed_seeds"]) for item in rows)
    print(f"Saved inner trial metadata: {completed_inner}/20; completed refit models: {completed_final}/15", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("Only complete-fold partial scores are shown; full OOF requires all five folds.", flush=True)


if __name__ == "__main__":
    show()

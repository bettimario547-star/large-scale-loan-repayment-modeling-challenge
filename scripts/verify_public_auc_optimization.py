"""Reload completed models and independently check OOF/test predictions."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from . import run_public_auc_optimization as optimizer
except ImportError:
    import run_public_auc_optimization as optimizer


def verify(root=optimizer.ROOT, sample_rows=512):
    if sample_rows < 1:
        raise ValueError("Sample size must be positive")
    root = Path(root)
    config = optimizer.checked_config(root)
    experiment = config["experiment"]
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    test_meta = pd.read_parquet(root / "data/processed/test.parquet", columns=["id"])
    folds = optimizer.base.align_validation_folds(
        meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    labels, fold_ids = meta.label.to_numpy(), folds.fold.to_numpy()
    raw = optimizer.trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/train_features.npy")
    raw_test = optimizer.trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/test_features.npy")
    rng = np.random.default_rng(8128)
    records = []
    for variant in optimizer.VARIANTS:
        for seed in (42, 2026, 3407):
            for fold in range(5):
                saved = optimizer.completed_job(root, variant, seed, fold, config["training_signature"])
                if saved is None:
                    continue
                _, prediction_path, model_path = optimizer.job_paths(root, variant, seed, fold)
                names = experiment["plans"][variant][str(fold)]["kept"]
                if saved["positive_class"] != 1 or saved["features"] != names:
                    raise RuntimeError("Checkpoint label direction or feature schema differs")
                if optimizer.CORE_DRIFT.intersection(names):
                    raise RuntimeError("Core drift feature was unexpectedly restored")
                values = optimizer.SubsetView(raw, experiment["features"], names)
                test_values = optimizer.SubsetView(raw_test, experiment["features"], names)
                with np.load(prediction_path, allow_pickle=False) as predictions:
                    valid = predictions["outer_valid_index"]
                    np.testing.assert_array_equal(valid, np.flatnonzero(fold_ids == fold))
                    expected_valid = predictions["p_repay_valid"]
                    expected_test = predictions["p_repay_test"]
                    if expected_valid.shape != valid.shape or expected_test.shape != (len(test_meta),):
                        raise RuntimeError("Checkpoint prediction shape differs")
                    for array in (expected_valid, expected_test):
                        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                            raise RuntimeError("Checkpoint contains invalid probabilities")
                    actual_metrics = optimizer.base.metrics(labels[valid], expected_valid)
                    for key, actual in actual_metrics.items():
                        np.testing.assert_allclose(actual, saved["metrics"][key], rtol=0, atol=1e-12)
                    valid_positions = np.sort(rng.choice(len(valid), min(sample_rows, len(valid)), replace=False))
                    test_positions = np.sort(rng.choice(len(test_meta), min(sample_rows, len(test_meta)), replace=False))
                    if variant.startswith("lgbm"):
                        model = lgb.Booster(model_file=str(model_path))
                        if model.feature_name() != names or model.current_iteration() != saved["rounds"]:
                            raise RuntimeError("Reloaded LightGBM schema/tree count differs")
                        actual_valid = model.predict(values[valid[valid_positions]], num_threads=1)
                        actual_test = model.predict(test_values[test_positions], num_threads=1)
                    else:
                        model = cb.CatBoostClassifier()
                        model.load_model(str(model_path))
                        if model.feature_names_ != names or model.tree_count_ != saved["rounds"]:
                            raise RuntimeError("Reloaded CatBoost schema/tree count differs")
                        actual_valid = model.predict_proba(
                            cb.Pool(values[valid[valid_positions]], feature_names=names, thread_count=1),
                            thread_count=1)[:, 1]
                        actual_test = model.predict_proba(
                            cb.Pool(test_values[test_positions], feature_names=names, thread_count=1),
                            thread_count=1)[:, 1]
                    np.testing.assert_allclose(actual_valid, expected_valid[valid_positions], rtol=0, atol=1e-10)
                    np.testing.assert_allclose(actual_test, expected_test[test_positions], rtol=0, atol=1e-10)
                    records.append({"variant": variant, "seed": seed, "fold": fold,
                                    "backend": saved["backend"], "valid_rows_checked": len(valid_positions),
                                    "test_rows_checked": len(test_positions), "all_oof_metrics_recomputed": True,
                                    "max_prediction_difference": float(max(
                                        np.max(np.abs(actual_valid - expected_valid[valid_positions])),
                                        np.max(np.abs(actual_test - expected_test[test_positions]))))})
                    del model
                    gc.collect()
                    print(f"Verified {variant}, seed {seed}, fold {fold}", flush=True)
    report, _ = optimizer.locations(root)
    result = {"status": "passed" if records else "no_completed_folds",
              "training_signature": config["training_signature"],
              "prediction_check_scope": "deterministic sampled OOF/test rows; full saved OOF metrics",
              "models_reloaded": len(records), "checks": records}
    optimizer.candidates.json_write(report / "model_reload_verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=optimizer.ROOT)
    parser.add_argument("--sample-rows", type=int, default=512)
    args = parser.parse_args()
    verify(args.root, args.sample_rows)


if __name__ == "__main__":
    main()

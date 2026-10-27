"""Fixed-cost CPU challenger excluding four drift columns, with full OOF."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from threadpoolctl import threadpool_limits

try:
    from . import run_public_auc_optimization as base
except ImportError:
    import run_public_auc_optimization as base

ROOT = base.ROOT
NAME = "xgboost_drift_fast_s42"


def locations(root):
    root = Path(root)
    return root / "reports/xgboost_drift_fast", root / "artifacts/xgboost_drift_fast"


def initialize(root):
    parent = base.checked_config(root)
    exp = {"parent_training_signature": parent["training_signature"],
           "source_sha256": base.candidates.sha256(Path(__file__)), "xgboost_version": xgb.__version__,
           "features": [name for name in parent["experiment"]["features"] if name not in base.CORE_DRIFT],
           "rounds": 600, "scope": "fixed recipe and tree count; outer validation not used for early stopping",
           "params": {"objective": "binary:logistic", "eval_metric": "auc", "booster": "gbtree",
               "tree_method": "hist", "device": "cpu", "grow_policy": "depthwise", "eta": .075,
               "max_depth": 6, "min_child_weight": 30., "lambda": 8., "alpha": .05,
               "subsample": .8, "colsample_bytree": .8, "max_bin": 63, "verbosity": 1}}
    signature = base.trees.signing_hash(exp)
    path = locations(root)[0] / "training_config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["training_signature"] != signature or saved["experiment"] != exp:
            raise RuntimeError("Fixed-cost XGBoost configuration changed")
        return saved
    config = {"training_signature": signature, "experiment": exp}
    base.candidates.json_write(path, config)
    return config


def paths(root, fold):
    directory = locations(root)[1] / "jobs"
    return directory / f"fold_{fold}.json", directory / f"fold_{fold}.npz", directory / f"fold_{fold}.ubj"


def checkpoint(root, fold, config):
    metadata, predictions, model = paths(root, fold)
    if not metadata.exists():
        return None
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    if (saved["training_signature"] != config["training_signature"] or saved["positive_class"] != 1
        or saved["features"] != config["experiment"]["features"] or saved["backend"] != "CPU"
        or base.candidates.sha256(predictions) != saved["prediction_sha256"]
        or base.candidates.sha256(model) != saved["model_sha256"]):
        raise RuntimeError("Fixed-cost XGBoost checkpoint differs")
    return saved


def context(root, config):
    root = Path(root)
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    test_meta = pd.read_parquet(root / "data/processed/test.parquet", columns=["id"])
    folds = base.base.align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    names = config["experiment"]["features"]
    full_names = base.checked_config(root)["experiment"]["features"]
    values = base.SubsetView(base.trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/train_features.npy"), full_names, names)
    tests = base.SubsetView(base.trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/test_features.npy"), full_names, names)
    return meta, test_meta, folds, values, tests


class Progress(xgb.callback.TrainingCallback):
    def __init__(self):
        self.started = time.monotonic()

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 50 == 0:
            print(f"XGBoost: trees={epoch+1}, fit_elapsed={time.monotonic()-self.started:.0f}s", flush=True)
        return False


def train_fold(root, fold, threads):
    config = initialize(root)
    if checkpoint(root, fold, config):
        return
    meta, test_meta, folds, values, tests = context(root, config)
    labels = meta.label.to_numpy(dtype=np.int8)
    fitting, valid = np.flatnonzero(folds.fold.to_numpy() != fold), np.flatnonzero(folds.fold.to_numpy() == fold)
    names = config["experiment"]["features"]
    params = dict(config["experiment"]["params"], nthread=threads, seed=42+fold)
    metadata, prediction_path, model_path = paths(root, fold)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"Start fixed-cost XGBoost fold {fold}: {len(fitting):,} rows, {len(names)} columns, {threads} threads", flush=True)
    with threadpool_limits(limits=threads):
        matrix = base.trees.xgb_data(values, labels, fitting, names, threads, 4096)
        np.testing.assert_array_equal(matrix.get_label(), labels[fitting])
        model = xgb.train(params, matrix, num_boost_round=config["experiment"]["rounds"], callbacks=[Progress()])
        del matrix
        gc.collect()
        saved_config = json.loads(model.save_config())["learner"]
        if saved_config["generic_param"]["device"] != "cpu" or saved_config["objective"]["name"] != "binary:logistic":
            raise RuntimeError("Unexpected XGBoost backend or objective")
        if model.feature_names != names or model.num_boosted_rounds() != config["experiment"]["rounds"]:
            raise RuntimeError("Unexpected XGBoost feature order/tree count")
        model.save_model(model_path)
        p_valid = base.trees.predict_rows("xgboost", model, values, names, 4096, valid)
        p_test = base.trees.predict_rows("xgboost", model, tests, names, 4096)
        np.savez_compressed(prediction_path, outer_valid_index=valid, p_repay_valid=p_valid, p_repay_test=p_test)
        base.candidates.json_write(metadata, {"training_signature": config["training_signature"], "fold": fold,
            "features": names, "params": params, "rounds": config["experiment"]["rounds"], "backend": "CPU", "positive_class": 1,
            "elapsed_seconds": time.monotonic()-started, "metrics": base.base.metrics(labels[valid], p_valid),
            "prediction_sha256": base.candidates.sha256(prediction_path), "model_sha256": base.candidates.sha256(model_path)})
    print(f"Completed fixed-cost XGBoost fold {fold}: AUC={base.base.metrics(labels[valid],p_valid)['repay_roc_auc']:.6f}", flush=True)


def collect(root):
    config = initialize(root)
    saved = [checkpoint(root, fold, config) for fold in range(5)]
    base.candidates.json_write(locations(root)[0] / "training_progress.json", {
        "completed_folds": sum(row is not None for row in saved), "complete_candidate": all(row is not None for row in saved)})
    if not all(row is not None for row in saved):
        return False
    meta, test_meta, folds, _, _ = context(root, config)
    scores, tests, coverage = np.full(len(meta), np.nan), np.zeros(len(test_meta)), np.zeros(len(meta), dtype=np.int8)
    for fold in range(5):
        with np.load(paths(root, fold)[1], allow_pickle=False) as stored:
            valid = stored["outer_valid_index"]
            np.testing.assert_array_equal(valid, np.flatnonzero(folds.fold.to_numpy() == fold))
            if stored["p_repay_test"].shape != tests.shape or stored["p_repay_valid"].shape != valid.shape:
                raise RuntimeError("Incorrect fixed-cost XGBoost prediction shape")
            scores[valid] = stored["p_repay_valid"]
            tests += stored["p_repay_test"] / 5
            coverage[valid] += 1
    if not (coverage == 1).all():
        raise RuntimeError("Incomplete fixed-cost XGBoost OOF coverage")
    for values in (scores, tests):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise RuntimeError("Invalid XGBoost probabilities")
    output = Path(root) / "artifacts/public_auc_optimization/completed" / NAME
    output.mkdir(parents=True, exist_ok=True)
    frame = folds.copy()
    frame["label"], frame["p_repay"] = meta.label.to_numpy(), scores
    frame.to_parquet(output / "oof_predictions.parquet", index=False)
    pd.DataFrame({"id": test_meta.id, "p_repay": tests}).to_parquet(output / "test_predictions.parquet", index=False)
    base.candidates.json_write(output / "model_summary.json", {"training_signature": config["training_signature"],
        "metrics": base.base.metrics(meta.label.to_numpy(), scores), "backend": "CPU", "public_auc": None})
    return True


def train(root, minutes, threads):
    config = initialize(root)
    if psutil.virtual_memory().available < 2 * 1024**3:
        raise RuntimeError("Fixed-cost XGBoost needs at least 2 GiB available RAM")
    logs = Path(root) / "logs/xgboost_drift_fast" / str(time.time_ns())
    logs.mkdir(parents=True)
    deadline = time.monotonic() + minutes * 60
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(threads)
    failures, stopped = [], []
    for fold in range(5):
        if checkpoint(root, fold, config):
            continue
        if time.monotonic() >= deadline:
            break
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "worker", "--root", str(root), "--fold", str(fold), "--threads", str(threads)]
        with (logs / f"fold_{fold}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            print(f"Launched fixed-cost XGBoost fold {fold}, PID {process.pid}", flush=True)
            try:
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.5)
                if process.poll() is None:
                    stopped.append(fold)
                elif process.returncode:
                    failures.append({"fold": fold, "exit_code": process.returncode})
                elif checkpoint(root, fold, config) is None:
                    raise RuntimeError("XGBoost worker exited without a completed checkpoint")
            finally:
                base.stop_worker_tree(process)
        if failures or stopped:
            break
    saved = [checkpoint(root, fold, config) for fold in range(5)]
    complete = all(row is not None for row in saved)
    base.candidates.json_write(locations(root)[0] / "training_progress.json", {
        "completed_folds": sum(row is not None for row in saved), "complete_candidate": complete})
    base.candidates.json_write(locations(root)[0] / "last_training_session.json", {
        "minutes_budget": minutes, "threads": threads, "complete_candidate": complete,
        "failures": failures, "stopped_at_deadline": stopped, "log_directory": str(logs)})
    # Defer prediction export and candidate rebuilding until all controllers
    # finish, so another model cannot read a partially written Parquet pair.
    print(f"Fixed-cost XGBoost session finished; complete five-fold candidate: {complete}", flush=True)


def verify(root):
    config = initialize(root)
    meta, test_meta, folds, values, tests = context(root, config)
    rng, records = np.random.default_rng(8128), []
    for fold in range(5):
        saved = checkpoint(root, fold, config)
        if saved is None:
            continue
        model = xgb.Booster(model_file=paths(root, fold)[2])
        if model.feature_names != saved["features"] or model.num_boosted_rounds() != saved["rounds"]:
            raise RuntimeError("Reloaded XGBoost schema/tree count differs")
        model.set_param({"nthread": 1})
        with np.load(paths(root, fold)[1], allow_pickle=False) as stored:
            valid = stored["outer_valid_index"]
            np.testing.assert_array_equal(valid, np.flatnonzero(folds.fold.to_numpy() == fold))
            metrics = base.base.metrics(meta.label.to_numpy()[valid], stored["p_repay_valid"])
            for key, value in metrics.items():
                np.testing.assert_allclose(value, saved["metrics"][key], rtol=0, atol=1e-12)
            vp = np.sort(rng.choice(len(valid), min(512, len(valid)), replace=False))
            tp = np.sort(rng.choice(len(test_meta), min(512, len(test_meta)), replace=False))
            pv = model.inplace_predict(np.array(values[valid[vp]], dtype=np.float32, order="C"))
            pt = model.inplace_predict(np.array(tests[tp], dtype=np.float32, order="C"))
            np.testing.assert_allclose(pv, stored["p_repay_valid"][vp], rtol=0, atol=1e-10)
            np.testing.assert_allclose(pt, stored["p_repay_test"][tp], rtol=0, atol=1e-10)
            records.append({"fold": fold, "sampled_valid_rows": len(vp), "sampled_test_rows": len(tp),
                "all_oof_metrics_recomputed": True, "max_prediction_difference": float(max(
                    np.max(np.abs(pv-stored["p_repay_valid"][vp])), np.max(np.abs(pt-stored["p_repay_test"][tp]))))})
        del model
        gc.collect()
    result = {"status": "passed" if records else "no_completed_folds", "models_reloaded": len(records), "checks": records}
    base.candidates.json_write(locations(root)[0] / "model_reload_verification.json", result)
    print(f"Verified {len(records)} fixed-cost XGBoost models", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "worker", "collect", "verify"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--time-budget-minutes", type=float, default=55)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    if not np.isfinite(args.time_budget_minutes) or args.time_budget_minutes <= 0 or not 1 <= args.threads <= 10:
        parser.error("Invalid time/CPU budget")
    if args.stage == "train":
        train(args.root, args.time_budget_minutes, args.threads)
    elif args.stage == "worker":
        if args.fold is None:
            parser.error("Worker requires a fold")
        train_fold(args.root, args.fold, args.threads)
    elif args.stage == "verify":
        verify(args.root)
    else:
        collect(args.root)


if __name__ == "__main__":
    main()

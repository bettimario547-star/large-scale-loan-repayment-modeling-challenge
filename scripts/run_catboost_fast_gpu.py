"""Budgeted CatBoost GPU folds using binary, column-major numerical input.

Keep this study separate from the original TSV-based experiment. Quantization
uses only fitting rows; held-out rows reuse the fitting quantization borders.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import catboost as cb
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

try:
    from . import run_public_auc_optimization as original
except ImportError:
    import run_public_auc_optimization as original

ROOT = original.ROOT
NAME = "catboost_fast_gpu_s42"


def locations(root):
    root = Path(root)
    return root / "reports/catboost_fast_gpu", root / "artifacts/catboost_fast_gpu"


def initialize(root):
    parent = original.checked_config(root)
    if not parent["gpu_probe"].get("hardware_available", parent["gpu_probe"]["available"]):
        raise RuntimeError("CatBoost GPU preflight failed")
    experiment = {"parent_training_signature": parent["training_signature"],
                  "source_sha256": original.candidates.sha256(Path(__file__)),
                  "params": parent["experiment"]["catboost_gpu_params"],
                  "feature_plans": parent["experiment"]["plans"]["catboost_drift_gpu"],
                  "data_input": "float32 Fortran-order mmap -> training-only CPU quantization -> saved binary Pool",
                  "catboost_version": cb.__version__}
    signature = original.trees.signing_hash(experiment)
    report, _ = locations(root)
    path = report / "training_config.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        if config["training_signature"] != signature:
            raise RuntimeError("Fast GPU study configuration changed; preserve the existing study")
        return config
    config = {"training_signature": signature, "experiment": experiment}
    original.candidates.json_write(path, config)
    return config


def binary_pool(root, values, labels, indices, names, fold, stage, threads, signature, ref_borders=None):
    """Bound raw memory with a disk mmap, then reload only quantized features."""
    _, artifacts = locations(root)
    directory = artifacts / "pools" / f"fold_{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    binary, borders = directory / f"{stage}.qbin", directory / f"{stage}.borders.tsv"
    metadata = directory / f"{stage}.json"
    expected = {"training_signature": signature, "rows_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "labels_sha256": hashlib.sha256(labels[indices].tobytes()).hexdigest(), "features": names,
                "reference_border_sha256": None if ref_borders is None else original.candidates.sha256(ref_borders)}
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["inputs"] != expected or original.candidates.sha256(binary) != saved["binary_sha256"] or original.candidates.sha256(borders) != saved["borders_sha256"]:
            raise RuntimeError("Binary Pool cache signature/checksum mismatch")
        return cb.Pool(f"quantized://{binary}"), borders
    temporary = directory / f"{stage}.float32.npy"
    started = time.monotonic()
    mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                      shape=(len(indices), len(names)), fortran_order=True)
    pool = None
    try:
        for section in original.base.blocks(len(indices), 4096):
            chunk = values[indices[section]]
            if np.isinf(chunk).any():
                raise ValueError("Infinite numerical input")
            mapped[section] = chunk
            if section.stop % (4096 * 32) == 0 or section.stop == len(indices):
                print(f"Binary {stage}: {section.stop:,}/{len(indices):,} rows, {time.monotonic()-started:.0f}s", flush=True)
        mapped.flush()
        # CatBoost borrows column-major float32 arrays without duplicating the
        # full raw matrix. Saving/reloading below releases all raw holders.
        pool = cb.Pool(mapped, label=labels[indices], feature_names=names, thread_count=threads)
        ignored = None
        if ref_borders is not None:
            with_borders = {int(line.split("\t")[0]) for line in ref_borders.read_text().splitlines() if line}
            ignored = [i for i in range(len(names)) if i not in with_borders]
        pool.quantize(border_count=63, nan_mode="Min", task_type="CPU", used_ram_limit="3gb",
                      random_seed=42 + fold, ignored_features=ignored,
                      input_borders=None if ref_borders is None else str(ref_borders))
        pool.save_quantization_borders(str(borders))
        if ref_borders is not None and borders.read_bytes() != ref_borders.read_bytes():
            raise RuntimeError("Held-out Pool invented different quantization borders")
        temporary_binary = binary.with_suffix(".qbin.tmp")
        pool.save(str(temporary_binary))
        temporary_binary.replace(binary)
    finally:
        del pool
        gc.collect()
        mapped._mmap.close()
        if not temporary.resolve().is_relative_to(artifacts.resolve()):
            raise RuntimeError("Unexpected temporary mmap path")
        temporary.unlink(missing_ok=True)
    pool = cb.Pool(f"quantized://{binary}")
    if not pool.is_quantized() or pool.num_row() != len(indices) or pool.get_feature_names() != names:
        raise RuntimeError("Reloaded binary Pool schema differs")
    np.testing.assert_array_equal(np.asarray(pool.get_label(), dtype=float), labels[indices])
    original.candidates.json_write(metadata, {"inputs": expected,
        "binary_sha256": original.candidates.sha256(binary), "borders_sha256": original.candidates.sha256(borders),
        "elapsed_seconds": time.monotonic() - started})
    print(f"Ready {stage} Pool: {time.monotonic()-started:.1f}s", flush=True)
    return pool, borders


def job_paths(root, fold):
    _, artifacts = locations(root)
    folder = artifacts / "jobs"
    return folder / f"fold_{fold}.json", folder / f"fold_{fold}.npz", folder / f"fold_{fold}.cbm"


def checkpoint(root, fold, signature):
    metadata, predictions, model = job_paths(root, fold)
    if not metadata.exists():
        return None
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    if saved["training_signature"] != signature or original.candidates.sha256(predictions) != saved["prediction_sha256"] or original.candidates.sha256(model) != saved["model_sha256"]:
        raise RuntimeError("Fast GPU fold checksum/signature mismatch")
    return saved


def train_fold(root, fold, threads):
    root = Path(root)
    process = psutil.Process()
    allowed = process.cpu_affinity()
    if len(allowed) < threads:
        raise RuntimeError("Requested CPU budget exceeds this worker's available processors")
    # Pool.quantize has no thread_count argument. CPU affinity also caps its
    # native helper threads to the requested logical CPU budget on Windows.
    process.cpu_affinity(allowed[:threads])
    config = initialize(root)
    if checkpoint(root, fold, config["training_signature"]):
        return
    parent = original.checked_config(root)["experiment"]
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    folds = original.base.align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    labels, fold_ids = meta.label.to_numpy(dtype=np.int8), folds.fold.to_numpy()
    names = config["experiment"]["feature_plans"][str(fold)]["kept"]
    values = original.SubsetView(original.trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/train_features.npy"), parent["features"], names)
    test_values = original.SubsetView(original.trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/test_features.npy"), parent["features"], names)
    inner, stop, valid = original.base.split_outer_fit(labels, fold_ids, fold, .15)
    fitting = np.flatnonzero(fold_ids != fold)
    started = time.monotonic()
    params = dict(config["experiment"]["params"])
    params.update(thread_count=threads, random_seed=42 + fold)
    print(f"Start fast CatBoost GPU fold {fold}: {len(fitting):,} fitting rows", flush=True)
    with threadpool_limits(limits=threads):
        inner_pool, borders = binary_pool(root, values, labels, inner, names, fold, "inner", threads, config["training_signature"])
        stop_pool, _ = binary_pool(root, values, labels, stop, names, fold, "stop", threads, config["training_signature"], borders)
        selected = cb.CatBoostClassifier(**params)
        selected.fit(inner_pool, eval_set=stop_pool, use_best_model=True, verbose=100)
        rounds = int(selected.best_iteration_) + 1
        inner_auc = float(selected.best_score_["validation"]["AUC"])
        del selected, inner_pool, stop_pool
        gc.collect()
        full_pool, _ = binary_pool(root, values, labels, fitting, names, fold, "refit", threads, config["training_signature"])
        params.update(iterations=rounds)
        params.pop("od_type")
        params.pop("od_wait")
        model = cb.CatBoostClassifier(**params)
        model.fit(full_pool, verbose=100)
        if model.get_all_params().get("task_type") != "GPU":
            raise RuntimeError("CatBoost silently changed training backend")
        np.testing.assert_array_equal(np.asarray(model.classes_, dtype=float), [0, 1])
        del full_pool
        gc.collect()
        metadata, prediction_path, model_path = job_paths(root, fold)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        p_valid = original.trees.predict_rows("catboost", model, values, names, 4096, valid)
        p_test = original.trees.predict_rows("catboost", model, test_values, names, 4096)
        np.savez_compressed(prediction_path, outer_valid_index=valid, p_repay_valid=p_valid, p_repay_test=p_test)
        original.candidates.json_write(metadata, {"training_signature": config["training_signature"],
            "fold": fold, "features": names, "params": params, "positive_class": 1, "backend": "GPU",
            "rounds": rounds, "inner_stop_auc": inner_auc, "elapsed_seconds": time.monotonic()-started,
            "metrics": original.base.metrics(labels[valid], p_valid),
            "prediction_sha256": original.candidates.sha256(prediction_path), "model_sha256": original.candidates.sha256(model_path)})
    print(f"Completed fast GPU fold {fold}: AUC={original.base.metrics(labels[valid],p_valid)['repay_roc_auc']:.6f}", flush=True)


def collect(root, config):
    root = Path(root)
    report, _ = locations(root)
    saved = [checkpoint(root, fold, config["training_signature"]) for fold in range(5)]
    original.candidates.json_write(report / "training_progress.json", {"completed_folds": sum(row is not None for row in saved),
                                                                      "complete_candidate": all(row is not None for row in saved)})
    if not all(row is not None for row in saved):
        return False
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    test_meta = pd.read_parquet(root / "data/processed/test.parquet", columns=["id"])
    folds = original.base.align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    scores, test_scores = np.full(len(meta), np.nan), np.zeros(len(test_meta))
    coverage = np.zeros(len(meta), dtype=np.int8)
    for fold in range(5):
        with np.load(job_paths(root, fold)[1], allow_pickle=False) as predictions:
            valid = predictions["outer_valid_index"]
            np.testing.assert_array_equal(valid, np.flatnonzero(folds.fold.to_numpy() == fold))
            scores[valid] = predictions["p_repay_valid"]
            test_scores += predictions["p_repay_test"] / 5
            coverage[valid] += 1
    if not (coverage == 1).all() or not np.isfinite(scores).all():
        raise RuntimeError("Fast GPU OOF coverage differs")
    output = root / "artifacts/public_auc_optimization/completed" / NAME
    output.mkdir(parents=True, exist_ok=True)
    oof = folds.copy()
    oof["label"], oof["p_repay"] = meta.label.to_numpy(), scores
    oof.to_parquet(output / "oof_predictions.parquet", index=False)
    pd.DataFrame({"id": test_meta.id, "p_repay": test_scores}).to_parquet(output / "test_predictions.parquet", index=False)
    original.candidates.json_write(output / "model_summary.json", {"training_signature": config["training_signature"],
        "metrics": original.base.metrics(meta.label.to_numpy(), scores), "public_auc": None, "backend": "GPU"})
    original.candidates.build(root)
    return True


def train(root, minutes, threads):
    config = initialize(root)
    if psutil.virtual_memory().available < 5 * 1024**3:
        raise RuntimeError("Fast GPU quantization requires 5 GiB available RAM")
    deadline = time.monotonic() + minutes * 60
    report, _ = locations(root)
    logs = Path(root) / "logs/catboost_fast_gpu" / str(time.time_ns())
    logs.mkdir(parents=True)
    failures, stopped = [], []
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(threads)
    for fold in range(5):
        if checkpoint(root, fold, config["training_signature"]):
            continue
        if time.monotonic() >= deadline:
            break
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "worker", "--root", str(root),
                   "--fold", str(fold), "--threads", str(threads)]
        with (logs / f"fold_{fold}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            print(f"Launched fast GPU fold {fold}, PID {process.pid}", flush=True)
            try:
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.5)
                if process.poll() is None:
                    stopped.append(fold)
                elif process.returncode:
                    failures.append({"fold": fold, "exit_code": process.returncode})
            finally:
                original.stop_worker_tree(process)
        collect(root, config)
        if failures or stopped:
            break
    complete = collect(root, config)
    original.candidates.json_write(report / "last_training_session.json", {"minutes_budget": minutes, "threads": threads,
        "complete_candidate": complete, "failures": failures, "stopped_at_deadline": stopped, "log_directory": str(logs)})
    print(f"Fast GPU session finished; complete five-fold candidate: {complete}", flush=True)


def verify(root, sample_rows=512):
    config = initialize(root)
    parent = original.checked_config(root)["experiment"]
    meta = pd.read_parquet(Path(root) / "data/processed/train.parquet", columns=["id", "label"])
    test_meta = pd.read_parquet(Path(root) / "data/processed/test.parquet", columns=["id"])
    folds = original.base.align_validation_folds(meta.id, pd.read_parquet(Path(root) / "data/interim/validation_folds.parquet"))
    raw = original.trees.BatchedFeatureFile(Path(root) / "artifacts/linear_baselines/cache/train_features.npy")
    raw_test = original.trees.BatchedFeatureFile(Path(root) / "artifacts/linear_baselines/cache/test_features.npy")
    rng, records = np.random.default_rng(8128), []
    for fold in range(5):
        saved = checkpoint(root, fold, config["training_signature"])
        if saved is None:
            continue
        names = config["experiment"]["feature_plans"][str(fold)]["kept"]
        if saved["features"] != names or saved["positive_class"] != 1 or saved["backend"] != "GPU":
            raise RuntimeError("Fast GPU checkpoint schema/label/backend differs")
        values = original.SubsetView(raw, parent["features"], names)
        test_values = original.SubsetView(raw_test, parent["features"], names)
        model = cb.CatBoostClassifier()
        model.load_model(str(job_paths(root, fold)[2]))
        np.testing.assert_array_equal(np.asarray(model.classes_, dtype=float), [0, 1])
        if model.feature_names_ != names or model.tree_count_ != saved["rounds"]:
            raise RuntimeError("Reloaded fast GPU model schema/tree count differs")
        with np.load(job_paths(root, fold)[1], allow_pickle=False) as predictions:
            valid = predictions["outer_valid_index"]
            np.testing.assert_array_equal(valid, np.flatnonzero(folds.fold.to_numpy() == fold))
            p_valid, p_test = predictions["p_repay_valid"], predictions["p_repay_test"]
            if p_valid.shape != valid.shape or p_test.shape != (len(test_meta),):
                raise RuntimeError("Fast GPU prediction shape differs")
            for array in (p_valid, p_test):
                if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                    raise RuntimeError("Invalid saved GPU probabilities")
            metrics = original.base.metrics(meta.label.to_numpy()[valid], p_valid)
            for key, actual in metrics.items():
                np.testing.assert_allclose(actual, saved["metrics"][key], rtol=0, atol=1e-12)
            vpos = np.sort(rng.choice(len(valid), min(len(valid), sample_rows), replace=False))
            tpos = np.sort(rng.choice(len(test_meta), min(len(test_meta), sample_rows), replace=False))
            actual_valid = model.predict_proba(cb.Pool(values[valid[vpos]], feature_names=names, thread_count=1), thread_count=1)[:, 1]
            actual_test = model.predict_proba(cb.Pool(test_values[tpos], feature_names=names, thread_count=1), thread_count=1)[:, 1]
            np.testing.assert_allclose(actual_valid, p_valid[vpos], rtol=0, atol=1e-10)
            np.testing.assert_allclose(actual_test, p_test[tpos], rtol=0, atol=1e-10)
            records.append({"fold": fold, "sampled_valid_rows": len(vpos), "sampled_test_rows": len(tpos),
                            "all_oof_metrics_recomputed": True, "max_prediction_difference": float(max(
                                np.max(np.abs(actual_valid-p_valid[vpos])), np.max(np.abs(actual_test-p_test[tpos]))))})
        del model
        gc.collect()
        print(f"Verified fast GPU fold {fold}", flush=True)
    report, _ = locations(root)
    original.candidates.json_write(report / "model_reload_verification.json", {
        "status": "passed" if records else "no_completed_folds", "models_reloaded": len(records),
        "prediction_check_scope": "sampled OOF/test rows; full saved OOF metrics", "checks": records})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "worker", "collect", "verify"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--time-budget-minutes", type=float, default=40)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    if not np.isfinite(args.time_budget_minutes) or args.time_budget_minutes <= 0 or not 1 <= args.threads <= 10:
        parser.error("Invalid CPU/time budget")
    if args.stage == "train":
        train(args.root, args.time_budget_minutes, args.threads)
    elif args.stage == "collect":
        collect(args.root, initialize(args.root))
    elif args.stage == "verify":
        verify(args.root)
    else:
        if args.fold is None:
            parser.error("Worker requires a fold")
        train_fold(args.root, args.fold, args.threads)


if __name__ == "__main__":
    main()

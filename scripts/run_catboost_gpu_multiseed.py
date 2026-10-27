"""Resume depth-eight GPU seeds while reusing certified training-only Pools.

Keep the completed seed-42 study immutable. New seeds have separate signed
checkpoints; only complete five-fold seeds enter the probability ensemble.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

try:
    from . import run_catboost_fast_gpu as engine
except ImportError:
    import run_catboost_fast_gpu as engine

SOURCE = Path(__file__).resolve()
BASE_SOURCE = Path(engine.__file__).resolve()
BASE_LOCATIONS, BASE_INITIALIZE, BASE_POOL = engine.locations, engine.initialize, engine.binary_pool
SEEDS = (2026, 3407)


def legacy_config(root):
    previous = engine.locations, engine.__file__
    engine.locations, engine.__file__ = BASE_LOCATIONS, str(BASE_SOURCE)
    try:
        return BASE_INITIALIZE(root)
    finally:
        engine.locations, engine.__file__ = previous


def capacity_config(root):
    legacy = legacy_config(root)
    path = Path(root) / "reports/catboost_gpu_capacity/training_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    exp = config["experiment"]
    expected_params = dict(legacy["experiment"]["params"])
    expected_params.update(depth=8, iterations=2000, l2_leaf_reg=8, gpu_ram_part=.60)
    if (engine.original.trees.signing_hash(exp) != config["training_signature"]
        or exp["parent_training_signature"] != legacy["training_signature"]
        or exp["engine_sha256"] != engine.original.candidates.sha256(BASE_SOURCE)
        or exp["source_sha256"] != engine.original.candidates.sha256(SOURCE.parent / "run_catboost_gpu_capacity.py")
        or exp["params"] != expected_params
        or exp["feature_plans"] != legacy["experiment"]["feature_plans"]):
        raise RuntimeError("Certified capacity recipe or Pool inputs changed")
    return config, legacy


def locations(root, seed):
    root = Path(root)
    return (root / "reports/catboost_gpu_multiseed" / f"seed_{seed}",
            root / "artifacts/catboost_gpu_multiseed" / f"seed_{seed}")


def seed_config(root, seed, capacity, legacy):
    engine.original.checked_config(root)
    exp = {"capacity_training_signature": capacity["training_signature"],
           "pool_training_signature": legacy["training_signature"],
           "source_sha256": engine.original.candidates.sha256(SOURCE),
           "base_seed": seed, "seed_rule": "base_seed + outer_fold",
           "params": capacity["experiment"]["params"],
           "feature_plans": capacity["experiment"]["feature_plans"],
           "catboost_version": engine.cb.__version__}
    signature = engine.original.trees.signing_hash(exp)
    path = locations(root, seed)[0] / "training_config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["training_signature"] != signature or saved["experiment"] != exp:
            raise RuntimeError("Multi-seed configuration changed; retain the completed run")
        return saved
    config = {"training_signature": signature, "experiment": exp}
    engine.original.candidates.json_write(path, config)
    return config


def configure(root, seed):
    capacity, legacy = capacity_config(root)

    def current_locations(dataset_root):
        return locations(dataset_root, seed)

    def initialize(dataset_root):
        return seed_config(dataset_root, seed, capacity, legacy)

    def binary_pool(dataset_root, values, labels, indices, names, fold, stage, threads, signature, ref_borders=None):
        engine.locations = BASE_LOCATIONS
        try:
            return BASE_POOL(dataset_root, values, labels, indices, names, fold, stage, threads,
                             legacy["training_signature"], ref_borders)
        finally:
            engine.locations = current_locations

    engine.locations, engine.initialize, engine.binary_pool = current_locations, initialize, binary_pool
    engine.train_fold = train_fold
    engine.NAME, engine.__file__ = f"catboost_gpu_capacity_s{seed}", str(SOURCE)
    os.environ["LOAN_CATBOOST_BASE_SEED"] = str(seed)
    return initialize(root)


def train_fold(root, fold, threads):
    root = Path(root)
    process = psutil.Process()
    allowed = process.cpu_affinity()
    if len(allowed) < threads:
        raise RuntimeError("Requested CPU budget exceeds available processors")
    process.cpu_affinity(allowed[:threads])
    config = engine.initialize(root)
    if engine.checkpoint(root, fold, config["training_signature"]):
        return
    parent = engine.original.checked_config(root)["experiment"]
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    folds = engine.original.base.align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    labels, fold_ids = meta.label.to_numpy(dtype=np.int8), folds.fold.to_numpy()
    names = config["experiment"]["feature_plans"][str(fold)]["kept"]
    values = engine.original.SubsetView(engine.original.trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/train_features.npy"), parent["features"], names)
    test_values = engine.original.SubsetView(engine.original.trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/test_features.npy"), parent["features"], names)
    inner, stop, valid = engine.original.base.split_outer_fit(labels, fold_ids, fold, .15)
    fitting = np.flatnonzero(fold_ids != fold)
    seed = config["experiment"]["base_seed"]
    started = time.monotonic()
    params = dict(config["experiment"]["params"])
    params.update(thread_count=threads, random_seed=seed + fold)
    print(f"Start depth-eight GPU seed {seed} fold {fold}", flush=True)
    with threadpool_limits(limits=threads):
        inner_pool, borders = engine.binary_pool(root, values, labels, inner, names, fold, "inner", threads, config["training_signature"])
        stop_pool, _ = engine.binary_pool(root, values, labels, stop, names, fold, "stop", threads, config["training_signature"], borders)
        selected = engine.cb.CatBoostClassifier(**params)
        selected.fit(inner_pool, eval_set=stop_pool, use_best_model=True, verbose=100)
        rounds = int(selected.best_iteration_) + 1
        inner_auc = float(selected.best_score_["validation"]["AUC"])
        del selected, inner_pool, stop_pool
        gc.collect()
        full_pool, _ = engine.binary_pool(root, values, labels, fitting, names, fold, "refit", threads, config["training_signature"])
        params.update(iterations=rounds)
        params.pop("od_type")
        params.pop("od_wait")
        model = engine.cb.CatBoostClassifier(**params)
        model.fit(full_pool, verbose=100)
        if model.get_all_params().get("task_type") != "GPU" or model.get_param("random_seed") != seed + fold:
            raise RuntimeError("Unexpected training backend or seed")
        np.testing.assert_array_equal(np.asarray(model.classes_, dtype=float), [0, 1])
        del full_pool
        gc.collect()
        metadata, prediction_path, model_path = engine.job_paths(root, fold)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        p_valid = engine.original.trees.predict_rows("catboost", model, values, names, 4096, valid)
        p_test = engine.original.trees.predict_rows("catboost", model, test_values, names, 4096)
        np.savez_compressed(prediction_path, outer_valid_index=valid, p_repay_valid=p_valid, p_repay_test=p_test)
        engine.original.candidates.json_write(metadata, {"training_signature": config["training_signature"],
            "base_seed": seed, "fold": fold, "features": names, "params": params, "positive_class": 1, "backend": "GPU",
            "rounds": rounds, "inner_stop_auc": inner_auc, "elapsed_seconds": time.monotonic()-started,
            "metrics": engine.original.base.metrics(labels[valid], p_valid),
            "prediction_sha256": engine.original.candidates.sha256(prediction_path), "model_sha256": engine.original.candidates.sha256(model_path)})
    print(f"Completed GPU seed {seed} fold {fold}: AUC={engine.original.base.metrics(labels[valid], p_valid)['repay_roc_auc']:.6f}", flush=True)


def collect_ensemble(root):
    root = Path(root)
    capacity, legacy = capacity_config(root)
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    test_meta = pd.read_parquet(root / "data/processed/test.parquet", columns=["id"])
    folds = engine.original.base.align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    completed, oof_scores, test_scores, signatures = [], [], [], {}
    for seed in (42, *SEEDS):
        report, artifacts = ((root / "reports/catboost_gpu_capacity", root / "artifacts/catboost_gpu_capacity")
                             if seed == 42 else locations(root, seed))
        if not (report / "training_config.json").exists() or not all((artifacts / "jobs" / f"fold_{fold}.json").exists() for fold in range(5)):
            continue
        config = capacity if seed == 42 else seed_config(root, seed, capacity, legacy)
        scores, tests, coverage = np.full(len(meta), np.nan), np.zeros(len(test_meta)), np.zeros(len(meta), dtype=np.int8)
        for fold in range(5):
            directory = artifacts / "jobs"
            record = json.loads((directory / f"fold_{fold}.json").read_text(encoding="utf-8"))
            path, model_path = directory / f"fold_{fold}.npz", directory / f"fold_{fold}.cbm"
            if (record["training_signature"] != config["training_signature"] or record["positive_class"] != 1
                or record["backend"] != "GPU" or record["params"]["random_seed"] != seed + fold
                or engine.original.candidates.sha256(path) != record["prediction_sha256"]
                or engine.original.candidates.sha256(model_path) != record["model_sha256"]):
                raise RuntimeError("Multi-seed checkpoint metadata or hash differs")
            with np.load(path, allow_pickle=False) as saved:
                valid = saved["outer_valid_index"]
                np.testing.assert_array_equal(valid, np.flatnonzero(folds.fold.to_numpy() == fold))
                if saved["p_repay_test"].shape != tests.shape or saved["p_repay_valid"].shape != valid.shape:
                    raise RuntimeError("Seed prediction shape differs")
                scores[valid] = saved["p_repay_valid"]
                tests += saved["p_repay_test"] / 5
                coverage[valid] += 1
        if not (coverage == 1).all():
            raise RuntimeError("Seed OOF coverage differs")
        for values in (scores, tests):
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise RuntimeError("Invalid seed probabilities")
        completed.append(seed)
        oof_scores.append(scores)
        test_scores.append(tests)
        signatures[str(seed)] = config["training_signature"]
    engine.original.candidates.json_write(root / "reports/catboost_gpu_multiseed/ensemble_progress.json", {
        "complete_seeds": completed, "expected_seeds": [42, *SEEDS], "complete_ensemble": len(completed) >= 2})
    if len(completed) < 2:
        return completed
    scores, tests = np.mean(oof_scores, axis=0), np.mean(test_scores, axis=0)
    output = root / "artifacts/public_auc_optimization/completed/catboost_gpu_capacity_ensemble"
    output.mkdir(parents=True, exist_ok=True)
    frame = folds.copy()
    frame["label"], frame["p_repay"] = meta.label.to_numpy(), scores
    frame.to_parquet(output / "oof_predictions.parquet", index=False)
    pd.DataFrame({"id": test_meta.id, "p_repay": tests}).to_parquet(output / "test_predictions.parquet", index=False)
    engine.original.candidates.json_write(output / "model_summary.json", {
        "metrics": engine.original.base.metrics(meta.label.to_numpy(), scores), "backend": "GPU",
        "complete_seeds": completed, "seed_training_signatures": signatures, "public_auc": None,
        "scope": "equal average of complete five-fold seeds; development OOF"})
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "worker", "collect", "verify"), required=True)
    parser.add_argument("--root", type=Path, default=engine.ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--time-budget-minutes", type=float, default=95)
    args = parser.parse_args()
    if not np.isfinite(args.time_budget_minutes) or args.time_budget_minutes <= 0 or not 1 <= args.threads <= 10:
        parser.error("Invalid time/CPU budget")
    if args.stage == "worker":
        seed = args.seed or int(os.environ.get("LOAN_CATBOOST_BASE_SEED", "2026"))
        if seed not in SEEDS or args.fold is None:
            parser.error("Worker requires a valid seed and fold")
        configure(args.root, seed)
        train_fold(args.root, args.fold, args.threads)
        return
    started = time.monotonic()
    deadline = started + args.time_budget_minutes * 60
    for seed in args.seeds:
        config = configure(args.root, seed)
        if args.stage == "train":
            remaining = (deadline - time.monotonic()) / 60
            if remaining <= 0:
                break
            engine.train(args.root, remaining, args.threads)
        elif args.stage == "verify":
            engine.verify(args.root)
        else:
            engine.collect(args.root, config)
        if args.stage != "verify":
            collect_ensemble(args.root)
            engine.original.candidates.build(args.root)
    if args.stage == "train":
        engine.original.candidates.json_write(Path(args.root) / "reports/catboost_gpu_multiseed/training_session.json", {
            "minutes_budget": args.time_budget_minutes, "threads": args.threads, "requested_seeds": args.seeds,
            "elapsed_seconds": time.monotonic()-started, "complete_seeds": collect_ensemble(args.root)})


if __name__ == "__main__":
    main()

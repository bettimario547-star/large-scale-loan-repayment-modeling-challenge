"""Budgeted drift-aware LightGBM and CatBoost GPU experiments.

Prioritize complete five-fold candidates, reuse fixed historical tuning settings,
and keep public feedback separate from development diagnostics. Child processes
isolate native memory; the parent owns the wall-clock deadline and collection.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

try:
    from . import run_lightgbm_baseline as base
    from . import run_tree_model_comparison as trees
    from . import build_public_auc_candidates as candidates
except ImportError:
    import run_lightgbm_baseline as base
    import run_tree_model_comparison as trees
    import build_public_auc_candidates as candidates

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT
VARIANTS = ("lgbm_drift_tuned", "catboost_drift_gpu", "lgbm_drift_available", "lgbm_drift_bins255")
CORE_DRIFT = {"feature_492", "feature_903", "feature_586", "feature_1039"}
CPU_BUDGET = 10


def locations(root):
    root = Path(root)
    return root / "reports/public_auc_optimization", root / "artifacts/public_auc_optimization"


def fingerprint(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class SubsetView:
    ndim = 2

    def __init__(self, values, original_names, kept_names):
        if len(set(kept_names)) != len(kept_names) or not kept_names or not set(kept_names).issubset(original_names):
            raise ValueError("Invalid feature subset")
        self.values = values
        self.columns = np.array([original_names.index(name) for name in kept_names])
        self.shape = (len(values), len(kept_names))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        raw = self.values[item]
        return raw[self.columns] if raw.ndim == 1 else raw[:, self.columns]


def validate_cache(root, dataset, names):
    """Compare every float32 cached value with Parquet once after migration.

    File timestamps were truncated by the move. Shape/size alone cannot certify
    equivalence, so do a full streamed value comparison rather than bypassing
    historical source checks or rewriting historical caches.
    """
    root = Path(root)
    report, artifacts = locations(root)
    source = root / f"data/processed/{dataset}.parquet"
    cache = root / f"artifacts/linear_baselines/cache/{dataset}_features.npy"
    certificate = artifacts / "cache" / f"{dataset}_validation.json"
    expected = {"source": fingerprint(source), "cache": fingerprint(cache), "features": names}
    if certificate.exists():
        saved = json.loads(certificate.read_text(encoding="utf-8"))
        if saved["fingerprints"] == expected:
            return saved
    values = trees.BatchedFeatureFile(cache)
    parquet = pq.ParquetFile(source)
    if values.shape != (parquet.metadata.num_rows, len(names)):
        raise RuntimeError("Historical cache has a different shape")
    offset = 0
    for batch in parquet.iter_batches(batch_size=4096, columns=names, use_threads=False):
        restored = np.column_stack([batch.column(i).to_numpy(zero_copy_only=False).astype(np.float32) for i in range(len(names))])
        np.testing.assert_array_equal(values[slice(offset, offset + len(batch))], restored)
        offset += len(batch)
    if offset != len(values):
        raise RuntimeError("Incomplete cache verification")
    saved = {"fingerprints": expected, "rows_checked": offset, "all_float32_values_equal": True,
             "source_sha256": candidates.sha256(source), "cache_sha256": candidates.sha256(cache),
             "cache_path": cache.relative_to(root).as_posix()}
    candidates.json_write(certificate, saved)
    print(f"Verified every cached value: {dataset}, {offset:,} rows", flush=True)
    return saved


def null_rates(path, names):
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    columns = {name: parquet.schema_arrow.names.index(name) for name in names}
    result = {}
    for name, column in columns.items():
        count = 0
        for group in range(metadata.num_row_groups):
            stats = metadata.row_group(group).column(column).statistics
            if stats is None or stats.null_count is None:
                raise RuntimeError("Parquet null counts are required for availability screening")
            count += stats.null_count
        result[name] = count / metadata.num_rows
    return result


def make_plans(names, folds, test_null, fold_missing):
    plans = {}
    for variant in VARIANTS:
        plans[variant] = {}
        for fold in range(5):
            removed = {name: "predeclared_core_drift" for name in CORE_DRIFT if name in names}
            if variant == "lgbm_drift_available":
                for name in names:
                    if test_null.get(name) == 1:
                        removed[name] = "test_all_null"
                    elif fold_missing[fold].get(name, 0) >= .90:
                        removed[name] = "outer_fit_missing_at_least_90_percent"
            kept = [name for name in names if name not in removed]
            if not kept:
                raise ValueError("Feature filtering removed all predictors")
            plans[variant][str(fold)] = {"kept": kept, "removed": removed,
                                       "fit_rows": int(np.sum(folds != fold))}
    return plans


def probe_catboost_gpu():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(256, 8)).astype(np.float32)
    labels = (values[:, 0] > 0).astype(np.int8)
    model = cb.CatBoostClassifier(task_type="GPU", devices="0", iterations=3, depth=2,
                                 allow_writing_files=False, verbose=False)
    try:
        pool = cb.Pool(values, label=labels)
        pool.quantize(border_count=63)
        model.fit(pool)
        if model.get_all_params().get("task_type") != "GPU":
            raise RuntimeError("CatBoost did not actually use GPU")
        return {"available": True, "backend": "GPU", "quantized_pool_test_passed": True}
    except Exception as error:
        return {"available": False, "error": str(error)}
    finally:
        del model
        gc.collect()


def initialize(root):
    root = Path(root)
    report, artifacts = locations(root)
    report.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    train, test = root / "data/processed/train.parquet", root / "data/processed/test.parquet"
    names = base.model_feature_columns(pq.read_schema(train).names, pq.read_schema(test).names)
    meta = pd.read_parquet(train, columns=["id", "label"])
    folds = base.align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    certificates = {dataset: validate_cache(root, dataset, names) for dataset in ("train", "test")}
    train_null, test_null = null_rates(train, names), null_rates(test, names)
    potential = [name for name in names if train_null[name] >= .85]
    missing = pd.read_parquet(train, columns=potential).isna() if potential else pd.DataFrame(index=meta.index)
    fold_missing = {fold: missing.loc[folds.fold.to_numpy() != fold].mean().to_dict() for fold in range(5)}
    plans = make_plans(names, folds.fold.to_numpy(), test_null, fold_missing)
    inherited, source_hashes = {}, {}
    for fold in range(5):
        path = root / f"artifacts/hyperparameter_seed_optimization/models/seed_42_fold_{fold}.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["positive_class"] != 1 or saved["features"] != names:
            raise RuntimeError("Historical tuned feature schema or label direction differs")
        inherited[str(fold)] = {"params": saved["params"], "rounds": saved["record"]["refit_rounds"]}
        source_hashes[path.relative_to(root).as_posix()] = candidates.sha256(path)
    experiment = {"features": names, "plans": plans, "inherited": inherited,
                  "inheritance_scope": "fixed historical parameter/tree-count recipe; no new outer-fold tuning",
                  "input_sha256": {name: cert["source_sha256"] for name, cert in certificates.items()},
                  "fold_sha256": candidates.sha256(root / "data/interim/validation_folds.parquet"),
                  "historical_parameter_sha256": source_hashes,
                  "code_sha256": {name: candidates.sha256(CODE_ROOT / "scripts" / name) for name in (
                      "run_public_auc_optimization.py", "run_lightgbm_baseline.py", "run_tree_model_comparison.py", "id_feature_contract.py")},
                  "versions": {"lightgbm": lgb.__version__, "catboost": cb.__version__, "numpy": np.__version__},
                  "catboost_gpu_params": {"task_type": "GPU", "devices": "0", "boosting_type": "Plain",
                      "loss_function": "Logloss", "eval_metric": "AUC", "depth": 6, "learning_rate": .05,
                      "l2_leaf_reg": 5, "bootstrap_type": "Bernoulli", "subsample": .8,
                      "border_count": 63, "nan_mode": "Min", "gpu_ram_part": .70,
                      "allow_writing_files": False, "iterations": 1200, "od_type": "Iter", "od_wait": 60},
                  "scope": "competition development; public AUC only known after submission"}
    signature = trees.signing_hash(experiment)
    config_path = report / "training_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["training_signature"] != signature:
            raise RuntimeError("Optimization inputs/code changed; retain this run and select a separate workspace/run before retraining")
        return config
    config = {"experiment": experiment, "training_signature": signature,
              "gpu_probe": probe_catboost_gpu(), "status": "initialized"}
    candidates.json_write(config_path, config)
    return config


def checked_config(root):
    report, artifacts = locations(root)
    config = json.loads((report / "training_config.json").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    if trees.signing_hash(experiment) != config["training_signature"]:
        raise RuntimeError("Training configuration was modified")
    for name, digest in experiment["code_sha256"].items():
        if candidates.sha256(CODE_ROOT / "scripts" / name) != digest:
            raise RuntimeError(f"Training source changed: {name}")
    for dataset in ("train", "test"):
        certificate = json.loads((artifacts / "cache" / f"{dataset}_validation.json").read_text())
        expected = certificate["fingerprints"]
        if fingerprint(Path(root) / f"data/processed/{dataset}.parquet") != expected["source"] or fingerprint(Path(root) / certificate["cache_path"]) != expected["cache"]:
            raise RuntimeError("Validated input/cache changed")
    if candidates.sha256(Path(root) / "data/interim/validation_folds.parquet") != experiment["fold_sha256"]:
        raise RuntimeError("Validation folds changed")
    if experiment["versions"] != {"lightgbm": lgb.__version__, "catboost": cb.__version__, "numpy": np.__version__}:
        raise RuntimeError("Training runtime changed")
    return config


def job_paths(root, variant, seed, fold):
    _, artifacts = locations(root)
    directory = artifacts / "jobs" / variant / f"seed_{seed}"
    return directory / f"fold_{fold}.json", directory / f"fold_{fold}.npz", directory / f"fold_{fold}.model"


def completed_job(root, variant, seed, fold, signature):
    metadata, predictions, model = job_paths(root, variant, seed, fold)
    if not metadata.exists():
        return None
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    if saved["training_signature"] != signature or candidates.sha256(predictions) != saved["prediction_sha256"] or candidates.sha256(model) != saved["model_sha256"]:
        raise RuntimeError("Checkpoint signature/checksum mismatch")
    return saved


def train_job(root, variant, seed, fold, threads):
    root = Path(root)
    config = checked_config(root)
    if completed_job(root, variant, seed, fold, config["training_signature"]):
        return
    experiment = config["experiment"]
    names, kept = experiment["features"], experiment["plans"][variant][str(fold)]["kept"]
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    folds = base.align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    labels, fold_id = meta.label.to_numpy(dtype=np.int8), folds.fold.to_numpy()
    raw = trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/train_features.npy")
    raw_test = trees.BatchedFeatureFile(root / "artifacts/linear_baselines/cache/test_features.npy")
    values, test_values = SubsetView(raw, names, kept), SubsetView(raw_test, names, kept)
    fitting, valid = np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold)
    metadata, predictions, model_path = job_paths(root, variant, seed, fold)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"Start {variant} seed {seed} fold {fold}: {len(fitting):,} rows, {len(kept)} features, {threads} threads", flush=True)
    with threadpool_limits(limits=threads):
        if variant.startswith("lgbm"):
            params = dict(experiment["inherited"][str(fold)]["params"])
            params.update(num_threads=threads, seed=seed + fold, feature_fraction_seed=seed + fold,
                          bagging_seed=seed + fold, data_random_seed=42 + fold, feature_pre_filter=False)
            if variant == "lgbm_drift_bins255":
                params.update(max_bin=255, bin_construct_sample_cnt=50_000)
            rounds = experiment["inherited"][str(fold)]["rounds"]
            dataset = base.make_dataset(values, labels, fitting, kept, params, 4096)
            model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[base.log_refit(variant)])
            del dataset
            gc.collect()
            model.save_model(str(model_path))
            p_valid = base.predict_rows(model, values, 4096, valid)
            p_test = base.predict_rows(model, test_values, 4096)
            backend = "CPU"
        else:
            if not config["gpu_probe"]["available"]:
                raise RuntimeError("CatBoost GPU preflight failed")
            _, artifacts = locations(root)
            trees.ARTIFACTS = artifacts / "gpu_work" / f"seed_{seed}_fold_{fold}"
            trees.ARTIFACTS.mkdir(parents=True, exist_ok=True)
            inner, stop, _ = base.split_outer_fit(labels, fold_id, fold, .15)
            params = dict(experiment["catboost_gpu_params"])
            params.update(random_seed=seed + fold, thread_count=threads)
            inner_pool, borders = trees.cat_data(values, labels, inner, kept, fold, "inner", threads, 1024)
            stop_pool, _ = trees.cat_data(values, labels, stop, kept, fold, "stop", threads, 1024, borders)
            selected = cb.CatBoostClassifier(**params)
            selected.fit(inner_pool, eval_set=stop_pool, use_best_model=True, verbose=100)
            rounds = int(selected.best_iteration_) + 1
            del selected, inner_pool, stop_pool
            gc.collect()
            full_pool, _ = trees.cat_data(values, labels, fitting, kept, fold, "refit", threads, 1024)
            params.update(iterations=rounds)
            params.pop("od_type")
            params.pop("od_wait")
            model = cb.CatBoostClassifier(**params)
            model.fit(full_pool, verbose=100)
            if model.get_all_params().get("task_type") != "GPU":
                raise RuntimeError("Requested GPU training silently changed backend")
            del full_pool
            gc.collect()
            model.save_model(str(model_path))
            p_valid = trees.predict_rows("catboost", model, values, kept, 4096, valid)
            p_test = trees.predict_rows("catboost", model, test_values, kept, 4096)
            backend = "GPU"
        # Checkpoint metadata is the final completion marker.
        np.savez_compressed(predictions, outer_valid_index=valid, p_repay_valid=p_valid, p_repay_test=p_test)
        saved = {"training_signature": config["training_signature"], "variant": variant, "seed": seed,
                 "fold": fold, "params": params, "features": kept, "rounds": rounds, "backend": backend,
                 "positive_class": 1, "worker_pid": os.getpid(), "threads": threads,
                 "elapsed_seconds": time.monotonic() - started,
                 "metrics": base.metrics(labels[valid], p_valid),
                 "prediction_sha256": candidates.sha256(predictions), "model_sha256": candidates.sha256(model_path)}
        candidates.json_write(metadata, saved)
    print(f"Completed {variant} seed {seed} fold {fold}: OOF AUC={saved['metrics']['repay_roc_auc']:.6f}", flush=True)


def collect(root, config):
    root = Path(root)
    report, artifacts = locations(root)
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    test_meta = pd.read_parquet(root / "data/processed/test.parquet", columns=["id"])
    folds = base.align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    completed, records = {}, []
    for variant in VARIANTS:
        seed_scores, seed_tests = [], []
        for seed in (42, 2026, 3407):
            saved = [completed_job(root, variant, seed, fold, config["training_signature"]) for fold in range(5)]
            records.extend(row for row in saved if row is not None)
            if not all(row is not None for row in saved):
                continue
            score, test_score, coverage = np.full(len(meta), np.nan), np.zeros(len(test_meta)), np.zeros(len(meta), dtype=np.int8)
            for fold in range(5):
                with np.load(job_paths(root, variant, seed, fold)[1], allow_pickle=False) as prediction:
                    valid = prediction["outer_valid_index"]
                    np.testing.assert_array_equal(valid, np.flatnonzero(folds.fold.to_numpy() == fold))
                    score[valid] = prediction["p_repay_valid"]
                    test_score += prediction["p_repay_test"] / 5
                    coverage[valid] += 1
            if not (coverage == 1).all() or not np.isfinite(score).all():
                raise RuntimeError("Invalid complete-model OOF coverage")
            seed_scores.append(score)
            seed_tests.append(test_score)
            completed[f"{variant}_s{seed}"] = (score, test_score, 1)
        if len(seed_scores) >= 2:
            completed[f"{variant}_ensemble"] = (np.mean(seed_scores, axis=0), np.mean(seed_tests, axis=0), len(seed_scores))
    for name, (score, test_score, seeds) in completed.items():
        output = artifacts / "completed" / name
        output.mkdir(parents=True, exist_ok=True)
        oof = folds.copy()
        oof["label"], oof["p_repay"] = meta.label.to_numpy(), score
        oof.to_parquet(output / "oof_predictions.parquet", index=False)
        pd.DataFrame({"id": test_meta.id, "p_repay": test_score}).to_parquet(output / "test_predictions.parquet", index=False)
        candidates.json_write(output / "model_summary.json", {"seeds": seeds, "metrics": base.metrics(meta.label.to_numpy(), score),
                                                               "public_auc": None, "training_signature": config["training_signature"]})
    candidates.json_write(report / "training_progress.json", {"completed_folds": len(records), "completed_candidates": list(completed),
                                                              "completed_jobs": [{key: row[key] for key in ("variant", "seed", "fold", "backend", "elapsed_seconds")} for row in records]})
    return list(completed)


def task_order(gpu_available):
    order = [("lgbm_drift_tuned", 42, fold) for fold in range(5)]
    if gpu_available:
        order += [("catboost_drift_gpu", 42, fold) for fold in range(5)]
    for seed in (2026, 3407):
        order += [("lgbm_drift_tuned", seed, fold) for fold in range(5)]
    for variant in ("lgbm_drift_available", "lgbm_drift_bins255"):
        order += [(variant, 42, fold) for fold in range(5)]
    return order


def stop_worker_tree(process):
    """Stop a worker and its venv-launched Python descendants on Windows."""
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        targets = list(reversed(parent.children(recursive=True))) + [parent]
    except psutil.NoSuchProcess:
        process.wait()
        return
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)
    process.wait(timeout=5)


def train(root, minutes, parallel):
    root = Path(root)
    config = checked_config(root)
    tasks = [task for task in task_order(config["gpu_probe"]["available"]) if not completed_job(root, *task, config["training_signature"])]
    max_workers = 2 if parallel else 1
    threads = CPU_BUDGET // max_workers
    if psutil.cpu_count() < CPU_BUDGET:
        raise RuntimeError("At least 10 logical CPUs are required by this run's CPU budget")
    if parallel and psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError("Parallel preprocessing/training requires at least 6 GiB available RAM")
    log_directory = root / "logs/public_auc_optimization" / str(time.time_ns())
    log_directory.mkdir(parents=True)
    deadline = time.monotonic() + minutes * 60
    active, failures = {}, []
    exported = set(collect(root, config))
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(threads)
    print(f"Training budget: {minutes} minutes; {max_workers} processes x {threads} threads; GPU concurrency capped at one", flush=True)
    try:
        while (tasks or active) and time.monotonic() < deadline:
            while tasks and len(active) < max_workers:
                gpu_active = any(task[0] == "catboost_drift_gpu" for _, _, task in active.values())
                index = next((i for i, task in enumerate(tasks) if task[0] != "catboost_drift_gpu" or not gpu_active), None)
                if index is None:
                    break
                task = tasks.pop(index)
                variant, seed, fold = task
                log = (log_directory / f"{variant}_s{seed}_fold{fold}.log").open("w", encoding="utf-8")
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "worker", "--root", str(root),
                           "--variant", variant, "--seed", str(seed), "--fold", str(fold), "--threads", str(threads)]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                active[process.pid] = (process, log, task)
                print(f"Launched {variant} seed {seed} fold {fold}, PID {process.pid}", flush=True)
            for pid, (process, log, task) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[pid]
                if status:
                    failures.append({"task": task, "exit_code": status})
                    print(f"Failed {task}; log: {log_directory}", flush=True)
                    if task[0] == "catboost_drift_gpu":
                        tasks = [job for job in tasks if job[0] != "catboost_drift_gpu"]
                else:
                    print(f"Saved {task}", flush=True)
                    complete = set(collect(root, config))
                    if complete != exported:
                        candidates.build(root)
                        exported = complete
            time.sleep(.5)
    finally:
        for process, log, task in active.values():
            stop_worker_tree(process)
            log.close()
        complete = collect(root, config)
        candidates.build(root)
        report, _ = locations(root)
        candidates.json_write(report / "last_training_session.json", {"minutes_budget": minutes,
            "parallel": parallel, "threads_per_worker": threads, "log_directory": str(log_directory),
            "completed_candidates": complete, "failures": failures, "budget_exhausted": bool(tasks or active)})
    print(f"Training session finished; complete five-fold candidates: {complete}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("init", "train", "worker", "collect"), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--time-budget-minutes", type=float, default=90)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=(42, 2026, 3407), default=42)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    if args.time_budget_minutes <= 0 or not 1 <= args.threads <= CPU_BUDGET:
        parser.error("Invalid time or thread budget")
    if args.stage == "init":
        initialize(args.root)
    elif args.stage == "train":
        train(args.root, args.time_budget_minutes, args.parallel)
    elif args.stage == "worker":
        if args.variant is None or args.fold is None:
            parser.error("Worker requires --variant and --fold")
        train_job(args.root, args.variant, args.seed, args.fold, args.threads)
    else:
        collect(args.root, checked_config(args.root))
        candidates.build(args.root)


if __name__ == "__main__":
    main()

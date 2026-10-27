"""Full-data XGBoost/CatBoost controls on the existing development folds.

Each engine has one predeclared configuration and one seed. Only inner AUC
selects rounds. Refit on all four fitting folds, then predict outer/test rows.
XGBoost reads bounded batches; CatBoost quantizes temporary numeric TSV files
without loading the complete raw matrix. Existing LightGBM artifacts are read-only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import gc
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import catboost as cb
from catboost.utils import quantize
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

try:
    from . import run_lightgbm_baseline as base
except ImportError:
    import run_lightgbm_baseline as base

ROOT = base.ROOT
REPORT = ROOT / "reports/tree_model_comparison"
ARTIFACTS = ROOT / "artifacts/tree_model_comparison"
TUNED = ROOT / "artifacts/hyperparameter_seed_optimization"
TUNED_REPORT = ROOT / "reports/hyperparameter_seed_optimization"
ENGINES = ("xgboost", "catboost")
CODE_FILES = ("run_tree_model_comparison.py", "run_lightgbm_baseline.py",
              "run_linear_baselines.py", "id_feature_contract.py")
METRICS = ("repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score")
PARALLEL_CPU_BUDGET = 10
PARALLEL_WORKERS = 2
_WORKER_CONTEXT = None
_WORKER_CONFIG = None


def configure_run(run_name: str):
    """Keep separate experiments in separate report/artifact directories."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_name) or run_name in {
        "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }:
        raise ValueError("Run name must be a safe lowercase directory name (1-64 characters)")
    global REPORT, ARTIFACTS
    REPORT = ROOT / "reports" / run_name
    ARTIFACTS = ROOT / "artifacts" / run_name


def submission_path(engine):
    suffix = "" if REPORT.name == "tree_model_comparison" else f"_{REPORT.name}"
    return ROOT / "submissions" / f"{engine}_baseline{suffix}.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signing_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def params_for(engine: str, fold: int, threads: int) -> dict:
    if engine not in ENGINES or fold not in range(5) or threads < 1:
        raise ValueError("Invalid engine, fold or thread count")
    if engine == "xgboost":
        return {"objective": "binary:logistic", "eval_metric": "auc", "booster": "gbtree",
                "tree_method": "hist", "device": "cpu", "grow_policy": "depthwise",
                "eta": 0.05, "max_depth": 6, "min_child_weight": 20.0,
                "lambda": 5.0, "alpha": 0.0, "subsample": 0.8,
                "colsample_bytree": 0.8, "max_bin": 63, "nthread": threads,
                "seed": 42 + fold, "verbosity": 1}
    return {"loss_function": "Logloss", "eval_metric": "AUC", "task_type": "CPU",
            "boosting_type": "Plain", "grow_policy": "SymmetricTree", "depth": 6,
            "learning_rate": 0.05, "l2_leaf_reg": 5.0, "bootstrap_type": "Bernoulli",
            "subsample": 0.8, "rsm": 0.8, "random_strength": 1.0,
            "border_count": 63, "nan_mode": "Min", "boost_from_average": True,
            "leaf_estimation_iterations": 10, "thread_count": threads,
            "random_seed": 42 + fold, "allow_writing_files": False,
            "use_best_model": False}


class AUCStop:
    """A round is accepted only when AUC exceeds the retained best by min_delta."""

    def __init__(self, patience: int, min_delta: float):
        if patience < 1 or min_delta < 0:
            raise ValueError("Invalid stopping rule")
        self.patience, self.min_delta = patience, min_delta
        self.best_auc, self.best_round = -np.inf, 0
        self.history: list[float] = []

    def update(self, score: float) -> bool:
        if not np.isfinite(score):
            raise ValueError("Non-finite AUC")
        self.history.append(float(score))
        if score > self.best_auc + self.min_delta:
            self.best_auc, self.best_round = float(score), len(self.history)
        return len(self.history) - self.best_round < self.patience


class CatStop:
    def __init__(self, patience: int, min_delta: float):
        self.state = AUCStop(patience, min_delta)
        self.started = time.perf_counter()

    def after_iteration(self, info):
        score = info.metrics["validation"]["AUC"][-1]
        keep_training = self.state.update(score)
        if info.iteration % 50 == 0:
            print(f"CatBoost inner: trees={info.iteration}, AUC={score:.6f}, "
                  f"elapsed={time.perf_counter() - self.started:.0f}s", flush=True)
        return keep_training


class BatchIter(xgb.DataIter):
    def __init__(self, values, labels, indices, names, batch_rows):
        super().__init__(release_data=True)
        self.values, self.labels = values, labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.names, self.batch_rows, self.offset = names, batch_rows, 0

    def reset(self):
        self.offset = 0

    def next(self, input_data):
        if self.offset >= len(self.indices):
            return False
        rows = self.indices[self.offset:self.offset + self.batch_rows]
        data = np.array(self.values[rows], dtype=np.float32, order="C", copy=True)
        if np.isinf(data).any():
            raise ValueError("Infinite features")
        input_data(data=data, label=self.labels[rows], feature_names=self.names)
        self.offset += len(rows)
        return True


def xgb_data(values, labels, indices, names, threads, batch_rows, ref=None):
    iterator = BatchIter(values, labels, indices, names, batch_rows)
    return xgb.QuantileDMatrix(iterator, missing=np.nan, max_bin=63, nthread=threads, ref=ref)


class BatchedFeatureFile:
    """Copy each selected batch, then release its read-only source mapping.

    Keeping the full raw mapping resident alongside XGBoost's quantized rows
    can exhaust a 16 GB desktop. Returned batches own their memory, so source
    mappings can close immediately; OS file-cache pages become reclaimable.
    """

    def __init__(self, path):
        self.path = Path(path)
        mapped = np.load(self.path, mmap_mode="r")
        try:
            if mapped.ndim != 2 or mapped.dtype != np.float32:
                raise ValueError("Expected a raw float32 feature cache")
            self.shape, self.dtype, self.ndim = mapped.shape, mapped.dtype, mapped.ndim
        finally:
            mapped._mmap.close()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        mapped = np.load(self.path, mmap_mode="r")
        try:
            return np.array(mapped[rows], dtype=np.float32, order="C", copy=True)
        finally:
            mapped._mmap.close()


def export_numeric_tsv(path: Path, values, labels, indices, names, batch_rows=1024):
    """Nine significant digits round-trip float32; zero and NaN stay distinct."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for section in base.blocks(len(indices), batch_rows):
            rows = indices[section]
            frame = pd.DataFrame(np.array(values[rows], dtype=np.float32, copy=True), columns=names)
            if np.isinf(frame.to_numpy()).any():
                raise ValueError("Infinite features")
            frame.insert(0, "label", labels[rows])
            frame.to_csv(handle, sep="\t", header=False, index=False, na_rep="nan", float_format="%.9g")
            if section.stop % (batch_rows * 32) == 0 or section.stop == len(indices):
                print(f"Export {path.name}: {section.stop:,}/{len(indices):,} rows", flush=True)
    return path


def cat_data(values, labels, indices, names, fold, stage, threads, batch_rows, ref_borders=None):
    scratch = ARTIFACTS / "scratch" / f"fold_{fold}"
    scratch.mkdir(parents=True, exist_ok=True)
    tsv, description = scratch / f"{stage}.tsv", scratch / "columns.cd"
    description.write_text("0\tLabel\n" + "".join(f"{i + 1}\tNum\t{name}\n" for i, name in enumerate(names)), encoding="utf-8")
    export_numeric_tsv(tsv, values, labels, indices, names, min(batch_rows, 1024))
    try:
        print(f"CatBoost quantize {stage}: {len(indices):,} rows", flush=True)
        # Features without training borders are constant/ignored in fitting.
        # Do not let validation values create new borders for those features.
        ignored = None
        if ref_borders is not None:
            with_borders = {int(line.split("\t")[0]) for line in ref_borders.read_text(encoding="utf-8").splitlines() if line}
            ignored = [i for i in range(len(names)) if i not in with_borders]
        pool = quantize(str(tsv), column_description=str(description), thread_count=threads,
                        border_count=63, nan_mode="Min", task_type="CPU", used_ram_limit="3gb",
                        random_seed=42 + fold, ignored_features=ignored,
                        input_borders=None if ref_borders is None else str(ref_borders))
    finally:
        # Only remove the exact temporary file created above, inside this study.
        if not tsv.resolve().is_relative_to((ARTIFACTS / "scratch").resolve()):
            raise RuntimeError("Unexpected scratch path")
        tsv.unlink(missing_ok=True)
    if pool.get_feature_names() != names or pool.num_row() != len(indices) or not pool.is_quantized():
        raise RuntimeError("Invalid quantized CatBoost Pool")
    borders = ARTIFACTS / "borders" / f"catboost_{stage}_fold_{fold}.tsv"
    borders.parent.mkdir(parents=True, exist_ok=True)
    pool.save_quantization_borders(str(borders))
    if ref_borders is not None and borders.read_bytes() != ref_borders.read_bytes():
        raise RuntimeError("CatBoost early-stop quantization differs from fitting borders")
    return pool, borders


def predict_rows(engine, model, values, names, batch_rows, indices=None):
    rows = np.arange(len(values)) if indices is None else np.asarray(indices)
    output = np.empty(len(rows), dtype=np.float64)
    for section in base.blocks(len(rows), batch_rows):
        chunk = np.array(values[rows[section]], dtype=np.float32, order="C", copy=True)
        if engine == "xgboost":
            output[section] = model.inplace_predict(chunk, validate_features=True, missing=np.nan)
        else:
            pool = cb.Pool(chunk, feature_names=names, thread_count=1)
            output[section] = model.predict_proba(pool, thread_count=model.get_param("thread_count") or 1)[:, 1]
    if not np.isfinite(output).all() or np.any((output < 0) | (output > 1)):
        raise RuntimeError("Invalid repayment probabilities")
    return output


def load_model(engine, path):
    if engine == "xgboost":
        model = xgb.Booster()
        model.load_model(path)
        return model
    model = cb.CatBoostClassifier()
    model.load_model(str(path))
    return model


def model_rounds(engine, model):
    return model.num_boosted_rounds() if engine == "xgboost" else model.tree_count_


def model_paths(engine, fold, stage):
    tag = f"{engine}_fold_{fold}"
    extension = "ubj" if engine == "xgboost" else "cbm"
    folder = "inner_models" if stage == "inner" else "models"
    return ARTIFACTS / folder / f"{tag}.{extension}", ARTIFACTS / folder / f"{tag}.json"


def checkpoint(metadata, prediction, signature):
    if not metadata.exists():
        return None
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    if saved["training_signature"] != signature:
        raise RuntimeError("Checkpoint belongs to a different experiment")
    if sha256(Path(saved["model_path"])) != saved["model_sha256"] or sha256(prediction) != saved["prediction_sha256"]:
        raise RuntimeError("Model or prediction checksum mismatch")
    if any(sha256(Path(path)) != checksum for path, checksum in saved.get("border_checksums", {}).items()):
        raise RuntimeError("Quantization border checksum mismatch")
    return saved["record"]


def save_checkpoint(engine, fold, stage, model, params, names, record, predictions, signature, borders=()):
    path, metadata = model_paths(engine, fold, stage)
    model.save_model(str(path))
    base.json_write(metadata, {"training_signature": signature, "engine": engine, "positive_class": 1,
                              "features": names, "params": params, "record": record,
                              "model_path": str(path), "model_sha256": sha256(path),
                              "prediction_sha256": sha256(predictions),
                              "border_checksums": {str(border): sha256(border) for border in borders}})


def run_fold(context, engine, fold, config):
    _, _, _, labels, fold_id, names, values, test_values, _ = context
    experiment, signature = config["experiment"], config["training_signature"]
    execution = config.get("execution", {})
    threads = execution.get("threads_per_worker", experiment["threads"])
    batch = experiment["batch_rows"]
    inner, stop, valid = base.split_outer_fit(labels, fold_id, fold, 0.15)
    all_fit = np.flatnonzero(fold_id != fold)
    split = ARTIFACTS / "inner_splits" / f"fold_{fold}.npz"
    if split.exists():
        with np.load(split) as saved:
            for key, rows in (("inner_fit_index", inner), ("early_stop_index", stop), ("outer_valid_index", valid)):
                np.testing.assert_array_equal(saved[key], rows)
    else:
        np.savez_compressed(split, inner_fit_index=inner, early_stop_index=stop, outer_valid_index=valid)
    params = params_for(engine, fold, threads)
    _, metadata = model_paths(engine, fold, "inner")
    inner_predictions = ARTIFACTS / "inner_predictions" / f"{engine}_fold_{fold}.npz"
    selection = checkpoint(metadata, inner_predictions, signature)
    if selection is None:
        started = time.perf_counter()
        print(f"Start {engine} fold {fold} inner: fit={len(inner):,}, stop={len(stop):,}", flush=True)
        borders = []
        if engine == "xgboost":
            train_data = xgb_data(values, labels, inner, names, threads, batch)
            stop_data = xgb_data(values, labels, stop, names, threads, batch, ref=train_data)
            history = {}
            callback = xgb.callback.EarlyStopping(rounds=experiment["patience"], min_delta=experiment["min_delta"],
                        metric_name="auc", data_name="inner_early_stop", maximize=True, save_best=True)
            model = xgb.train(params, train_data, num_boost_round=experiment["max_rounds"],
                              evals=[(stop_data, "inner_early_stop")], evals_result=history,
                              callbacks=[callback], verbose_eval=50)
            curve = list(history["inner_early_stop"]["auc"])
            rounds = int(model.best_iteration) + 1
        else:
            train_data, train_borders = cat_data(values, labels, inner, names, fold, "inner", threads, batch)
            stop_data, stop_borders = cat_data(values, labels, stop, names, fold, "stop", threads, batch, ref_borders=train_borders)
            borders = [train_borders, stop_borders]
            callback = CatStop(experiment["patience"], experiment["min_delta"])
            model = cb.CatBoostClassifier(**params, iterations=experiment["max_rounds"])
            model.fit(train_data, eval_set=stop_data, callbacks=[callback], verbose=False)
            curve, rounds = callback.state.history, callback.state.best_round
            model.shrink(rounds)
        del train_data, stop_data
        gc.collect()
        p_stop = predict_rows(engine, model, values, names, batch, stop)
        state = AUCStop(experiment["patience"], experiment["min_delta"])
        for score in curve:
            keep = state.update(score)
            if not keep and len(state.history) != len(curve):
                raise RuntimeError("Training continued past the stopping rule")
        if rounds != state.best_round or model_rounds(engine, model) != rounds:
            raise RuntimeError("Wrong selected tree count")
        score = float(roc_auc_score(labels[stop], p_stop))
        np.testing.assert_allclose(score, state.best_auc, atol=2e-7, rtol=0)
        selection = {"engine": engine, "fold": fold, "selected_rounds": rounds,
                     "inner_auc": score, "evaluated_rounds": len(curve),
                     "hit_round_cap": len(curve) >= experiment["max_rounds"],
                     "inner_fit_rows": len(inner), "early_stop_rows": len(stop),
                     "elapsed_seconds": time.perf_counter() - started,
                     "training_threads": threads, "execution_mode": execution.get("mode", "single"),
                     "worker_pid": os.getpid()}
        np.savez_compressed(inner_predictions, early_stop_index=stop, p_repay=p_stop)
        base.json_write(REPORT / "inner_curves" / f"{engine}_fold_{fold}.json", {"auc": curve})
        save_checkpoint(engine, fold, "inner", model, params, names, selection, inner_predictions, signature, borders)
        print(f"Selected {engine} fold {fold}: trees={rounds}, inner AUC={score:.6f}", flush=True)
        del model, p_stop
        gc.collect()
    _, metadata = model_paths(engine, fold, "refit")
    predictions = ARTIFACTS / "fold_predictions" / f"{engine}_fold_{fold}.npz"
    record = checkpoint(metadata, predictions, signature)
    if record is not None:
        if record["refit_rounds"] != selection["selected_rounds"]:
            raise RuntimeError("Refit checkpoint differs from selected rounds")
        print(f"Reuse {engine} fold {fold}: AUC={record['repay_roc_auc']:.6f}", flush=True)
        return record
    started = time.perf_counter()
    print(f"Refit {engine} fold {fold}: rows={len(all_fit):,}, trees={selection['selected_rounds']}", flush=True)
    borders = []
    if engine == "xgboost":
        full_data = xgb_data(values, labels, all_fit, names, threads, batch)
        model = xgb.train(params, full_data, num_boost_round=selection["selected_rounds"],
                          callbacks=[XGBProgress(f"{engine} fold {fold} refit")], verbose_eval=False)
    else:
        full_data, full_borders = cat_data(values, labels, all_fit, names, fold, "refit", threads, batch)
        borders = [full_borders]
        model = cb.CatBoostClassifier(**params, iterations=selection["selected_rounds"])
        model.fit(full_data, verbose=50)
    del full_data
    gc.collect()
    p_valid = predict_rows(engine, model, values, names, batch, valid)
    p_test = predict_rows(engine, model, test_values, names, batch)
    np.savez_compressed(predictions, outer_valid_index=valid, p_repay_valid=p_valid, p_repay_test=p_test)
    record = {"engine": engine, "fold": fold, "refit_rounds": model_rounds(engine, model),
              "fit_rows": len(all_fit), **base.metrics(labels[valid], p_valid),
              "elapsed_seconds": time.perf_counter() - started,
              "training_threads": threads, "execution_mode": execution.get("mode", "single"),
              "worker_pid": os.getpid()}
    save_checkpoint(engine, fold, "refit", model, params, names, record, predictions, signature, borders)
    print(f"Finished {engine} fold {fold}: AUC={record['repay_roc_auc']:.6f}, "
          f"default AP={record['default_pr_auc_ap']:.6f}, logloss={record['log_loss']:.6f}", flush=True)
    del model, p_valid, p_test
    gc.collect()
    return record


class XGBProgress(xgb.callback.TrainingCallback):
    def __init__(self, label):
        self.label, self.started = label, time.perf_counter()

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 50 == 0:
            print(f"{self.label}: trees={epoch + 1}, elapsed={time.perf_counter() - self.started:.0f}s", flush=True)
        return False


def readonly_feature_cache(path, names):
    """Workers may only open a cache prepared by the parent; never rebuild it."""
    import pyarrow.parquet as pq
    cache = base.LINEAR_ARTIFACTS / "cache"
    array_path = cache / f"{path.stem}_features.npy"
    manifest_path = cache / f"{path.stem}_features.json"
    expected = {"source": base.source_signature(path), "features": names,
                "shape": [pq.ParquetFile(path).metadata.num_rows, len(names)], "dtype": "float32"}
    if not array_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Parent must prepare feature caches before starting workers")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("Feature cache changed after parent preparation")
    values = BatchedFeatureFile(array_path)
    if list(values.shape) != expected["shape"]:
        raise RuntimeError("Invalid feature cache shape")
    return values


def load_context(batch_rows, readonly_cache=False):
    import pyarrow.parquet as pq
    names = base.model_feature_columns(pq.read_schema(base.TRAIN).names, pq.read_schema(base.TEST).names)
    meta = pd.read_parquet(base.TRAIN, columns=["id", "label"], use_threads=False)
    test_meta = pd.read_parquet(base.TEST, columns=["id"], use_threads=False)
    folds = base.align_validation_folds(meta.id, pd.read_parquet(base.FOLDS, use_threads=False))
    if set(folds.fold.unique()) != set(range(5)) or not meta.label.isin([0, 1]).all():
        raise RuntimeError("Invalid labels/folds")
    original = pd.read_parquet(base.ARTIFACTS / "oof_predictions.parquet", use_threads=False)
    tuned = pd.read_parquet(TUNED / "oof_predictions.parquet", use_threads=False)
    for frame in (original, tuned):
        pd.testing.assert_frame_equal(frame[["id", "label"]], meta, check_dtype=False)
        pd.testing.assert_frame_equal(frame[list(folds.columns)], folds, check_dtype=False)
    references = {"lightgbm_baseline": original.p_repay.to_numpy(),
                  "lightgbm_tuned_seed_42": tuned.p_repay_seed_42.to_numpy(),
                  "lightgbm_tuned_ensemble": tuned.p_repay_ensemble.to_numpy()}
    if readonly_cache:
        values = readonly_feature_cache(base.TRAIN, names)
        test_values = readonly_feature_cache(base.TEST, names)
    else:
        raw_train = base.raw_feature_cache(base.TRAIN, names, batch_rows)
        raw_test = base.raw_feature_cache(base.TEST, names, batch_rows)
        train_path, test_path = raw_train.filename, raw_test.filename
        raw_train._mmap.close()
        raw_test._mmap.close()
        values, test_values = BatchedFeatureFile(train_path), BatchedFeatureFile(test_path)
    return meta, test_meta, folds, meta.label.to_numpy(dtype=np.int8), folds.fold.to_numpy(), names, values, test_values, references


def reference_paths():
    return [base.FOLDS, base.REPORT / "run_config.json", base.ARTIFACTS / "oof_predictions.parquet",
            base.ARTIFACTS / "test_predictions.parquet", TUNED_REPORT / "run_config.json",
            TUNED / "oof_predictions.parquet", TUNED / "test_predictions.parquet"]


def initialize(args):
    for path in (REPORT, ARTIFACTS, REPORT / "inner_curves", ARTIFACTS / "inner_models",
                 ARTIFACTS / "inner_predictions", ARTIFACTS / "inner_splits", ARTIFACTS / "models",
                 ARTIFACTS / "fold_predictions", ARTIFACTS / "borders", ARTIFACTS / "scratch"):
        path.mkdir(parents=True, exist_ok=True)
    context = load_context(args.batch_rows)
    meta, test_meta, _, _, _, names, _, _, _ = context
    for path in (base.REPORT / "run_config.json", TUNED_REPORT / "run_config.json"):
        if json.loads(path.read_text(encoding="utf-8"))["status"] != "completed":
            raise RuntimeError("Reference training is incomplete")
    experiment = {"protocol_version": 1, "train_rows": len(meta), "test_rows": len(test_meta),
        "features": names, "train_source": base.source_signature(base.TRAIN),
        "test_source": base.source_signature(base.TEST),
        "reference_sha256": {str(path): sha256(path) for path in reference_paths()},
        "code_sha256": {name: sha256(ROOT / "scripts" / name) for name in CODE_FILES},
        "versions": {"xgboost": xgb.__version__, "catboost": cb.__version__, "numpy": np.__version__},
        "engines": list(ENGINES), "params_fold_0": {engine: params_for(engine, 0, args.threads) for engine in ENGINES},
        "model_seed": "42+fold", "split_seed": "42+fold", "early_stop_fraction": 0.15,
        "max_rounds": args.max_rounds, "patience": args.patience, "min_delta": 0.0001,
        "threads": args.threads, "batch_rows": args.batch_rows,
        "features_policy": "all_1105_raw_float32_numeric_native_NaN_no_imputation_no_scaling_no_id",
        "selection": "one_predeclared_configuration_per_engine_inner_AUC_selects_rounds_then_full_four_fold_refit",
        "catboost_mode": "CPU_Plain_SymmetricTree_no_categorical_features",
        "quantization": "training_rows_only_XGB_ref_and_CatBoost_input_borders",
        "xgboost_memory": "CPU_QuantileDMatrix_with_raw_source_maps_released_after_every_batch",
        "test_prediction": "equal_mean_of_five_refit_models_per_engine_no_cross_engine_blend",
        "evaluation_scope": "reused_development_folds_not_untouched_final_test",
        "recommendation_policy": "replace_existing_ensemble_only_if_AUC_gain_ge_0.0005_AP_loss_le_0.001_logloss_increase_le_0.0005"}
    signature = signing_hash(experiment)
    path = REPORT / "run_config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["training_signature"] != signature:
            raise RuntimeError("Study differs; start a separate output directory")
        return saved
    saved = {"experiment": experiment, "training_signature": signature, "status": "initialized",
             "initial_available_memory_gb": psutil.virtual_memory().available / 1e9}
    base.json_write(path, saved)
    print(f"Initialized full-data controls: signature={signature}", flush=True)
    return saved


def checked_config():
    config = json.loads((REPORT / "run_config.json").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    if signing_hash(experiment) != config["training_signature"]:
        raise RuntimeError("Configuration changed")
    for name, expected in experiment["code_sha256"].items():
        if sha256(ROOT / "scripts" / name) != expected:
            raise RuntimeError(f"Training source changed: {name}. Keep existing artifacts and initialize "
                               "a new experiment with --run-name, e.g. tree_model_comparison_parallel10.")
    for path, expected in experiment["reference_sha256"].items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Reference changed: {path}")
    for path, key in ((base.TRAIN, "train_source"), (base.TEST, "test_source")):
        if base.source_signature(path) != experiment[key]:
            raise RuntimeError("Raw data changed")
    if experiment["versions"] != {"xgboost": xgb.__version__, "catboost": cb.__version__, "numpy": np.__version__}:
        raise RuntimeError("Runtime changed")
    return config


def pending_tasks(engines, folds, signature):
    """Check completed work before allocating large worker contexts."""
    pending = []
    for engine in engines:
        for fold in folds:
            _, inner_meta = model_paths(engine, fold, "inner")
            _, final_meta = model_paths(engine, fold, "refit")
            chosen = checkpoint(inner_meta, ARTIFACTS / "inner_predictions" / f"{engine}_fold_{fold}.npz", signature)
            record = checkpoint(final_meta, ARTIFACTS / "fold_predictions" / f"{engine}_fold_{fold}.npz", signature)
            if record is None:
                pending.append((engine, fold))
            elif chosen is None or chosen["selected_rounds"] != record["refit_rounds"]:
                raise RuntimeError(f"Inconsistent completed checkpoint: {engine} fold {fold}")
            else:
                print(f"Reuse {engine} fold {fold}: AUC={record['repay_roc_auc']:.6f}", flush=True)
    return pending


def parallel_plan(context, task_count):
    """Limit concurrent model threads to 10 logical CPUs; estimate RAM first."""
    if task_count < 1:
        raise ValueError("Need at least one pending task")
    available_cpus = psutil.cpu_count() or 1
    if available_cpus < PARALLEL_CPU_BUDGET:
        raise RuntimeError(f"--parallel requires at least {PARALLEL_CPU_BUDGET} logical CPUs; detected {available_cpus}")
    workers = min(PARALLEL_WORKERS, task_count)
    threads = PARALLEL_CPU_BUDGET // workers
    _, _, _, _, fold_id, names, _, _, _ = context
    max_fit_rows = max(int(np.sum(fold_id != fold)) for fold in range(5))
    # Conservative planning allowance for quantized rows, sketches, histograms
    # and native-library overhead. This is an estimate, not an OS memory limit.
    per_worker = max_fit_rows * len(names) * 4 * 1.25 + 1024**3
    required = workers * per_worker + 1024**3  # Reserve for the parent/OS.
    available = psutil.virtual_memory().available
    if available < required:
        raise RuntimeError(f"Insufficient available RAM for {workers} parallel workers: "
                           f"estimated {required / 1024**3:.1f} GiB, available {available / 1024**3:.1f} GiB. "
                           "Close memory-heavy applications and retry, or use single-process training.")
    return {"mode": "parallel", "workers": workers, "threads_per_worker": threads,
            "cpu_budget": PARALLEL_CPU_BUDGET, "estimated_required_memory_bytes": int(required),
            "available_memory_bytes": int(available)}


def prepare_splits(context, tasks):
    """The parent owns shared split files; workers only read them."""
    labels, fold_id = context[3], context[4]
    for fold in sorted({fold for _, fold in tasks}):
        inner, stop, valid = base.split_outer_fit(labels, fold_id, fold, 0.15)
        path = ARTIFACTS / "inner_splits" / f"fold_{fold}.npz"
        arrays = {"inner_fit_index": inner, "early_stop_index": stop, "outer_valid_index": valid}
        if path.exists():
            with np.load(path) as saved:
                for key, rows in arrays.items():
                    np.testing.assert_array_equal(saved[key], rows)
        else:
            np.savez_compressed(path, **arrays)


def worker_paths():
    # Only paths and small configuration objects cross the spawn boundary.
    return {"ROOT": str(ROOT), "REPORT": str(REPORT), "ARTIFACTS": str(ARTIFACTS),
            "TUNED": str(TUNED), "TRAIN": str(base.TRAIN), "TEST": str(base.TEST),
            "FOLDS": str(base.FOLDS), "BASELINE_ARTIFACTS": str(base.ARTIFACTS),
            "LINEAR_ARTIFACTS": str(base.LINEAR_ARTIFACTS)}


def initialize_worker(paths, config):
    global ROOT, REPORT, ARTIFACTS, TUNED, _WORKER_CONTEXT, _WORKER_CONFIG
    ROOT, REPORT, ARTIFACTS, TUNED = (Path(paths[key]) for key in ("ROOT", "REPORT", "ARTIFACTS", "TUNED"))
    base.TRAIN, base.TEST, base.FOLDS = (Path(paths[key]) for key in ("TRAIN", "TEST", "FOLDS"))
    base.ARTIFACTS = Path(paths["BASELINE_ARTIFACTS"])
    base.LINEAR_ARTIFACTS = Path(paths["LINEAR_ARTIFACTS"])
    _WORKER_CONFIG = config
    # The parent already built caches and splits. A child cannot rewrite them.
    with threadpool_limits(limits=config["execution"]["threads_per_worker"]):
        _WORKER_CONTEXT = load_context(config["experiment"]["batch_rows"], readonly_cache=True)


def train_worker(engine, fold, log_path):
    if _WORKER_CONTEXT is None or _WORKER_CONFIG is None:
        raise RuntimeError("Worker was not initialized")
    with Path(log_path).open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log), threadpool_limits(
            limits=_WORKER_CONFIG["execution"]["threads_per_worker"]
        ):
            return run_fold(_WORKER_CONTEXT, engine, fold, _WORKER_CONFIG)


def train_parallel(context, tasks, config):
    execution = parallel_plan(context, len(tasks))
    prepare_splits(context, tasks)
    log_directory = ROOT / "logs" / REPORT.name / f"parallel_{time.time_ns()}"
    log_directory.mkdir(parents=True)
    execution["log_directory"] = str(log_directory)
    config = {**config, "execution": execution}
    base.json_write(REPORT / "run_config.json", config)
    print(f"Parallel training: {execution['workers']} processes x {execution['threads_per_worker']} threads "
          f"= {PARALLEL_CPU_BUDGET} logical CPUs; {len(tasks)} pending tasks. Logs: {log_directory}", flush=True)
    # Set limits before spawn imports NumPy/native libraries. Restore the
    # parent's environment after all children finish, including on failure.
    variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {name: os.environ.get(name) for name in variables}
    try:
        for name in variables:
            os.environ[name] = str(execution["threads_per_worker"])
        with ProcessPoolExecutor(max_workers=execution["workers"], mp_context=mp.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(worker_paths(), config)) as pool:
            futures = {pool.submit(train_worker, engine, fold, str(log_directory / f"{engine}_fold_{fold}.log")):
                       (engine, fold) for engine, fold in tasks}
            for future in as_completed(futures):
                engine, fold = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    for other in futures:
                        other.cancel()
                    raise RuntimeError(f"Parallel task failed: {engine} fold {fold}; see "
                                       f"{log_directory / f'{engine}_fold_{fold}.log'}. "
                                       "Completed checkpoints are retained.") from error
                print(f"Completed {engine} fold {fold}: AUC={record['repay_roc_auc']:.6f} "
                      f"(PID {record['worker_pid']})", flush=True)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def group_metrics(labels, folds, predictions):
    masks = {"overall": np.ones(len(labels), dtype=bool),
             "feature_492_upper_tail": folds.stress_feature_492_upper_tail.to_numpy(dtype=bool),
             "feature_1039_observed": folds.stress_feature_1039_observed.to_numpy(dtype=bool)}
    return [{"group": group, **base.metrics(labels[mask], predictions[mask])} for group, mask in masks.items()]


def collect(context, config):
    meta, test_meta, folds, labels, fold_id, _, _, _, references = context
    oof, test = folds.copy(), test_meta.copy()
    oof["label"] = labels
    scores, search, records = dict(references), [], []
    signature = config["training_signature"]
    template = pd.read_parquet(ROOT / "data/processed/sample_submission.parquet")
    for engine in ENGINES:
        prediction, test_prediction = np.full(len(labels), np.nan), np.zeros(len(test_meta))
        coverage = np.zeros(len(labels), dtype=np.int8)
        for fold in range(5):
            _, metadata = model_paths(engine, fold, "inner")
            inner_path = ARTIFACTS / "inner_predictions" / f"{engine}_fold_{fold}.npz"
            chosen = checkpoint(metadata, inner_path, signature)
            _, metadata = model_paths(engine, fold, "refit")
            path = ARTIFACTS / "fold_predictions" / f"{engine}_fold_{fold}.npz"
            record = checkpoint(metadata, path, signature)
            if chosen is None or record is None or chosen["selected_rounds"] != record["refit_rounds"]:
                raise RuntimeError("Incomplete engine/fold")
            with np.load(path) as saved:
                valid = saved["outer_valid_index"]
                np.testing.assert_array_equal(valid, np.flatnonzero(fold_id == fold))
                prediction[valid] = saved["p_repay_valid"]
                coverage[valid] += 1
                test_prediction += saved["p_repay_test"] / 5
            search.append(chosen)
            records.append(record)
        if not (coverage == 1).all() or not np.isfinite(prediction).all():
            raise RuntimeError("Invalid OOF coverage")
        scores[engine] = prediction
        oof[f"p_repay_{engine}"] = prediction
        test[f"p_repay_{engine}"] = test_prediction
        submission = base.build_submission(test[["id", f"p_repay_{engine}"]].rename(columns={f"p_repay_{engine}": "p_repay"}), template)
        submission.to_csv(submission_path(engine), index=False)
    comparison = pd.DataFrame([{"model": model, **row} for model, p in scores.items() for row in group_metrics(labels, folds, p)])
    paired = []
    for fold in range(5):
        valid = fold_id == fold
        single = base.metrics(labels[valid], references["lightgbm_tuned_seed_42"][valid])
        ensemble = base.metrics(labels[valid], references["lightgbm_tuned_ensemble"][valid])
        for model, p in scores.items():
            row = base.metrics(labels[valid], p[valid])
            paired.append({"model": model, "fold": fold, **row,
                           **{f"delta_single_{metric}": row[metric] - single[metric] for metric in METRICS},
                           **{f"delta_ensemble_{metric}": row[metric] - ensemble[metric] for metric in METRICS}})
    overall = comparison.query("group == 'overall'").set_index("model")
    incumbent = overall.loc["lightgbm_tuned_ensemble"]
    eligible = [engine for engine in ENGINES if
                overall.loc[engine, "repay_roc_auc"] - incumbent.repay_roc_auc >= 0.0005 and
                overall.loc[engine, "default_pr_auc_ap"] - incumbent.default_pr_auc_ap >= -0.001 and
                overall.loc[engine, "log_loss"] - incumbent.log_loss <= 0.0005]
    recommendation = max(eligible, key=lambda engine: overall.loc[engine, "repay_roc_auc"]) if eligible else "lightgbm_tuned_ensemble"
    oof.to_parquet(ARTIFACTS / "oof_predictions.parquet", index=False)
    test.to_parquet(ARTIFACTS / "test_predictions.parquet", index=False)
    comparison.to_csv(REPORT / "comparison_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(REPORT / "paired_fold_metrics.csv", index=False)
    pd.DataFrame(search).to_csv(REPORT / "selection_records.csv", index=False)
    pd.DataFrame(records).to_csv(REPORT / "refit_records.csv", index=False)
    pd.DataFrame(scores).corr().to_csv(REPORT / "prediction_correlations.csv")
    summary = {"training_signature": signature, "overall": overall.reset_index().to_dict(orient="records"),
               "selected_rounds": {engine: [r["selected_rounds"] for r in search if r["engine"] == engine] for engine in ENGINES},
               "hit_round_cap": {engine: sum(r["hit_round_cap"] for r in search if r["engine"] == engine) for engine in ENGINES},
               "folds_above_tuned_single_auc": {engine: sum(r["delta_single_repay_roc_auc"] > 0 for r in paired if r["model"] == engine) for engine in ENGINES},
               "recommended_development_model": recommendation, "official_test_score": None,
               "submissions_uploaded": False, "no_cross_engine_blend": True,
               "model_stage_seconds_sum": float(sum(r["elapsed_seconds"] for r in search + records))}
    base.json_write(REPORT / "summary.json", summary)
    config["status"] = "completed"
    base.json_write(REPORT / "run_config.json", config)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("init", "train", "collect"), required=True)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--max-rounds", type=int, default=900)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--parallel", action="store_true",
                        help="Train independent folds with 2 processes x 5 threads (10 logical CPUs); train stage only")
    parser.add_argument("--run-name", default="tree_model_comparison",
                        help="Separate report/artifact directory name; default: tree_model_comparison")
    args = parser.parse_args()
    if args.max_rounds < 1 or args.patience < 1 or args.threads < 1 or args.batch_rows < 1 or any(f not in range(5) for f in args.folds):
        parser.error("Invalid positive setting or fold")
    if len(set(args.folds)) != len(args.folds) or len(set(args.engines)) != len(args.engines):
        parser.error("Duplicate engine/fold")
    if args.parallel and args.stage != "train":
        parser.error("--parallel applies only to --stage train")
    try:
        configure_run(args.run_name)
    except ValueError as error:
        parser.error(str(error))
    if args.stage == "init":
        initialize(args)
        return
    config = checked_config()
    with threadpool_limits(limits=config["experiment"]["threads"]):
        if args.stage == "train":
            tasks = pending_tasks(args.engines, args.folds, config["training_signature"])
            if not tasks:
                print("All requested folds already completed; no training needed", flush=True)
                return
            context = load_context(config["experiment"]["batch_rows"])
            if args.parallel:
                train_parallel(context, tasks, config)
            else:
                config = {**config, "execution": {"mode": "single", "workers": 1,
                          "threads_per_worker": config["experiment"]["threads"],
                          "cpu_budget": config["experiment"]["threads"]}}
                base.json_write(REPORT / "run_config.json", config)
                for engine, fold in tasks:
                    run_fold(context, engine, fold, config)
        else:
            context = load_context(config["experiment"]["batch_rows"])
            collect(context, config)


if __name__ == "__main__":
    mp.freeze_support()
    main()

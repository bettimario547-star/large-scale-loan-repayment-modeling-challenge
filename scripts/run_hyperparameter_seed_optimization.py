"""Nested finite LightGBM search and a predeclared three-seed ensemble.

Each outer fold chooses parameters / rounds using its fitting rows only.
All candidates share inner splits and bins. Final seeds share the chosen tree
count and full-fitting bins; only model sampling seeds change. No lucky-seed
selection. Existing baseline / screening code and artifacts remain immutable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

try:
    from . import run_lightgbm_baseline as base
except ImportError:
    import run_lightgbm_baseline as base


ROOT = base.ROOT
REPORT = ROOT / "reports/hyperparameter_seed_optimization"
ARTIFACTS = ROOT / "artifacts/hyperparameter_seed_optimization"
SEEDS = (42, 2026, 3407)
TIE_TOLERANCE = 0.0001
# A small, fixed joint search, not a full Cartesian grid or an Optuna study.
# Earlier entries take priority if within TIE_TOLERANCE of the best inner AUC.
CANDIDATES = (
    {"name": "baseline_extended", "learning_rate": 0.05, "num_leaves": 31,
     "min_data_in_leaf": 200, "lambda_l2": 1.0, "feature_fraction": 0.8, "bagging_fraction": 0.8},
    {"name": "regularized", "learning_rate": 0.05, "num_leaves": 31,
     "min_data_in_leaf": 400, "lambda_l2": 5.0, "feature_fraction": 0.75, "bagging_fraction": 0.85},
    {"name": "wider", "learning_rate": 0.05, "num_leaves": 63,
     "min_data_in_leaf": 300, "lambda_l2": 5.0, "feature_fraction": 0.8, "bagging_fraction": 0.8},
    {"name": "faster_rate", "learning_rate": 0.08, "num_leaves": 31,
     "min_data_in_leaf": 300, "lambda_l2": 3.0, "feature_fraction": 0.8, "bagging_fraction": 0.8},
)
CODE_FILES = ("run_hyperparameter_seed_optimization.py", "run_lightgbm_baseline.py",
              "run_linear_baselines.py", "id_feature_contract.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signing_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def params_for(candidate: dict, fold: int, seed: int, threads: int) -> dict:
    if candidate not in CANDIDATES or fold not in range(5) or seed not in SEEDS or threads < 1:
        raise ValueError("Unknown candidate, fold, seed, or invalid thread count")
    params = base.model_params(fold, threads)
    params.update({key: value for key, value in candidate.items() if key != "name"})
    params.update(seed=seed + fold, feature_fraction_seed=seed + fold,
                  bagging_seed=seed + fold, data_random_seed=42 + fold,
                  feature_pre_filter=False)
    # Dataset bins stay identical across candidates and final sampling seeds.
    # pre_filter=False allows changing min_data_in_leaf on one constructed Dataset.
    return params


def choose_candidate(records: list[dict], tolerance: float = TIE_TOLERANCE) -> dict:
    if tolerance < 0 or len(records) != len(CANDIDATES):
        raise ValueError("Expected one result per candidate and nonnegative tolerance")
    indexed = {item["candidate"]: item for item in records}
    if set(indexed) != {item["name"] for item in CANDIDATES}:
        raise ValueError("Missing or duplicate candidate")
    if not all(np.isfinite(item["inner_auc"]) and item["selected_rounds"] >= 1 for item in records):
        raise ValueError("Invalid candidate scores or tree counts")
    best = max(item["inner_auc"] for item in records)
    return next(indexed[item["name"]] for item in CANDIDATES
                if indexed[item["name"]]["inner_auc"] >= best - tolerance)


def probability_mean(predictions: list[np.ndarray]) -> np.ndarray:
    if not predictions or len({np.asarray(item).shape for item in predictions}) != 1:
        raise ValueError("Need predictions with matching shapes")
    stacked = np.stack(predictions)
    if stacked.ndim != 2 or not np.isfinite(stacked).all() or np.any((stacked < 0) | (stacked > 1)):
        raise ValueError("Expected one-dimensional finite probabilities")
    return stacked.mean(axis=0)


def load_context(batch_rows: int):
    names = base.model_feature_columns(pq.read_schema(base.TRAIN).names, pq.read_schema(base.TEST).names)
    meta = pd.read_parquet(base.TRAIN, columns=["id", "label"])
    test_meta = pd.read_parquet(base.TEST, columns=["id"])
    folds = base.align_validation_folds(meta.id, pd.read_parquet(base.FOLDS))
    if set(folds.fold.unique()) != set(range(5)) or not meta.label.isin([0, 1]).all():
        raise RuntimeError("Invalid fixed folds or labels")
    baseline = pd.read_parquet(base.ARTIFACTS / "oof_predictions.parquet")
    pd.testing.assert_frame_equal(baseline[["id", "label"]], meta, check_dtype=False)
    pd.testing.assert_frame_equal(baseline[list(folds.columns)], folds, check_dtype=False)
    values = base.raw_feature_cache(base.TRAIN, names, batch_rows)
    test_values = base.raw_feature_cache(base.TEST, names, batch_rows)
    return meta, test_meta, folds, meta.label.to_numpy(dtype=np.int8), folds.fold.to_numpy(), names, values, test_values, baseline


def initialize(args):
    for directory in (REPORT, ARTIFACTS, REPORT / "inner_curves", ARTIFACTS / "inner_models",
                      ARTIFACTS / "inner_predictions", ARTIFACTS / "inner_splits",
                      ARTIFACTS / "models", ARTIFACTS / "fold_predictions", REPORT / "selections"):
        directory.mkdir(parents=True, exist_ok=True)
    meta, test_meta, _, _, _, names, _, _, _ = load_context(args.batch_rows)
    reference = json.loads((base.REPORT / "run_config.json").read_text(encoding="utf-8"))
    assert reference["status"] == "completed"
    assert reference["fold_sha256"] == sha256(base.FOLDS)
    assert reference["train_source"] == base.source_signature(base.TRAIN)
    assert reference["test_source"] == base.source_signature(base.TEST)
    experiment = {
        "protocol_version": 1, "train_rows": len(meta), "test_rows": len(test_meta),
        "features": names, "train_source": base.source_signature(base.TRAIN),
        "test_source": base.source_signature(base.TEST), "fold_sha256": sha256(base.FOLDS),
        "baseline_config_sha256": sha256(base.REPORT / "run_config.json"),
        "baseline_oof_sha256": sha256(base.ARTIFACTS / "oof_predictions.parquet"),
        "baseline_test_sha256": sha256(base.ARTIFACTS / "test_predictions.parquet"),
        "code_sha256": {name: sha256(ROOT / "scripts" / name) for name in CODE_FILES},
        "candidates": list(CANDIDATES), "model_seeds": list(SEEDS), "split_seed": 42,
        "data_bin_seed": "42+fold_fixed_across_model_seeds",
        "model_seed": "seed+fold_for_seed_bagging_and_feature_sampling",
        "early_stop_fraction": 0.15, "max_rounds": args.max_rounds,
        "patience": args.patience, "min_delta": 0.0001, "tie_tolerance": TIE_TOLERANCE,
        "threads": args.threads, "batch_rows": args.batch_rows,
        "lightgbm_version": lgb.__version__, "numpy_version": np.__version__,
        "selection": "inner_AUC_then_predeclared_candidate_priority_within_tolerance",
        "seed_rounds": "same_selected_rounds_for_all_three_refit_seeds",
        "test_ensemble": "equal_mean_of_3_seeds_times_5_outer_refit_models",
        "features_policy": "all_1105_raw_float32_native_missing_no_imputation_no_scaling",
        "evaluation_scope": "reused_development_folds_with_fold_local_current_search",
        "recommendation_policy": "ensemble_if_auc_gain_ge_0.0005_AP_loss_le_0.001_logloss_increase_le_0.0005_else_baseline",
    }
    signature = signing_hash(experiment)
    path = REPORT / "run_config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["training_signature"] != signature:
            raise RuntimeError("Existing study differs; use a new output directory for a new protocol")
        print("Study already initialized; matching signature", flush=True)
        return saved
    config = {"experiment": experiment, "training_signature": signature, "status": "initialized",
              "initial_available_memory_gb": psutil.virtual_memory().available / 1e9}
    base.json_write(path, config)
    print(json.dumps({key: value for key, value in experiment.items()
                      if key not in {"features", "code_sha256"}}, ensure_ascii=False, indent=2), flush=True)
    return config


def checked_config() -> dict:
    config = json.loads((REPORT / "run_config.json").read_text(encoding="utf-8"))
    experiment = config["experiment"]
    if signing_hash(experiment) != config["training_signature"]:
        raise RuntimeError("Study configuration checksum differs")
    for name, checksum in experiment["code_sha256"].items():
        if sha256(ROOT / "scripts" / name) != checksum:
            raise RuntimeError(f"Training source changed: {name}")
    for path, signature in ((base.TRAIN, experiment["train_source"]), (base.TEST, experiment["test_source"])):
        if base.source_signature(path) != signature:
            raise RuntimeError("Raw input changed")
    checks = ((base.FOLDS, "fold_sha256"), (base.REPORT / "run_config.json", "baseline_config_sha256"),
              (base.ARTIFACTS / "oof_predictions.parquet", "baseline_oof_sha256"),
              (base.ARTIFACTS / "test_predictions.parquet", "baseline_test_sha256"))
    for path, key in checks:
        if sha256(path) != experiment[key]:
            raise RuntimeError(f"Reference changed: {key}")
    if lgb.__version__ != experiment["lightgbm_version"] or np.__version__ != experiment["numpy_version"]:
        raise RuntimeError("Runtime version changed")
    return config


def checkpoint_record(metadata_path: Path, prediction_path: Path, signature: str) -> dict | None:
    if not metadata_path.exists() or not prediction_path.exists():
        return None
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    if saved["training_signature"] != signature or sha256(Path(saved["model_path"])) != saved["model_sha256"]:
        raise RuntimeError("Completed checkpoint configuration or model checksum differs")
    if sha256(prediction_path) != saved["prediction_sha256"]:
        raise RuntimeError("Completed checkpoint prediction checksum differs")
    return saved["record"]


def run_fold(context, fold: int, config: dict):
    _, _, _, labels, fold_id, names, values, test_values, _ = context
    experiment, signature = config["experiment"], config["training_signature"]
    threads, batch = experiment["threads"], experiment["batch_rows"]
    inner, stop, valid = base.split_outer_fit(labels, fold_id, fold, experiment["early_stop_fraction"])
    all_fit = np.flatnonzero(fold_id != fold)
    split_path = ARTIFACTS / f"inner_splits/fold_{fold}.npz"
    if split_path.exists():
        with np.load(split_path) as saved:
            for key, expected in (("inner_fit_index", inner), ("early_stop_index", stop), ("outer_valid_index", valid)):
                np.testing.assert_array_equal(saved[key], expected)
    else:
        np.savez_compressed(split_path, inner_fit_index=inner, early_stop_index=stop, outer_valid_index=valid)
    print(f"Fold {fold}: inner={len(inner):,}, stop={len(stop):,}, outer={len(valid):,}", flush=True)
    inner_data = stop_data = None
    records = []
    for candidate in CANDIDATES:
        tag = f"{candidate['name']}_fold_{fold}"
        model_path = ARTIFACTS / f"inner_models/{tag}.txt"
        metadata = ARTIFACTS / f"inner_models/{tag}.json"
        predictions = ARTIFACTS / f"inner_predictions/{tag}.npz"
        record = checkpoint_record(metadata, predictions, signature)
        if record is not None:
            print(f"Reuse inner {tag}", flush=True)
            records.append(record)
            continue
        started = time.perf_counter()
        params = params_for(candidate, fold, SEEDS[0], threads)
        if inner_data is None:
            # Build once per fold, on inner fitting rows only. Dataset construction
            # parameters (including bin seed) are fixed across the four candidates.
            inner_data = base.make_dataset(values, labels, inner, names, params, batch)
            stop_data = base.make_dataset(values, labels, stop, names, params, batch, reference=inner_data)
            inner_data.construct()
            stop_data.construct()
        history = {}
        selected = lgb.train(
            params, inner_data, num_boost_round=experiment["max_rounds"],
            valid_sets=[stop_data], valid_names=["inner_early_stop"],
            callbacks=[lgb.early_stopping(experiment["patience"], first_metric_only=True,
                                         min_delta=experiment["min_delta"]),
                       lgb.log_evaluation(100), lgb.record_evaluation(history)],
        )
        p_stop = base.predict_rows(selected, values, batch, stop)
        rounds = int(selected.best_iteration)
        evaluated = len(history["inner_early_stop"]["auc"])
        score = float(roc_auc_score(labels[stop], p_stop))
        np.testing.assert_allclose(score, selected.best_score["inner_early_stop"]["auc"], atol=1e-12)
        record = {"fold": fold, "candidate": candidate["name"], "inner_auc": score,
                  "selected_rounds": rounds, "evaluated_rounds": evaluated,
                  "hit_round_cap": evaluated >= experiment["max_rounds"],
                  "inner_fit_rows": len(inner), "early_stop_rows": len(stop),
                  "elapsed_seconds": time.perf_counter() - started}
        selected.save_model(str(model_path))
        np.savez_compressed(predictions, early_stop_index=stop, p_repay=p_stop)
        base.json_write(REPORT / f"inner_curves/{tag}.json", history)
        base.json_write(metadata, {"training_signature": signature, "positive_class": 1,
                                  "features": names, "params": params, "record": record,
                                  "model_path": str(model_path), "model_sha256": sha256(model_path),
                                  "prediction_sha256": sha256(predictions)})
        records.append(record)
        print(f"Inner {tag}: AUC={score:.6f}, rounds={rounds}, seconds={record['elapsed_seconds']:.0f}", flush=True)
        del selected, p_stop
        gc.collect()
    del inner_data, stop_data
    gc.collect()
    chosen = choose_candidate(records, experiment["tie_tolerance"])
    candidate = next(item for item in CANDIDATES if item["name"] == chosen["candidate"])
    base.json_write(REPORT / f"selections/fold_{fold}.json",
                    {"training_signature": signature, "records": records, "chosen": chosen,
                     "rule": experiment["selection"]})
    print(f"Fold {fold}: choose {chosen['candidate']}, rounds={chosen['selected_rounds']}; refit three seeds", flush=True)
    full_data = None
    for seed in SEEDS:
        tag = f"seed_{seed}_fold_{fold}"
        model_path = ARTIFACTS / f"models/{tag}.txt"
        metadata = ARTIFACTS / f"models/{tag}.json"
        predictions = ARTIFACTS / f"fold_predictions/{tag}.npz"
        record = checkpoint_record(metadata, predictions, signature)
        if record is not None:
            if record["candidate"] != chosen["candidate"] or record["refit_rounds"] != chosen["selected_rounds"]:
                raise RuntimeError("Refit checkpoint differs from chosen candidate")
            print(f"Reuse final {tag}", flush=True)
            continue
        started = time.perf_counter()
        params = params_for(candidate, fold, seed, threads)
        if full_data is None:
            full_data = base.make_dataset(values, labels, all_fit, names, params, batch)
            full_data.construct()
        # The outer fold never enters train/early stopping/candidate choice.
        model = lgb.train(params, full_data, num_boost_round=chosen["selected_rounds"],
                          callbacks=[base.log_refit(tag)])
        p_valid = base.predict_rows(model, values, batch, valid)
        p_test = base.predict_rows(model, test_values, batch)
        model.save_model(str(model_path))
        np.savez_compressed(predictions, outer_valid_index=valid, p_repay_valid=p_valid, p_repay_test=p_test)
        record = {"fold": fold, "seed": seed, "candidate": chosen["candidate"],
                  "refit_rounds": model.current_iteration(), "fit_rows": len(all_fit),
                  **base.metrics(labels[valid], p_valid), "elapsed_seconds": time.perf_counter() - started}
        base.json_write(metadata, {"training_signature": signature, "positive_class": 1,
                                  "features": names, "params": params, "record": record,
                                  "model_path": str(model_path), "model_sha256": sha256(model_path),
                                  "prediction_sha256": sha256(predictions)})
        print(f"Final {tag}: AUC={record['repay_roc_auc']:.6f}, AP={record['default_pr_auc_ap']:.6f}, "
              f"logloss={record['log_loss']:.6f}", flush=True)
        del model, p_valid, p_test
        gc.collect()
    del full_data
    gc.collect()


def group_metrics(labels, folds, score):
    masks = {"overall": np.ones(len(labels), dtype=bool),
             "feature_492_upper_tail": folds.stress_feature_492_upper_tail.to_numpy(dtype=bool),
             "feature_1039_observed": folds.stress_feature_1039_observed.to_numpy(dtype=bool)}
    return [{"group": group, **base.metrics(labels[mask], score[mask])} for group, mask in masks.items()]


def collect(context, config):
    meta, test_meta, folds, labels, fold_id, _, _, _, baseline = context
    signature = config["training_signature"]
    oof, test = folds.copy(), test_meta.copy()
    oof["label"] = labels
    records, search, selections = [], [], []
    for fold in range(5):
        saved = json.loads((REPORT / f"selections/fold_{fold}.json").read_text(encoding="utf-8"))
        assert saved["training_signature"] == signature
        assert saved["chosen"] == choose_candidate(saved["records"])
        selections.append(saved["chosen"])
        search.extend(saved["records"])
    for seed in SEEDS:
        score, test_sum = np.full(len(labels), np.nan), np.zeros(len(test_meta))
        coverage = np.zeros(len(labels), dtype=np.int8)
        for fold in range(5):
            tag = f"seed_{seed}_fold_{fold}"
            prediction_path = ARTIFACTS / f"fold_predictions/{tag}.npz"
            record = checkpoint_record(ARTIFACTS / f"models/{tag}.json", prediction_path, signature)
            if record is None:
                raise RuntimeError(f"Incomplete final model: {tag}")
            with np.load(prediction_path) as saved:
                valid = saved["outer_valid_index"]
                np.testing.assert_array_equal(valid, np.flatnonzero(fold_id == fold))
                assert saved["p_repay_test"].shape == (len(test_meta),)
                score[valid] = saved["p_repay_valid"]
                coverage[valid] += 1
                test_sum += saved["p_repay_test"] / 5
            records.append(record)
        if not (coverage == 1).all() or not np.isfinite(score).all():
            raise RuntimeError("Incomplete or overlapping OOF predictions")
        oof[f"p_repay_seed_{seed}"] = score
        test[f"p_repay_seed_{seed}"] = test_sum
    columns = [f"p_repay_seed_{seed}" for seed in SEEDS]
    oof["p_repay_ensemble"] = probability_mean([oof[column].to_numpy() for column in columns])
    test["p_repay_ensemble"] = probability_mean([test[column].to_numpy() for column in columns])
    oof.to_parquet(ARTIFACTS / "oof_predictions.parquet", index=False)
    test.to_parquet(ARTIFACTS / "test_predictions.parquet", index=False)
    template = pd.read_parquet(ROOT / "data/processed/sample_submission.parquet")
    for column in columns + ["p_repay_ensemble"]:
        frame = test[["id", column]].rename(columns={column: "p_repay"})
        suffix = column.removeprefix("p_repay_")
        base.build_submission(frame, template).to_csv(ROOT / f"submissions/lightgbm_tuned_{suffix}.csv", index=False)
    all_scores = {"baseline": baseline.p_repay.to_numpy(),
                  **{f"tuned_seed_{seed}": oof[f"p_repay_seed_{seed}"].to_numpy() for seed in SEEDS},
                  "tuned_seed_ensemble": oof.p_repay_ensemble.to_numpy()}
    comparison = pd.DataFrame([{"model": name, **item} for name, score in all_scores.items()
                               for item in group_metrics(labels, folds, score)])
    comparison.to_csv(REPORT / "comparison_metrics.csv", index=False)
    pd.DataFrame(search).to_csv(REPORT / "search_results.csv", index=False)
    pd.DataFrame(selections).to_csv(REPORT / "selected_candidates.csv", index=False)
    pd.DataFrame(records).to_csv(REPORT / "refit_records.csv", index=False)
    paired, seed_spread = [], []
    for fold in range(5):
        valid = fold_id == fold
        baseline_metrics = base.metrics(labels[valid], all_scores["baseline"][valid])
        seed_aucs = [base.metrics(labels[valid], all_scores[f"tuned_seed_{seed}"][valid])["repay_roc_auc"] for seed in SEEDS]
        for name, score in all_scores.items():
            result = base.metrics(labels[valid], score[valid])
            paired.append({"model": name, "fold": fold, **result,
                           **{f"delta_{metric}": result[metric] - baseline_metrics[metric]
                              for metric in ("repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score")}})
        predictions = np.stack([all_scores[f"tuned_seed_{seed}"][valid] for seed in SEEDS])
        correlations = np.corrcoef(predictions)[np.triu_indices(3, 1)]
        seed_spread.append({"fold": fold, "seed_auc_mean": float(np.mean(seed_aucs)),
                            "seed_auc_std_ddof1": float(np.std(seed_aucs, ddof=1)),
                            "seed_auc_min": min(seed_aucs), "seed_auc_max": max(seed_aucs),
                            "mean_row_probability_std_ddof0": float(predictions.std(axis=0).mean()),
                            "mean_pairwise_prediction_correlation": float(correlations.mean())})
    pd.DataFrame(paired).to_csv(REPORT / "paired_fold_metrics.csv", index=False)
    pd.DataFrame(seed_spread).to_csv(REPORT / "seed_stability.csv", index=False)
    overall = comparison.query("group == 'overall'").set_index("model")
    original, ensemble = overall.loc["baseline"], overall.loc["tuned_seed_ensemble"]
    delta = {metric: float(ensemble[metric] - original[metric])
             for metric in ("repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score")}
    recommend = (delta["repay_roc_auc"] >= 0.0005 and delta["default_pr_auc_ap"] >= -0.001
                 and delta["log_loss"] <= 0.0005)
    per_fold = pd.DataFrame(paired)
    summary = {"training_signature": signature, "overall": overall.reset_index().to_dict(orient="records"),
               "ensemble_minus_baseline": delta, "chosen_candidates_by_fold": [item["candidate"] for item in selections],
               "chosen_rounds_by_fold": [item["selected_rounds"] for item in selections],
               "search_hit_cap_count": sum(item["hit_round_cap"] for item in search),
               "chosen_hit_cap_count": sum(item["hit_round_cap"] for item in selections),
               "ensemble_folds_with_positive_auc_delta": int((per_fold.query("model == 'tuned_seed_ensemble'").delta_repay_roc_auc > 0).sum()),
               "recommended_development_model": "tuned_seed_ensemble" if recommend else "baseline",
               "no_seed_selected_by_outer_auc": True, "official_test_score": None, "submissions_uploaded": False,
               "total_recorded_training_seconds": float(sum(item["elapsed_seconds"] for item in search + records))}
    base.json_write(REPORT / "summary.json", summary)
    config.update(status="completed", completed_inner_models=len(search), completed_refit_models=len(records))
    base.json_write(REPORT / "run_config.json", config)
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("init", "train", "collect"), required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--max-rounds", type=int, default=900)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args()
    if min(args.max_rounds, args.patience, args.threads, args.batch_rows) < 1 or len(args.folds) != len(set(args.folds)):
        parser.error("Counts must be positive and folds unique")
    if args.stage == "init":
        initialize(args)
        return
    config = checked_config()
    experiment = config["experiment"]
    with threadpool_limits(limits=experiment["threads"]):
        context = load_context(experiment["batch_rows"])
        if args.stage == "train":
            # Different local worker processes must receive disjoint --folds.
            for fold in args.folds:
                run_fold(context, fold, config)
        else:
            collect(context, config)


if __name__ == "__main__":
    main()

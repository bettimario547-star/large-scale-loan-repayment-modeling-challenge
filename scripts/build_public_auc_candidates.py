"""Build traceable AUC submissions from existing and completed new models.

Public scores are observations supplied by the user, never inferred from OOF.
Every prediction is aligned by dataset-local id before any blending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .id_feature_contract import align_validation_folds, build_submission
except ImportError:
    from id_feature_contract import align_validation_folds, build_submission

ROOT = Path(__file__).resolve().parents[1]
OBSERVED_PUBLIC_SCORES = {
    "lightgbm_baseline.csv": 0.73212,
    "lightgbm_tuned_ensemble.csv": 0.73453,
    "feature_screening_drift_core_filter.csv": 0.73501,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def aligned_values(frame, ids, column):
    if frame.id.isna().any() or not frame.id.is_unique or len(frame) != len(ids):
        raise ValueError("Prediction IDs must be non-null, unique and cover the dataset")
    if set(frame.id) != set(ids):
        raise ValueError("Prediction IDs do not match the dataset")
    values = frame.set_index("id").loc[ids, column].to_numpy()
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Predictions must be finite scores within [0, 1]")
    return values


def probability_blend(scores, weights):
    if not weights or any(name not in scores for name in weights):
        raise ValueError("Missing blend components")
    if any(not np.isfinite(w) or w < 0 for w in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Nonnegative weights must sum to one")
    if len({np.asarray(scores[name]).shape for name in weights}) != 1:
        raise ValueError("Blend components have different shapes")
    result = sum(weight * np.asarray(scores[name], dtype=np.float64) for name, weight in weights.items())
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Invalid blended scores")
    return result


def rank_blend(scores, weights):
    """Average empirical percentile ranks; these scores are not calibrated probabilities."""
    ranked = {}
    for name in weights:
        if name not in scores:
            raise ValueError("Missing rank blend component")
        values = np.asarray(scores[name])
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Rank blend components must be finite, nonempty vectors")
        ranked[name] = (pd.Series(values).rank(method="average").to_numpy() - .5) / len(values)
    return probability_blend(ranked, weights)


def record_public_score(root, filename, score, evidence):
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Public AUC must lie in [0, 1]")
    root = Path(root)
    source = (root / "submissions" / filename).resolve()
    if not source.is_relative_to((root / "submissions").resolve()) or not source.is_file():
        raise ValueError("Submission must exist inside this project's submissions directory")
    path = root / "reports/public_auc_optimization/submission_history.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    entry = {"filename": source.relative_to(root / "submissions").as_posix(),
             "sha256": sha256(source), "public_auc": float(score), "evidence": evidence}
    if entry not in history:
        history.append(entry)
        json_write(path, history)
    return history


def load_predictions(root):
    root = Path(root)
    meta = pd.read_parquet(root / "data/processed/train.parquet", columns=["id", "label"])
    test_meta = pd.read_parquet(root / "data/processed/test.parquet", columns=["id"])
    folds = align_validation_folds(meta.id, pd.read_parquet(root / "data/interim/validation_folds.parquet"))
    template = pd.read_parquet(root / "data/processed/sample_submission.parquet")
    oof, test, sources = {}, {}, {}

    def add(name, folder, column="p_repay"):
        folder = root / folder
        paths = [folder / "oof_predictions.parquet", folder / "test_predictions.parquet"]
        if not all(path.is_file() for path in paths):
            return
        frame, test_frame = (pd.read_parquet(path) for path in paths)
        for key in ("label", "fold", "stress_feature_492_upper_tail", "stress_feature_1039_observed"):
            expected = meta.label.to_numpy() if key == "label" else folds[key].to_numpy()
            actual = frame.set_index("id").loc[meta.id, key].to_numpy()
            np.testing.assert_array_equal(actual, expected)
        oof[name] = aligned_values(frame, meta.id, column)
        test[name] = aligned_values(test_frame, test_meta.id, column)
        sources[name] = {str(path.relative_to(root)): sha256(path) for path in paths}

    add("drift_core", "artifacts/feature_screening/drift_core_filter")
    add("tuned", "artifacts/hyperparameter_seed_optimization", "p_repay_ensemble")
    add("availability", "artifacts/feature_screening/availability_filter")
    add("baseline", "artifacts/lightgbm_baseline")
    folder = root / "artifacts/tree_model_comparison"
    if all((folder / f"fold_predictions/xgboost_fold_{k}.npz").is_file() for k in range(5)):
        prediction, test_prediction = np.full(len(meta), np.nan), np.zeros(len(test_meta))
        coverage = np.zeros(len(meta), dtype=np.int8)
        hashes = {}
        for fold in range(5):
            path = folder / f"fold_predictions/xgboost_fold_{fold}.npz"
            metadata = json.loads((folder / f"models/xgboost_fold_{fold}.json").read_text(encoding="utf-8"))
            if sha256(path) != metadata["prediction_sha256"] or metadata["positive_class"] != 1:
                raise RuntimeError("XGBoost prediction checksum or positive class differs")
            with np.load(path, allow_pickle=False) as saved:
                valid = saved["outer_valid_index"]
                np.testing.assert_array_equal(valid, np.flatnonzero(folds.fold.to_numpy() == fold))
                if saved["p_repay_test"].shape != (len(test_meta),):
                    raise ValueError("Incorrect XGBoost test prediction shape")
                prediction[valid] = saved["p_repay_valid"]
                test_prediction += saved["p_repay_test"] / 5
                coverage[valid] += 1
            hashes[str(path.relative_to(root))] = sha256(path)
        if not (coverage == 1).all():
            raise RuntimeError("Incomplete XGBoost OOF coverage")
        oof["xgboost"], test["xgboost"], sources["xgboost"] = prediction, test_prediction, hashes
    new_root = root / "artifacts/public_auc_optimization/completed"
    for folder in sorted(new_root.glob("*")):
        if folder.is_dir():
            add(folder.name, folder.relative_to(root))
    if "drift_core" not in oof or "tuned" not in oof:
        raise RuntimeError("The public-best drift model and tuned reference must be present")
    return meta, test_meta, folds, template, oof, test, sources


def export_candidate(root, name, values, ids, template):
    submission = build_submission(pd.DataFrame({"id": ids, "p_repay": values}), template)
    directory = Path(root) / "submissions/public_auc"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{name}.tmp.csv"
    submission.to_csv(temporary, index=False)
    digest = sha256(temporary)
    target = directory / f"{name}_{digest[:10]}.csv"
    if target.exists() and sha256(target) != digest:
        raise RuntimeError("Existing submission fingerprint differs")
    temporary.replace(target)
    return target, digest


def build(root=ROOT):
    root = Path(root)
    report = root / "reports/public_auc_optimization"
    report.mkdir(parents=True, exist_ok=True)
    history_path = report / "submission_history.json"
    if not history_path.exists():
        for name, score in OBSERVED_PUBLIC_SCORES.items():
            record_public_score(root, name, score, "用户提供的 Kaggle 公榜截图；观察日期 2026-10-25")
    meta, test_meta, folds, template, oof, test, sources = load_predictions(root)
    recipes = {
        "existing_blend_drift50_tuned50": {"drift_core": .5, "tuned": .5},
        "existing_blend_drift70_tuned30": {"drift_core": .7, "tuned": .3},
        "existing_xgboost": {"xgboost": 1.0},
        "existing_blend_drift50_xgb50": {"drift_core": .5, "xgboost": .5},
        "existing_availability": {"availability": 1.0},
    }
    rank_recipes = {
        "existing_rank_drift50_tuned50": {"drift_core": .5, "tuned": .5},
        "existing_rank_drift50_xgb50": {"drift_core": .5, "xgboost": .5},
    }
    for name in oof:
        if name in {"drift_core", "tuned", "availability", "baseline", "xgboost"}:
            continue
        recipes[name] = {name: 1.0}
        recipes[f"blend_{name}_with_public_best"] = {name: .7, "drift_core": .3}
    champion = next((name for name in ("lgbm_drift_tuned_ensemble", "lgbm_drift_tuned_s42") if name in oof), None)
    challenger = next((name for name in ("catboost_gpu_capacity_ensemble", "catboost_gpu_capacity_s42", "catboost_fast_gpu_s42", "catboost_drift_gpu_s42") if name in oof), None)
    if champion and challenger:
        recipes["blend_drift_lgbm70_catboost30"] = {champion: .7, challenger: .3}
        rank_recipes["rank_drift_lgbm50_catboost50"] = {champion: .5, challenger: .5}
    # Fixed broad blends compare complementary tree architectures. They do not
    # fit weights to test labels or search many near-identical public entries.
    if champion and "xgboost" in oof:
        recipes["blend_lgbm50_xgb50"] = {champion: .5, "xgboost": .5}
        rank_recipes["rank_lgbm50_xgb50"] = {champion: .5, "xgboost": .5}
    if challenger and "xgboost" in oof:
        recipes["blend_catboost50_xgb50"] = {challenger: .5, "xgboost": .5}
        recipes["blend_catboost25_xgb75"] = {challenger: .25, "xgboost": .75}
        recipes["blend_catboost75_xgb25"] = {challenger: .75, "xgboost": .25}
        rank_recipes["rank_catboost50_xgb50"] = {challenger: .5, "xgboost": .5}
        if "xgboost_drift_fast_s42" in oof:
            recipes["blend_catboost50_xgb_mix50"] = {challenger: .5,
                "xgboost": .25, "xgboost_drift_fast_s42": .25}
        if "catboost_gpu_depth9_s42" in oof:
            recipes["blend_catboost_diverse50_xgb50"] = {challenger: .375,
                "catboost_gpu_depth9_s42": .125, "xgboost": .5}
            # Coarse model-family balance checks preserve the same 3:1
            # capacity-ensemble/depth-nine split inside the CatBoost share.
            recipes["blend_catboost_diverse25_xgb75"] = {challenger: .1875,
                "catboost_gpu_depth9_s42": .0625, "xgboost": .75}
            recipes["blend_catboost_diverse75_xgb25"] = {challenger: .5625,
                "catboost_gpu_depth9_s42": .1875, "xgboost": .25}
            rank_recipes["rank_catboost_diverse50_xgb50"] = {challenger: .375,
                "catboost_gpu_depth9_s42": .125, "xgboost": .5}
            if "xgboost_drift_fast_s42" in oof:
                weights = {challenger: .375, "catboost_gpu_depth9_s42": .125,
                           "xgboost": .25, "xgboost_drift_fast_s42": .25}
                recipes["blend_catboost_diverse50_xgb_mix50"] = weights
                rank_recipes["rank_catboost_diverse50_xgb_mix50"] = weights
    if "catboost_gpu_capacity_s42" in oof and "xgboost" in oof:
        # Keep coarse controls anchored to the two original components that
        # produced the observed 0.74009, even as challenger selects new seeds.
        recipes["reference_catboost25_xgb75"] = {"catboost_gpu_capacity_s42": .25, "xgboost": .75}
        recipes["reference_catboost75_xgb25"] = {"catboost_gpu_capacity_s42": .75, "xgboost": .25}
    if champion and challenger and "xgboost" in oof:
        weights = {champion: .5, challenger: .3, "xgboost": .2}
        recipes["blend_lgbm50_catboost30_xgb20"] = weights
        rank_recipes["rank_lgbm50_catboost30_xgb20"] = weights
        recipes["blend_lgbm20_catboost40_xgb40"] = {champion: .2, challenger: .4, "xgboost": .4}
        if "lgbm_drift_bins255_s42" in oof:
            recipes["blend_lgbm35_bins15_catboost30_xgb20"] = {champion: .35,
                "lgbm_drift_bins255_s42": .15, challenger: .3, "xgboost": .2}
        if "catboost_gpu_depth9_s42" in oof:
            recipes["blend_lgbm50_catboost21_depth9_9_xgb20"] = {champion: .5,
                challenger: .21, "catboost_gpu_depth9_s42": .09, "xgboost": .2}
            if "lgbm_drift_bins255_s42" in oof:
                recipes["blend_diverse_challengers"] = {champion: .35,
                    "lgbm_drift_bins255_s42": .15, challenger: .21,
                    "catboost_gpu_depth9_s42": .09, "xgboost": .2}
    recipes.update(rank_recipes)
    masks = {"overall": np.ones(len(meta), dtype=bool),
             "feature_492_upper_tail": folds.stress_feature_492_upper_tail.to_numpy(dtype=bool),
             "feature_1039_observed": folds.stress_feature_1039_observed.to_numpy(dtype=bool),
             "joint_drift_stress": (folds.stress_feature_492_upper_tail.to_numpy(dtype=bool)
                                    & folds.stress_feature_1039_observed.to_numpy(dtype=bool))}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best_observed = max(history, key=lambda entry: entry["public_auc"])
    observed_path = (root / "submissions" / best_observed["filename"]).resolve()
    if not observed_path.is_relative_to((root / "submissions").resolve()) or sha256(observed_path) != best_observed["sha256"]:
        raise RuntimeError("Best observed submission no longer matches its recorded hash")
    observed_column = next(name for name in template.columns if name != "id")
    observed_test = aligned_values(pd.read_csv(observed_path), test_meta.id, observed_column)
    rows, manifest, fold_rows = [], [], []
    y = meta.label.to_numpy()
    for name, weights in recipes.items():
        if not set(weights).issubset(oof):
            continue
        method = "rank_average" if name in rank_recipes else "probability_average"
        blend = rank_blend if name in rank_recipes else probability_blend
        prediction, test_prediction = blend(oof, weights), blend(test, weights)
        path, digest = export_candidate(root, name, test_prediction, test_meta.id, template)
        observed = [entry["public_auc"] for entry in history if entry["sha256"] == digest]
        row = {"candidate": name, "submission": str(path.relative_to(root)),
               "public_auc": observed[-1] if observed else None, "blend_method": method}
        for group, mask in masks.items():
            row[f"{group}_rows"] = int(mask.sum())
            row[f"{group}_oof_auc"] = (float(roc_auc_score(y[mask], prediction[mask]))
                                       if np.unique(y[mask]).size == 2 else None)
        row["default_ap"] = float(average_precision_score(1-y, 1-prediction))
        row["test_rank_correlation_to_public_best"] = float(pd.Series(test_prediction).corr(pd.Series(test["drift_core"]), method="spearman"))
        row["test_rank_correlation_to_best_observed_submission"] = float(pd.Series(test_prediction).corr(pd.Series(observed_test), method="spearman"))
        fold_aucs = []
        for fold in sorted(folds.fold.unique()):
            fold_mask = folds.fold.to_numpy() == fold
            metrics = {"candidate": name, "fold": int(fold), "rows": int(fold_mask.sum())}
            for group, group_mask in masks.items():
                mask = fold_mask & group_mask
                metrics[f"{group}_rows"] = int(mask.sum())
                metrics[f"{group}_oof_auc"] = (float(roc_auc_score(y[mask], prediction[mask]))
                                                if np.unique(y[mask]).size == 2 else None)
            fold_aucs.append(metrics["overall_oof_auc"])
            metrics["overall_auc_delta_to_public_best"] = (metrics["overall_oof_auc"]
                - float(roc_auc_score(y[fold_mask], oof["drift_core"][fold_mask])))
            fold_rows.append(metrics)
        row["fold_auc_mean"] = float(np.mean(fold_aucs))
        row["fold_auc_min"] = float(np.min(fold_aucs))
        row["fold_auc_std"] = float(np.std(fold_aucs))
        rows.append(row)
        manifest.append({**row, "sha256": digest, "weights": weights, "positive_class": 1,
                         "calibrated_probability": False if name in rank_recipes else "not_assessed",
                         "scope": "development OOF; public AUC observed for this exact hash" if observed else "development OOF; public AUC pending submission",
                         "best_observed_submission_sha256": best_observed["sha256"],
                         "source_sha256": {component: sources[component] for component in weights}})
    pd.DataFrame(rows).to_csv(report / "candidate_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(report / "candidate_fold_metrics.csv", index=False)
    json_write(report / "candidate_manifest.json", manifest)
    new_names = ["catboost_gpu_capacity_ensemble", "lgbm_drift_tuned_ensemble", "catboost_gpu_capacity_s42", "catboost_fast_gpu_s42", "catboost_drift_gpu_s42",
                 "lgbm_drift_available_s42", "lgbm_drift_bins255_s42"]
    priorities = ["blend_catboost_diverse50_xgb_mix50", "rank_catboost_diverse50_xgb_mix50",
                  "blend_catboost_diverse50_xgb50", "rank_catboost_diverse50_xgb50",
                  "blend_catboost_diverse25_xgb75", "blend_catboost_diverse75_xgb25",
                  "blend_catboost50_xgb_mix50",
                  "blend_catboost50_xgb50", "rank_catboost50_xgb50",
                  "blend_catboost25_xgb75", "blend_catboost75_xgb25",
                  "reference_catboost25_xgb75", "reference_catboost75_xgb25",
                  "blend_diverse_challengers", "blend_lgbm35_bins15_catboost30_xgb20",
                  "blend_lgbm50_catboost21_depth9_9_xgb20", "blend_lgbm50_catboost30_xgb20",
                  "blend_lgbm20_catboost40_xgb40", "blend_drift_lgbm70_catboost30"] + new_names + ["blend_lgbm50_xgb50",
                              "rank_lgbm50_catboost30_xgb20", "existing_blend_drift50_tuned50",
                              "existing_xgboost", "existing_rank_drift50_xgb50", "existing_availability"]
    queue = [next(row for row in rows if row["candidate"] == name) for name in priorities if any(row["candidate"] == name and row["public_auc"] is None for row in rows)]
    pd.DataFrame(queue, columns=pd.DataFrame(rows).columns).to_csv(report / "submission_queue.csv", index=False)
    json_write(report / "summary.json", {"observed_best_public_auc": max(r["public_auc"] for r in history),
                                          "candidates": len(rows), "all_submissions_uploaded_by_this_script": False})
    print(f"Built {len(rows)} verified candidates; queue: {report / 'submission_queue.csv'}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--record-submission")
    parser.add_argument("--public-score", type=float)
    args = parser.parse_args()
    if (args.record_submission is None) != (args.public_score is None):
        parser.error("Provide both --record-submission and --public-score")
    if args.record_submission:
        record_public_score(args.root, args.record_submission, args.public_score, "用户录入的 Kaggle 公榜结果")
    build(args.root)


if __name__ == "__main__":
    main()

"""Verify saved LightGBM predictions, independent holdouts, and baseline comparison."""

from __future__ import annotations

import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

try:
    from .run_lightgbm_baseline import (
        ARTIFACTS, REPORT, LINEAR_ARTIFACTS, TRAIN, TEST, FOLDS, ROOT,
        comparison_metrics, json_write, predict_rows, split_outer_fit, write_report,
        source_signature,
    )
except ImportError:
    from run_lightgbm_baseline import (
        ARTIFACTS, REPORT, LINEAR_ARTIFACTS, TRAIN, TEST, FOLDS, ROOT,
        comparison_metrics, json_write, predict_rows, split_outer_fit, write_report,
        source_signature,
    )


def verify() -> dict:
    config = json.loads((REPORT / "run_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "completed"
    assert hashlib.sha256(FOLDS.read_bytes()).hexdigest() == config["fold_sha256"]
    assert source_signature(TRAIN) == config["train_source"]
    assert source_signature(TEST) == config["test_source"]
    assert source_signature(LINEAR_ARTIFACTS / "oof_predictions.parquet") == config["reference_oof_source"]
    train_meta = pd.read_parquet(TRAIN, columns=["id", "label"])
    test_meta = pd.read_parquet(TEST, columns=["id"])
    oof = pd.read_parquet(ARTIFACTS / "oof_predictions.parquet")
    test_predictions = pd.read_parquet(ARTIFACTS / "test_predictions.parquet")
    pd.testing.assert_frame_equal(oof[["id", "label"]], train_meta, check_dtype=False)
    pd.testing.assert_frame_equal(test_predictions[["id"]], test_meta)
    assert oof["id"].is_unique and test_predictions["id"].is_unique
    folds = pd.read_parquet(FOLDS).set_index("id").loc[oof["id"]].reset_index()
    pd.testing.assert_frame_equal(oof[list(folds.columns)], folds, check_dtype=False)
    labels = train_meta["label"].to_numpy()
    fold_id = oof["fold"].to_numpy()
    for frame in [oof, test_predictions]:
        assert np.isfinite(frame["p_repay"]).all() and frame["p_repay"].between(0, 1).all()
    values = np.load(LINEAR_ARTIFACTS / "cache/train_features.npy", mmap_mode="r")
    test_values = np.load(LINEAR_ARTIFACTS / "cache/test_features.npy", mmap_mode="r")
    # Validate representative raw cache columns directly against Parquet.
    names = ["feature_1", "feature_492", "feature_1039", "feature_1105"]
    for source, cache in [(TRAIN, values), (TEST, test_values)]:
        raw = pd.read_parquet(source, columns=names).to_numpy(dtype=np.float32)
        for column, name in enumerate(names):
            index = config["feature_names"].index(name)
            np.testing.assert_allclose(cache[:, index], raw[:, column], atol=0, rtol=0, equal_nan=True)
    coverage = np.zeros(len(oof), dtype=np.int8)
    reloaded_test = np.zeros(len(test_meta))
    max_valid_difference = 0.0
    for fold in range(5):
        with np.load(ARTIFACTS / f"inner_splits/fold_{fold}.npz") as saved:
            actual = [saved["inner_fit_index"], saved["early_stop_index"], saved["outer_valid_index"]]
        expected = split_outer_fit(labels, fold_id, fold, config["early_stop_fraction"])
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)
        inner, stop, valid = actual
        assert not len(np.intersect1d(inner, stop))
        assert not len(np.intersect1d(inner, valid))
        assert not len(np.intersect1d(stop, valid))
        np.testing.assert_array_equal(np.sort(np.r_[inner, stop]), np.flatnonzero(fold_id != fold))
        metadata = json.loads((ARTIFACTS / f"models/fold_{fold}.json").read_text(encoding="utf-8"))
        path = ARTIFACTS / f"models/fold_{fold}.txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["model_sha256"]
        assert metadata["training_signature"] == config["training_signature"]
        assert metadata["features"] == config["feature_names"] and metadata["positive_class"] == 1
        record = metadata["record"]
        history = json.loads((REPORT / f"learning_curves/fold_{fold}.json").read_text(encoding="utf-8"))
        curve = history["inner_early_stop"]["auc"]
        selected = record["selected_rounds"]
        assert 1 <= selected <= len(curve) <= config["max_rounds"]
        np.testing.assert_allclose(curve[selected - 1], record["inner_early_stop_auc"], atol=1e-12)
        assert record["outer_fit_rows"] == len(inner) + len(stop)
        model = lgb.Booster(model_file=str(path))
        assert model.feature_name() == config["feature_names"]
        assert model.current_iteration() == record["refit_rounds"]
        p_valid = predict_rows(model, values, config["batch_rows"], valid)
        np.testing.assert_allclose(p_valid, oof.loc[valid, "p_repay"], atol=1e-12)
        max_valid_difference = max(max_valid_difference, float(np.max(np.abs(p_valid - oof.loc[valid, "p_repay"]))))
        coverage[valid] += 1
        reloaded_test += predict_rows(model, test_values, config["batch_rows"]) / 5
        print(f"Verified saved fold {fold + 1}/5: split isolation, learning curve, OOF and test inference", flush=True)
    assert (coverage == 1).all()
    np.testing.assert_allclose(reloaded_test, test_predictions["p_repay"], atol=1e-12)
    comparison = pd.read_csv(REPORT / "comparison_metrics.csv")
    recomputed = comparison_metrics(train_meta, folds, oof["p_repay"].to_numpy())
    pd.testing.assert_frame_equal(comparison, recomputed, check_dtype=False, atol=1e-12, rtol=1e-12)
    template = pd.read_parquet(ROOT / "data/processed/sample_submission.parquet")
    csv = pd.read_csv(ROOT / "submissions/lightgbm_baseline.csv")
    assert list(csv.columns) == ["id", "prediction"]
    pd.testing.assert_series_equal(csv["id"], template["id"], check_names=False)
    expected_probabilities = test_predictions.set_index("id").loc[template["id"], "p_repay"]
    np.testing.assert_allclose(csv["prediction"], expected_probabilities, atol=1e-12)
    importance = pd.read_csv(REPORT / "feature_importance.csv")
    assert set(importance["feature"]) == set(config["feature_names"])
    assert importance["feature"].is_unique
    np.testing.assert_allclose(importance["mean_gain_share"].sum(), 1, atol=1e-10)
    result = {"status": "passed", "training_signature": config["training_signature"],
              "train_rows": len(oof), "test_rows": len(test_predictions), "fold_file_unchanged": True,
              "inner_splits_reproduced_and_disjoint_from_outer_validation": True,
              "binning_raw_cache_checks_passed": True, "best_rounds_match_inner_learning_curves": True,
              "saved_models_reproduce_oof": True, "saved_models_reproduce_test_ensemble": True,
              "max_oof_probability_difference": max_valid_difference,
              "max_test_probability_difference": float(np.max(np.abs(reloaded_test - test_predictions["p_repay"]))),
              "all_comparison_metrics_recomputed": True, "submission_ids_and_probabilities_match": True,
              "feature_importance_has_exact_whitelist_and_normalized_gain": True,
              "oof_coverage_min": int(coverage.min()), "oof_coverage_max": int(coverage.max()),
              "test_probability_quantiles_0_1_50_99_100_percent": test_predictions["p_repay"].quantile([0, .01, .5, .99, 1]).tolist()}
    json_write(REPORT / "artifact_verification.json", result)
    write_report(comparison, pd.read_csv(REPORT / "fold_metrics.csv"), importance, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        verify()

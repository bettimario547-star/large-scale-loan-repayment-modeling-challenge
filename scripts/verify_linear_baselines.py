"""Audit completed OOF, saved-model predictions, metrics, and submission joins."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

try:
    from .run_linear_baselines import ARTIFACTS, FOLDS, REPORT, ROOT, TEST, TRAIN, json_write, metrics, predict
except ImportError:
    from run_linear_baselines import ARTIFACTS, FOLDS, REPORT, ROOT, TEST, TRAIN, json_write, metrics, predict


def verify() -> dict:
    config = json.loads((REPORT / "run_config.json").read_text(encoding="utf-8"))
    if config["status"] != "completed":
        raise AssertionError("Training has not completed")
    assert hashlib.sha256(FOLDS.read_bytes()).hexdigest() == config["fold_sha256"]
    train = pd.read_parquet(TRAIN, columns=["id", "label"])
    oof = pd.read_parquet(ARTIFACTS / "oof_predictions.parquet")
    folds = pd.read_parquet(FOLDS)
    test = pd.read_parquet(TEST, columns=["id"])
    predictions = pd.read_parquet(ARTIFACTS / "test_predictions.parquet")
    template = pd.read_parquet(ROOT / "data/processed/sample_submission.parquet")
    pd.testing.assert_frame_equal(oof[["id", "label"]], train, check_dtype=False)
    aligned = folds.set_index("id").loc[oof["id"]].reset_index()
    pd.testing.assert_frame_equal(oof[folds.columns], aligned, check_dtype=False)
    pd.testing.assert_frame_equal(predictions[["id"]], test)
    assert oof["id"].is_unique and predictions["id"].is_unique
    # Check representative cache columns against the actual Parquet sources,
    # including the two drift-group columns and the ends of the feature schema.
    check_names = ["feature_1", "feature_492", "feature_1039", "feature_1105"]
    check_indices = [config["feature_names"].index(name) for name in check_names]
    for source in [TRAIN, TEST]:
        raw = pd.read_parquet(source, columns=check_names).to_numpy(dtype=np.float32)
        cached = np.load(ARTIFACTS / f"cache/{source.stem}_features.npy", mmap_mode="r")
        for column, index in enumerate(check_indices):
            np.testing.assert_allclose(cached[:, index], raw[:, column], atol=0, rtol=0, equal_nan=True)
        del cached, raw
    y = oof["label"].to_numpy()
    for col in ["p_repay_constant", "p_repay_logistic"]:
        for frame in [oof, predictions]:
            p = frame[col].to_numpy()
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    for fold in range(5):
        valid = oof["fold"].eq(fold).to_numpy()
        np.testing.assert_allclose(oof.loc[valid, "p_repay_constant"], y[~valid].mean(), atol=1e-12)
    summary = pd.read_csv(REPORT / "summary_metrics.csv")
    masks = {"overall": np.ones(len(y), dtype=bool),
             "feature_492_upper_tail": oof["stress_feature_492_upper_tail"].to_numpy(),
             "feature_1039_observed": oof["stress_feature_1039_observed"].to_numpy()}
    for row in summary.to_dict(orient="records"):
        col = "p_repay_constant" if row["model"] == "constant" else "p_repay_logistic"
        mask = masks[row["group"]]
        recomputed = metrics(y[mask], oof[col].to_numpy()[mask])
        for name, value in recomputed.items():
            np.testing.assert_allclose(row[name], value, atol=1e-12)
    # Independently reload each saved model and reproduce its ensemble output.
    test_values = np.load(ARTIFACTS / "cache/test_features.npy", mmap_mode="r")
    ensemble = np.zeros(len(test))
    for fold in range(5):
        info = json.loads((ARTIFACTS / f"models/fold_{fold}.json").read_text(encoding="utf-8"))
        assert info["features"] == config["feature_names"] and info["classes"] == [0, 1]
        assert info["optimizer"]["converged"]
        with np.load(ARTIFACTS / f"models/fold_{fold}.npz") as model:
            assert len(model["coef"]) == 1105
            prep = {"mean": model["mean"], "scale": model["scale"]}
            assert np.isfinite(prep["mean"]).all() and (prep["scale"] > 0).all()
            theta = np.r_[model["coef"], model["intercept"]]
            assert int(model["fit_rows"]) == int(oof["fold"].ne(fold).sum())
            ensemble += predict(test_values, theta, prep, config["batch_rows"]) / 5
    np.testing.assert_allclose(ensemble, predictions["p_repay_logistic"], atol=1e-12)
    np.testing.assert_allclose(predictions["p_repay_constant"], y.mean(), atol=1e-12)
    indexed = predictions.set_index("id").loc[template["id"]]
    for model, col in [("constant", "p_repay_constant"), ("logistic_regression", "p_repay_logistic")]:
        csv = pd.read_csv(ROOT / f"submissions/{model}_baseline.csv")
        assert list(csv.columns) == ["id", "prediction"]
        pd.testing.assert_series_equal(csv["id"], template["id"], check_names=False)
        np.testing.assert_allclose(csv["prediction"], indexed[col], atol=1e-12)
    result = {"status": "passed", "train_rows": len(train), "test_rows": len(test),
              "fold_file_unchanged": True, "oof_ids_labels_and_folds_match": True,
              "representative_cache_columns_match_parquet_sources": True,
              "constant_probabilities_use_fit_only_labels": True, "summary_metrics_recomputed": True,
              "saved_models_all_converged": True, "saved_models_reproduce_test_predictions": True,
              "maximum_reloaded_test_probability_difference": float(np.max(np.abs(ensemble - predictions["p_repay_logistic"]))),
              "both_submissions_match_template_ids_and_saved_probabilities": True,
              "test_probability_quantiles": {col: predictions[col].quantile([0, 0.01, 0.5, 0.99, 1]).tolist()
                                             for col in ["p_repay_constant", "p_repay_logistic"]}}
    json_write(REPORT / "artifact_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        verify()

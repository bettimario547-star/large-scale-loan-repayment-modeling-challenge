"""Replay fold-local feature plans and saved-model predictions for every ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

try:
    from . import run_feature_screening as screening
except ImportError:
    import run_feature_screening as screening


def verify():
    base = screening.base
    report, artifacts = screening.REPORT, screening.ARTIFACTS
    config = json.loads((report / "run_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "completed"
    signing = {key: value for key, value in config.items()
               if key not in {"status", "training_signature", "elapsed_seconds", "completed_variant_folds"}}
    assert hashlib.sha256(json.dumps(signing, sort_keys=True).encode()).hexdigest() == config["training_signature"]
    assert config["fold_sha256"] == screening.sha256(base.FOLDS)
    assert config["train_source"] == base.source_signature(base.TRAIN)
    assert config["test_source"] == base.source_signature(base.TEST)
    assert config["baseline_oof_sha256"] == screening.sha256(base.ARTIFACTS / "oof_predictions.parquet")
    assert config["baseline_config_sha256"] == screening.sha256(base.REPORT / "run_config.json")
    for fold, checksum in enumerate(config["baseline_model_checksums"]):
        assert screening.sha256(base.ARTIFACTS / f"models/fold_{fold}.txt") == checksum
    for name, checksum in config["code_sha256"].items():
        assert screening.sha256(screening.ROOT / "scripts" / name) == checksum
    context = screening.load_context(config["batch_rows"])
    _, meta, test_meta, folds, labels, fold_id, names, values, test_values, baseline_oof = context
    expected_profile = screening.profile_all_folds(values, labels, fold_id, names)
    saved_profile = pd.read_csv(report / "fold_feature_statistics.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(saved_profile, expected_profile, check_dtype=False, atol=1e-14, rtol=1e-14)
    marker = json.loads((report / "profile_signature.json").read_text())
    assert marker["profile_sha256"] == screening.sha256(report / "fold_feature_statistics.csv")
    assert marker["signature"] == config["training_signature"]
    saved_plans = json.loads((report / "feature_plans.json").read_text(encoding="utf-8"))
    assert len(saved_plans) == len(screening.VARIANTS) * 5
    baseline_metrics = pd.read_csv(base.REPORT / "fold_metrics.csv").set_index("fold")
    comparison = [{"variant": "baseline", **item} for item in screening.group_metrics(labels, folds, baseline_oof.p_repay.to_numpy())]
    records, audit_records = [], []
    max_oof_difference = max_test_difference = 0.0
    template = pd.read_parquet(screening.ROOT / "data/processed/sample_submission.parquet")
    for variant in screening.VARIANTS:
        output = artifacts / variant
        oof = pd.read_parquet(output / "oof_predictions.parquet")
        test = pd.read_parquet(output / "test_predictions.parquet")
        pd.testing.assert_frame_equal(oof[["id", "label"]], meta, check_dtype=False)
        pd.testing.assert_frame_equal(oof[list(folds.columns)], folds, check_dtype=False)
        pd.testing.assert_frame_equal(test[["id"]], test_meta)
        predicted_oof, test_sum = np.full(len(labels), np.nan), np.zeros(len(test_meta))
        coverage = np.zeros(len(labels), dtype=np.int8)
        for fold in range(5):
            plan = screening.fit_plan(expected_profile.loc[expected_profile.fold == fold], names, variant)
            plan["fold"] = fold
            matches = [item for item in saved_plans if item["variant"] == variant and item["fold"] == fold]
            assert len(matches) == 1 and matches[0] == plan
            metadata = json.loads((output / f"models/fold_{fold}.json").read_text(encoding="utf-8"))
            assert metadata["plan"] == plan and metadata["positive_class"] == 1
            assert metadata["training_signature"] == config["training_signature"]
            assert metadata["params"] == base.model_params(fold, config["threads"])
            model_path = Path(metadata["model_path"])
            expected_path = (base.ARTIFACTS if plan["identity"] else output) / f"models/fold_{fold}.txt"
            assert model_path.resolve() == expected_path.resolve()
            assert screening.sha256(model_path) == metadata["model_sha256"]
            if plan["identity"]:
                assert metadata["model_sha256"] == config["baseline_model_checksums"][fold]
            model = lgb.Booster(model_file=str(model_path))
            assert model.feature_name() == plan["output_features"]
            assert model.current_iteration() == config["fixed_rounds"][fold] == int(baseline_metrics.loc[fold, "refit_rounds"])
            valid = np.flatnonzero(fold_id == fold)
            p_valid = base.predict_rows(model, screening.FeatureView(values, plan), config["batch_rows"], valid)
            p_test = base.predict_rows(model, screening.FeatureView(test_values, plan), config["batch_rows"])
            with np.load(output / f"fold_predictions/fold_{fold}.npz") as checkpoint:
                np.testing.assert_array_equal(checkpoint["outer_valid_index"], valid)
                np.testing.assert_allclose(p_valid, checkpoint["p_repay_valid"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(p_test, checkpoint["p_repay_test"], atol=1e-12, rtol=0)
            predicted_oof[valid] = p_valid
            coverage[valid] += 1
            test_sum += p_test / 5
            recomputed = base.metrics(labels[valid], p_valid)
            record = metadata["record"]
            for metric, score in recomputed.items():
                np.testing.assert_allclose(score, record[metric], atol=1e-12, rtol=0)
            assert record["fit_rows"] == int((fold_id != fold).sum()) == plan["fit_rows"]
            assert record["output_features"] == len(plan["output_features"])
            records.append(record)
            if not plan["identity"]:
                importance = pd.read_csv(report / f"importance/{variant}_fold_{fold}.csv")
                assert importance["feature"].tolist() == plan["output_features"]
                np.testing.assert_allclose(importance.gain_share.sum(), 1, atol=1e-12)
            del model
            print(f"Verified {variant} fold {fold}: plan, tree count, model, OOF, test", flush=True)
        assert (coverage == 1).all()
        np.testing.assert_allclose(predicted_oof, oof.p_repay, atol=1e-12, rtol=0)
        np.testing.assert_allclose(test_sum, test.p_repay, atol=1e-12, rtol=0)
        if all(item["identity"] for item in saved_plans if item["variant"] == variant):
            np.testing.assert_array_equal(oof.p_repay, baseline_oof.p_repay)
        submission = pd.read_csv(screening.ROOT / f"submissions/feature_screening_{variant}.csv")
        assert submission.columns.tolist() == ["id", "prediction"]
        pd.testing.assert_series_equal(submission.id, template.id, check_names=False)
        expected = test.set_index("id").loc[template.id, "p_repay"].to_numpy()
        np.testing.assert_allclose(submission.prediction, expected, atol=1e-12, rtol=0)
        difference_oof = float(np.max(np.abs(predicted_oof - oof.p_repay.to_numpy())))
        difference_test = float(np.max(np.abs(test_sum - test.p_repay.to_numpy())))
        max_oof_difference, max_test_difference = max(max_oof_difference, difference_oof), max(max_test_difference, difference_test)
        comparison.extend({"variant": variant, **item} for item in screening.group_metrics(labels, folds, predicted_oof))
        audit_records.append({"variant": variant, "models_verified": 5, "oof_coverage_exactly_once": True,
                              "max_oof_difference": difference_oof, "max_test_difference": difference_test,
                              "submission_aligned": True})
    frame = pd.DataFrame(records)
    pd.testing.assert_frame_equal(pd.read_csv(report / "fold_metrics.csv"), frame, check_dtype=False, atol=1e-12, rtol=1e-12)
    table = pd.DataFrame(comparison)
    pd.testing.assert_frame_equal(pd.read_csv(report / "comparison_metrics.csv"), table, check_dtype=False, atol=1e-12, rtol=1e-12)
    paired = pd.read_csv(report / "paired_fold_deltas.csv")
    for metric in ["repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score"]:
        expected = paired[metric] - paired["fold"].map(baseline_metrics[metric])
        np.testing.assert_allclose(expected, paired[f"delta_{metric}"], atol=1e-12, rtol=0)
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    best = table.query("group == 'overall'").sort_values("repay_roc_auc", ascending=False, kind="stable").iloc[0]
    assert summary["best_development_variant_by_auc"] == best["variant"]
    np.testing.assert_allclose(summary["best_development_auc"], best["repay_roc_auc"], atol=1e-12)
    result = {"status": "passed", "training_signature": config["training_signature"],
              "raw_inputs_and_folds_unchanged": True, "full_fold_profiles_recomputed": True,
              "all_feature_plans_refitted_and_reproduced": True, "fixed_tree_counts_match_baseline": True,
              "outer_validation_not_used_in_rule_fit": True,
              "saved_models_reproduce_all_predictions": True, "all_metrics_recomputed": True,
              "all_submissions_aligned": True, "max_oof_difference": max_oof_difference,
              "max_test_difference": max_test_difference, "variants": audit_records}
    base.json_write(report / "artifact_verification.json", result)
    screening.write_report(table, frame, saved_plans, config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        verify()

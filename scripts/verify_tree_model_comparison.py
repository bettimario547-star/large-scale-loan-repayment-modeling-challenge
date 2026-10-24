"""Replay inner stopping and reload all 20 native models and their predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

try:
    from . import run_tree_model_comparison as study
except ImportError:
    import run_tree_model_comparison as study


def verify():
    config = study.checked_config()
    if config["status"] != "completed":
        raise RuntimeError("All five folds of both engines must be collected first")
    experiment, signature = config["experiment"], config["training_signature"]
    meta, test_meta, folds, labels, fold_id, names, values, test_values, references = study.load_context(experiment["batch_rows"])
    saved_oof = pd.read_parquet(study.ARTIFACTS / "oof_predictions.parquet")
    saved_test = pd.read_parquet(study.ARTIFACTS / "test_predictions.parquet")
    pd.testing.assert_frame_equal(saved_oof[["id", "label"]], meta, check_dtype=False)
    pd.testing.assert_frame_equal(saved_oof[list(folds.columns)], folds, check_dtype=False)
    pd.testing.assert_frame_equal(saved_test[["id"]], test_meta)
    template = pd.read_parquet(study.ROOT / "data/processed/sample_submission.parquet")
    search, records, scores, audit = [], [], dict(references), []
    maximum_inner = maximum_oof = maximum_test = 0.0
    border_count = 0
    for engine in study.ENGINES:
        score, test_score = np.full(len(labels), np.nan), np.zeros(len(test_meta))
        coverage = np.zeros(len(labels), dtype=np.int8)
        for fold in range(5):
            inner, stop, valid = study.base.split_outer_fit(labels, fold_id, fold, 0.15)
            with np.load(study.ARTIFACTS / "inner_splits" / f"fold_{fold}.npz") as saved:
                for key, rows in (("inner_fit_index", inner), ("early_stop_index", stop), ("outer_valid_index", valid)):
                    np.testing.assert_array_equal(saved[key], rows)
            assert len(np.intersect1d(inner, stop)) == len(np.intersect1d(inner, valid)) == len(np.intersect1d(stop, valid)) == 0
            np.testing.assert_array_equal(np.sort(np.r_[inner, stop]), np.flatnonzero(fold_id != fold))
            inner_path = study.ARTIFACTS / "inner_predictions" / f"{engine}_fold_{fold}.npz"
            path, metadata_path = study.model_paths(engine, fold, "inner")
            selection = study.checkpoint(metadata_path, inner_path, signature)
            assert selection is not None
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert metadata["params"] == study.params_for(engine, fold, experiment["threads"])
            assert metadata["features"] == names and metadata["positive_class"] == 1
            assert Path(metadata["model_path"]).resolve() == path.resolve()
            curve = json.loads((study.REPORT / "inner_curves" / f"{engine}_fold_{fold}.json").read_text())["auc"]
            state = study.AUCStop(experiment["patience"], experiment["min_delta"])
            for index, value in enumerate(curve):
                keep = state.update(value)
                if not keep:
                    assert index == len(curve) - 1
            assert state.best_round == selection["selected_rounds"]
            assert len(curve) == selection["evaluated_rounds"] <= experiment["max_rounds"]
            assert selection["hit_round_cap"] == (len(curve) >= experiment["max_rounds"])
            assert selection["inner_fit_rows"] == len(inner) and selection["early_stop_rows"] == len(stop)
            model = study.load_model(engine, path)
            if engine == "xgboost":
                model.set_param({"nthread": experiment["threads"]})
                assert model.feature_names == names
                assert model.num_features() == len(names)
                assert int(model.best_iteration) + 1 == state.best_round
            else:
                assert model.feature_names_ == names
                params = model.get_all_params()
                for key in ("depth", "boosting_type", "grow_policy", "random_seed", "bootstrap_type", "nan_mode", "leaf_estimation_iterations"):
                    assert params[key] == metadata["params"][key], (key, params[key], metadata["params"][key])
                # File-based Pools can store numeric labels as strings.
                assert np.asarray(model.classes_, dtype=np.float64).tolist() == [0.0, 1.0]
                train_borders = study.ARTIFACTS / "borders" / f"catboost_inner_fold_{fold}.tsv"
                stop_borders = study.ARTIFACTS / "borders" / f"catboost_stop_fold_{fold}.tsv"
                assert train_borders.read_bytes() == stop_borders.read_bytes()
                border_count += len(metadata["border_checksums"])
            assert study.model_rounds(engine, model) == state.best_round
            p_stop = study.predict_rows(engine, model, values, names, experiment["batch_rows"], stop)
            with np.load(inner_path) as saved:
                np.testing.assert_array_equal(saved["early_stop_index"], stop)
                np.testing.assert_allclose(p_stop, saved["p_repay"], atol=1e-12, rtol=0)
                maximum_inner = max(maximum_inner, float(np.max(np.abs(p_stop - saved["p_repay"]))))
            np.testing.assert_allclose(roc_auc_score(labels[stop], p_stop), selection["inner_auc"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(state.best_auc, selection["inner_auc"], atol=2e-7, rtol=0)
            search.append(selection)
            del model
            path, metadata_path = study.model_paths(engine, fold, "refit")
            predictions_path = study.ARTIFACTS / "fold_predictions" / f"{engine}_fold_{fold}.npz"
            record = study.checkpoint(metadata_path, predictions_path, signature)
            assert record is not None
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert metadata["params"] == study.params_for(engine, fold, experiment["threads"])
            assert metadata["features"] == names and metadata["positive_class"] == 1
            model = study.load_model(engine, path)
            if engine == "xgboost":
                model.set_param({"nthread": experiment["threads"]})
                assert model.feature_names == names
            else:
                assert model.feature_names_ == names
                assert np.asarray(model.classes_, dtype=np.float64).tolist() == [0.0, 1.0]
                border_count += len(metadata["border_checksums"])
            assert study.model_rounds(engine, model) == record["refit_rounds"] == selection["selected_rounds"]
            assert record["fit_rows"] == int((fold_id != fold).sum())
            p_valid = study.predict_rows(engine, model, values, names, experiment["batch_rows"], valid)
            p_test = study.predict_rows(engine, model, test_values, names, experiment["batch_rows"])
            with np.load(predictions_path) as saved:
                np.testing.assert_array_equal(saved["outer_valid_index"], valid)
                np.testing.assert_allclose(p_valid, saved["p_repay_valid"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(p_test, saved["p_repay_test"], atol=1e-12, rtol=0)
            for key, result in study.base.metrics(labels[valid], p_valid).items():
                np.testing.assert_allclose(result, record[key], atol=1e-12, rtol=0)
            score[valid], coverage[valid] = p_valid, coverage[valid] + 1
            test_score += p_test / 5
            records.append(record)
            del model
            print(f"Verified {engine} fold {fold}: inner stopping, {selection['selected_rounds']} trees, OOF/test", flush=True)
        assert (coverage == 1).all() and np.isfinite(score).all()
        np.testing.assert_allclose(score, saved_oof[f"p_repay_{engine}"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(test_score, saved_test[f"p_repay_{engine}"], atol=1e-12, rtol=0)
        maximum_oof = max(maximum_oof, float(np.max(np.abs(score - saved_oof[f"p_repay_{engine}"]))))
        maximum_test = max(maximum_test, float(np.max(np.abs(test_score - saved_test[f"p_repay_{engine}"]))))
        scores[engine] = score
        submission = pd.read_csv(study.ROOT / "submissions" / f"{engine}_baseline.csv")
        assert submission.columns.tolist() == ["id", "prediction"] and len(submission) == len(test_meta)
        pd.testing.assert_series_equal(submission.id, template.id, check_names=False)
        aligned = pd.Series(test_score, index=test_meta.id).loc[template.id].to_numpy()
        np.testing.assert_allclose(submission.prediction, aligned, atol=1e-12, rtol=0)
        audit.append({"engine": engine, "inner_models": 5, "refit_models": 5, "oof_coverage_exactly_once": True})
    comparison = pd.DataFrame([{"model": model, **row} for model, p in scores.items() for row in study.group_metrics(labels, folds, p)])
    paired = []
    for fold in range(5):
        valid = fold_id == fold
        single = study.base.metrics(labels[valid], references["lightgbm_tuned_seed_42"][valid])
        ensemble = study.base.metrics(labels[valid], references["lightgbm_tuned_ensemble"][valid])
        for model, p in scores.items():
            row = study.base.metrics(labels[valid], p[valid])
            paired.append({"model": model, "fold": fold, **row,
                           **{f"delta_single_{key}": row[key] - single[key] for key in study.METRICS},
                           **{f"delta_ensemble_{key}": row[key] - ensemble[key] for key in study.METRICS}})
    for filename, expected in (("comparison_metrics.csv", comparison), ("paired_fold_metrics.csv", pd.DataFrame(paired)),
                               ("selection_records.csv", pd.DataFrame(search)), ("refit_records.csv", pd.DataFrame(records))):
        pd.testing.assert_frame_equal(pd.read_csv(study.REPORT / filename), expected, check_dtype=False, atol=1e-12, rtol=1e-12)
    correlations = pd.DataFrame(scores).corr()
    pd.testing.assert_frame_equal(pd.read_csv(study.REPORT / "prediction_correlations.csv", index_col=0), correlations, atol=1e-12, rtol=1e-12)
    summary = json.loads((study.REPORT / "summary.json").read_text())
    assert summary["training_signature"] == signature and summary["official_test_score"] is None
    assert summary["submissions_uploaded"] is False and summary["no_cross_engine_blend"] is True
    pd.testing.assert_frame_equal(pd.DataFrame(summary["overall"]), comparison.query("group == 'overall'").reset_index(drop=True), check_dtype=False)
    incumbent = comparison.query("group == 'overall' and model == 'lightgbm_tuned_ensemble'").iloc[0]
    eligible = comparison.query("group == 'overall' and model in @study.ENGINES")
    eligible = eligible.loc[(eligible.repay_roc_auc - incumbent.repay_roc_auc >= .0005) &
                            (eligible.default_pr_auc_ap - incumbent.default_pr_auc_ap >= -.001) &
                            (eligible.log_loss - incumbent.log_loss <= .0005)]
    expected_model = eligible.sort_values("repay_roc_auc", ascending=False).iloc[0].model if len(eligible) else "lightgbm_tuned_ensemble"
    assert summary["recommended_development_model"] == expected_model
    for engine in study.ENGINES:
        assert summary["selected_rounds"][engine] == [r["selected_rounds"] for r in search if r["engine"] == engine]
        assert summary["hit_round_cap"][engine] == sum(r["hit_round_cap"] for r in search if r["engine"] == engine)
        assert summary["folds_above_tuned_single_auc"][engine] == sum(r["delta_single_repay_roc_auc"] > 0 for r in paired if r["model"] == engine)
    np.testing.assert_allclose(summary["model_stage_seconds_sum"], sum(r["elapsed_seconds"] for r in search + records), atol=1e-8)
    result = {"status": "passed", "training_signature": signature, "engines": audit,
              "inner_models_reloaded": 10, "refit_models_reloaded": 10,
              "all_splits_reproduced_and_disjoint": True, "all_stopping_rules_replayed": True,
              "all_1105_numeric_features_and_native_missing_preserved": True,
              "all_model_prediction_and_reference_checksums_verified": True,
              "catboost_border_files_verified": border_count, "catboost_stop_borders_equal_training": True,
              "all_metrics_correlations_and_recommendation_recomputed": True,
              "both_submissions_aligned": True,
              "max_inner_prediction_difference": maximum_inner, "max_oof_difference": maximum_oof,
              "max_test_difference": maximum_test}
    study.base.json_write(study.REPORT / "artifact_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        verify()

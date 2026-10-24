"""Reload all 35 models and independently audit the nested search / ensembles."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

try:
    from . import run_hyperparameter_seed_optimization as tuning
except ImportError:
    import run_hyperparameter_seed_optimization as tuning


def early_stop_choice(history, patience, min_delta):
    """Replay the single-AUC improvement rule, including the final-round cap."""
    best_value, best_round = -np.inf, 0
    for round_number, score in enumerate(history, 1):
        if score > best_value + min_delta:
            best_value, best_round = score, round_number
        if round_number >= best_round + patience:
            assert round_number == len(history)
            break
    return best_round, best_value


def verify():
    config = tuning.checked_config()
    assert config["status"] == "completed"
    experiment, signature = config["experiment"], config["training_signature"]
    report, artifacts = tuning.REPORT, tuning.ARTIFACTS
    context = tuning.load_context(experiment["batch_rows"])
    meta, test_meta, folds, labels, fold_id, names, values, test_values, baseline = context
    batch = experiment["batch_rows"]
    search, chosen, records, audit, reference_curves = [], [], [], [], []
    max_inner = max_oof = max_test = 0.0
    for fold in range(5):
        inner, stop, valid = tuning.base.split_outer_fit(labels, fold_id, fold, experiment["early_stop_fraction"])
        with np.load(artifacts / f"inner_splits/fold_{fold}.npz") as saved:
            for key, expected in (("inner_fit_index", inner), ("early_stop_index", stop), ("outer_valid_index", valid)):
                np.testing.assert_array_equal(saved[key], expected)
        assert len(np.intersect1d(inner, stop)) == len(np.intersect1d(inner, valid)) == len(np.intersect1d(stop, valid)) == 0
        np.testing.assert_array_equal(np.sort(np.r_[inner, stop]), np.flatnonzero(fold_id != fold))
        fold_records = []
        for candidate in tuning.CANDIDATES:
            tag = f"{candidate['name']}_fold_{fold}"
            metadata_path = artifacts / f"inner_models/{tag}.json"
            prediction_path = artifacts / f"inner_predictions/{tag}.npz"
            record = tuning.checkpoint_record(metadata_path, prediction_path, signature)
            assert record is not None
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert metadata["features"] == names and metadata["positive_class"] == 1
            assert metadata["params"] == tuning.params_for(candidate, fold, 42, experiment["threads"])
            assert Path(metadata["model_path"]).resolve() == (artifacts / f"inner_models/{tag}.txt").resolve()
            history = json.loads((report / f"inner_curves/{tag}.json").read_text())["inner_early_stop"]["auc"]
            if candidate["name"] == "baseline_extended":
                previous_curve = json.loads((tuning.base.REPORT / f"learning_curves/fold_{fold}.json").read_text())["inner_early_stop"]["auc"]
                common = min(len(previous_curve), len(history))
                reference_curves.append({"fold": fold, "common_rounds": common,
                                         "max_auc_difference": float(np.max(np.abs(np.asarray(previous_curve[:common]) - history[:common])))})
            expected_round, expected_auc = early_stop_choice(history, experiment["patience"], experiment["min_delta"])
            assert expected_round == record["selected_rounds"]
            np.testing.assert_allclose(expected_auc, record["inner_auc"], atol=1e-12, rtol=0)
            assert len(history) == record["evaluated_rounds"] <= experiment["max_rounds"]
            assert record["hit_round_cap"] == (len(history) >= experiment["max_rounds"])
            assert record["inner_fit_rows"] == len(inner) and record["early_stop_rows"] == len(stop)
            model = lgb.Booster(model_file=metadata["model_path"])
            assert model.current_iteration() == expected_round and model.feature_name() == names
            p_stop = tuning.base.predict_rows(model, values, batch, stop)
            with np.load(prediction_path) as checkpoint:
                np.testing.assert_array_equal(checkpoint["early_stop_index"], stop)
                np.testing.assert_allclose(p_stop, checkpoint["p_repay"], atol=1e-12, rtol=0)
                max_inner = max(max_inner, float(np.max(np.abs(p_stop - checkpoint["p_repay"]))))
            np.testing.assert_allclose(roc_auc_score(labels[stop], p_stop), record["inner_auc"], atol=1e-12, rtol=0)
            fold_records.append(record)
            del model
            print(f"Verified inner {tag}: model, stopping rule, predictions, AUC", flush=True)
        selection = json.loads((report / f"selections/fold_{fold}.json").read_text())
        assert selection["training_signature"] == signature and selection["records"] == fold_records
        expected = tuning.choose_candidate(fold_records, experiment["tie_tolerance"])
        assert selection["chosen"] == expected
        search.extend(fold_records)
        chosen.append(expected)
    stored_oof = pd.read_parquet(artifacts / "oof_predictions.parquet")
    stored_test = pd.read_parquet(artifacts / "test_predictions.parquet")
    pd.testing.assert_frame_equal(stored_oof[["id", "label"]], meta, check_dtype=False)
    pd.testing.assert_frame_equal(stored_oof[list(folds.columns)], folds, check_dtype=False)
    pd.testing.assert_frame_equal(stored_test[["id"]], test_meta)
    scores, test_scores = [], []
    template = pd.read_parquet(tuning.ROOT / "data/processed/sample_submission.parquet")
    for seed in tuning.SEEDS:
        score, test_sum = np.full(len(labels), np.nan), np.zeros(len(test_meta))
        coverage = np.zeros(len(labels), dtype=np.int8)
        for fold in range(5):
            tag = f"seed_{seed}_fold_{fold}"
            metadata_path = artifacts / f"models/{tag}.json"
            prediction_path = artifacts / f"fold_predictions/{tag}.npz"
            record = tuning.checkpoint_record(metadata_path, prediction_path, signature)
            assert record is not None
            metadata = json.loads(metadata_path.read_text())
            candidate = next(item for item in tuning.CANDIDATES if item["name"] == chosen[fold]["candidate"])
            assert metadata["params"] == tuning.params_for(candidate, fold, seed, experiment["threads"])
            assert metadata["features"] == names and metadata["positive_class"] == 1
            assert Path(metadata["model_path"]).resolve() == (artifacts / f"models/{tag}.txt").resolve()
            assert record["candidate"] == candidate["name"] and record["seed"] == seed and record["fold"] == fold
            assert record["fit_rows"] == int((fold_id != fold).sum())
            model = lgb.Booster(model_file=metadata["model_path"])
            assert model.current_iteration() == record["refit_rounds"] == chosen[fold]["selected_rounds"]
            assert model.feature_name() == names
            valid = np.flatnonzero(fold_id == fold)
            p_valid = tuning.base.predict_rows(model, values, batch, valid)
            p_test = tuning.base.predict_rows(model, test_values, batch)
            with np.load(prediction_path) as checkpoint:
                np.testing.assert_array_equal(checkpoint["outer_valid_index"], valid)
                np.testing.assert_allclose(p_valid, checkpoint["p_repay_valid"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(p_test, checkpoint["p_repay_test"], atol=1e-12, rtol=0)
            score[valid], coverage[valid] = p_valid, coverage[valid] + 1
            test_sum += p_test / 5
            for name, result in tuning.base.metrics(labels[valid], p_valid).items():
                np.testing.assert_allclose(result, record[name], atol=1e-12, rtol=0)
            records.append(record)
            del model
            print(f"Verified final {tag}: selected parameters, fixed rounds, OOF and test", flush=True)
        assert (coverage == 1).all() and np.isfinite(score).all()
        column = f"p_repay_seed_{seed}"
        np.testing.assert_allclose(score, stored_oof[column], atol=1e-12, rtol=0)
        np.testing.assert_allclose(test_sum, stored_test[column], atol=1e-12, rtol=0)
        max_oof = max(max_oof, float(np.max(np.abs(score - stored_oof[column]))))
        max_test = max(max_test, float(np.max(np.abs(test_sum - stored_test[column]))))
        scores.append(score)
        test_scores.append(test_sum)
        audit.append({"seed": seed, "refit_models": 5, "oof_coverage_exactly_once": True})
    ensemble = tuning.probability_mean(scores)
    test_ensemble = tuning.probability_mean(test_scores)
    np.testing.assert_allclose(ensemble, stored_oof.p_repay_ensemble, atol=1e-12, rtol=0)
    np.testing.assert_allclose(test_ensemble, stored_test.p_repay_ensemble, atol=1e-12, rtol=0)
    for suffix, prediction in [(f"seed_{seed}", test_scores[i]) for i, seed in enumerate(tuning.SEEDS)] + [("ensemble", test_ensemble)]:
        submission = pd.read_csv(tuning.ROOT / f"submissions/lightgbm_tuned_{suffix}.csv")
        assert submission.columns.tolist() == ["id", "prediction"] and len(submission) == len(test_meta)
        pd.testing.assert_series_equal(submission.id, template.id, check_names=False)
        aligned = pd.Series(prediction, index=test_meta.id).loc[template.id].to_numpy()
        np.testing.assert_allclose(submission.prediction, aligned, atol=1e-12, rtol=0)
    all_scores = {"baseline": baseline.p_repay.to_numpy(),
                  **{f"tuned_seed_{seed}": scores[i] for i, seed in enumerate(tuning.SEEDS)},
                  "tuned_seed_ensemble": ensemble}
    comparison = pd.DataFrame([{"model": name, **item} for name, score in all_scores.items()
                               for item in tuning.group_metrics(labels, folds, score)])
    for filename, frame in (("comparison_metrics.csv", comparison), ("search_results.csv", pd.DataFrame(search)),
                            ("selected_candidates.csv", pd.DataFrame(chosen)), ("refit_records.csv", pd.DataFrame(records))):
        pd.testing.assert_frame_equal(pd.read_csv(report / filename), frame, check_dtype=False, atol=1e-12, rtol=1e-12)
    paired, stability = [], []
    for fold in range(5):
        valid = fold_id == fold
        reference = tuning.base.metrics(labels[valid], all_scores["baseline"][valid])
        for name, score in all_scores.items():
            result = tuning.base.metrics(labels[valid], score[valid])
            paired.append({"model": name, "fold": fold, **result,
                           **{f"delta_{metric}": result[metric] - reference[metric]
                              for metric in ("repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score")}})
        predictions = np.stack([score[valid] for score in scores])
        aucs = [roc_auc_score(labels[valid], p) for p in predictions]
        stability.append({"fold": fold, "seed_auc_mean": float(np.mean(aucs)),
                          "seed_auc_std_ddof1": float(np.std(aucs, ddof=1)),
                          "seed_auc_min": min(aucs), "seed_auc_max": max(aucs),
                          "mean_row_probability_std_ddof0": float(predictions.std(axis=0).mean()),
                          "mean_pairwise_prediction_correlation": float(np.corrcoef(predictions)[np.triu_indices(3, 1)].mean())})
    for filename, frame in (("paired_fold_metrics.csv", pd.DataFrame(paired)), ("seed_stability.csv", pd.DataFrame(stability))):
        pd.testing.assert_frame_equal(pd.read_csv(report / filename), frame, check_dtype=False, atol=1e-12, rtol=1e-12)
    summary = json.loads((report / "summary.json").read_text())
    overall = comparison.query("group == 'overall'").set_index("model")
    deltas = {name: float(overall.loc["tuned_seed_ensemble", name] - overall.loc["baseline", name])
              for name in ("repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score")}
    assert summary["ensemble_minus_baseline"] == deltas
    assert summary["chosen_candidates_by_fold"] == [item["candidate"] for item in chosen]
    assert summary["chosen_rounds_by_fold"] == [item["selected_rounds"] for item in chosen]
    assert summary["search_hit_cap_count"] == sum(item["hit_round_cap"] for item in search)
    assert summary["chosen_hit_cap_count"] == sum(item["hit_round_cap"] for item in chosen)
    recommendation = (deltas["repay_roc_auc"] >= 0.0005 and deltas["default_pr_auc_ap"] >= -0.001
                      and deltas["log_loss"] <= 0.0005)
    assert summary["recommended_development_model"] == ("tuned_seed_ensemble" if recommendation else "baseline")
    result = {"status": "passed", "training_signature": signature, "inner_models_reloaded": 20,
              "refit_models_reloaded": 15, "all_splits_reproduced_and_disjoint": True,
              "all_early_stopping_choices_replayed": True, "all_inner_selections_reproduced": True,
              "all_model_and_prediction_checksums_verified": True,
              "all_1105_features_and_native_missing_preserved": True,
              "all_metrics_and_seed_spreads_recomputed": True, "all_four_submissions_aligned": True,
              "equal_weight_ensemble_reproduced": True, "max_inner_prediction_difference": max_inner,
              "max_oof_difference": max_oof, "max_test_difference": max_test, "seeds": audit}
    result["baseline_reference_common_learning_curves"] = reference_curves
    tuning.base.json_write(report / "artifact_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        verify()

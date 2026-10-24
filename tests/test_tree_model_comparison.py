"""Exercise native libraries, quantization isolation and the actual fold runner."""

import json
import shutil
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_tree_model_comparison as study


class TreeModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.parent = study.ROOT / "artifacts/test_tmp"
        self.root = self.parent / f"tree_controls_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.parent.resolve()))
        shutil.rmtree(self.root)

    def test_stop_rule_and_zero_based_xgboost_round_conversion(self):
        state = study.AUCStop(3, 0.0001)
        self.assertEqual([state.update(v) for v in [.8, .80005, .80006, .80007]], [True, True, True, False])
        self.assertEqual(state.best_round, 1)
        improved = study.AUCStop(3, 0.0001)
        for value in [.8, .80011, .80012]:
            improved.update(value)
        self.assertEqual(improved.best_round, 2)
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                improved.update(invalid)

    def test_numeric_export_round_trips_float32_nan_and_real_zero(self):
        values = np.array([[0, np.nan, 1.23456789, -1e30], [-0.0, 1e-30, 16777216, 3.5]], dtype=np.float32)
        path = self.root / "numeric.tsv"
        study.export_numeric_tsv(path, values, np.array([0, 1]), np.arange(2), ["a", "b", "c", "d"], 1)
        restored = pd.read_csv(path, sep="\t", header=None, dtype=np.float32).iloc[:, 1:].to_numpy()
        np.testing.assert_array_equal(restored, values)
        self.assertTrue(np.isnan(restored[0, 1]))
        self.assertEqual(restored[0, 0], 0)

    def test_same_split_seed_numeric_only_and_explicit_training_modes(self):
        for fold in range(5):
            xgb = study.params_for("xgboost", fold, 1)
            cat = study.params_for("catboost", fold, 1)
            self.assertEqual(xgb["seed"], cat["random_seed"])
            self.assertEqual(xgb["tree_method"], "hist")
            self.assertEqual(cat["boosting_type"], "Plain")
            self.assertEqual(cat["grow_policy"], "SymmetricTree")
            self.assertNotIn("cat_features", cat)
            self.assertFalse(cat["use_best_model"])

    def test_native_quantization_preserves_training_only_borders(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(600, 4)).astype(np.float32)
        values[:400, 1] = 0  # Constant in fitting, variable in early-stop rows.
        values[:400, 2] = np.nan  # Missing in fitting, observed in early-stop rows.
        values[::8, 3] = np.nan
        labels = (values[:, 0] > 0).astype(np.int8)
        names = [f"feature_{i}" for i in range(4)]
        with patch.object(study, "ARTIFACTS", self.root):
            fitting, borders = study.cat_data(values, labels, np.arange(400), names, 0, "inner", 1, 64)
            stop, stop_borders = study.cat_data(values, labels, np.arange(400, 600), names, 0, "stop", 1, 64, borders)
        self.assertEqual(borders.read_bytes(), stop_borders.read_bytes())
        self.assertEqual(fitting.get_feature_names(), names)
        self.assertTrue(stop.is_quantized())
        self.assertFalse((self.root / "scratch/fold_0/stop.tsv").exists())
        data = study.xgb_data(values, labels, np.arange(400), names, 1, 64)
        stop_data = study.xgb_data(values, labels, np.arange(400, 600), names, 1, 64, ref=data)
        cut_pointer, cut_values = data.get_quantile_cut()
        stop_pointer, stop_values = stop_data.get_quantile_cut()
        np.testing.assert_array_equal(cut_pointer, stop_pointer)
        np.testing.assert_array_equal(cut_values, stop_values)

    def test_actual_fold_models_unchanged_when_outer_features_and_labels_change(self):
        rng = np.random.default_rng(78)
        values = rng.normal(size=(1600, 7)).astype(np.float32)
        values[::9, 3] = np.nan
        labels = (values[:, 0] + rng.normal(size=len(values)) > 0).astype(np.int8)
        folds = np.arange(len(values)) % 5
        changed_values, changed_labels = values.copy(), labels.copy()
        changed_values[folds == 0] = 10000
        changed_labels[folds == 0] = 1 - labels[folds == 0]
        names = [f"feature_{i}" for i in range(values.shape[1])]
        config = {"training_signature": "synthetic_control_isolation", "experiment": {
            "threads": 1, "batch_rows": 128, "max_rounds": 12, "patience": 4, "min_delta": .0001}}
        probe = values[:100].copy()
        for engine in study.ENGINES:
            output = []
            for tag, matrix, y in (("original", values, labels), ("changed", changed_values, changed_labels)):
                artifacts, report = self.root / engine / tag / "artifacts", self.root / engine / tag / "report"
                for folder in ("inner_models", "inner_predictions", "inner_splits", "models", "fold_predictions", "borders"):
                    (artifacts / folder).mkdir(parents=True)
                (report / "inner_curves").mkdir(parents=True)
                context = (None, None, None, y, folds, names, matrix, probe, None)
                with patch.object(study, "ARTIFACTS", artifacts), patch.object(study, "REPORT", report):
                    study.run_fold(context, engine, 0, config)
                    study.run_fold(context, engine, 0, config)  # Checkpoint reuse checks signatures and files.
                    path, metadata = study.model_paths(engine, 0, "refit")
                    saved = json.loads(metadata.read_text())
                    model = study.load_model(engine, path)
                    predictions = study.predict_rows(engine, model, probe, names, 17)
                    self.assertEqual(study.model_rounds(engine, model), saved["record"]["refit_rounds"])
                    output.append((saved["record"]["refit_rounds"], predictions))
            self.assertEqual(output[0][0], output[1][0])
            np.testing.assert_array_equal(output[0][1], output[1][1])

    def test_full_native_pipeline_collect_reload_and_report_on_small_fixture(self):
        from scripts import verify_tree_model_comparison as verifier
        from scripts import report_tree_model_comparison as reporting
        rng = np.random.default_rng(21)
        values = rng.normal(size=(600, 4)).astype(np.float32)
        values[::7, 3] = np.nan
        labels = (values[:, 0] + rng.normal(size=len(values)) > 0).astype(np.int8)
        fold_id = np.arange(len(values)) % 5
        names = [f"feature_{i}" for i in range(4)]
        meta = pd.DataFrame({"id": np.arange(len(values)), "label": labels})
        test_meta = pd.DataFrame({"id": np.arange(30)})
        folds = pd.DataFrame({"id": meta.id, "fold": fold_id,
                              "stress_feature_492_upper_tail": values[:, 0] > .5,
                              "stress_feature_1039_observed": np.arange(len(values)) % 3 == 0})
        references = {"lightgbm_baseline": np.clip(.5 + .1 * values[:, 0], .01, .99),
                      "lightgbm_tuned_seed_42": np.clip(.5 + .11 * values[:, 0], .01, .99),
                      "lightgbm_tuned_ensemble": np.clip(.5 + .12 * values[:, 0], .01, .99)}
        np.save(self.root / "features.npy", values)
        source = study.BatchedFeatureFile(self.root / "features.npy")
        np.testing.assert_array_equal(source[np.arange(30)], values[:30])
        context = (meta, test_meta, folds, labels, fold_id, names, source, values[:30], references)
        artifacts, report = self.root / "artifacts", self.root / "report"
        for folder in ("inner_models", "inner_predictions", "inner_splits", "models", "fold_predictions", "borders"):
            (artifacts / folder).mkdir(parents=True)
        (report / "inner_curves").mkdir(parents=True)
        (self.root / "submissions").mkdir()
        (self.root / "data/processed").mkdir(parents=True)
        pd.DataFrame({"id": np.arange(29, -1, -1), "prediction": 0.0}).to_parquet(
            self.root / "data/processed/sample_submission.parquet", index=False)
        config = {"training_signature": "synthetic_full_pipeline", "status": "initialized", "experiment": {
            "threads": 1, "batch_rows": 64, "max_rounds": 3, "patience": 2, "min_delta": .0001,
            "train_rows": 600, "test_rows": 30, "features": names}}
        with patch.object(study, "ROOT", self.root), patch.object(study, "ARTIFACTS", artifacts), \
                patch.object(study, "REPORT", report), patch.object(study, "checked_config", return_value=config), \
                patch.object(study, "load_context", return_value=context):
            for engine in study.ENGINES:
                for fold in range(5):
                    study.run_fold(context, engine, fold, config)
            study.collect(context, config)
            verifier.verify()
            reporting.report()
        audit = json.loads((report / "artifact_verification.json").read_text())
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["max_oof_difference"], 0.0)
        self.assertTrue((report / "tree_model_comparison_report.md").exists())
        self.assertEqual(len(list((report / "figures").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()

"""Test AUC exports and real drift-aware training on an isolated small dataset."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest
import uuid
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

from scripts import build_public_auc_candidates as builder
from scripts import run_public_auc_optimization as optimizer
from scripts import verify_public_auc_optimization as verifier
from scripts import verify_public_auc_candidates as file_verifier
from scripts import run_catboost_fast_gpu as fast_gpu
from scripts import run_public_auc_cpu_refinements as cpu_refiner
import catboost as cb


class PublicAucTests(unittest.TestCase):
    def setUp(self):
        self.parent = optimizer.CODE_ROOT / "artifacts/test_tmp"
        self.root = self.parent / f"public_auc_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.parent.resolve()))
        shutil.rmtree(self.root)

    def test_deadline_cleanup_stops_real_descendant_processes(self):
        ready = self.root / "child_pid.txt"
        source = self.root / "spawn_child.py"
        source.write_text(
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
            "creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)\n"
            "Path(sys.argv[1]).write_text(str(child.pid))\n"
            "time.sleep(60)\n", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(source), str(ready)],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        descendants = []
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(ready.exists(), "Descendant failed to start")
            child_pid = int(ready.read_text())
            descendants = psutil.Process(process.pid).children(recursive=True)
            self.assertIn(child_pid, [child.pid for child in descendants])
            optimizer.stop_worker_tree(process)
            self.assertIsNotNone(process.poll())
            self.assertFalse(any(child.is_running() for child in descendants))
        finally:
            optimizer.stop_worker_tree(process)
            for child in descendants:
                if child.is_running():
                    child.kill()

    def fixture(self):
        rng = np.random.default_rng(7)
        names = [f"feature_{i}" for i in range(1, 1106)]
        values = np.zeros((500, len(names)), dtype=np.float32)
        values[:, :6] = rng.normal(size=(500, 6))
        values[::9, 3] = np.nan
        labels = (values[:, 0] + rng.normal(size=500) > 0).astype(np.int8)
        meta = pd.DataFrame({"id": np.arange(500), "label": labels})
        fold_id = np.arange(500) % 5
        folds = pd.DataFrame({"id": meta.id, "fold": fold_id,
                             "stress_feature_492_upper_tail": values[:, 0] > .5,
                             "stress_feature_1039_observed": np.arange(500) % 3 == 0})
        data = self.root / "data/processed"
        data.mkdir(parents=True)
        (self.root / "data/interim").mkdir()
        train = pd.DataFrame(values, columns=names)
        train.insert(0, "id", meta.id)
        train["label"] = labels
        train.to_parquet(data / "train.parquet", index=False)
        train.iloc[:30].drop(columns="label").to_parquet(data / "test.parquet", index=False)
        folds.to_parquet(self.root / "data/interim/validation_folds.parquet", index=False)
        template = pd.DataFrame({"id": np.arange(29, -1, -1), "prediction": 0.0})
        template.to_parquet(data / "sample_submission.parquet", index=False)
        cache = self.root / "artifacts/linear_baselines/cache"
        cache.mkdir(parents=True)
        np.save(cache / "train_features.npy", values)
        np.save(cache / "test_features.npy", values[:30])
        for fold in range(5):
            path = self.root / f"artifacts/hyperparameter_seed_optimization/models/seed_42_fold_{fold}.json"
            params = optimizer.base.model_params(fold, 1)
            params.update(min_data_in_leaf=15, num_leaves=7)
            builder.json_write(path, {"features": names, "positive_class": 1,
                                      "params": params, "record": {"refit_rounds": 6}})
        oof = folds.copy()
        oof["label"] = labels
        oof["p_repay"] = np.clip(.5 + .1 * values[:, 0], .01, .99)
        predictions = pd.DataFrame({"id": np.arange(30), "p_repay": oof.p_repay.to_numpy()[:30]})
        for folder in ("feature_screening/drift_core_filter", "feature_screening/availability_filter", "lightgbm_baseline", "hyperparameter_seed_optimization"):
            target = self.root / "artifacts" / folder
            target.mkdir(parents=True, exist_ok=True)
            reference, reference_test = oof.copy(), predictions.copy()
            if folder == "hyperparameter_seed_optimization":
                reference["p_repay_ensemble"] = reference.p_repay
                reference_test["p_repay_ensemble"] = reference_test.p_repay
            reference.sample(frac=1, random_state=1).to_parquet(target / "oof_predictions.parquet", index=False)
            reference_test.iloc[::-1].to_parquet(target / "test_predictions.parquet", index=False)
        submissions = self.root / "submissions"
        submissions.mkdir()
        for filename in builder.OBSERVED_PUBLIC_SCORES:
            optimizer.base.build_submission(predictions, template).to_csv(submissions / filename, index=False)
        return values, meta, folds, names, template

    def test_blends_align_by_id_and_reject_invalid_weights_and_scores(self):
        frame = pd.DataFrame({"id": [2, 0, 1], "score": [.8, .1, .4]})
        np.testing.assert_array_equal(builder.aligned_values(frame, [0, 1, 2], "score"), [.1, .4, .8])
        scores = {"a": np.array([.1, .9]), "b": np.array([.3, .7])}
        np.testing.assert_allclose(builder.probability_blend(scores, {"a": .5, "b": .5}), [.2, .8])
        for weights in ({"a": 1.1}, {"a": -1, "b": 2}, {"missing": 1}):
            with self.assertRaises(ValueError):
                builder.probability_blend(scores, weights)
        with self.assertRaises(ValueError):
            builder.aligned_values(frame, [0, 1, 3], "score")

    def test_rank_blending_handles_ties_and_ignores_monotone_score_scales(self):
        scores = {"a": np.array([.1, .1, .7, .9]), "b": np.array([.8, .3, .3, .9])}
        expected = .5 * np.array([.25, .25, .625, .875]) + .5 * np.array([.625, .25, .25, .875])
        actual = builder.rank_blend(scores, {"a": .5, "b": .5})
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, builder.rank_blend(
            {"a": scores["a"] ** 3, "b": np.exp(scores["b"])}, {"a": .5, "b": .5}))
        for bad in ([], [.1, np.nan], [[.1, .9]]):
            with self.assertRaises(ValueError):
                builder.rank_blend({"a": np.array(bad)}, {"a": 1})

    def test_binary_catboost_pool_matches_tsv_and_keeps_training_borders(self):
        values, meta, _, names, _ = self.fixture()
        indices = np.arange(300, dtype=np.int64)
        labels = meta.label.to_numpy(dtype=np.int8)
        binary, borders = fast_gpu.binary_pool(self.root, values, labels, indices, names,
                                              0, "inner", 1, "synthetic")
        with patch.object(optimizer.trees, "ARTIFACTS", self.root / "artifacts/tsv_reference"):
            reference, reference_borders = optimizer.trees.cat_data(
                values, labels, indices, names, 0, "inner", 1, 128)
        self.assertEqual(borders.read_bytes(), reference_borders.read_bytes())
        np.testing.assert_array_equal(np.array(binary.get_label(), dtype=float), labels[indices])
        params = {"iterations": 5, "depth": 3, "thread_count": 1, "verbose": False,
                  "random_seed": 42, "allow_writing_files": False}
        a, b = cb.CatBoostClassifier(**params), cb.CatBoostClassifier(**params)
        a.fit(binary)
        b.fit(reference)
        np.testing.assert_array_equal(a.predict_proba(values), b.predict_proba(values))
        heldout = values.copy()
        heldout[300:, 7] = 1000
        _, stop_borders = fast_gpu.binary_pool(self.root, heldout, labels,
                                               np.arange(300, 500, dtype=np.int64), names,
                                               0, "stop", 1, "synthetic", borders)
        self.assertEqual(borders.read_bytes(), stop_borders.read_bytes())
        reloaded, _ = fast_gpu.binary_pool(self.root, values, labels, indices, names,
                                           0, "inner", 1, "synthetic")
        np.testing.assert_array_equal(a.predict_proba(reloaded), a.predict_proba(binary))
        _, artifacts = fast_gpu.locations(self.root)
        self.assertEqual(list(artifacts.rglob("*.float32.npy")), [])
        self.assertFalse(fast_gpu.collect(self.root, {"training_signature": "synthetic"}))
        self.assertFalse((self.root / "artifacts/public_auc_optimization/completed" / fast_gpu.NAME).exists())

    def test_cpu_refinement_spawns_real_folds_and_defers_shared_collection(self):
        self.fixture()
        with patch.object(optimizer, "probe_catboost_gpu", return_value={"available": False}):
            config = optimizer.initialize(self.root)
        cache = self.root / "artifacts/linear_baselines/cache/train_features.npy"
        digest = builder.sha256(cache)
        # The small fixture needs little memory; test the same two-worker path.
        with patch.object(cpu_refiner.psutil, "virtual_memory") as memory:
            memory.return_value.available = 10 * 1024**3
            cpu_refiner.train(self.root, 2, "availability", parallel=True)
        self.assertEqual(builder.sha256(cache), digest)
        for fold in range(5):
            self.assertIsNotNone(optimizer.completed_job(self.root, "lgbm_drift_available", 42,
                                                         fold, config["training_signature"]))
        self.assertFalse((self.root / "artifacts/public_auc_optimization/completed").exists())
        self.assertEqual(optimizer.collect(self.root, config), ["lgbm_drift_available_s42"])

    def test_feature_subsets_preserve_nan_zero_and_exclude_metadata(self):
        values = np.array([[0, np.nan, 2], [3, 4, 5]], dtype=np.float32)
        view = optimizer.SubsetView(values, ["feature_1", "feature_2", "feature_3"], ["feature_2", "feature_1"])
        np.testing.assert_array_equal(view[:], [[np.nan, 0], [4, 3]])
        with self.assertRaises(ValueError):
            optimizer.SubsetView(values, ["feature_1", "feature_2", "feature_3"], ["id"])

    def test_full_cache_check_rejects_a_wrong_value(self):
        values, _, _, names, _ = self.fixture()
        values[10, 1] += 1
        np.save(self.root / "artifacts/linear_baselines/cache/train_features.npy", values)
        with self.assertRaises(AssertionError):
            optimizer.validate_cache(self.root, "train", names)

    def test_submission_hashes_preserve_versions_and_public_feedback_is_observed(self):
        self.fixture()
        rows = builder.build(self.root)
        row = next(row for row in rows if row["candidate"] == "existing_blend_drift50_tuned50")
        self.assertIsNone(row["public_auc"])
        csv = pd.read_csv(self.root / row["submission"])
        self.assertEqual(csv.id.tolist(), list(range(29, -1, -1)))
        filename = str(Path(row["submission"]).relative_to("submissions"))
        builder.record_public_score(self.root, filename, .736, "fixture feedback")
        again = builder.build(self.root)
        self.assertEqual(next(item for item in again if item["candidate"] == row["candidate"])["public_auc"], .736)
        pending = pd.read_csv(self.root / "reports/public_auc_optimization/submission_queue.csv")
        self.assertNotIn(row["candidate"], pending.candidate.tolist())
        old_path = self.root / row["submission"]
        old_hash = builder.sha256(old_path)
        new_path, _ = builder.export_candidate(self.root, row["candidate"], np.full(30, .2), np.arange(30), csv)
        self.assertNotEqual(old_path, new_path)
        self.assertEqual(builder.sha256(old_path), old_hash)

    def test_submission_reconstruction_rejects_reversed_predictions_even_with_updated_hash(self):
        self.fixture()
        rows = builder.build(self.root)
        result = file_verifier.verify(self.root)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["submissions_checked"], len(rows))
        report = self.root / "reports/public_auc_optimization"
        manifest = json.loads((report / "candidate_manifest.json").read_text(encoding="utf-8"))
        path = self.root / manifest[0]["submission"]
        reversed_scores = pd.read_csv(path)
        reversed_scores["prediction"] = 1 - reversed_scores.prediction
        reversed_scores.to_csv(path, index=False)
        # A matching file hash alone must not certify the wrong target direction.
        manifest[0]["sha256"] = builder.sha256(path)
        builder.json_write(report / "candidate_manifest.json", manifest)
        with self.assertRaises(AssertionError):
            file_verifier.verify(self.root)

    def test_empty_joint_stress_group_keeps_auc_unknown(self):
        self.fixture()
        loaded = list(builder.load_predictions(self.root))
        folds = loaded[2].copy()
        folds["stress_feature_492_upper_tail"] = folds.id % 2 == 0
        folds["stress_feature_1039_observed"] = folds.id % 2 == 1
        loaded[2] = folds
        with patch.object(builder, "load_predictions", return_value=tuple(loaded)):
            rows = builder.build(self.root)
        self.assertTrue(rows)
        self.assertTrue(all(row["joint_drift_stress_rows"] == 0 for row in rows))
        self.assertTrue(all(row["joint_drift_stress_oof_auc"] is None for row in rows))

    def test_real_training_reload_and_complete_only_collection(self):
        values, _, folds, names, _ = self.fixture()
        with patch.object(optimizer, "probe_catboost_gpu", return_value={"available": False}):
            config = optimizer.initialize(self.root)
        for fold in range(5):
            optimizer.train_job(self.root, "lgbm_drift_tuned", 42, fold, 1)
            if fold < 4:
                self.assertEqual(optimizer.collect(self.root, config), [])
        complete = optimizer.collect(self.root, config)
        self.assertEqual(complete, ["lgbm_drift_tuned_s42"])
        prediction = pd.read_parquet(self.root / "artifacts/public_auc_optimization/completed/lgbm_drift_tuned_s42/oof_predictions.parquet")
        for fold in range(5):
            metadata, _, model_path = optimizer.job_paths(self.root, "lgbm_drift_tuned", 42, fold)
            saved = json.loads(metadata.read_text())
            self.assertFalse(set(optimizer.CORE_DRIFT) & set(saved["features"]))
            model = lgb.Booster(model_file=str(model_path))
            indices = np.flatnonzero(folds.fold.to_numpy() == fold)
            view = optimizer.SubsetView(values, names, saved["features"])
            actual = optimizer.base.predict_rows(model, view, 128, indices)
            np.testing.assert_array_equal(actual, prediction.p_repay.to_numpy()[indices])
        self.assertTrue(optimizer.completed_job(self.root, "lgbm_drift_tuned", 42, 0, config["training_signature"]))
        verification = verifier.verify(self.root, sample_rows=20)
        self.assertEqual(verification["status"], "passed")
        self.assertEqual(verification["models_reloaded"], 5)
        self.assertEqual(max(row["max_prediction_difference"] for row in verification["checks"]), 0)
        metadata, _, _ = optimizer.job_paths(self.root, "lgbm_drift_tuned", 42, 0)
        saved = json.loads(metadata.read_text())
        saved["metrics"]["repay_roc_auc"] = .5
        builder.json_write(metadata, saved)
        with self.assertRaises(AssertionError):
            verifier.verify(self.root, sample_rows=20)


if __name__ == "__main__":
    unittest.main()

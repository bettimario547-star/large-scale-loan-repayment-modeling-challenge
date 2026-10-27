"""Exercise the real Windows spawn path without loading the competition data."""

import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_tree_model_comparison as study


class TreeModelParallelTests(unittest.TestCase):
    def setUp(self):
        self.parent = study.ROOT / "artifacts/test_tmp"
        self.root = self.parent / f"parallel_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.report, self.artifacts = self.root / "reports/run", self.root / "artifacts/run"
        for folder in ("inner_models", "inner_predictions", "inner_splits", "models", "fold_predictions", "borders"):
            (self.artifacts / folder).mkdir(parents=True)
        (self.report / "inner_curves").mkdir(parents=True)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.parent.resolve()))
        shutil.rmtree(self.root)

    def small_context(self):
        rng = np.random.default_rng(68)
        values = rng.normal(size=(600, 4)).astype(np.float32)
        labels = (values[:, 0] + rng.normal(size=len(values)) > 0).astype(np.int8)
        folds = np.arange(len(values)) % 5
        return (None, None, None, labels, folds, ["a", "b", "c", "d"], values, values[:30], None)

    def test_cpu_budget_and_low_memory_fail_before_process_launch(self):
        context = self.small_context()
        with patch.object(study.psutil, "cpu_count", return_value=16), patch.object(
            study.psutil, "virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)
        ):
            plan = study.parallel_plan(context, 5)
            self.assertEqual((plan["workers"], plan["threads_per_worker"], plan["cpu_budget"]), (2, 5, 10))
            one = study.parallel_plan(context, 1)
            self.assertEqual((one["workers"], one["threads_per_worker"]), (1, 10))
        with patch.object(study.psutil, "cpu_count", return_value=8):
            with self.assertRaisesRegex(RuntimeError, "at least 10 logical CPUs"):
                study.parallel_plan(context, 5)
        with patch.object(study.psutil, "cpu_count", return_value=16), patch.object(
            study.psutil, "virtual_memory", return_value=SimpleNamespace(available=1024**3)
        ), patch.object(study, "ProcessPoolExecutor") as pool, patch.object(study.base, "json_write") as write:
            with self.assertRaisesRegex(RuntimeError, "Insufficient available RAM"):
                study.train_parallel(context, [("xgboost", 0), ("xgboost", 1)], {})
            pool.assert_not_called()
            write.assert_not_called()

    def test_default_cli_stays_single_process(self):
        config = {"training_signature": "cli", "experiment": {"threads": 1, "batch_rows": 32}}
        with patch("sys.argv", ["runner", "--stage", "train"]), patch.object(study, "configure_run"), \
                patch.object(study, "checked_config", return_value=config), \
                patch.object(study, "pending_tasks", return_value=[("xgboost", 0)]), \
                patch.object(study, "load_context", return_value=self.small_context()), \
                patch.object(study.base, "json_write"), patch.object(study, "run_fold") as serial, \
                patch.object(study, "train_parallel") as parallel:
            study.main()
        serial.assert_called_once()
        parallel.assert_not_called()
        self.assertEqual(serial.call_args.args[3]["execution"]["mode"], "single")

    def test_completed_checkpoints_do_not_start_workers(self):
        with patch.object(study, "checkpoint", side_effect=[{"selected_rounds": 3},
                    {"refit_rounds": 3, "repay_roc_auc": .7}]):
            self.assertEqual(study.pending_tasks(["xgboost"], [0], "signed"), [])
        with patch("sys.argv", ["runner", "--stage", "train", "--parallel"]), \
                patch.object(study, "configure_run"), patch.object(study, "checked_config", return_value={
                    "training_signature": "signed", "experiment": {"threads": 4}}), \
                patch.object(study, "pending_tasks", return_value=[]), \
                patch.object(study, "load_context") as load, patch.object(study, "train_parallel") as parallel:
            study.main()
        load.assert_not_called()
        parallel.assert_not_called()

    def test_run_name_isolates_outputs_and_submission_files(self):
        with patch.object(study, "ROOT", self.root), patch.object(study, "REPORT"), patch.object(study, "ARTIFACTS"):
            study.configure_run("new_run10")
            self.assertEqual(study.REPORT, self.root / "reports/new_run10")
            self.assertEqual(study.submission_path("xgboost").name, "xgboost_baseline_new_run10.csv")
            for invalid in ("../old", "C:/outside", "con", "", "A name"):
                with self.assertRaises(ValueError):
                    study.configure_run(invalid)

    def test_spawned_engines_reproduce_serial_models_and_only_read_shared_caches(self):
        # Keep the production 1,105-column contract; only four columns vary.
        small = self.small_context()
        labels, fold_id = small[3:5]
        names = [f"feature_{i}" for i in range(1, 1106)]
        values = np.zeros((len(labels), len(names)), dtype=np.float32)
        values[:, :4] = small[6]
        values[::7, 3] = np.nan
        meta = pd.DataFrame({"id": np.arange(len(labels)), "label": labels})
        folds = pd.DataFrame({"id": meta.id, "fold": fold_id,
                             "stress_feature_492_upper_tail": values[:, 0] > .5,
                             "stress_feature_1039_observed": np.arange(len(labels)) % 3 == 0})
        data = self.root / "data/processed"
        data.mkdir(parents=True)
        train, test, fold_path = data / "train.parquet", data / "test.parquet", self.root / "folds.parquet"
        train_frame = pd.DataFrame(values, columns=names)
        train_frame.insert(0, "id", meta.id)
        train_frame["label"] = labels
        train_frame.to_parquet(train, index=False)
        train_frame.iloc[:30].drop(columns="label").to_parquet(test, index=False)
        folds.to_parquet(fold_path, index=False)
        baseline, tuned = self.root / "baseline", self.root / "tuned"
        baseline.mkdir()
        tuned.mkdir()
        reference = folds.copy()
        reference["label"] = labels
        reference["p_repay"] = np.clip(.5 + .1 * values[:, 0], .01, .99)
        reference.to_parquet(baseline / "oof_predictions.parquet", index=False)
        reference["p_repay_seed_42"] = reference.p_repay
        reference["p_repay_ensemble"] = reference.p_repay
        reference.to_parquet(tuned / "oof_predictions.parquet", index=False)
        linear = self.root / "linear"
        cache = linear / "cache"
        cache.mkdir(parents=True)
        for path, array in ((train, values), (test, values[:30])):
            np.save(cache / f"{path.stem}_features.npy", array)
            (cache / f"{path.stem}_features.json").write_text(json.dumps({
                "source": study.base.source_signature(path), "features": names,
                "shape": list(array.shape), "dtype": "float32"}), encoding="utf-8")
        initial_hashes = {path: study.sha256(path) for path in cache.iterdir()}
        config = {"training_signature": "spawn_fixture", "status": "initialized", "experiment": {
            "threads": 5, "batch_rows": 64, "max_rounds": 3, "patience": 2, "min_delta": .0001}}
        tasks = [(engine, fold) for fold in (0, 1) for engine in study.ENGINES]
        environment = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        with patch.object(study, "ROOT", self.root), patch.object(study, "REPORT", self.report), \
                patch.object(study, "ARTIFACTS", self.artifacts), patch.object(study, "TUNED", tuned), \
                patch.object(study.base, "TRAIN", train), patch.object(study.base, "TEST", test), \
                patch.object(study.base, "FOLDS", fold_path), patch.object(study.base, "ARTIFACTS", baseline), \
                patch.object(study.base, "LINEAR_ARTIFACTS", linear), \
                patch.object(study.psutil, "cpu_count", return_value=16), patch.object(
                    study.psutil, "virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3)):
            context = study.load_context(64, readonly_cache=True)
            study.train_parallel(context, tasks, config)
            pids = set()
            for engine, fold in tasks:
                path, metadata_path = study.model_paths(engine, fold, "refit")
                saved = json.loads(metadata_path.read_text())
                self.assertEqual(saved["record"]["training_threads"], 5)
                self.assertEqual(saved["record"]["execution_mode"], "parallel")
                self.assertEqual(saved["params"], study.params_for(engine, fold, 5))
                pids.add(saved["record"]["worker_pid"])
                self.assertNotEqual(saved["record"]["worker_pid"], os.getpid())
                parallel = study.load_model(engine, path)
                parallel_score = study.predict_rows(engine, parallel, values[:30], names, 64)
                serial_artifacts, serial_report = self.root / f"serial_{engine}_{fold}", self.root / f"report_{engine}_{fold}"
                for folder in ("inner_models", "inner_predictions", "inner_splits", "models", "fold_predictions", "borders"):
                    (serial_artifacts / folder).mkdir(parents=True)
                (serial_report / "inner_curves").mkdir(parents=True)
                with patch.object(study, "ARTIFACTS", serial_artifacts), patch.object(study, "REPORT", serial_report):
                    study.run_fold(context, engine, fold, config)
                    serial_path, _ = study.model_paths(engine, fold, "refit")
                    serial = study.load_model(engine, serial_path)
                    np.testing.assert_array_equal(study.predict_rows(engine, serial, values[:30], names, 64), parallel_score)
            self.assertEqual(len(pids), 2)
            self.assertEqual(study.pending_tasks(list(study.ENGINES), [0, 1], config["training_signature"]), [])
        self.assertEqual({path: study.sha256(path) for path in cache.iterdir()}, initial_hashes)
        self.assertEqual({name: os.environ.get(name) for name in environment}, environment)
        self.assertEqual(len(list((self.root / "logs/run").glob("*/**/*.log"))), 4)


if __name__ == "__main__":
    unittest.main()

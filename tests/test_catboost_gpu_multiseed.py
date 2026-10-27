"""Check seed identity, cache reuse and full-OOF averaging with real GPU fits."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_catboost_fast_gpu as engine
from scripts import run_catboost_gpu_capacity as capacity
from scripts import run_catboost_gpu_multiseed as study
from scripts import build_public_auc_candidates as builder
from tests.test_public_auc_optimization import PublicAucTests


class CatBoostMultiSeedTests(unittest.TestCase):
    def test_real_gpu_seeds_preserve_pools_and_average_complete_oof_only(self):
        helper = PublicAucTests()
        helper.setUp()
        root = helper.root
        keys = ("locations", "initialize", "binary_pool", "train_fold", "NAME", "__file__")
        previous = {key: getattr(engine, key) for key in keys}
        try:
            values, _, _, _, _ = helper.fixture()
            engine.original.initialize(root)
            capacity.configure(root)
            engine.initialize(root)

            def tiny_config(dataset_root, seed, base, legacy):
                # Reduce the fixture's iteration ceiling; production still
                # inherits the signed 2,000-round capacity recipe unchanged.
                exp = {"capacity_training_signature": base["training_signature"],
                       "pool_training_signature": legacy["training_signature"],
                       "source_sha256": builder.sha256(study.SOURCE), "base_seed": seed,
                       "seed_rule": "base_seed + outer_fold", "params": dict(base["experiment"]["params"]),
                       "feature_plans": base["experiment"]["feature_plans"],
                       "catboost_version": engine.cb.__version__}
                exp["params"].update(iterations=8, od_wait=2)
                config = {"training_signature": engine.original.trees.signing_hash(exp), "experiment": exp}
                path = study.locations(dataset_root, seed)[0] / "training_config.json"
                builder.json_write(path, config)
                return config

            with patch.object(study, "seed_config", side_effect=tiny_config):
                study.configure(root, 2026)
                study.train_fold(root, 0, 1)
                study.collect_ensemble(root)
                target = root / "artifacts/public_auc_optimization/completed/catboost_gpu_capacity_ensemble"
                self.assertFalse(target.exists(), "A partial seed must never export an ensemble")
                for fold in range(1, 5):
                    study.train_fold(root, fold, 1)
                engine.collect(root, engine.initialize(root))
                pools = {path: builder.sha256(path) for path in (root / "artifacts/catboost_fast_gpu/pools").rglob("*.qbin")}
                self.assertEqual(len(pools), 15)
                study.configure(root, 3407)
                for fold in range(5):
                    study.train_fold(root, fold, 1)
                engine.collect(root, engine.initialize(root))
                self.assertEqual(study.collect_ensemble(root), [2026, 3407])
                self.assertEqual(pools, {path: builder.sha256(path) for path in pools})
                ensemble = pd.read_parquet(target / "oof_predictions.parquet").set_index("id")
                components = []
                for seed in (2026, 3407):
                    folder = root / f"artifacts/public_auc_optimization/completed/catboost_gpu_capacity_s{seed}"
                    components.append(pd.read_parquet(folder / "oof_predictions.parquet").set_index("id").loc[ensemble.index, "p_repay"].to_numpy())
                    study.configure(root, seed)
                    checks = engine.verify(root, sample_rows=20)
                    self.assertEqual(len(checks), 5)
                    for fold in range(5):
                        model = engine.cb.CatBoostClassifier()
                        model.load_model(str(engine.job_paths(root, fold)[2]))
                        self.assertEqual(model.get_param("random_seed"), seed + fold)
                        self.assertEqual(model.get_all_params()["task_type"], "GPU")
                np.testing.assert_array_equal(ensemble.p_repay.to_numpy(), np.mean(components, axis=0))
                self.assertEqual(len(ensemble), len(values))
        finally:
            for key, value in previous.items():
                setattr(engine, key, value)
            helper.tearDown()


if __name__ == "__main__":
    unittest.main()

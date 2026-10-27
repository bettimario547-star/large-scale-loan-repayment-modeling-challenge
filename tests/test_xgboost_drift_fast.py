"""Check real spawned CPU fits, source integrity and complete-only exports."""
import json
import unittest
from unittest.mock import patch

from scripts import run_xgboost_drift_fast as study
from scripts import build_public_auc_candidates as builder
from tests.test_public_auc_optimization import PublicAucTests


class XGBoostDriftFastTests(unittest.TestCase):
    def test_spawned_full_folds_defer_exports_and_reproduce_predictions(self):
        helper = PublicAucTests()
        helper.setUp()
        try:
            helper.fixture()
            root = helper.root
            with patch.object(study.base, "probe_catboost_gpu", return_value={"available": False}):
                study.base.initialize(root)
            config = study.initialize(root)
            cache = root / "artifacts/linear_baselines/cache/train_features.npy"
            digest = builder.sha256(cache)
            self.assertFalse(study.collect(root))
            target = root / "artifacts/public_auc_optimization/completed" / study.NAME
            study.train(root, 2, 1)
            self.assertFalse(target.exists(), "The controller must defer shared prediction exports")
            self.assertTrue(study.collect(root))
            result = study.verify(root)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["models_reloaded"], 5)
            self.assertEqual(max(row["max_prediction_difference"] for row in result["checks"]), 0)
            self.assertEqual(builder.sha256(cache), digest)
            for fold in range(5):
                saved = study.checkpoint(root, fold, config)
                self.assertFalse(study.base.CORE_DRIFT.intersection(saved["features"]))
                self.assertNotIn("id", saved["features"])
                self.assertEqual(saved["rounds"], 600)
                model = study.xgb.Booster(model_file=study.paths(root, fold)[2])
                actual = json.loads(model.save_config())["learner"]
                self.assertEqual(actual["generic_param"]["device"], "cpu")
                self.assertEqual(actual["objective"]["name"], "binary:logistic")
        finally:
            helper.tearDown()


if __name__ == "__main__":
    unittest.main()

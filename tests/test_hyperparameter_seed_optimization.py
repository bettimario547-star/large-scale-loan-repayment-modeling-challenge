"""Audit selection isolation, seed roles, equal averaging and Dataset reuse."""

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np

from scripts import run_hyperparameter_seed_optimization as tuning
from scripts.verify_hyperparameter_seed_optimization import early_stop_choice


class HyperparameterSeedTests(unittest.TestCase):
    def test_early_stop_audit_ignores_tiny_improvements_and_keeps_best_at_cap(self):
        # Numeric maximum is at round 4; no improvement exceeds min_delta.
        self.assertEqual(early_stop_choice([0.8, 0.80005, 0.80006, 0.80007], 3, 0.0001), (1, 0.8))
        # Budget expires before patience; retained best is round 2, not last.
        self.assertEqual(early_stop_choice([0.8, 0.80011, 0.80012], 3, 0.0001), (2, 0.80011))

    def test_model_seed_does_not_change_split_or_bin_seed(self):
        for fold in range(5):
            first = tuning.params_for(tuning.CANDIDATES[0], fold, 42, 1)
            second = tuning.params_for(tuning.CANDIDATES[0], fold, 2026, 1)
            differences = {key for key in first if first[key] != second[key]}
            self.assertEqual(differences, {"seed", "bagging_seed", "feature_fraction_seed"})
            self.assertEqual(first["data_random_seed"], 42 + fold)
            self.assertFalse(first["feature_pre_filter"])
            self.assertFalse(first["zero_as_missing"])
            self.assertTrue(first["deterministic"])

    def test_selection_uses_inner_score_with_predeclared_tie_priority(self):
        records = [{"candidate": item["name"], "inner_auc": score, "selected_rounds": 200,
                    "outer_auc": 1 - score}
                   for item, score in zip(tuning.CANDIDATES, (0.80000, 0.80008, 0.799, 0.798))]
        self.assertEqual(tuning.choose_candidate(records)["candidate"], "baseline_extended")
        records[2]["inner_auc"] = 0.81
        self.assertEqual(tuning.choose_candidate(records)["candidate"], "wider")
        changed = [dict(item, outer_auc=0.2) for item in records]
        self.assertEqual(tuning.choose_candidate(records)["candidate"], tuning.choose_candidate(changed)["candidate"])
        with self.assertRaises(ValueError):
            tuning.choose_candidate(records[:-1])
        invalid = [dict(item, inner_auc=float("nan")) for item in records]
        with self.assertRaises(ValueError):
            tuning.choose_candidate(invalid)

    def test_ensemble_is_probability_mean_not_majority_vote_or_best_seed(self):
        inputs = [np.array([0.2, 0.9]), np.array([0.4, 0.6]), np.array([0.9, 0.3])]
        np.testing.assert_allclose(tuning.probability_mean(inputs), [0.5, 0.6], atol=1e-15)
        for invalid in ([], [np.array([0.2]), np.array([0.2, 0.3])],
                        [np.array([1.1])], [np.array([np.nan])]):
            with self.assertRaises(ValueError):
                tuning.probability_mean(invalid)

    def test_end_to_end_outer_features_and_labels_cannot_change_fitted_models(self):
        rng = np.random.default_rng(73)
        values = rng.normal(size=(4000, 6)).astype(np.float32)
        values[::7, 2] = np.nan
        labels = (values[:, 0] + rng.normal(size=4000) > 0).astype(np.int8)
        folds = np.arange(len(labels)) % 5
        outer = folds == 0
        changed_values = values.copy()
        changed_values[outer] = 1e5
        changed_labels = labels.copy()
        changed_labels[outer] = 1 - labels[outer]
        names = [f"feature_{i}" for i in range(6)]
        config = {"training_signature": "synthetic_isolation_test", "experiment": {
            "threads": 1, "batch_rows": 211, "early_stop_fraction": 0.15,
            "max_rounds": 12, "patience": 4, "min_delta": 0.0001,
            "tie_tolerance": tuning.TIE_TOLERANCE, "selection": "inner_only"}}
        fixture_parent = tuning.ROOT / "artifacts/test_tmp"
        root = fixture_parent / f"tuning_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        self.assertTrue(root.resolve().is_relative_to(fixture_parent.resolve()))
        try:
            outputs = []
            for suffix, matrix, y in (("original", values, labels), ("changed", changed_values, changed_labels)):
                artifacts, report = root / suffix / "artifacts", root / suffix / "report"
                for path in (artifacts / "inner_models", artifacts / "inner_predictions", artifacts / "inner_splits",
                             artifacts / "models", artifacts / "fold_predictions", report / "inner_curves", report / "selections"):
                    path.mkdir(parents=True)
                context = (None, None, None, y, folds, names, matrix, values[:20], None)
                with patch.object(tuning, "ARTIFACTS", artifacts), patch.object(tuning, "REPORT", report):
                    tuning.run_fold(context, 0, config)
                    # Checkpoint reuse also validates models / probabilities.
                    tuning.run_fold(context, 0, config)
                selection = json.loads((report / "selections/fold_0.json").read_text())
                predictions = []
                for seed in tuning.SEEDS:
                    metadata = json.loads((artifacts / f"models/seed_{seed}_fold_0.json").read_text())
                    self.assertEqual(metadata["record"]["refit_rounds"], selection["chosen"]["selected_rounds"])
                    model = lgb.Booster(model_file=metadata["model_path"])
                    predictions.append(model.predict(values))
                outputs.append((selection["chosen"]["candidate"], predictions))
            self.assertEqual(outputs[0][0], outputs[1][0])
            for first, second in zip(outputs[0][1], outputs[1][1]):
                np.testing.assert_array_equal(first, second)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()

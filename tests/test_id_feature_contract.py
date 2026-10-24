"""Small, synthetic checks for the id/feature/submission contract."""

import unittest

import numpy as np
import pandas as pd

from scripts.id_feature_contract import (
    align_validation_folds,
    build_submission,
    model_feature_columns,
    select_model_features,
)


FEATURES = [f"feature_{i}" for i in range(1, 1106)]


class IdFeatureContractTests(unittest.TestCase):
    def test_feature_whitelist_excludes_identifier_target_and_fold(self):
        train_columns = ["id", "label", *FEATURES]
        test_columns = ["id", *FEATURES]
        names = model_feature_columns(train_columns, test_columns)
        self.assertEqual(names, FEATURES)
        frame = pd.DataFrame([[7, 1, 3, *([0] * 1104)]],
                             columns=["id", "label", *FEATURES])
        selected = select_model_features(frame, names)
        self.assertEqual(list(selected.columns), FEATURES)
        self.assertEqual(selected.iloc[0, 0], 3)
        self.assertNotIn("id", selected)
        self.assertNotIn("label", selected)

    def test_schema_drift_and_metadata_as_predictor_are_rejected(self):
        with self.assertRaises(ValueError):
            model_feature_columns(["id", "label", *FEATURES, "fold"], ["id", *FEATURES])
        with self.assertRaises(ValueError):
            model_feature_columns(["id", "label", *FEATURES[:-1]], ["id", *FEATURES])
        frame = pd.DataFrame(columns=["id", "label", "fold", *FEATURES])
        with self.assertRaises(ValueError):
            select_model_features(frame, ["id", *FEATURES[1:]])
        with self.assertRaises(ValueError):
            select_model_features(frame, [*FEATURES[:-1], "feature_1106"])

    def test_submission_uses_id_not_prediction_row_position(self):
        template = pd.DataFrame({"id": [2, 0, 1], "prediction": [0.5] * 3})
        predictions = pd.DataFrame({"id": [1, 2, 0], "p_repay": [0.1, 0.9, 0.7]})
        submission = build_submission(predictions, template)
        self.assertEqual(list(submission.columns), ["id", "prediction"])
        self.assertEqual(submission["id"].tolist(), [2, 0, 1])
        self.assertEqual(submission["prediction"].tolist(), [0.9, 0.7, 0.1])

    def test_fold_alignment_uses_id_not_row_position(self):
        train_ids = pd.Series([12, 10, 11])
        folds = pd.DataFrame({
            "id": [10, 11, 12], "fold": [0, 1, 2],
            "stress_feature_492_upper_tail": [False, True, False],
            "stress_feature_1039_observed": [True, False, False],
        })
        aligned = align_validation_folds(train_ids, folds)
        self.assertEqual(aligned["id"].tolist(), [12, 10, 11])
        self.assertEqual(aligned["fold"].tolist(), [2, 0, 1])
        with self.assertRaises(ValueError):
            align_validation_folds(train_ids, folds.assign(id=[10, 11, 99]))

    def test_submission_rejects_bad_ids(self):
        template = pd.DataFrame({"id": [0, 1], "prediction": [0.5, 0.5]})
        for ids in ([0, 0], [0, 2], [0, None]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_submission(pd.DataFrame({"id": ids, "p_repay": [0.2, 0.8]}), template)

    def test_submission_rejects_bad_probabilities(self):
        template = pd.DataFrame({"id": [0, 1], "prediction": [0.5, 0.5]})
        for scores in ([0.2, -0.1], [0.2, 1.1], [0.2, np.nan], [0.2, np.inf]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                build_submission(pd.DataFrame({"id": [0, 1], "p_repay": scores}), template)


if __name__ == "__main__":
    unittest.main()

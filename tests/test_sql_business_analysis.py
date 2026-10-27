"""Check SQL semantics that can otherwise silently bias business conclusions."""
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

from scripts import business_analysis_common as common
from scripts.verify_sql_business_analysis import verify_connection


def fixture_predictions():
    # Ties include both classes. Dataset IDs intentionally overlap and input
    # row order is shuffled. Test score range differs strongly from training.
    labels = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    scores = [.1, .1, .3, .3, .5, .5, .7, .7, .9, .9]
    rows = []
    for role, version in (("champion", "A"), ("challenger", "B")):
        for dataset in ("train", "test"):
            count = 10 if dataset == "train" else 3
            for index in range(count):
                score = scores[index] if dataset == "train" else [.01, .02, .99][index]
                rows.append({"dataset": dataset, "id": index, "model_version": version,
                    "model_role": role, "prediction_kind": "oof" if dataset == "train" else "test_ensemble",
                    "label": labels[index] if dataset == "train" else None,
                    "fold": index % 5 if dataset == "train" else None,
                    "stress_feature_492_upper_tail": index % 2 == 0 if dataset == "train" else None,
                    "stress_feature_1039_observed": index % 3 == 0 if dataset == "train" else None,
                    "p_repay": score, "p_default": 1 - score})
    return pd.DataFrame(rows, columns=common.PREDICTION_COLUMNS).sample(frac=1, random_state=42).reset_index(drop=True)


class SQLBusinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.connection = duckdb.connect()

    def tearDown(self):
        self.connection.close()
        self.tmp.cleanup()

    def build(self, predictions):
        predictions.to_parquet(self.root / "predictions.parquet", index=False)
        pd.DataFrame({"model_version": ["A", "B"], "model_role": ["champion", "challenger"]}).to_parquet(self.root / "registry.parquet", index=False)
        common.execute_analysis_sql(self.connection, self.root / "predictions.parquet", self.root / "registry.parquet", [0., .5, .9, 1.])

    def test_ties_null_test_outcomes_and_zero_approval_match_independent_references(self):
        frame = fixture_predictions()
        self.build(frame)
        result = verify_connection(self.connection, frame, [0., .5, .9, 1.])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(self.connection.execute("SELECT count(*) FROM model_predictions WHERE id = 0").fetchone()[0], 4)
        # Threshold equality is approved: train scores include exactly 0.5.
        self.assertEqual(self.connection.execute("SELECT approved_rows FROM approval_policy_results WHERE dataset='train' AND model_version='A' AND threshold=.5").fetchone()[0], 6)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM approval_policy_results WHERE dataset='test' AND observed_default_rate_approved IS NOT NULL").fetchone()[0], 0)

    def test_constant_scores_have_auc_half_ap_prevalence_and_no_artificial_decile_split(self):
        frame = fixture_predictions()
        frame["p_repay"], frame["p_default"] = .8, .2
        self.build(frame)
        result = verify_connection(self.connection, frame, [0., .5, .9, 1.])
        self.assertTrue(result["risk_bands_match_numpy_and_preserve_ties"])
        row = self.connection.execute("SELECT repay_roc_auc, default_pr_auc_ap, ks FROM model_performance_summary WHERE model_version='A'").fetchone()
        np.testing.assert_allclose(row, [.5, .4, 0])
        self.assertEqual(self.connection.execute("SELECT count(*) FROM risk_band_summary WHERE row_count=0").fetchone()[0], 36)

    def test_test_changes_cannot_change_train_band_boundaries(self):
        frame = fixture_predictions()
        self.build(frame)
        original = self.connection.table("risk_band_boundaries").fetchdf().sort_values("model_version").reset_index(drop=True)
        self.connection.close()
        self.connection = duckdb.connect()
        test_mask = frame.dataset.eq("test")
        frame.loc[test_mask, "p_repay"] = .999
        frame.loc[test_mask, "p_default"] = .001
        self.build(frame)
        actual = self.connection.table("risk_band_boundaries").fetchdf().sort_values("model_version").reset_index(drop=True)
        pd.testing.assert_frame_equal(original, actual)

    def test_invalid_probabilities_duplicate_keys_and_fake_test_labels_are_rejected(self):
        original = fixture_predictions()
        common.validate_predictions(original)
        duplicate = pd.concat([original, original.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError): common.validate_predictions(duplicate)
        for score in [np.nan, np.inf, -.01, 1.01]:
            invalid = original.copy()
            invalid.loc[0, "p_repay"] = score
            with self.assertRaises(ValueError): common.validate_predictions(invalid)
        invalid = original.copy()
        invalid.loc[invalid.dataset.eq("test"), "label"] = 0
        with self.assertRaises(ValueError): common.validate_predictions(invalid)
        invalid = original.copy()
        invalid.loc[invalid.dataset.eq("train"), "prediction_kind"] = "in_sample"
        with self.assertRaises(ValueError): common.validate_predictions(invalid)

    def test_id_alignment_and_run_names(self):
        frame = pd.DataFrame({"id": [2, 0, 1], "score": [.2, .0, .1]})
        aligned = common.align_by_id(frame, pd.Series([0, 1, 2]))
        self.assertEqual(aligned.score.tolist(), [.0, .1, .2])
        with self.assertRaises(ValueError): common.align_by_id(frame, pd.Series([0, 1, 3]))
        for name in ["../other", "A", "con", "a/b"]:
            with self.assertRaises(ValueError): common.Paths(self.root, name)


if __name__ == "__main__": unittest.main()

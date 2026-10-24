"""Verify streaming row alignment, holdout isolation, and native missing-value modeling."""

import unittest
import uuid
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.run_lightgbm_baseline import IndexedSequence, make_dataset, model_params, predict_rows, split_outer_fit


class LightGBMBaselineTests(unittest.TestCase):
    def test_sequence_keeps_selected_rows_order_nan_and_actual_zero(self):
        values = np.array([[0, np.nan], [2, 3], [4, 5], [6, 7]], dtype=np.float32)
        seq = IndexedSequence(values, np.array([2, 0, 3]), 2)
        self.assertEqual(len(seq), 3)
        np.testing.assert_equal(seq[0], values[2])
        np.testing.assert_equal(seq[0:2], values[[2, 0]])
        np.testing.assert_equal(seq[[2, 0]], values[[3, 2]])
        self.assertEqual(seq[1][0], 0)
        self.assertTrue(np.isnan(seq[1][1]))
        self.assertEqual(seq[0:2].dtype, np.float64)

    def test_inner_split_excludes_outer_validation_and_is_reproducible(self):
        labels = (np.arange(500) % 7 != 0).astype(np.int8)
        folds = np.arange(500) % 5
        for fold in range(5):
            inner, stop, valid = split_outer_fit(labels, folds, fold)
            self.assertEqual(len(np.intersect1d(inner, stop)), 0)
            self.assertEqual(len(np.intersect1d(inner, valid)), 0)
            self.assertEqual(len(np.intersect1d(stop, valid)), 0)
            np.testing.assert_array_equal(np.sort(np.r_[inner, stop]), np.flatnonzero(folds != fold))
            np.testing.assert_array_equal(valid, np.flatnonzero(folds == fold))
            second = split_outer_fit(labels, folds, fold)
            for left, right in zip((inner, stop, valid), second):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(set(labels[inner]), {0, 1})
            self.assertEqual(set(labels[stop]), {0, 1})

    def test_sequence_training_matches_dense_training_and_saved_probability_direction(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(400, 5)).astype(np.float32)
        labels = (values[:, 0] > 0).astype(np.int8)
        values[::5, 1] = np.nan
        rows = np.arange(len(values))
        names = [f"feature_{i + 1}" for i in range(5)]
        params = model_params(0, 1)
        params.update(min_data_in_leaf=5, feature_fraction=1.0, bagging_fraction=1.0,
                      bagging_freq=0, bin_construct_sample_cnt=1000)
        streamed = make_dataset(values, labels, rows, names, params, 31)
        dense = lgb.Dataset(values.astype(np.float64), label=labels, feature_name=names, params=params)
        stream_model = lgb.train(params, streamed, num_boost_round=20)
        dense_model = lgb.train(params, dense, num_boost_round=20)
        p = predict_rows(stream_model, values, 37)
        np.testing.assert_allclose(p, dense_model.predict(values), atol=1e-12)
        self.assertGreater(roc_auc_score(labels, p), 0.99)
        self.assertTrue(np.isfinite(p).all())
        self.assertFalse(params["zero_as_missing"])
        # Use a regular file in the workspace. Windows sandbox sessions can
        # deny access to the owner-only directories created by mkdtemp().
        fixture_root = Path(__file__).resolve().parents[1] / "artifacts/test_tmp"
        fixture_root.mkdir(parents=True, exist_ok=True)
        path = fixture_root / f"model_{uuid.uuid4().hex}.txt"
        self.assertTrue(path.resolve().is_relative_to(fixture_root.resolve()))
        try:
            stream_model.save_model(str(path))
            restored = lgb.Booster(model_file=str(path))
            np.testing.assert_allclose(p, predict_rows(restored, values, 53), atol=1e-12)
        finally:
            if path.exists():
                path.unlink()

    def test_native_binning_ignores_outer_rows_and_references_inner_bins(self):
        rng = np.random.default_rng(43)
        values = rng.normal(size=(300, 4)).astype(np.float32)
        labels = (values[:, 0] > 0).astype(np.int8)
        inner = np.arange(200)
        stop = np.arange(200, 250)
        outer = np.arange(250, 300)
        names = [f"feature_{i + 1}" for i in range(4)]
        params = model_params(0, 1)
        params.update(min_data_in_leaf=5, feature_fraction=1.0, bagging_fraction=1.0,
                      bagging_freq=0, bin_construct_sample_cnt=1000)
        first = make_dataset(values, labels, inner, names, params, 37)
        early = make_dataset(values, labels, stop, names, params, 37, reference=first)
        self.assertIs(early.reference, first)
        np.testing.assert_array_equal(first.data.indices, inner)
        np.testing.assert_array_equal(early.data.indices, stop)
        np.testing.assert_array_equal(early.get_label(), labels[stop])
        model_a = lgb.train(params, first, num_boost_round=15, valid_sets=[early])
        changed = values.copy()
        changed[outer] = 1e6
        second = make_dataset(changed, labels, inner, names, params, 37)
        second_early = make_dataset(changed, labels, stop, names, params, 37, reference=second)
        model_b = lgb.train(params, second, num_boost_round=15, valid_sets=[second_early])
        np.testing.assert_allclose(model_a.predict(values), model_b.predict(values), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

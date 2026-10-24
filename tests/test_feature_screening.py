"""Tests of fold-local filtering, missing semantics and feature-view inference."""

import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.run_feature_screening import FeatureView, column_statistics, fit_plan
from scripts.run_lightgbm_baseline import make_dataset, model_params, predict_rows


class FeatureScreeningTests(unittest.TestCase):
    def profile(self, values, labels, rows, names):
        return pd.DataFrame([{"feature": name, **column_statistics(values[:, j], labels, rows)}
                             for j, name in enumerate(names)])

    def test_selector_ignores_holdout_features_and_labels(self):
        values = np.array([[np.nan, 1], [np.nan, 2], [3, 3], [4, 4], [8, 9], [9, 8]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1, 0, 1])
        rows, names = np.arange(4), ["feature_1", "feature_2"]
        first = self.profile(values, labels, rows, names)
        changed, changed_labels = values.copy(), labels.copy()
        changed[4:] = np.nan
        changed_labels[4:] = 1 - changed_labels[4:]
        second = self.profile(changed, changed_labels, rows, names)
        pd.testing.assert_frame_equal(first, second)
        plan = fit_plan(first, names, "missing_patterns")
        self.assertEqual(plan["missing_indicators"], ["feature_1"])
        self.assertEqual(plan, fit_plan(second, names, "missing_patterns"))

    def test_low_information_rule_is_scale_invariant_and_keeps_missing_signal(self):
        values = np.array([[1, 7, 0], [1, np.nan, 1e-8], [1, 7, 2e-8], [1, np.nan, 3e-8]], dtype=np.float32)
        labels, rows = np.array([0, 1, 0, 1]), np.arange(4)
        names = ["feature_1", "feature_2", "feature_3"]
        first = self.profile(values, labels, rows, names)
        plan = fit_plan(first, names, "low_information_filter")
        self.assertEqual(plan["kept_features"], ["feature_2", "feature_3"])
        self.assertEqual(first.loc[1, "observed_variance"], 0)
        scaled = values.copy()
        scaled[:, 2] *= 1e8
        second = fit_plan(self.profile(scaled, labels, rows, names), names, "low_information_filter")
        self.assertEqual(plan["kept_features"], second["kept_features"])

    def test_availability_drift_rules_and_feature_order(self):
        values = np.array([[np.nan, 1, 2], [np.nan, 2, 3], [np.nan, 3, 4], [np.nan, 4, 5]], dtype=np.float32)
        labels, rows = np.array([0, 1, 0, 1]), np.arange(4)
        names = ["feature_522", "feature_364", "feature_492"]
        profile = self.profile(values, labels, rows, names)
        availability = fit_plan(profile, names, "availability_filter")
        self.assertEqual(availability["kept_features"], ["feature_492"])
        drift = fit_plan(profile, names, "drift_core_filter")
        self.assertEqual(drift["kept_features"], ["feature_522", "feature_364"])
        np.testing.assert_equal(FeatureView(values, availability)[[3, 0]], values[[3, 0], 2:])

    def test_view_preserves_nan_zero_and_counts_original_missing_columns(self):
        values = np.array([[np.nan, 0], [1, np.nan], [2, 3], [np.nan, np.nan]], dtype=np.float32)
        labels, rows, names = np.array([0, 1, 1, 0]), np.arange(4), ["feature_1", "feature_2"]
        plan = fit_plan(self.profile(values, labels, rows, names), names, "missing_patterns")
        view = FeatureView(values, plan)
        dense = view[:]
        np.testing.assert_equal(dense[:, :2], values)
        self.assertEqual(dense[0, 1], 0)
        np.testing.assert_equal(dense[:, -1], [1, 1, 0, 2])
        for j, name in enumerate(plan["missing_indicators"], start=2):
            np.testing.assert_equal(dense[:, j], np.isnan(values[:, names.index(name)]).astype(np.float32))
        np.testing.assert_equal(view[0], dense[0])
        np.testing.assert_equal(view[[3, 0]], dense[[3, 0]])

    def test_native_view_training_and_batch_inference_match_dense(self):
        rng = np.random.default_rng(20261022)
        values = rng.normal(size=(400, 4)).astype(np.float32)
        labels = (values[:, 0] > 0).astype(np.int8)
        values[labels == 0, 1] = np.nan
        names, rows = [f"feature_{j}" for j in range(1, 5)], np.arange(400)
        plan = fit_plan(self.profile(values, labels, rows, names), names, "missing_patterns")
        view = FeatureView(values, plan)
        params = model_params(0, 1)
        params.update(min_data_in_leaf=5, feature_fraction=1.0, bagging_fraction=1.0,
                      bagging_freq=0, bin_construct_sample_cnt=1000)
        streamed = make_dataset(view, labels, rows, plan["output_features"], params, 31)
        dense = lgb.Dataset(view[:].astype(np.float64), label=labels,
                            feature_name=plan["output_features"], params=params)
        a = lgb.train(params, streamed, num_boost_round=15)
        b = lgb.train(params, dense, num_boost_round=15)
        np.testing.assert_allclose(predict_rows(a, view, 37), b.predict(view[:]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()

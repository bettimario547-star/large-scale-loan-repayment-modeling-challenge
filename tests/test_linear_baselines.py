"""Check leakage boundaries and the chunked full-batch logistic objective."""

import unittest

import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.run_linear_baselines import (
    constant_predictions, fit_logistic, fit_preprocessor, logistic_loss_gradient, metrics, transform,
)


class LinearBaselineTests(unittest.TestCase):
    def test_preprocessing_matches_fit_only_sklearn_and_ignores_validation(self):
        values = np.array([[1, np.nan, 7], [3, 8, 7], [5, 10, 7], [999, -999, 7]], dtype=np.float32)
        fit = np.array([True, True, True, False])
        prep = fit_preprocessor(values, fit, 2)
        imputer = SimpleImputer(strategy="mean")
        filled = imputer.fit_transform(values[fit])
        scaler = StandardScaler().fit(filled)
        np.testing.assert_allclose(transform(values, prep), scaler.transform(imputer.transform(values)), rtol=1e-6)
        changed = values.copy()
        changed[~fit] = [np.nan, 1e9, -1e9]
        second = fit_preprocessor(changed, fit, 1)
        np.testing.assert_allclose(prep["mean"], second["mean"])
        np.testing.assert_allclose(prep["scale"], second["scale"])

    def test_all_missing_fitting_column_remains_finite_and_keeps_schema(self):
        values = np.array([[np.nan, 2], [np.nan, 2], [4, np.nan]], dtype=np.float32)
        prep = fit_preprocessor(values, np.array([True, True, False]), 1)
        np.testing.assert_allclose(prep["mean"], [0, 2])
        np.testing.assert_allclose(prep["scale"], [1, 1])
        result = transform(values, prep)
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_allclose(result, [[0, 0], [0, 0], [4, 0]])

    def test_loss_gradient_numerically_and_does_not_penalize_intercept(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(17, 4))
        labels = np.array([0, 1] * 8 + [1])
        theta = rng.normal(size=5)
        loss = lambda t: logistic_loss_gradient(t, values, labels, 0.7, 5)[0]
        gradient = lambda t: logistic_loss_gradient(t, values, labels, 0.7, 5)[1]
        self.assertLess(check_grad(loss, gradient, theta), 1e-6)
        full = logistic_loss_gradient(theta, values, labels, 0.7, 100)
        chunked = logistic_loss_gradient(theta, values, labels, 0.7, 3)
        self.assertAlmostEqual(full[0], chunked[0], places=12)
        np.testing.assert_allclose(full[1], chunked[1], atol=1e-12)
        self.assertAlmostEqual(gradient(theta)[-1], np.mean(expit(values @ theta[:-1] + theta[-1]) - labels))

    def test_optimizer_predicts_repayment_like_sklearn(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(300, 5)).astype(np.float32)
        p = expit(values @ np.array([1, -2, 0.5, 0, 0.1]) + 1)
        labels = rng.binomial(1, p)
        theta, state = fit_logistic(values, labels, 0.7, 31, 500, 1e-7)
        reference = LogisticRegression(C=0.7, solver="lbfgs", tol=1e-7, max_iter=500).fit(values, labels)
        column = int(np.flatnonzero(reference.classes_ == 1)[0])
        np.testing.assert_allclose(expit(values @ theta[:-1] + theta[-1]), reference.predict_proba(values)[:, column], atol=2e-5)
        self.assertTrue(state["converged"])

    def test_constant_and_default_metrics_have_correct_target_direction(self):
        labels = np.array([0, 1, 1, 1])
        constant = constant_predictions(np.array([0, 1, 1, 1, 1]), len(labels))
        np.testing.assert_allclose(constant, 0.8)
        scored = metrics(labels, constant)
        self.assertEqual(scored["repay_roc_auc"], 0.5)
        self.assertEqual(scored["default_pr_auc_ap"], 0.25)
        self.assertEqual(scored["accuracy_at_0_5"], 0.75)
        perfect = metrics(labels, np.array([0.1, 0.9, 0.8, 0.7]))
        self.assertEqual(perfect["repay_roc_auc"], 1)
        self.assertEqual(perfect["default_pr_auc_ap"], 1)
        with self.assertRaises(ValueError):
            metrics(labels, np.array([np.nan, 0.1, 0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()

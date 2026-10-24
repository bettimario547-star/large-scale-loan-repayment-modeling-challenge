"""Keep identifiers, repayment labels, and model predictors in separate roles.

Use ``id`` to align a row with folds, predictions, and the submission template.
Only ``feature_1`` through ``feature_1105`` may enter a repayment model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


FEATURE_PATTERN = re.compile(r"^feature_([1-9][0-9]*)$")
EXPECTED_FEATURE_COUNT = 1_105


def model_feature_columns(
    train_columns: Sequence[str], test_columns: Sequence[str]
) -> list[str]:
    """Return the ordered, shared predictor whitelist or fail on schema drift."""
    train = list(train_columns)
    test = list(test_columns)
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ValueError("Duplicate column names are not allowed")
    train_features = {name for name in train if FEATURE_PATTERN.fullmatch(name)}
    test_features = {name for name in test if FEATURE_PATTERN.fullmatch(name)}
    expected = {f"feature_{number}" for number in range(1, EXPECTED_FEATURE_COUNT + 1)}
    if train_features != expected or test_features != expected:
        raise ValueError("Train and test must share feature_1 through feature_1105")
    if set(train) - train_features != {"id", "label"}:
        raise ValueError("Unexpected training columns outside id, label, and features")
    if set(test) - test_features != {"id"}:
        raise ValueError("Unexpected test columns outside id and features")
    return [f"feature_{number}" for number in range(1, EXPECTED_FEATURE_COUNT + 1)]


def select_model_features(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Select explicit predictors; never use a broad drop-based fallback."""
    names = list(feature_names)
    if len(names) != EXPECTED_FEATURE_COUNT or len(names) != len(set(names)):
        raise ValueError("Expected exactly 1,105 distinct predictor names")
    expected = {f"feature_{number}" for number in range(1, EXPECTED_FEATURE_COUNT + 1)}
    if set(names) != expected:
        raise ValueError("Only feature_1 through feature_1105 may enter the model")
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("Duplicate DataFrame column names are not allowed")
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)[:5]}")
    predictors = frame.loc[:, names]
    if list(predictors.columns) != names:
        raise AssertionError("Predictor column order changed unexpectedly")
    return predictors


def align_validation_folds(train_ids: pd.Series, fold_table: pd.DataFrame) -> pd.DataFrame:
    """Return fold metadata in training-row order, matching on id rather than position."""
    required = [
        "id", "fold", "stress_feature_492_upper_tail", "stress_feature_1039_observed"
    ]
    if set(fold_table.columns) != set(required) or len(fold_table.columns) != len(required):
        raise ValueError("Unexpected validation-fold columns")
    if train_ids.isna().any() or train_ids.duplicated().any():
        raise ValueError("Training IDs must be non-null and unique")
    if fold_table["id"].isna().any() or fold_table["id"].duplicated().any():
        raise ValueError("Fold IDs must be non-null and unique")
    if len(train_ids) != len(fold_table):
        raise ValueError("Training and fold tables have different row counts")
    left = pd.DataFrame({"id": train_ids.to_numpy(), "_row_order": np.arange(len(train_ids))})
    aligned = left.merge(
        fold_table[required], on="id", how="left", sort=False,
        validate="one_to_one", indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Fold IDs do not exactly match training IDs")
    if aligned["fold"].isna().any() or not aligned["fold"].isin(range(5)).all():
        raise ValueError("Fold values must be 0 through 4")
    return aligned.sort_values("_row_order")[required].reset_index(drop=True)


def build_submission(
    test_predictions: pd.DataFrame, sample_template: pd.DataFrame
) -> pd.DataFrame:
    """Align test repayment probabilities by id in template order.

    ``test_predictions`` must have ``id,p_repay``. The Kaggle template has
    ``id,prediction``; its placeholder prediction values are never reused.
    This function does not save a file, so it cannot create a fake submission.
    """
    if list(test_predictions.columns) != ["id", "p_repay"]:
        raise ValueError("Predictions must have exactly id and p_repay columns")
    if list(sample_template.columns) != ["id", "prediction"]:
        raise ValueError("Template must have exactly id and prediction columns")
    for name, frame in [("predictions", test_predictions), ("template", sample_template)]:
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"{name} IDs must be non-null and unique")
    score = pd.to_numeric(test_predictions["p_repay"], errors="raise").to_numpy()
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("Repayment probabilities must be finite and within [0, 1]")
    if len(test_predictions) != len(sample_template):
        raise ValueError("Predictions and template have different row counts")
    aligned = sample_template[["id"]].merge(
        test_predictions, on="id", how="left", sort=False,
        validate="one_to_one", indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Prediction IDs do not exactly match template IDs")
    submission = aligned[["id", "p_repay"]].rename(columns={"p_repay": "prediction"})
    if submission["prediction"].isna().any():
        raise AssertionError("Submission alignment introduced missing scores")
    return submission

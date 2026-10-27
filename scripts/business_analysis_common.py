"""Frozen prediction inputs and reusable SQL analysis; no model fitting/inference.

Historical training metadata may contain a previous machine's absolute paths.
Resolve known artifact filenames under the current root and verify their hashes;
never rewrite historical records or weaken their recorded checksums.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sql_business_analysis.json"
SQL_DIR = ROOT / "sql/03_analysis"
SQL_FILES = tuple(SQL_DIR / name for name in (
    "01_model_predictions.sql", "02_model_performance.sql", "03_risk_bands.sql",
    "04_approval_thresholds.sql", "05_prediction_distribution.sql"))
EXPORT_TABLES = (
    "model_registry", "data_quality_summary", "feature_drift_summary",
    "model_performance_summary", "risk_band_boundaries", "risk_band_summary",
    "approval_threshold_grid", "approval_policy_results", "champion_challenger_comparison",
    "prediction_score_summary", "prediction_drift_summary")
PREDICTION_COLUMNS = ["dataset", "id", "model_version", "model_role", "prediction_kind",
    "label", "fold", "stress_feature_492_upper_tail", "stress_feature_1039_observed",
    "p_repay", "p_default"]
STRESS_COLUMNS = ["stress_feature_492_upper_tail", "stress_feature_1039_observed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    run_name: str = "sql_business_analysis"

    def __post_init__(self):
        reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                    *(f"lpt{i}" for i in range(1, 10))}
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.run_name) or self.run_name in reserved:
            raise ValueError("Use a safe lowercase run name")

    @property
    def artifacts(self): return self.root / "artifacts" / self.run_name
    @property
    def reports(self): return self.root / "reports" / self.run_name
    @property
    def database(self): return self.root / "data/interim" / f"{self.run_name}.duckdb"
    @property
    def predictions(self): return self.artifacts / "frozen_predictions.parquet"
    @property
    def registry(self): return self.artifacts / "model_registry.parquet"
    @property
    def manifest(self): return self.artifacts / "model_freeze_manifest.json"


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_config(path: Path = CONFIG) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    thresholds = config["approval_thresholds"]
    if not thresholds or thresholds != sorted(set(thresholds)) or not all(0 <= v <= 1 for v in thresholds):
        raise ValueError("Thresholds must be unique, ordered probabilities")
    if len(config["models"]) != 2 or {m["model_role"] for m in config["models"]} != {"champion", "challenger"}:
        raise ValueError("Need exactly one Champion and one Challenger")
    if len({m["model_version"] for m in config["models"]}) != 2:
        raise ValueError("Model versions must differ")
    return config


def align_by_id(frame: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Reject ambiguous/missing IDs, then align by key, never input row position."""
    if frame.id.isna().any() or frame.id.duplicated().any() or ids.isna().any() or ids.duplicated().any():
        raise ValueError("IDs must be unique and present within each dataset")
    if len(frame) != len(ids) or set(frame.id) != set(ids):
        raise ValueError("Prediction IDs do not match dataset IDs")
    return frame.set_index("id").loc[ids.to_numpy()].reset_index()


def validate_predictions(frame: pd.DataFrame) -> None:
    if list(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError("Unexpected normalized prediction schema")
    if frame.empty or frame[["dataset", "id", "model_version", "model_role", "prediction_kind"]].isna().any().any():
        raise ValueError("Missing prediction identity")
    if not frame.dataset.isin(["train", "test"]).all() or not frame.model_role.isin(["champion", "challenger"]).all():
        raise ValueError("Unknown dataset or model role")
    if frame.duplicated(["dataset", "id", "model_version"]).any():
        raise ValueError("Duplicate prediction key")
    scores = frame[["p_repay", "p_default"]].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid probability")
    np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-12, rtol=0)
    train = frame.loc[frame.dataset.eq("train")]
    test = frame.loc[frame.dataset.eq("test")]
    if not train.prediction_kind.eq("oof").all() or not train.label.isin([0, 1]).all() or not train.fold.isin(range(5)).all():
        raise ValueError("Training analysis must use labeled OOF rows")
    if train[STRESS_COLUMNS].isna().any().any():
        raise ValueError("Missing OOF stress flags")
    if not test.prediction_kind.eq("test_ensemble").all() or test[["label", "fold", *STRESS_COLUMNS]].notna().any().any():
        raise ValueError("Test labels and fold/stress metadata must be absent")
    for version, rows in frame.groupby("model_version"):
        if rows.model_role.nunique() != 1 or set(rows.dataset) != {"train", "test"}:
            raise ValueError(f"Inconsistent model identity: {version}")


def source_files(paths: Paths, config: dict, config_path: Path) -> dict:
    files = [config_path, paths.root / "data/interim/validation_folds.parquet",
             paths.root / "data/processed/sample_submission.parquet",
             paths.root / "data/interim/loan_repayment.duckdb"]
    for spec in config["models"]:
        artifact, report = paths.root / spec["artifact_directory"], paths.root / spec["report_directory"]
        files += [artifact / "oof_predictions.parquet", artifact / "test_predictions.parquet",
                  report / "run_config.json", report / "artifact_verification.json"]
        for seed in spec["seeds"]:
            for fold in range(5):
                tag = f"fold_{fold}" if len(spec["seeds"]) == 1 else f"seed_{seed}_fold_{fold}"
                files += [artifact / "models" / f"{tag}.txt", artifact / "models" / f"{tag}.json",
                          artifact / "fold_predictions" / f"{tag}.npz"]
    return {relative(p, paths.root): sha256(p) for p in files}


def metadata_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def dataset_metadata(paths: Paths, config: dict):
    train = pd.read_parquet(paths.root / "data/processed/train.parquet", columns=["id", "label"])
    test = pd.read_parquet(paths.root / "data/processed/test.parquet", columns=["id"])
    folds = align_by_id(pd.read_parquet(paths.root / "data/interim/validation_folds.parquet"), train.id)
    template = pd.read_parquet(paths.root / "data/processed/sample_submission.parquet", columns=["id"])
    align_by_id(template, test.id)
    if (len(train), len(test)) != (config["expected_train_rows"], config["expected_test_rows"]):
        raise ValueError("Unexpected dataset row counts")
    if not train.label.isin([0, 1]).all() or set(folds.fold) != set(range(5)):
        raise ValueError("Invalid labels/folds")
    return train, test, folds


def validate_model_artifacts(paths: Paths, spec: dict, train, test, folds) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    artifact, report = paths.root / spec["artifact_directory"], paths.root / spec["report_directory"]
    config = json.loads((report / "run_config.json").read_text(encoding="utf-8"))
    audit = json.loads((report / "artifact_verification.json").read_text(encoding="utf-8"))
    signature = config["training_signature"]
    if config["status"] != "completed" or audit["status"] != "passed" or audit["training_signature"] != signature:
        raise ValueError("Completed, historically audited training is required")
    fold_sha = config.get("fold_sha256", config.get("experiment", {}).get("fold_sha256"))
    if fold_sha != sha256(paths.root / "data/interim/validation_folds.parquet"):
        raise ValueError("Fold file differs from the recorded training folds")
    oof = align_by_id(pd.read_parquet(artifact / "oof_predictions.parquet"), train.id)
    test_scores = align_by_id(pd.read_parquet(artifact / "test_predictions.parquet"), test.id)
    pd.testing.assert_frame_equal(oof[["id", "label"]], train, check_dtype=False)
    pd.testing.assert_frame_equal(oof[["id", "fold", *STRESS_COLUMNS]], folds, check_dtype=False)
    seed_oof, seed_test = [], []
    maximum = 0.0
    for seed in spec["seeds"]:
        rebuilt_oof, rebuilt_test = np.full(len(train), np.nan), np.zeros(len(test))
        coverage = np.zeros(len(train), dtype=np.int8)
        for fold in range(5):
            tag = f"fold_{fold}" if len(spec["seeds"]) == 1 else f"seed_{seed}_fold_{fold}"
            record = json.loads((artifact / "models" / f"{tag}.json").read_text(encoding="utf-8"))
            model_file, checkpoint = artifact / "models" / f"{tag}.txt", artifact / "fold_predictions" / f"{tag}.npz"
            if record["training_signature"] != signature or record["positive_class"] != 1:
                raise ValueError("Model metadata differs from training identity")
            if record["features"] != [f"feature_{i}" for i in range(1, 1106)]:
                raise ValueError("Model predictor whitelist differs from the original 1,105 features")
            if sha256(model_file) != record["model_sha256"]:
                raise ValueError(f"Model checksum mismatch: {tag}")
            if "prediction_sha256" in record and sha256(checkpoint) != record["prediction_sha256"]:
                raise ValueError(f"Prediction checkpoint checksum mismatch: {tag}")
            with np.load(checkpoint) as saved:
                valid = np.flatnonzero(folds.fold.to_numpy() == fold)
                np.testing.assert_array_equal(saved["outer_valid_index"], valid)
                rebuilt_oof[valid] = saved["p_repay_valid"]
                coverage[valid] += 1
                rebuilt_test += saved["p_repay_test"] / 5
        if not (coverage == 1).all():
            raise ValueError("Each OOF row must be predicted exactly once per seed")
        column = spec["prediction_column"] if len(spec["seeds"]) == 1 else f"p_repay_seed_{seed}"
        for rebuilt, saved in ((rebuilt_oof, oof[column].to_numpy()), (rebuilt_test, test_scores[column].to_numpy())):
            np.testing.assert_allclose(rebuilt, saved, atol=1e-12, rtol=0)
            maximum = max(maximum, float(np.max(np.abs(rebuilt - saved))))
        seed_oof.append(rebuilt_oof)
        seed_test.append(rebuilt_test)
    for rebuilt, saved in ((np.mean(seed_oof, axis=0), oof[spec["prediction_column"]].to_numpy()),
                           (np.mean(seed_test, axis=0), test_scores[spec["prediction_column"]].to_numpy())):
        np.testing.assert_allclose(rebuilt, saved, atol=1e-12, rtol=0)
        maximum = max(maximum, float(np.max(np.abs(rebuilt - saved))))
    registry = {"model_version": spec["model_version"], "model_role": spec["model_role"],
        "training_signature": signature, "prediction_column": spec["prediction_column"],
        "oof_models_per_row": len(spec["seeds"]), "test_ensemble_models": len(spec["seeds"]) * 5,
        "historical_native_reload_audit": "passed", "current_checkpoint_reconstruction": "passed",
        "max_checkpoint_difference": maximum, "evaluation_scope": "reused_development_oof",
        "feature_count": 1105, "probability_calibrated": False, "production_deployed": False}
    return oof, test_scores, registry


def normalized_rows(oof, test, spec) -> pd.DataFrame:
    train_rows = oof[["id", "label", "fold", *STRESS_COLUMNS]].copy()
    test_rows = test[["id"]].copy()
    for name in ["label", "fold"]:
        test_rows[name] = pd.Series(pd.NA, index=test_rows.index, dtype="Int8")
    for name in STRESS_COLUMNS:
        test_rows[name] = pd.Series(pd.NA, index=test_rows.index, dtype="boolean")
    for frame, source, dataset, kind in ((train_rows, oof, "train", "oof"), (test_rows, test, "test", "test_ensemble")):
        frame["dataset"], frame["prediction_kind"] = dataset, kind
        frame["model_version"], frame["model_role"] = spec["model_version"], spec["model_role"]
        frame["p_repay"] = source[spec["prediction_column"]].to_numpy(dtype=np.float64)
        frame["p_default"] = 1 - frame.p_repay
        frame["label"], frame["fold"] = frame.label.astype("Int8"), frame.fold.astype("Int8")
        for name in STRESS_COLUMNS: frame[name] = frame[name].astype("boolean")
    return pd.concat([train_rows[PREDICTION_COLUMNS], test_rows[PREDICTION_COLUMNS]], ignore_index=True)


def freeze_models(paths: Paths, config: dict, config_path: Path = CONFIG) -> dict:
    """Freeze portable, immutable analysis snapshots; changed inputs need a new run."""
    inputs = source_files(paths, config, config_path)
    train, test, folds = dataset_metadata(paths, config)
    digests = {"train_id_label": metadata_digest(train), "test_id": metadata_digest(test)}
    if paths.manifest.exists():
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if inputs != manifest["input_sha256"] or digests != manifest["dataset_metadata_digests"]:
            raise ValueError("Frozen inputs changed; use a new --run-name to preserve this snapshot")
        for name, checksum in manifest["output_sha256"].items():
            if sha256(paths.root / name) != checksum:
                raise ValueError("Frozen snapshot checksum mismatch")
        print("Reusing unchanged frozen model snapshots", flush=True)
        return manifest
    rows, registry = [], []
    for spec in config["models"]:
        oof, scores, record = validate_model_artifacts(paths, spec, train, test, folds)
        rows.append(normalized_rows(oof, scores, spec))
        registry.append(record)
        print(f"Frozen {spec['model_role']}: {spec['model_version']}, {record['test_ensemble_models']} models", flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    validate_predictions(predictions)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    for frame, path in ((predictions, paths.predictions), (pd.DataFrame(registry), paths.registry)):
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    manifest = {"protocol_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": registry, "prediction_rows": len(predictions), "input_sha256": inputs,
        "dataset_metadata_digests": digests,
        "output_sha256": {relative(p, paths.root): sha256(p) for p in (paths.predictions, paths.registry)},
        "verification_scope": "current model hashes, checkpoint reconstruction and ID/label/fold alignment; historical native reload audit retained",
        "raw_features_reinferred": False, "new_model_training": False,
        "historical_absolute_paths_preserved": True, "test_labels_available": False,
        "next_stage": "approval_economics_and_experiment_design_not_started"}
    write_json(paths.manifest, manifest)
    return manifest


def execute_analysis_sql(connection, prediction_path: Path, registry_path: Path, thresholds: list[float]) -> None:
    frame = pd.DataFrame({"threshold": thresholds}, dtype=np.float64)
    connection.register("_approval_thresholds", frame)
    replacements = {"__PREDICTIONS_PARQUET__": prediction_path.resolve().as_posix().replace("'", "''"),
                    "__REGISTRY_PARQUET__": registry_path.resolve().as_posix().replace("'", "''")}
    for path in SQL_FILES:
        sql = path.read_text(encoding="utf-8")
        for token, value in replacements.items(): sql = sql.replace(token, value)
        connection.execute(sql).fetchall()
    connection.unregister("_approval_thresholds")


def copy_quality_tables(connection, source: Path) -> None:
    """Reuse physical quality tables; migrated raw CSV views need not be opened."""
    with duckdb.connect(str(source), read_only=True) as original:
        for target, source_table in (("data_quality_summary", "dq_quality_overview"),
                                     ("feature_drift_summary", "dq_train_test_drift")):
            frame = original.execute(f"SELECT * FROM {source_table}").fetchdf()
            connection.register("_quality_source", frame)
            connection.execute(f"CREATE TABLE {target} AS SELECT * FROM _quality_source").fetchall()
            connection.unregister("_quality_source")


def markdown_table(frame: pd.DataFrame, rates=(), digits=6) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for column, value in zip(frame.columns, row):
            if pd.isna(value): rendered = "n.a."
            elif column in rates: rendered = f"{value:.2%}"
            elif isinstance(value, (float, np.floating)): rendered = f"{value:.{digits}f}"
            else: rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

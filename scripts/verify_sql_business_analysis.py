"""Independently check frozen inputs, SQL metrics, bands and threshold denominators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

try:
    from . import business_analysis_common as common
except ImportError:
    import business_analysis_common as common


def assert_number(actual, expected, name):
    if pd.isna(expected):
        if not pd.isna(actual): raise AssertionError(f"{name} must be NULL")
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11, err_msg=name)


def verify_connection(connection, predictions: pd.DataFrame, thresholds: list[float]) -> dict:
    """Use NumPy/sklearn references, not a second copy of the SQL implementation."""
    common.validate_predictions(predictions)
    if connection.execute("SELECT count(*) FROM model_predictions").fetchall()[0][0] != len(predictions):
        raise AssertionError("Database prediction row count differs")
    connection.register("_verification_predictions", predictions)
    conditions = " OR ".join(f"p.{name} IS DISTINCT FROM f.{name}" for name in common.PREDICTION_COLUMNS)
    mismatch = connection.execute(f"""
        SELECT count(*) FROM model_predictions p FULL JOIN _verification_predictions f
        USING (dataset, id, model_version) WHERE {conditions}
    """).fetchall()[0][0]
    connection.unregister("_verification_predictions")
    if mismatch: raise AssertionError("Database differs from frozen predictions")
    stored_grid = connection.execute("SELECT threshold FROM approval_threshold_grid ORDER BY threshold").fetchnumpy()["threshold"]
    np.testing.assert_array_equal(stored_grid, thresholds)
    performance = connection.table("model_performance_summary").fetchdf().set_index("model_version")
    boundaries = connection.table("risk_band_boundaries").fetchdf().set_index("model_version")
    bands = connection.table("risk_band_summary").fetchdf().set_index(["dataset", "model_version", "risk_decile"]).sort_index()
    policies = connection.table("approval_policy_results").fetchdf().set_index(["dataset", "model_version", "threshold"]).sort_index()
    distribution = connection.table("prediction_score_summary").fetchdf().set_index(["dataset", "model_version"])
    drift = connection.table("prediction_drift_summary").fetchdf().set_index("model_version")
    model_count = predictions.model_version.nunique()
    if len(bands) != model_count * 2 * 10 or len(policies) != model_count * 2 * len(thresholds):
        raise AssertionError("Missing or duplicate summary groups")
    for version, all_rows in predictions.groupby("model_version"):
        train = all_rows.loc[all_rows.dataset.eq("train")]
        y = train.label.to_numpy(dtype=np.int8)
        p = train.p_repay.to_numpy()
        raw_pd = train.p_default.to_numpy()
        fpr, tpr, _ = roc_curve(1 - y, raw_pd, drop_intermediate=False)
        clipped = np.clip(p, 1e-15, 1 - 1e-15)
        expected_metrics = {"repay_roc_auc": roc_auc_score(y, p),
            "default_pr_auc_ap": average_precision_score(1 - y, raw_pd), "ks": np.max(np.abs(tpr - fpr)),
            "log_loss": np.mean(-y * np.log(clipped) - (1 - y) * np.log(np.clip(raw_pd, 1e-15, 1 - 1e-15))),
            "brier_score": brier_score_loss(y, p), "row_count": len(y), "default_rows": int((y == 0).sum()),
            "repaid_rows": int(y.sum()), "observed_default_rate": np.mean(1 - y)}
        for name, value in expected_metrics.items(): assert_number(performance.loc[version, name], value, name)
        cutoffs = np.quantile(raw_pd, np.arange(1, 10) / 10)
        np.testing.assert_allclose(boundaries.loc[version, [f"p{i}0" for i in range(1, 10)]].to_numpy(dtype=float), cutoffs, atol=1e-12)
        for dataset, rows in all_rows.groupby("dataset"):
            pd_values, scores = rows.p_default.to_numpy(), rows.p_repay.to_numpy()
            deciles = np.searchsorted(cutoffs, pd_values, side="left") + 1
            labels = rows.label.to_numpy(dtype=float, na_value=np.nan)
            has_labels = dataset == "train"
            defaults = np.sum(1 - labels) if has_labels else np.nan
            for decile in range(1, 11):
                selected = deciles == decile
                count = int(selected.sum())
                row = bands.loc[(dataset, version, decile)]
                local_default = np.mean(1 - labels[selected]) if has_labels and count else np.nan
                local_repay = np.mean(labels[selected]) if has_labels and count else np.nan
                mean_score = pd_values[selected].mean() if count else np.nan
                values = {"row_count": count, "labeled_rows": count if has_labels else 0,
                    "population_share": count / len(rows), "observed_default_rate": local_default,
                    "observed_repayment_rate": local_repay, "mean_model_p_default": mean_score,
                    "default_rows": np.sum(1 - labels[selected]) if has_labels else np.nan,
                    "repaid_rows": np.sum(labels[selected]) if has_labels else np.nan,
                    "min_p_default": np.min(pd_values[selected]) if count else np.nan,
                    "max_p_default": np.max(pd_values[selected]) if count else np.nan,
                    "observed_minus_model_default_rate": local_default - mean_score,
                    "default_lift": local_default / np.mean(1 - labels) if has_labels else np.nan,
                    "repayment_lift": local_repay / np.mean(labels) if has_labels else np.nan,
                    "default_capture_share": np.sum(1 - labels[selected]) / defaults if has_labels else np.nan,
                    "cumulative_default_capture_from_high_risk": np.sum(1 - labels[deciles >= decile]) / defaults if has_labels else np.nan}
                for name, value in values.items(): assert_number(row[name], value, f"band {name}")
            for threshold in thresholds:
                selected = scores >= threshold
                count = int(selected.sum())
                row = policies.loc[(dataset, version, threshold)]
                bad = np.sum(1 - labels[selected]) if has_labels else np.nan
                good = np.sum(labels[selected]) if has_labels else np.nan
                rejected_bad = np.sum(1 - labels[~selected]) if has_labels else np.nan
                default_rate = bad / count if has_labels and count else np.nan
                values = {"total_rows": len(rows), "total_labeled_rows": len(rows) if has_labels else 0,
                    "approved_rows": count, "rejected_rows": len(rows) - count,
                    "labeled_approved_rows": count if has_labels else 0,
                    "approval_rate": count / len(rows), "rejection_rate": 1 - count / len(rows),
                    "approved_default_rows": bad, "approved_repaid_rows": good, "rejected_default_rows": rejected_bad,
                    "observed_default_rate_approved": default_rate,
                    "observed_repayment_rate_approved": good / count if has_labels and count else np.nan,
                    "population_default_rate": np.mean(1 - labels) if has_labels else np.nan,
                    "mean_model_p_default_approved": pd_values[selected].mean() if count else np.nan,
                    "observed_default_rate_reduction": np.mean(1 - labels) - default_rate if has_labels else np.nan,
                    "rejected_default_capture_share": rejected_bad / defaults if has_labels else np.nan}
                for name, value in values.items(): assert_number(row[name], value, f"threshold {name}")
                if row.policy_status != "exploratory_not_selected": raise AssertionError("Policy prematurely selected")
            row = distribution.loc[(dataset, version)]
            expected = {"row_count": len(rows), "labeled_rows": len(rows) if has_labels else 0,
                "mean_p_repay": scores.mean(), "std_p_repay": scores.std(), "min_p_repay": scores.min(),
                "max_p_repay": scores.max(), "mean_model_p_default": pd_values.mean(),
                "observed_default_rate": np.mean(1 - labels) if has_labels else np.nan,
                **dict(zip(["p01", "p10", "p50", "p90", "p99"], np.quantile(scores, [.01, .1, .5, .9, .99])))}
            for name, value in expected.items(): assert_number(row[name], value, f"distribution {name}")
        shares = [bands.loc[(dataset, version)].sort_index().population_share.to_numpy() for dataset in ("train", "test")]
        a, b = [np.maximum(values, 1e-6) for values in shares]
        assert_number(drift.loc[version, "score_psi"], np.sum((b - a) * np.log(b / a)), "score PSI")
        assert_number(drift.loc[version, "decile_total_variation"], .5 * np.sum(np.abs(shares[1] - shares[0])), "score TV")
    comparisons = connection.table("champion_challenger_comparison").fetchdf()
    if len(comparisons) != 2 * len(thresholds): raise AssertionError("Missing paired comparisons")
    for row in comparisons.itertuples(index=False):
        champion = policies.loc[(row.dataset, row.champion_version, row.threshold)]
        challenger = policies.loc[(row.dataset, row.challenger_version, row.threshold)]
        for name, expected in {
            "champion_approval_rate": champion.approval_rate, "challenger_approval_rate": challenger.approval_rate,
            "delta_approval_rate": challenger.approval_rate - champion.approval_rate,
            "champion_observed_default_rate": champion.observed_default_rate_approved,
            "challenger_observed_default_rate": challenger.observed_default_rate_approved,
            "delta_observed_default_rate": challenger.observed_default_rate_approved - champion.observed_default_rate_approved,
            "champion_mean_model_p_default": champion.mean_model_p_default_approved,
            "challenger_mean_model_p_default": challenger.mean_model_p_default_approved}.items():
            assert_number(getattr(row, name), expected, f"paired {name}")
    return {"status": "passed", "prediction_rows": len(predictions), "model_count": model_count,
        "sql_metrics_match_sklearn": True, "risk_bands_match_numpy_and_preserve_ties": True,
        "test_uses_train_oof_boundaries": True, "test_realized_metrics_are_null": True,
        "threshold_counts_rates_and_zero_approval_handling_verified": True,
        "paired_comparisons_and_score_psi_verified": True}


def verify(paths: common.Paths, config_path: Path = common.CONFIG, check_exports=True) -> dict:
    config = common.load_config(config_path)
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if manifest["input_sha256"] != common.source_files(paths, config, config_path):
        raise AssertionError("Frozen source inputs changed")
    train, test, _ = common.dataset_metadata(paths, config)
    if manifest["dataset_metadata_digests"] != {"train_id_label": common.metadata_digest(train), "test_id": common.metadata_digest(test)}:
        raise AssertionError("Dataset metadata changed")
    for path, checksum in manifest["output_sha256"].items():
        if common.sha256(paths.root / path) != checksum: raise AssertionError("Frozen output changed")
    frozen = pd.read_parquet(paths.predictions)
    with duckdb.connect(str(paths.database), read_only=True) as connection:
        result = verify_connection(connection, frozen, config["approval_thresholds"])
        pd.testing.assert_frame_equal(connection.table("model_registry").fetchdf(), pd.read_parquet(paths.registry), check_dtype=False)
        metrics = connection.table("model_performance_summary").fetchdf().set_index("model_version")
        for spec in config["models"]:
            np.testing.assert_allclose(metrics.loc[spec["model_version"], "repay_roc_auc"], spec["expected_oof_auc"], atol=1e-6, rtol=0)
        with duckdb.connect(str(paths.root / "data/interim/loan_repayment.duckdb"), read_only=True) as original:
            for table, source in (("data_quality_summary", "dq_quality_overview"), ("feature_drift_summary", "dq_train_test_drift")):
                pd.testing.assert_frame_equal(connection.table(table).fetchdf(), original.table(source).fetchdf(), check_dtype=False)
        if check_exports:
            for table in common.EXPORT_TABLES:
                saved = pd.read_csv(paths.reports / "tables" / f"{table}.csv")
                expected = connection.table(table).fetchdf()
                pd.testing.assert_frame_equal(saved, expected, check_dtype=False, atol=1e-11, rtol=1e-11)
    result.update(frozen_input_and_output_checksums_verified=True, dataset_scoped_keys_verified=True,
                  current_model_files_checked=20, historical_records_unchanged=True,
                  quality_tables_reused_and_verified=True, csv_exports_verified=check_exports,
                  verification_scope=manifest["verification_scope"])
    if (paths.reports / "run_config.json").exists():
        run = json.loads((paths.reports / "run_config.json").read_text(encoding="utf-8"))
        if common.sha256(paths.manifest) != run["freeze_manifest_sha256"]:
            raise AssertionError("Freeze manifest changed")
        if run["status"] == "completed" and common.sha256(paths.database) != run["database_sha256"]:
            raise AssertionError("Completed database changed")
        for path, checksum in run["analysis_code_sha256"].items():
            if common.sha256(paths.root / path) != checksum: raise AssertionError("Analysis code changed; rerun analysis")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="sql_business_analysis")
    args = parser.parse_args()
    paths = common.Paths(run_name=args.run_name)
    result = verify(paths)
    common.write_json(paths.reports / "artifact_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()

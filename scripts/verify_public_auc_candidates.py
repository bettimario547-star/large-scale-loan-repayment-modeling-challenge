"""Independently reload submissions and reconstruct their scores from sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from . import build_public_auc_candidates as builder
except ImportError:
    import build_public_auc_candidates as builder


def verify(root=builder.ROOT):
    root = Path(root).resolve()
    report = root / "reports/public_auc_optimization"
    manifest = json.loads((report / "candidate_manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise RuntimeError("No candidates to verify")
    meta, test_meta, _, template, oof, test, _ = builder.load_predictions(root)
    ensemble_checks = []
    ensemble_name = "catboost_gpu_capacity_ensemble"
    if ensemble_name in oof:
        summary = json.loads((root / "artifacts/public_auc_optimization/completed" / ensemble_name / "model_summary.json").read_text(encoding="utf-8"))
        seeds = summary["complete_seeds"]
        names = [f"catboost_gpu_capacity_s{seed}" for seed in seeds]
        if len(seeds) < 2 or len(set(seeds)) != len(seeds) or not all(name in oof for name in names):
            raise RuntimeError("Incomplete or duplicate CatBoost ensemble components")
        np.testing.assert_allclose(oof[ensemble_name], np.mean([oof[name] for name in names], axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(test[ensemble_name], np.mean([test[name] for name in names], axis=0), rtol=0, atol=1e-12)
        ensemble_checks.append({"candidate": ensemble_name, "complete_seeds": seeds, "oof_and_test_means_verified": True})
    checked_sources, records = {}, []
    for entry in manifest:
        path = (root / entry["submission"]).resolve()
        if not path.is_relative_to(root / "submissions") or builder.sha256(path) != entry["sha256"]:
            raise RuntimeError("Submission path/checksum differs from its manifest")
        if entry["positive_class"] != 1 or entry["blend_method"] not in ("probability_average", "rank_average"):
            raise RuntimeError("Unknown positive class or blending method")
        for sources in entry["source_sha256"].values():
            for relative, expected_hash in sources.items():
                source = (root / relative).resolve()
                if not source.is_relative_to(root / "artifacts"):
                    raise RuntimeError("Prediction source is outside the artifacts directory")
                if source not in checked_sources:
                    checked_sources[source] = builder.sha256(source)
                if checked_sources[source] != expected_hash:
                    raise RuntimeError("Source prediction changed after candidate generation")
        blend = builder.rank_blend if entry["blend_method"] == "rank_average" else builder.probability_blend
        expected_scores = blend(test, entry["weights"])
        expected = builder.build_submission(
            pd.DataFrame({"id": test_meta.id, "p_repay": expected_scores}), template)
        actual = pd.read_csv(path)
        if actual.columns.tolist() != template.columns.tolist() or len(actual) != len(template):
            raise RuntimeError("Submission schema/row count differs from the competition template")
        if actual.id.isna().any() or not actual.id.is_unique:
            raise RuntimeError("Missing or duplicate submission IDs")
        np.testing.assert_array_equal(actual.id.to_numpy(), template.id.to_numpy())
        column = next(name for name in template.columns if name != "id")
        scores = actual[column].to_numpy()
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise RuntimeError("Invalid submission scores")
        np.testing.assert_allclose(scores, expected[column].to_numpy(), rtol=0, atol=1e-12)
        auc = float(roc_auc_score(meta.label.to_numpy(), blend(oof, entry["weights"])))
        np.testing.assert_allclose(auc, entry["overall_oof_auc"], rtol=0, atol=1e-12)
        records.append({"candidate": entry["candidate"], "submission": entry["submission"],
                        "sha256": entry["sha256"], "rows": len(actual), "positive_class": 1,
                        "min_score": float(scores.min()), "max_score": float(scores.max()),
                        "overall_oof_auc_recomputed": auc,
                        "max_csv_prediction_difference": float(np.max(np.abs(scores - expected[column].to_numpy())))})
    result = {"status": "passed", "submissions_checked": len(records),
              "distinct_source_files_checked": len(checked_sources),
              "ensemble_checks": ensemble_checks,
              "scope": "all current candidate CSV rows, source hashes and full OOF AUC; public AUC remains unobserved for unsubmitted files",
              "checks": records}
    builder.json_write(report / "candidate_file_verification.json", result)
    print(f"Verified {len(records)} complete candidate files and {len(checked_sources)} source files", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=builder.ROOT)
    verify(parser.parse_args().root)

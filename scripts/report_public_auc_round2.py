"""Compare verified round-two candidates with the exact observed public champion."""
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


def report(root, preferred):
    root = Path(root)
    directory = root / "reports/public_auc_optimization"
    history = json.loads((directory / "submission_history.json").read_text(encoding="utf-8"))
    observed = max(history, key=lambda entry: entry["public_auc"])
    manifest = json.loads((directory / "candidate_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((directory / "candidate_file_verification.json").read_text(encoding="utf-8"))
    checked = {(entry["candidate"], entry["sha256"]) for entry in verification["checks"]}
    if verification["status"] != "passed" or any((entry["candidate"], entry["sha256"]) not in checked for entry in manifest):
        raise RuntimeError("Current candidates must pass independent verification first")
    if any(entry["best_observed_submission_sha256"] != observed["sha256"] for entry in manifest):
        raise RuntimeError("Rebuild candidates after a new public score is recorded")

    meta, test_meta, folds, template, oof, test, _ = builder.load_predictions(root)
    current = next((entry for entry in manifest if entry["sha256"] == observed["sha256"]), None)
    weights = current["weights"] if current else {"catboost_gpu_capacity_s42": .5, "xgboost": .5}
    method = current["blend_method"] if current else "probability_average"
    blend = builder.rank_blend if method == "rank_average" else builder.probability_blend
    source = root / "submissions" / observed["filename"]
    if builder.sha256(source) != observed["sha256"]:
        raise RuntimeError("The recorded public champion changed")
    label_column = next(name for name in template.columns if name != "id")
    actual = builder.aligned_values(pd.read_csv(source), test_meta.id, label_column)
    expected = blend(test, weights)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12,
                               err_msg="This round's reference recipe no longer matches the public champion")
    reference = blend(oof, weights)
    y = meta.label.to_numpy()
    masks = {"overall": np.ones(len(meta), dtype=bool),
             "feature_492_upper_tail": folds.stress_feature_492_upper_tail.to_numpy(dtype=bool),
             "feature_1039_observed": folds.stress_feature_1039_observed.to_numpy(dtype=bool)}
    masks["joint_drift_stress"] = masks["feature_492_upper_tail"] & masks["feature_1039_observed"]
    reference_aucs = {group: float(roc_auc_score(y[mask], reference[mask]))
                      if np.unique(y[mask]).size == 2 else None for group, mask in masks.items()}
    fold_reference = {int(fold): float(roc_auc_score(y[folds.fold.to_numpy() == fold],
                        reference[folds.fold.to_numpy() == fold])) for fold in sorted(folds.fold.unique())}
    fold_metrics = pd.read_csv(directory / "candidate_fold_metrics.csv")
    rows = []
    for entry in manifest:
        row = {key: entry[key] for key in ("candidate", "submission", "sha256", "overall_oof_auc",
               "public_auc", "blend_method", "test_rank_correlation_to_best_observed_submission")}
        for group, auc in reference_aucs.items():
            value = entry[f"{group}_oof_auc"]
            row[f"{group}_auc_delta_to_observed_champion"] = value-auc if value is not None and auc is not None else None
        deltas = [float(record.overall_oof_auc)-fold_reference[int(record.fold)]
                  for record in fold_metrics.loc[fold_metrics.candidate == entry["candidate"]].itertuples()]
        if len(deltas) != 5:
            raise RuntimeError("Five complete validation fold metrics are required")
        row["folds_above_observed_champion"] = sum(delta > 0 for delta in deltas)
        row["min_fold_auc_delta_to_observed_champion"] = min(deltas)
        rows.append(row)
    by_name = {entry["candidate"]: entry for entry in manifest}
    if len(set(preferred)) != len(preferred) or any(name not in by_name for name in preferred):
        raise ValueError("Preferred candidates must be distinct and complete")
    result = {"status": "complete", "observed_champion": observed,
              "reference_weights": weights, "reference_blend_method": method, "reference_oof_auc": reference_aucs,
              "reference_fold_auc": fold_reference,
              "recommended_order": [by_name[name] for name in preferred if by_name[name]["public_auc"] is None],
              "verified_candidate_count": len(manifest),
              "scope": "development OOF diagnostics; new public scores require actual submission; recommendations can change after feedback"}
    pd.DataFrame(rows).to_csv(directory / "round2_candidate_comparison.csv", index=False)
    queue = pd.read_csv(directory / "submission_queue.csv")
    available = [name for name in preferred if name in set(queue.candidate)]
    order = available + [name for name in queue.candidate if name not in available]
    queue.set_index("candidate").loc[order].reset_index().to_csv(directory / "submission_queue.csv", index=False)
    builder.json_write(directory / "round2_completion_summary.json", result)
    print(f"Compared {len(rows)} verified candidates with public AUC {observed['public_auc']:.5f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=builder.ROOT)
    parser.add_argument("--preferred", nargs="+", default=["blend_catboost_diverse50_xgb_mix50",
                        "rank_catboost_diverse50_xgb_mix50", "blend_catboost_diverse50_xgb50"])
    args = parser.parse_args()
    report(args.root, args.preferred)

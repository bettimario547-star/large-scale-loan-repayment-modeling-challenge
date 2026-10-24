"""Render full-OOF metric curves and inner early-stopping curves for learning."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "logs/matplotlib_cache"))
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .run_lightgbm_baseline import ARTIFACTS, LINEAR_ARTIFACTS, REPORT, write_report
except ImportError:
    from run_lightgbm_baseline import ARTIFACTS, LINEAR_ARTIFACTS, REPORT, write_report


def sampled_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Metrics are calculated from all predictions. Only render fewer vertices.
    selected = np.unique(np.r_[np.arange(min(100, len(x))),
                               np.linspace(0, len(x) - 1, min(5000, len(x))).astype(int)])
    return x[selected], y[selected]


def render() -> None:
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    example_model = lgb.Booster(model_file=str(ARTIFACTS / "models/fold_0.txt"))
    example = example_model.dump_model()["tree_info"][0]
    names = example_model.feature_name()
    root = example["tree_structure"]

    def describe(node):
        if "split_feature" in node:
            return {"feature": names[node["split_feature"]], "threshold": node["threshold"],
                    "default_left": node["default_left"], "missing_type": node["missing_type"]}
        return {"leaf_value": node["leaf_value"], "leaf_count": node["leaf_count"]}

    tree_example = {"fold": 0, "tree_index": 0, "num_leaves": example["num_leaves"],
                    "root": describe(root), "left_child": describe(root["left_child"]),
                    "right_child": describe(root["right_child"])}
    (REPORT / "tree_example.json").write_text(json.dumps(tree_example, indent=2) + "\n", encoding="utf-8")
    del example_model, example
    fold_metrics = pd.read_csv(REPORT / "fold_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for fold in range(5):
        history = json.loads((REPORT / f"learning_curves/fold_{fold}.json").read_text(encoding="utf-8"))
        auc = np.asarray(history["inner_early_stop"]["auc"])
        rounds = np.arange(1, len(auc) + 1)
        chosen = int(fold_metrics.loc[fold_metrics["fold"] == fold, "selected_rounds"].iloc[0])
        line, = ax.plot(rounds, auc, label=f"Fold {fold}, selected={chosen}", linewidth=1.5)
        ax.scatter([chosen], [auc[chosen - 1]], color=line.get_color(), s=35)
    ax.set(xlabel="Boosting rounds", ylabel="Inner early-stop repayment ROC-AUC",
           title="Tree-count selection uses only inner holdouts")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(figures / "inner_learning_curves.png", dpi=160)
    plt.close(fig)

    oof = pd.read_parquet(ARTIFACTS / "oof_predictions.parquet")
    reference = pd.read_parquet(LINEAR_ARTIFACTS / "oof_predictions.parquet")
    pd.testing.assert_frame_equal(oof[["id", "label"]], reference[["id", "label"]])
    y = oof["label"].to_numpy()
    scores = {"Constant": reference["p_repay_constant"].to_numpy(),
              "Logistic regression": reference["p_repay_logistic"].to_numpy(),
              "LightGBM": oof["p_repay"].to_numpy()}
    metrics = pd.read_csv(REPORT / "comparison_metrics.csv").query("group == 'overall'").set_index("model")
    keys = {"Constant": "constant", "Logistic regression": "logistic_regression", "LightGBM": "lightgbm"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for name, score in scores.items():
        record = metrics.loc[keys[name]]
        fpr, tpr, _ = roc_curve(y, score)
        x, z = sampled_curve(fpr, tpr)
        axes[0].plot(x, z, label=f"{name} AUC={record['repay_roc_auc']:.4f}", linewidth=1.5)
        precision, recall, _ = precision_recall_curve(1 - y, 1 - score, drop_intermediate=True)
        x, z = sampled_curve(recall, precision)
        # Descending recall with post steps matches AP's constant-precision
        # intervals and avoids drawing a spurious triangle from the (0, 1)
        # synthetic endpoint of a constant predictor.
        axes[1].step(x, z, where="post", label=f"{name} AP={record['default_pr_auc_ap']:.4f}", linewidth=1.5)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", alpha=.5)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Repayment ROC (all OOF)")
    axes[1].set(xlabel="Default recall", ylabel="Default precision", title="Default precision-recall (all OOF)")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(figures / "oof_metric_curves.png", dpi=160)
    plt.close(fig)
    write_report(pd.read_csv(REPORT / "comparison_metrics.csv"), fold_metrics,
                 pd.read_csv(REPORT / "feature_importance.csv"),
                 json.loads((REPORT / "run_config.json").read_text(encoding="utf-8")))
    print(f"Figures saved to {figures}")


if __name__ == "__main__":
    render()

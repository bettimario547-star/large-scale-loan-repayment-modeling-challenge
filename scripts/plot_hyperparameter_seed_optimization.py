"""Plot recorded inner search, paired OOF fold gains, and model-seed stability."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import run_hyperparameter_seed_optimization as tuning
except ImportError:
    import run_hyperparameter_seed_optimization as tuning


def plot():
    report = tuning.REPORT
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    search = pd.read_csv(report / "search_results.csv")
    selected = pd.read_csv(report / "selected_candidates.csv")
    paired = pd.read_csv(report / "paired_fold_metrics.csv")
    overall = pd.read_csv(report / "comparison_metrics.csv").query("group == 'overall'").set_index("model")
    config = json.loads((report / "run_config.json").read_text())["experiment"]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 130})
    colors = ("#52657b", "#d69a23", "#167c80", "#a866b0")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for i, candidate in enumerate(tuning.CANDIDATES):
        rows = search.loc[search.candidate == candidate["name"]]
        axes[0].plot(rows.fold + 1, rows.inner_auc, marker="o", color=colors[i], label=candidate["name"])
    axes[0].scatter(selected.fold + 1, selected.inner_auc, s=100, facecolors="none", edgecolors="black", zorder=5,
                    label="Chosen (tie priority)")
    axes[0].set(title="Inner AUC used for candidate selection", xlabel="Outer fold", ylabel="Inner repayment ROC-AUC")
    axes[0].set_xticks(range(1, 6))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    for fold in range(5):
        candidate = selected.loc[selected.fold == fold, "candidate"].iloc[0]
        tag = f"{candidate}_fold_{fold}"
        auc = json.loads((report / f"inner_curves/{tag}.json").read_text())["inner_early_stop"]["auc"]
        rounds = int(selected.loc[selected.fold == fold, "selected_rounds"].iloc[0])
        axes[1].plot(np.arange(1, len(auc) + 1), auc, linewidth=1.2, label=f"Fold {fold + 1}: {candidate}")
        axes[1].scatter(rounds, auc[rounds - 1], s=25)
    axes[1].set(title="Chosen candidates: inner learning curves", xlabel="Boosting rounds", ylabel="Inner repayment ROC-AUC")
    axes[1].axvline(config["max_rounds"], color="grey", linestyle="--", alpha=0.6, label="Round budget")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.2)
    fig.savefig(figures / "inner_search.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for i, name in enumerate([f"tuned_seed_{seed}" for seed in tuning.SEEDS] + ["tuned_seed_ensemble"]):
        rows = paired.loc[paired.model == name].sort_values("fold")
        label = name.replace("tuned_seed_", "Seed ").replace("Seed ensemble", "3-seed mean")
        axes[0].plot(rows.fold + 1, rows.delta_repay_roc_auc, marker="o", color=colors[i], label=label)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Paired OOF AUC change vs original baseline", xlabel="Outer fold", ylabel="AUC difference")
    axes[0].set_xticks(range(1, 6))
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)
    for i, seed in enumerate(tuning.SEEDS):
        rows = paired.loc[paired.model == f"tuned_seed_{seed}"].sort_values("fold")
        axes[1].scatter(rows.fold + 1 + (i - 1) * 0.07, rows.repay_roc_auc, color=colors[i], label=f"Seed {seed}", s=35)
    rows = paired.loc[paired.model == "tuned_seed_ensemble"].sort_values("fold")
    axes[1].plot(rows.fold + 1, rows.repay_roc_auc, marker="D", color=colors[3], label="3-seed mean")
    axes[1].set(title="Same fold, same configuration: seed variation", xlabel="Outer fold", ylabel="OOF repayment ROC-AUC")
    axes[1].set_xticks(range(1, 6))
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.2)
    fig.savefig(figures / "seed_stability.png", dpi=160)
    plt.close(fig)
    models = ["baseline"] + [f"tuned_seed_{seed}" for seed in tuning.SEEDS] + ["tuned_seed_ensemble"]
    labels = ["Original", "Seed 42", "Seed 2026", "Seed 3407", "3-seed mean"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    for ax, metric, title in zip(axes, ("repay_roc_auc", "default_pr_auc_ap", "log_loss"),
                                 ("Repayment ROC-AUC (higher better)", "Default AP (higher better)", "Log loss (lower better)")):
        values = overall.loc[models, metric].to_numpy()
        ax.barh(np.arange(5), values, color=["#52657b", "#167c80", "#167c80", "#167c80", "#a866b0"])
        ax.set_yticks(np.arange(5), labels)
        ax.invert_yaxis()
        # Full zero-based bar axis; printed values show small differences honestly.
        ax.set_xlim(0, values.max() * 1.34)
        ax.set_title(title, fontsize=10)
        for i, value in enumerate(values):
            ax.text(value + values.max() * 0.015, i, f"{value:.6f}", va="center", fontsize=9)
    fig.savefig(figures / "overall_metrics.png", dpi=160)
    plt.close(fig)
    print(f"Created figures: {figures}", flush=True)


if __name__ == "__main__":
    plot()

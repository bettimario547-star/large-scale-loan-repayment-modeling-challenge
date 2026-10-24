"""Plot full-OOF ablation deltas and fold-local screening diagnostics."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import run_feature_screening as screening
except ImportError:
    import run_feature_screening as screening


def plot():
    report = screening.REPORT
    config = json.loads((report / "run_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "completed"
    comparison = pd.read_csv(report / "comparison_metrics.csv")
    paired = pd.read_csv(report / "paired_fold_deltas.csv")
    plans = json.loads((report / "feature_plans.json").read_text(encoding="utf-8"))
    output = report / "figures"
    output.mkdir(exist_ok=True)
    colors = ["#2375A4", "#A96927", "#B34D5F", "#607780"]
    positions = np.arange(len(screening.VARIANTS))
    names = [screening.DISPLAY[name] for name in screening.VARIANTS]
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 8.8))
    for ax, metric, title in [(axes[0, 0], "repay_roc_auc", "Overall OOF AUC change (higher is better)"),
                              (axes[0, 1], "default_pr_auc_ap", "Default AP change (higher is better)"),
                              (axes[1, 0], "log_loss", "Log-loss change (lower is better)")]:
        groups = ["overall"] if metric != "default_pr_auc_ap" else ["overall", "feature_492_upper_tail", "feature_1039_observed"]
        for group_index, group in enumerate(groups):
            frame = comparison.loc[comparison.group == group].set_index("variant")
            delta = frame.loc[list(screening.VARIANTS), metric] - frame.loc["baseline", metric]
            width = 0.22 if len(groups) > 1 else 0.60
            offset = (group_index - (len(groups) - 1) / 2) * width
            label = {"overall": "Overall", "feature_492_upper_tail": "492 upper tail", "feature_1039_observed": "1039 observed"}[group]
            ax.bar(positions + offset, delta, width=width,
                   color=["#2375A4", "#A96927", "#B34D5F"][group_index] if len(groups) > 1 else colors,
                   label=label if len(groups) > 1 else None)
            if len(groups) == 1:
                for x, value in zip(positions, delta):
                    ax.annotate(f"{value:+.6f}", (x, value), xytext=(0, 6 if value >= 0 else -15),
                                textcoords="offset points", ha="center", fontsize=9)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(positions, names, rotation=14, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.margins(y=0.22)
        if len(groups) > 1:
            ax.legend(fontsize=8)
    ax = axes[1, 1]
    for x, (variant, color) in enumerate(zip(screening.VARIANTS, colors)):
        values = paired.loc[paired.variant == variant, "delta_repay_roc_auc"].to_numpy()
        ax.scatter(x + np.linspace(-.12, .12, len(values)), values, color=color, s=42)
        ax.hlines(values.mean(), x - .22, x + .22, color=color, linewidth=2)
    ax.axhline(0, color="#333333", linewidth=.8)
    ax.set_xticks(positions, names, rotation=14, ha="right")
    ax.set_title("Paired fold AUC changes: dots = folds, line = mean")
    ax.grid(axis="y", alpha=.2)
    figure.suptitle("Feature screening at fixed baseline tree counts\nFull-data OOF changes vs raw LightGBM; rules fitted within each training fold", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output / "ablation_comparison.png", dpi=160)
    plt.close(figure)

    profile = pd.read_csv(report / "fold_feature_statistics.csv")
    first = profile.loc[profile.fold == 0].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    missing = first.nlargest(12, "abs_missing_label_gap").sort_values("abs_missing_label_gap")
    y = np.arange(len(missing))
    axes[0].barh(y - .18, missing.default_missing_rate, height=.36, label="Default rows", color="#B34D5F")
    axes[0].barh(y + .18, missing.repay_missing_rate, height=.36, label="Repaid rows", color="#2375A4")
    axes[0].set_yticks(y, missing.feature)
    axes[0].set_xlabel("Missing rate within fitting four folds")
    axes[0].set_title("Fold 0: strongest label-associated missing patterns")
    axes[0].legend(fontsize=8)
    axes[1].scatter(np.sqrt(first.observed_variance), first.dominant_state_share, s=10, alpha=.45, color="#2375A4")
    near = first.loc[first.dominant_state_share >= screening.RULES["dominant_state_share"]]
    axes[1].scatter(np.sqrt(near.observed_variance), near.dominant_state_share, s=45, color="#B34D5F", zorder=3)
    for row in near.itertuples():
        axes[1].annotate(row.feature, (np.sqrt(row.observed_variance), row.dominant_state_share),
                         xytext=(5, -14), textcoords="offset points", fontsize=8)
    axes[1].set_xscale("log")
    axes[1].axhline(screening.RULES["dominant_state_share"], color="#B34D5F", linestyle="--", label="99.5% removal rule")
    axes[1].set_xlabel("Observed standard deviation (log scale; different units)")
    axes[1].set_ylabel("Largest observed-value or NaN state share")
    axes[1].set_title("Small raw variance does not imply a constant feature")
    axes[1].set_ylim(0, 1.03)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "fold_feature_diagnostics.png", dpi=160)
    plt.close(figure)
    screening.write_report(comparison, pd.read_csv(report / "fold_metrics.csv"), plans, config)
    precision_path = report / "precision_audit.json"
    if precision_path.exists():
        precision = json.loads(precision_path.read_text(encoding="utf-8"))
        rows = [{"feature": item["feature"],
                 "original_unique": item["original_float64"]["observed_unique"],
                 "float32_unique": item["model_float32"]["observed_unique"],
                 "original_dominant_state": f"{item['original_float64']['dominant_state_share']:.4%}",
                 "float32_dominant_state": f"{item['model_float32']['dominant_state_share']:.4%}"}
                for item in precision["records"]]
        note = f"""\n## 输入精度核查\n\n{screening.base.markdown_table(pd.DataFrame(rows))}\n\n
全训练原始 DOUBLE 与基线 float32 缓存的对照说明，相近值的舍入可以明显增加状态集中度。
这一核查用于解释筛选候选，没有改动折内名单。本轮各方案均采用基线相同输入精度，
不能把 float32 中的近常数结论直接归给原始高精度列。是否恢复高精度需要单独的输入实验。

![折内特征诊断]({(output / 'fold_feature_diagnostics.png').as_posix()})\n"""
        path = report / "feature_screening_report.md"
        path.write_text(path.read_text(encoding="utf-8") + note, encoding="utf-8")
    importance = pd.concat([pd.read_csv(report / f"importance/missing_patterns_fold_{fold}.csv").assign(fold=fold)
                            for fold in range(5)])
    derived = importance.loc[importance.feature.str.startswith("missing__") | importance.feature.eq("row_missing_count")]
    gain_summary = pd.DataFrame({"derived_gain_share": derived.groupby("fold").gain_share.sum(),
                                 "row_missing_count_gain_share": importance.loc[importance.feature.eq("row_missing_count")]
                                     .set_index("fold").gain_share}).reset_index()
    gain_summary.to_csv(report / "missing_feature_gain_summary.csv", index=False)
    displayed = gain_summary.copy()
    for column in ["derived_gain_share", "row_missing_count_gain_share"]:
        displayed[column] = displayed[column].map(lambda value: f"{value:.4%}")
    note = f"\n## 新增缺失列的训练重要性\n\n{screening.base.markdown_table(displayed)}\n\n"
    note += "gain 占比只描述训练分裂对损失的改善，不表示新增特征的折外收益或因果效应。\n"
    path = report / "feature_screening_report.md"
    path.write_text(path.read_text(encoding="utf-8") + note, encoding="utf-8")
    print(f"Saved plots in {output}", flush=True)


if __name__ == "__main__":
    plot()

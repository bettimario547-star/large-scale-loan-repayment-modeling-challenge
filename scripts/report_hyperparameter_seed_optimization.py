"""Generate a beginner-readable report from completed, audited study outputs."""

import json

import pandas as pd

try:
    from . import run_hyperparameter_seed_optimization as tuning
except ImportError:
    import run_hyperparameter_seed_optimization as tuning


def table(frame, digits=6):
    frame = frame.copy()
    for column in frame.select_dtypes(include="floating").columns:
        frame[column] = frame[column].map(lambda value: f"{value:.{digits}f}")
    return tuning.base.markdown_table(frame)


def report():
    root = tuning.REPORT
    config = tuning.checked_config()
    assert config["status"] == "completed"
    experiment = config["experiment"]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "artifact_verification.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed" and audit["training_signature"] == config["training_signature"]
    comparison = pd.read_csv(root / "comparison_metrics.csv")
    selected = pd.read_csv(root / "selected_candidates.csv")
    search = pd.read_csv(root / "search_results.csv")
    paired = pd.read_csv(root / "paired_fold_metrics.csv")
    spread = pd.read_csv(root / "seed_stability.csv")
    overall = comparison.query("group == 'overall'")
    columns = ["model", "repay_roc_auc", "default_pr_auc_ap", "log_loss", "brier_score"]
    indexed = overall.set_index("model")
    ensemble, baseline = indexed.loc["tuned_seed_ensemble"], indexed.loc["baseline"]
    deltas = summary["ensemble_minus_baseline"]
    fold_table = paired.query("model == 'tuned_seed_ensemble'")[["fold", "repay_roc_auc", "delta_repay_roc_auc",
                                                               "default_pr_auc_ap", "delta_default_pr_auc_ap", "log_loss"]]
    aucs = indexed.loc[[f"tuned_seed_{seed}" for seed in tuning.SEEDS], "repay_roc_auc"]
    recommendation = summary["recommended_development_model"]
    recommendation_text = ("三种子集成满足预声明的开发推荐规则，可以作为下一步开发候选。"
                           if recommendation == "tuned_seed_ensemble" else
                           "三种子集成没有满足预声明的全部开发推荐条件，当前仍保留原 LightGBM 基线。")
    selected_cap = selected.loc[selected.hit_round_cap, "fold"].tolist()
    curve_difference = max(item["max_auc_difference"] for item in audit["baseline_reference_common_learning_curves"])
    pressure = comparison.loc[comparison.group != 'overall', ['model', 'group', 'rows', 'default_rate',
                                                             'repay_roc_auc', 'default_pr_auc_ap', 'log_loss', 'brier_score']].copy()
    pressure['default_rate'] = pressure['default_rate'].map(lambda value: f"{value:.4%}")
    notes = f"""# 第三步：超参数与随机种子优化

使用 **{experiment['train_rows']:,}** 条训练记录、**{experiment['test_rows']:,}** 条测试记录、
**{len(experiment['features']):,}** 个原始 float32 特征。`label=1` 是还款，所有模型输出还款概率。
固定已有五折、原始 NaN 处理和输入精度，比较 4 组联合配置，再对每折选中的配置重训 3 个种子。
共完成 **20 个内部搜索模型、15 个完整四折重训模型**。

三种子集成的开发 OOF AUC 为 **{ensemble.repay_roc_auc:.6f}**，相对原基线
**{baseline.repay_roc_auc:.6f}** 的变化为 **{deltas['repay_roc_auc']:+.6f}**；
违约 AP 为 **{ensemble.default_pr_auc_ap:.6f}**，变化 **{deltas['default_pr_auc_ap']:+.6f}**；
log loss 为 **{ensemble.log_loss:.6f}**，变化 **{deltas['log_loss']:+.6f}**。
{recommendation_text}
这是重复使用既有折表的开发验证，没有官方测试成绩，也未上传 Kaggle。

## 一、提前固定的实验设计

{table(pd.DataFrame(tuning.CANDIDATES))}

这是人工制定的有限候选搜索，没有调用 Optuna，也没有搜索整个参数空间。
多项参数同时变化，结果不能证明某个单独参数造成了收益。
原最多 600 轮提高到 **{experiment['max_rounds']}** 轮；连续 **{experiment['patience']}** 轮没有
超过 **{experiment['min_delta']}** 的有效 AUC 改善时早停。因此本轮也比较了增加训练预算的效果。
`max_bin=63`、分箱抽样 20,000 行、4 线程、原有行列抽样语义均沿用。
20,000 是估计边界的抽样量，不是整体模型只训练 20,000 行；分箱精度本轮没有优化。

每个外层拟合四折内部按标签分层，85% 训练、15% 早停。
四组候选共用内部划分和分箱；早停 Dataset 引用内部训练分箱。
只按内部早停 AUC 选候选，在最佳内部 AUC 的 {experiment['tie_tolerance']} 范围内
优先使用候选表中较早的配置。优先级及容忍量都提前固定，不是显著性检验。
选好树数后在完整拟合四折从头重训三个模型，外层验证不传给训练。

## 二、实际选中的配置与树数

{table(selected[['fold', 'candidate', 'inner_auc', 'selected_rounds', 'evaluated_rounds', 'hit_round_cap']])}

内部搜索达到 900 轮上限：**{summary['search_hit_cap_count']}/20**；选中配置达到上限：
**{summary['chosen_hit_cap_count']}/5**（折号：{selected_cap}）。
达到上限只是用完预算，不代表训练轮数已最优。
不同折独立选择配置，本轮评价的是这个选择流程，并不是全局统一一组参数的单模型。
结果表和代码使用折号 0–4，图中用 1–5，二者是一一对应的同一份固定划分。

![内部搜索与学习曲线]({(root / 'figures/inner_search.png').as_posix()})

## 三、完整 OOF 比较

{table(overall[columns])}

AUC、违约 AP 越高越好，log loss、Brier 越低越好。
违约 AP 使用 `average_precision_score(1-label, 1-p_repay)`；它不是梯形积分的 PR 面积。
OOF 每一行由排除该行所在折的模型预测，三个种子分别完整覆盖全量训练行一次。
总 OOF AUC 不是五折 AUC 的简单平均，因为它还包含不同折样本间的排序比较。

{table(fold_table)}

三种子集成在 **{summary['ensemble_folds_with_positive_auc_delta']}/5** 折的 AUC 高于原基线。
这些折的拟合数据相互重叠，不能把折标准差直接当成独立重复实验的置信区间。

## 四、随机种子和概率平均

模型种子提前固定为 **42、2026、3407**，每折实际建树种子是 `seed + fold`。
外层折表不变，内部划分种子固定 `42 + fold`，分箱种子也固定 `42 + fold`。
只改变 `seed`、`bagging_seed`、`feature_fraction_seed`，不改变输入、分箱、超参数和树数。
树数由种子 42 的内部搜索选出并共用，所以不是为每个种子单独寻找最佳树数。

{table(spread)}

三个完整 OOF AUC 的范围为 **{aucs.min():.6f}–{aucs.max():.6f}**。
表中的种子标准差描述同一折的建树随机性，不是正式置信区间；三个种子不足以刻画全部随机风险。
不以外层分数挑最高分种子，不搜索混合权重，全部使用等权概率平均。
测试预测是 3 个种子 × 5 折 = **15 个模型平均**，每个权重 1/15。
OOF 每条记录平均三个同折模型的概率，不混入曾经见过该记录的其他折模型。

![种子稳定性与逐折变化]({(root / 'figures/seed_stability.png').as_posix()})

![整体指标]({(root / 'figures/overall_metrics.png').as_posix()})

## 五、既有压力组

{table(pressure)}

压力组来自之前的预声明规则，本轮没有根据它们重新选候选。
两组有重叠；不同违约率影响 AP，不能直接把不同组的 AP 当成模型排名。
测试 `feature_492` 大量值超过训练取值范围；随机五折和这些压力组不能证明未见范围的预测有效。

## 六、开发推荐和边界

预声明规则：集成 AUC 比原基线至少高 0.0005，违约 AP 下降不超过 0.001，
log loss 增加不超过 0.0005 才推荐集成。这是工程选择阈值，不是业务批准或统计显著性。
本次推荐为 **{recommendation}**。
本轮选择在开发验证上发生，不能把结果称为独立最终测试成绩。
增加种子提升了模型数量与推理成本；生产是否采用需进一步验证收益与成本。

继续使用原始 float32 输入。已有 float32 舍入不会因为批次转换为 float64 而恢复，
`feature_51`、`feature_118` 的原始 DOUBLE 变化是否有用仍未验证。
没有做概率校准、阈值成本选择、时间外推验证，也没有重新删除漂移特征。

## 七、代码、运行与核查

主训练：`scripts/run_hyperparameter_seed_optimization.py`。
核查：`scripts/verify_hyperparameter_seed_optimization.py`。
图表与报告：`scripts/plot_hyperparameter_seed_optimization.py`、
`scripts/report_hyperparameter_seed_optimization.py`。
逐段代码、参数原理和面试回答：`docs/hyperparameter_seed_optimization.md`。

模型在 `artifacts/hyperparameter_seed_optimization/inner_models` 和 `models`；
分割与概率 checkpoint 在 `inner_splits`、`inner_predictions`、`fold_predictions`；
汇总文件为 `oof_predictions.parquet` 和 `test_predictions.parquet`。
三个单种子及集成本地提交为 `submissions/lightgbm_tuned_seed_*.csv`、
`submissions/lightgbm_tuned_ensemble.csv`，按官方模板 id 对齐，未上传。

核查已通过：重载 **{audit['inner_models_reloaded']} 个内部模型、{audit['refit_models_reloaded']} 个最终模型**，
重放全部早停选择和候选选择，核对分割无交叉、模型和预测校验值、OOF 恰好覆盖一次、
全部指标与稳定性、三个单种子和集成提交。
内部、OOF、测试重载预测最大差分别为 **{audit['max_inner_prediction_difference']}**、
**{audit['max_oof_difference']}**、**{audit['max_test_difference']}**。
原配置扩展与旧基线在共同训练轮数范围内，内部 AUC 曲线最大差为 **{curve_difference:.12g}**；
后续树数变化仍需结合本轮增加的轮数上限和早停耐心理解。
训练阶段记录耗时合计 **{summary['total_recorded_training_seconds']:.1f} 秒**，
这是各模型耗时之和，含共享 Dataset 的首次构建和预测；并行时不等于墙钟耗时。
已有 LightGBM 基线和特征筛选训练代码没有修改，历史结果可继续核查。
"""
    path = root / "hyperparameter_seed_report.md"
    path.write_text(notes, encoding="utf-8")
    print(f"Created report: {path}", flush=True)


if __name__ == "__main__":
    report()

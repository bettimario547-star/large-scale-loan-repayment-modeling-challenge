"""Run strict offline A/A validation without modifying any frozen A/B policies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import aa_validation_common as aa
    from .verify_ab_test_design import verify as verify_parent
    from .verify_aa_validation import verify, validate_metrics
except ImportError:
    import aa_validation_common as aa
    from verify_ab_test_design import verify as verify_parent
    from verify_aa_validation import verify, validate_metrics


def progress(message): print(message,flush=True)


def figures(output,calibration,histogram,peeking,replays):
    cache = output.artifacts/"matplotlib_cache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output.reports/"figures"
    folder.mkdir(exist_ok=True)
    selected = calibration.loc[calibration.required & calibration.alpha.eq(.05) & calibration.endpoint.eq("profit_itt")].sort_values("case_id")
    figure,axes = plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    axes[0].errorbar(np.arange(len(selected)),selected.false_positive_rate*100,
        yerr=np.vstack([(selected.false_positive_rate-selected.fpr_simultaneous_ci_low)*100,
                        (selected.fpr_simultaneous_ci_high-selected.false_positive_rate)*100]),fmt="o",capsize=5)
    axes[0].axhspan(4.5,5.5,alpha=.15,color="green",label="Predeclared acceptance envelope")
    axes[0].axhline(5,color="black",linestyle=":")
    axes[0].set_xticks(np.arange(len(selected)),selected.case_id,rotation=15,ha="right")
    axes[0].set(title="A/A profit test: familywise MC intervals",ylabel="False positive rate (%)")
    axes[0].legend(fontsize=8)
    hist = histogram.loc[histogram.case_id.eq("finite_full_current") & histogram.endpoint.eq("profit_itt")].sort_values("p_bin_lower")
    axes[1].bar(hist.p_bin_lower+.05,hist.fraction,width=.085,color="#3266a8")
    axes[1].axhline(.1,color="black",linestyle=":",label="Uniform reference; diagnostic")
    axes[1].set(title="Same Champion on both arms: p-value distribution",xlabel="p-value",ylabel="Fraction",xlim=(0,1))
    axes[1].legend(fontsize=8)
    for axis in axes: axis.grid(alpha=.2)
    figure.savefig(folder/"aa_calibration.png",dpi=160); plt.close(figure)
    figure,axes = plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    axes[0].plot(peeking.look_fraction*100,peeking.single_look_two_sided_fpr*100,marker="o",label="Each individual look")
    axes[0].plot(peeking.look_fraction*100,peeking.naive_any_look_two_sided_fpr*100,marker="o",label="Any unadjusted look")
    axes[0].axhline(5,color="gray",linestyle=":")
    axes[0].set(title="A/A: naive repeated efficacy looks inflate errors",xlabel="Enrollment progress (%)",ylabel="False positive rate (%)")
    axes[0].legend(fontsize=8)
    axes[1].errorbar(replays.replicate,replays.delta,yerr=np.vstack([replays.delta-replays.ci_low,replays.ci_high-replays.delta]),fmt=".",alpha=.65,color="#3266a8")
    axes[1].axhline(0,color="black",linestyle=":")
    axes[1].set(title="Actual hashing, same fixed Champion outcomes",xlabel="Predeclared hash replicate",ylabel="Simulated profit difference with 95% CI")
    for axis in axes: axis.grid(alpha=.2)
    figure.savefig(folder/"aa_hash_and_peeking.png",dpi=160); plt.close(figure)


def write_report(output,config,champion,assigned,metrics,quality,balance,calibration,srm,peeking,replays,negative,bootstrap,planned_n):
    required = calibration.loc[calibration.required]
    required_passed = bool(required.calibration_passed.all())
    familywise_cutoff = quality.srm_alpha.iloc[0]/len(quality)
    quality_passed = bool(quality.srm_p_value.ge(familywise_cutoff).all())
    balance_passed = bool(balance.standardized_mean_difference.abs().le(config["max_absolute_covariate_smd"]).all())
    hash_rejections = int(replays.reject_at_005.sum())
    hash_consistency_p = float(stats.binomtest(hash_rejections,len(replays),.05).pvalue)
    passed = required_passed and quality_passed and balance_passed and bool(srm.calibration_passed.all()) and bool(negative.detected_as_expected.all()) and hash_consistency_p >= .001
    a,b = [assigned.loc[assigned.arm.eq(arm),"simulated_observed_net_profit"].to_numpy() for arm in ("A","B")]
    snapshot = aa.experiment.mean_comparison(a,b)
    diagnostic = aa.experiment.decision_diagnostics(assigned,quality,aa.pipeline_config(aa.experiment.load_config(),config))
    readiness = aa.analysis_readiness(len(assigned),planned_n,False,True,False)
    stress_failures = calibration.loc[~calibration.required & ~calibration.calibration_passed,["case_id","endpoint","alpha","false_positive_rate","fpr_simultaneous_ci_low","fpr_simultaneous_ci_high"]]
    summary = {"status":"completed","offline_aa_validation_passed":bool(passed),"required_regimes_calibrated":required_passed,
        "scope":config["scope"],"champion_definition":champion,"champion_retained":True,"challenger_promoted":False,
        "live_experiment_completed":False,"production_rollout_ready":False,"aa_snapshot_rows":len(assigned),
        "snapshot_welch":snapshot,"snapshot_quality_familywise_passed":quality_passed,"snapshot_balance_passed":balance_passed,
        "actual_hash_repetitions":len(replays),"actual_hash_profit_rejections":hash_rejections,
        "actual_hash_fpr_consistency_binomial_p":hash_consistency_p,"actual_hash_check_precision_is_limited":True,
        "calibration_repetitions_per_case":config["calibration_repetitions"],"required_calibration_rows":len(required),
        "stress_regimes_not_certified":stress_failures.astype(object).where(pd.notna(stress_failures),None).to_dict(orient="records"),
        "snapshot_guardrails_are_not_zero_effect_equality_tests":diagnostic["guardrails"],
        "live_analysis_readiness":readiness,"all_negative_controls_detected":bool(negative.detected_as_expected.all()),
        "paired_zero_is_identity_check_only":True,
        "next_step":"AB_feasibility_redesign_pending_no_automatic_run"}
    aa.business.write_json(output.reports/"summary.json",summary)
    aa.business.write_json(output.artifacts/"model_governance_decision.json",{
        "decision":"retain_champion_do_not_promote_challenger","champion_definition":champion,
        "basis":"human_decision_and_prior_ab_evidence_not_sufficient_for_promotion",
        "aa_validation_scope":config["scope"],"offline_aa_validation_passed":bool(passed),
        "new_ab_simulation_completed":False,"production_action_taken":False})
    aa.business.write_json(output.artifacts/"aa_protocol.json",{
        "configuration":config,"champion_definition":champion,"same_champion_both_arms":True,
        "eligibility":"unchanged_frozen_parent_AB_common_gate_and_strata",
        "score_b_is_eligibility_covariate_not_treatment_score":True,"registration":"local_predeclared_design_not_live_preregistration",
        "live_experiment_completed":False,"champion_retained":True,"challenger_promoted":False})
    display = required.loc[required.alpha.eq(.05),["case_id","endpoint","n_total","false_positive_rate","fpr_simultaneous_ci_low","fpr_simultaneous_ci_high","zero_effect_interval_coverage","calibration_passed"]]
    stressed = calibration.loc[~calibration.required & calibration.alpha.eq(.05),display.columns]
    report = f"""# 严格 A/A 离线校验：Champion 保留与实验管线校准

## 结果与治理决定

已完成同一 Champion 的 A/A 管线校验、真实哈希重复、零效应假阳性率与区间覆盖率、SRM 校准、独立组 Bootstrap、负向控制和停止逻辑演练。
**Champion 保留，Challenger 暂不晋级；本轮没有优化 A/B 方案、重新选择阈值或重做 Challenger 对照。**
主实验场景的离线 A/A 校验通过状态为 **{passed}**，必检场景校准通过状态为 **{required_passed}**。
这是针对当前历史分布及统计实现的离线校验，不是线上 A/A、业务安全证明或真实因果实验。

两组内部仍使用 `A/B` 字段以复用现有代码，展示时理解为 A1/A2；两组均为
**{champion['model_version']}，阈值 {champion['threshold']:.16f}**。
每一申请两组的处理分数、审批决定和潜在现金流完全相同；逐申请潜在收益差为零。
原 A/B 的共同入组资格与分数层保持固定。原 `score_b` 仅为冻结的入组协变量，
本轮 B 组实际处理分数 `treatment_score_b` 与 Champion 分数完全相同。

## 1. 一次完整管线 A/A

使用代理合格的策略评价 **{len(assigned):,}** 行。开发行和无标签测试行不进入实验分组。
SHA256 分流概率、共享准入范围、分数层、ITT 全申请分母沿用原协议；两组借贷现金流均来自 Champion。
获批还款/违约/拒绝收益仍为 690/-8510/-10 元，收回本金不是收益，拒绝成本进入主指标。
真实 DPD、客户标识和成熟日期仍不可观测，不填零或伪造。

{aa.business.markdown_table(metrics, rates=('approval_rate','simulated_default_incidence_itt','simulated_default_rate_approved','simulated_repayment_rate_approved'), digits=2)}

一次独立组 Welch 模拟差值为 **{snapshot['delta']:+.2f} 元**，95% 区间 **[{snapshot['ci_low']:.2f}, {snapshot['ci_high']:.2f}]**，p={snapshot['simulation_p_value']:.4f}。
一次不显著不能证明管线可靠；零效应下也允许按预定概率出现显著结果，不要求每个随机种子都不显著，也不挑选“好看”的盐值。
配对零差只检查两组策略身份一致，不作为独立组 SE、功效、p 值或有效置信区间的依据。
独立组 Bootstrap 稳健性演练：

{aa.business.markdown_table(bootstrap, digits=3)}

## 2. 分流与协变量质量

{aa.business.markdown_table(quality, rates=('expected_b_fraction','actual_b_fraction'), digits=6)}

本次单一快照对整体和分层 SRM 共 {len(quality)} 次检查，采用 Bonferroni 阈值
{familywise_cutoff:.6f} 控制其家族误报率；原表的 `srm_flag` 仍保留原逐项阈值，不能与家族规则混用。
分组前协变量的标准化均值差仅作为质量检查，阈值 |SMD|≤{config['max_absolute_covariate_smd']:.2f}。
不使用标签进行分层、配额调整或重新分组。

{aa.business.markdown_table(balance, digits=6)}

真实哈希重复 **{len(replays)} 次**，每个盐值均调用原 `assign()`，重新核验键、组概率、同一政策与现金流；
收益检验 p<0.05 共 **{hash_rejections} 次**，按零效应 5% 的精确二项一致性 p={hash_consistency_p:.4f}。
{len(replays)} 次真实哈希只是实现抽查，误报率精度有限，不能单独宣称“误报率严格等于 5%”。
所有预声明盐值均记录，不按结果剔除或重抽。

## 3. 重复零效应统计校准

每个场景 **{config['calibration_repetitions']:,} 次**。所有分布来自同一 Champion，禁止使用不同策略分布再减去均值差冒充 A/A。

必检场景：固定当前评价总体的 Bernoulli 随机化；同一 Champion 经验分布的独立组当前规模；同一分布的原 A/B 规划规模。
有限总体模拟对每个相同现金流/审批/违约事件单元抽 B 组二项人数，A 为互补人数，等价于理想独立 Bernoulli 逐行分流；
它不会重跑 SHA256，因此与真实哈希抽查分开报告。
独立组场景分别从同一 Champion 经验分布有放回抽样，固定组人数近似规划比例。模拟比历史规模大不意味着新增真实客户。

收益使用与原代码一致的 Welch 检验；审批率和 ITT 违约发生率用原护栏标准误对应的双侧零差 Wald 检验。
这是 equality 零效应校准；原非劣效护栏的零差位于备择内部，护栏通过率不能冒充其边界 Type I error 校准。
本轮没有证明非劣效边界检验、相关客户/时间数据或未来优化估计器已经校准。

检验水平 1%、5%、10%；可接受假阳性率偏离分别为 ±0.3、±0.5、±0.7 个百分点。
每一项用 Clopper–Pearson 精确二项区间，并对全部场景/端点/alpha/SRM 检查进行 Bonferroni 校正，
同时置信水平至少 {1-config['calibration_familywise_error']:.0%}。
只有整个区间落入预声明容许范围且没有无效重复，才认证该场景；不能只看点估计或发现不显著就宣布通过。
区间覆盖率是相应零差检验接受率的补数，不是第二份独立证据。

5% 水平的必检结果：

{aa.business.markdown_table(display, rates=('false_positive_rate','fpr_simultaneous_ci_low','fpr_simultaneous_ci_high','zero_effect_interval_coverage'), digits=4)}

SRM 的有限总体二项校准：

{aa.business.markdown_table(srm, rates=('nominal_srm_alpha','false_srm_flag_rate','simultaneous_ci_low','simultaneous_ci_high','upper_limit'), digits=5)}

压力场景仅用于识别使用限制，未被事后移除或混入“全部通过”的说法；未认证可能是失准或区间不足以证明精度。

{aa.business.markdown_table(stressed, rates=('false_positive_rate','fpr_simultaneous_ci_low','fpr_simultaneous_ci_high','zero_effect_interval_coverage'), digits=4)}

完整 1%/5%/10% 表与 p 值分箱见 CSV。p 值直方图以均匀分布为诊断参考；离散性/近似检验会影响形状，不单凭图形认证。

![A/A 校准与 p 值诊断](figures/aa_calibration.png)

## 4. 负向控制与停止逻辑

{aa.business.markdown_table(negative)}

故意注入 B 组收益偏移、55% 分流、标签决定分组/准入、重复 ID、伪造 DPD 与获批口径分母，确认相应核查能拒绝错误。
这些是隔离的故障注入测试，不进入实际 A/A 快照、不用于政策收益分析。
注入 +50 元仅测试收益检验敏感性，不是本轮优化目标或新的 MDE。

停止规则负向控制分别检查样本不足、未成熟、SRM 未解决和提前收益分析。
当前真实分析就绪始终为 false：历史没有成熟日期，当前规模也未达到原固定样本量。
安全暂停可保留；提前安全暂停不授权提前宣布收益成功。

在同一零效应流量上演练未经调整的 5 次查看，使用累计样本，而不是 5 次独立实验：

{aa.business.markdown_table(peeking, rates=('look_fraction','single_look_two_sided_fpr','naive_any_look_two_sided_fpr','single_look_positive_ci_rate','naive_any_look_positive_ci_rate'), digits=4)}

双侧错误与“B 正向区间超过零”错误分别对应 5% 与 2.5% 单次名义水平，不混淆二者。
这张表演示可选停止的危害，不为真实实验引入任何未经 alpha 分配的中期收益检验。

![真实哈希重复与提前查看](figures/aa_hash_and_peeking.png)

## 5. 边界与后续

SQL、键/概率、逐行现金流、零效应模拟和预声明校验规则可核验，不等于证明真实准入、客户去重、现金流成熟或生产风险。
Kaggle 申请行无法识别重复客户，既有 OOF 多轮用于开发；校准条件于固定历史分布，不覆盖未知相关性、选择偏差、未校准分数和未来漂移。
测试全体审批率约 17%、原共同代理测试覆盖率约 23% 的漂移问题仍然存在。
本轮治理文件 `model_governance_decision.json` 记录保留 Champion；未来优化 A/B 时若换估计器、目标人群、分组或停止规则，应重新做匹配的 A/A 校准。
下一步 A/B 可行性优化与重新模拟尚未执行。

复现：`docs/aa_validation.md`。数据库 `data/interim/{output.run_name}.duckdb`；13 份 CSV、快照/重复样本 Parquet、两张图与冻结清单。
"""
    (output.reports/"aa_validation_report.md").write_text(report,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="aa_validation")
    args = parser.parse_args()
    output,config = aa.paths(args.run_name),aa.load_config()
    start = time.perf_counter()
    progress("Verifying the frozen A/B stage (read-only)")
    verify_parent(aa.paths(config["parent_run_name"],output.root))
    if aa.check_freeze(output,config):
        audit = verify(output,progress=progress)
        progress(f"Reused frozen A/A validation: {output.reports/'aa_validation_report.md'}")
        return
    output.artifacts.mkdir(parents=True,exist_ok=True)
    output.reports.mkdir(parents=True,exist_ok=True)
    (output.reports/"tables").mkdir(exist_ok=True)
    run = {"status":"running","config":config,"input_sha256":aa.input_hashes(output,config),"protocol_sha256":aa.protocol_hashes(output)}
    aa.business.write_json(output.reports/"run_config.json",run)
    source,population,parent_config,champion,planned_n = aa.load_inputs(output,config)
    derived = aa.pipeline_config(parent_config,config)
    assigned = aa.experiment.assign(population,derived)
    aa.validate_assignments(source,population,assigned,parent_config,config,champion)
    progress(f"A/A snapshot: {len(assigned):,} rows; Champion on both arms")
    cells = aa.null_cells(population)
    cal,hist,srm,draws = aa.calibration(cells,planned_n,parent_config,config,progress)
    replays = aa.hash_replays(population,source,parent_config,config,champion,progress)
    peeking = aa.peek_simulation(cells,config)
    quality,balance = aa.experiment.assignment_quality(assigned,derived),aa.balance(assigned)
    bootstrap,bootdraws = aa.snapshot_bootstrap(assigned,config)
    temporary = output.database.with_name(f".{output.run_name}_{time.time_ns()}.duckdb")
    output.database.parent.mkdir(parents=True,exist_ok=True)
    try:
        with duckdb.connect(str(temporary)) as connection:
            connection.execute("SET threads=4")
            connection.execute("SET memory_limit='1GB'")
            connection.register("_population",population)
            connection.register("_assignments",assigned)
            connection.execute(aa.SQL_FILE.read_text(encoding="utf-8"))
            metrics = connection.table("experiment_metrics").fetchdf().sort_values("arm")
            validate_metrics(metrics,assigned)
            negative = aa.negative_controls(source,population,assigned,parent_config,config,champion,metrics,validate_metrics)
            frames = {"assignment_quality":quality,"aa_covariate_balance":balance,"aa_null_cells":cells,
                "aa_hash_replays":replays,"aa_calibration_results":cal,"aa_pvalue_histogram":hist,
                "aa_srm_calibration":srm,"aa_optional_stopping":peeking,"aa_negative_controls":negative,
                "aa_snapshot_bootstrap":bootstrap}
            for table,frame in frames.items():
                connection.register("_frame",frame)
                connection.execute(f"CREATE TABLE {table} AS SELECT * FROM _frame")
                connection.unregister("_frame")
            for table in [*aa.SNAPSHOT_TABLES,*aa.EXTRA_TABLES,"aa_snapshot_bootstrap"]:
                connection.execute(f"COPY {table} TO ? (FORMAT CSV,HEADER)",[str(output.reports/"tables"/f"{table}.csv")])
            connection.execute("CHECKPOINT")
        temporary.replace(output.database)
    finally:
        if temporary.exists(): temporary.unlink()
    assigned.to_parquet(output.artifacts/"aa_assignments.parquet",index=False)
    draws.to_parquet(output.artifacts/"aa_calibration_draws.parquet",index=False)
    bootdraws.to_parquet(output.artifacts/"aa_bootstrap_draws.parquet",index=False)
    figures(output,cal,hist,peeking,replays)
    write_report(output,config,champion,assigned,metrics,quality,balance,cal,srm,peeking,replays,negative,bootstrap,planned_n)
    progress("Independently reproducing A/A evidence before freezing")
    verify(output,require_freeze=False,deep=True,progress=progress)
    files = [output.database,*sorted(output.artifacts.glob("*.parquet")),output.artifacts/"aa_protocol.json",
             output.artifacts/"model_governance_decision.json",output.reports/"summary.json",output.reports/"aa_validation_report.md",
             *sorted((output.reports/"tables").glob("*.csv")),*sorted((output.reports/"figures").glob("*.png"))]
    manifest = {"scope":config["scope"],"input_sha256":aa.input_hashes(output,config),"protocol_sha256":aa.protocol_hashes(output),
        "output_sha256":{aa.business.relative(path,output.root):aa.business.sha256(path) for path in files},
        "champion_retained":True,"challenger_promoted":False}
    aa.business.write_json(output.artifacts/"aa_freeze_manifest.json",manifest)
    audit = verify(output,deep=False)
    aa.business.write_json(output.reports/"artifact_verification.json",audit)
    run.update(status="completed",elapsed_seconds=time.perf_counter()-start,
        manifest_sha256=aa.business.sha256(output.artifacts/"aa_freeze_manifest.json"))
    aa.business.write_json(output.reports/"run_config.json",run)
    progress(f"A/A completed and verified: {output.reports/'aa_validation_report.md'}")


if __name__ == "__main__": main()

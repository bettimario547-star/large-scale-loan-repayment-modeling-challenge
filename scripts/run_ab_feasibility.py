"""Freeze a development-only design, re-calibrate it, then simulate Challenger."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

try:
    from . import ab_feasibility_common as f
except ImportError:
    import ab_feasibility_common as f


def progress(message): print(message,flush=True)


def lock_design(output,config,protocol,strata,candidates):
    files = [output.artifacts/"design_protocol.json",output.reports/"planning_strata.csv",output.reports/"design_candidates.csv"]
    path = output.artifacts/"preanalysis_design_lock.json"
    if path.exists():
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock["input_sha256"]!=f.input_hashes(config) or lock["protocol_sha256"]!=f.protocol_hashes():
            raise AssertionError("Preanalysis design changed: use a new run-name, not a new seed")
        if json.loads(files[0].read_text(encoding="utf-8"))!=protocol: raise AssertionError("Planning changed")
        for name,digest in lock["design_file_sha256"].items():
            if f.business.sha256(f.ROOT/name)!=digest: raise AssertionError("Design file changed")
        return lock
    f.business.write_json(files[0],protocol)
    strata.to_csv(files[1],index=False)
    candidates.to_csv(files[2],index=False)
    lock = {"input_sha256":f.input_hashes(config),"protocol_sha256":f.protocol_hashes(),
        "design_file_sha256":{f.business.relative(p,f.ROOT):f.business.sha256(p) for p in files},
        "evaluation_outcomes_used_to_select_design":False,
        "registration":"local_development_only_lock_before_new_calibration_and_AB_replay_not_independent_preregistration"}
    f.business.write_json(path,lock)
    return lock


def plots(output,candidates,calibration,simulation,draws):
    os.environ["MPLCONFIGDIR"] = str(output.artifacts/"matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output.reports/"figures"
    folder.mkdir(exist_ok=True)
    fig,ax = plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    labels = ["Original all eligible","D-only / 50% exposure","D-only / 100% exposure"]
    ax[0].bar(labels[:len(candidates)],candidates.enrollment_days,color=["#a6a6a6","#4278ab","#39866b"][:len(candidates)])
    ax[0].set_yscale("log")
    ax[0].axhline(365,linestyle=":",color="black",label="365-day enrollment limit")
    ax[0].set(ylabel="Assumed enrollment days (log scale)",title="Same 5-unit precision gap; maturity unchanged")
    ax[0].tick_params(axis="x",labelrotation=15)
    ax[0].legend(fontsize=8)
    selected = calibration.loc[calibration.alpha.eq(.05) & calibration.endpoint.eq("profit_equality")]
    ax[1].errorbar(np.arange(len(selected)),selected.false_positive_rate*100,
        yerr=np.vstack([(selected.false_positive_rate-selected.simultaneous_ci_low)*100,
                       (selected.simultaneous_ci_high-selected.false_positive_rate)*100]),fmt="o",capsize=5)
    ax[1].axhspan(4.5,5.5,color="green",alpha=.15)
    ax[1].axhline(5,color="black",linestyle=":")
    ax[1].set_xticks(np.arange(len(selected)),selected.case_id,rotation=15,ha="right")
    ax[1].set(title="Matched same-Champion A/A calibration",ylabel="False positive rate (%)")
    fig.savefig(folder/"feasibility_and_calibration.png",dpi=160); plt.close(fig)
    fig,ax = plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    for case,label in (("historical_at_planned","Historical empirical effect"),("synthetic_true_gain_10","Synthetic +10 planning alternative")):
        values = draws.loc[draws.case_id.eq(case),"profit_delta"]
        ax[0].hist(values,bins=55,density=True,alpha=.45,label=label)
    ax[0].axvline(5,color="black",linestyle=":",label="Business boundary +5")
    ax[0].set(title="Independent stratified A/B simulations",xlabel="Full eligible profit gain per application",ylabel="Density")
    ax[0].legend(fontsize=8)
    cases = simulation.case_id.tolist()
    x = np.arange(len(cases))
    ax[1].bar(x-.18,simulation.positive_zero_rate,width=.35,label="Lower CI > 0")
    ax[1].bar(x+.18,simulation.joint_success_rate,width=.35,label="Lower CI > 5 and all risk guards")
    ax[1].set_xticks(x,cases,rotation=25,ha="right")
    ax[1].set(ylabel="Simulation probability",ylim=(0,1),title="Statistical gain and business evidence differ")
    ax[1].legend(fontsize=8)
    fig.savefig(folder/"ab_business_evidence.png",dpi=160); plt.close(fig)


def report(output,config,protocol,candidates,calibration,simulation,est,quality,coverage,boot,duration):
    chosen = candidates.loc[candidates.selected].iloc[0]
    target = duration.loc[duration.coverage_source.eq("planning_development") & duration.daily_applications.eq(config["primary_daily_applications"]) & duration.eligible_fraction.eq(config["primary_eligible_fraction"])].iloc[0]
    risk_passed = est["default_ni_passed"] and est["approval_ni_passed"] and est["absolute_risk_passed"]
    data = {"status":"completed","scope":config["scope"],"protocol":protocol,"replay":est,
        "matched_aa_and_boundary_calibration_passed":bool(calibration.passed.all()),
        "quality_passed":bool(quality.passed.all()),"all_risk_guards_passed_in_simulation":bool(risk_passed),
        "required_fixed_sample_reached":int(coverage.loc[coverage.policy_cohort.eq("policy_evaluation"),"disagreements"].iloc[0])>=protocol["required_randomized_disagreements"],
        "live_outcome_maturity_validated":False,"champion_retained":True,"challenger_promoted":False,
        "live_experiment_completed":False,"production_rollout_ready":False,
        "traffic_assumptions_unchanged_except_disagreement_exposure":True,
        "primary_assumed_enrollment_days":int(target.enrollment_days),"primary_assumed_final_analysis_days":int(target.final_analysis_days)}
    f.business.write_json(output.reports/"summary.json",data)
    f.business.write_json(output.artifacts/"model_governance_decision.json",{
        "decision":"retain_champion","challenger_promoted":False,
        "business_lower_bound_exceeds_5_in_simulation":est["business_value_demonstrated_in_simulation"],
        "mature_live_outcomes_available":False,"production_action_taken":False,
        "basis":"require business evidence, matching calibration, full risk guards, fixed sample and real maturity"})
    diagnostics = pd.DataFrame([est])
    text = f"""# A/B 可行性优化与新方案离线模拟

## 结论

新方案已完成匹配的 A/A 与边界校准、固定盐值实际分流及新的 A/B 模拟。Champion 继续保留，Challenger 不晋级。
匹配校准全部通过：**{bool(calibration.passed.all())}**。本轮属于固定历史标签和假设现金流的离线设计，未完成真实因果实验或生产安全验证。

原方案需 {int(candidates.iloc[0].required_randomized):,} 个随机申请，原假设下入组 {int(candidates.iloc[0].enrollment_days):,} 天。
新方案需 **{protocol['required_randomized_disagreements']:,} 个分歧申请**，在每日 {config['primary_daily_applications']:,} 申请、共同合格占比 {config['primary_eligible_fraction']:.0%} 的原流量假设下，
选用合格分歧申请 {chosen.disagreement_exposure:.0%} 暴露、A:B=1:1，预计入组 **{int(target.enrollment_days)} 天**。
成熟期仍为 {config['outcome_maturity_days_assumed']} 天，宽限 {config['data_grace_days']} 天；最终分析约 **{int(target.final_analysis_days)} 天**，并未变成短周期贷款实验。
这是条件可行性，不是已确认流量或上线批准；低流量情景仍可能不可行。

## 1. 业务问题和成功契约

主问题：新政策在完整共同合格申请人群的平均净收益，是否比 Champion 高出 **5 元/申请**以上？
主检验 H0: Δ收益≤5；使用 95% 双侧区间的下界>5，等价于单侧 α=2.5%。下界>0 仅为次级描述。
规划备择为 Δ=10，距离业务边界仍为 5 元，边际功效 80%，方差安全系数 1.25；未通过放宽收益门槛或风险护栏来改善可行性。
原 MDE=5 针对零差，不是证明 Δ>5 的 80% 功效点。真值恰好为 5 时，业务成功误报上限应是 2.5%，不是 80%。

保持 ITT 违约发生率非劣效界值 0.2 个百分点、审批率降幅界值 2 个百分点、完整 B 获批组合违约率上界 5%。
联合成功率单独模拟，不把各项边际 80% 功效当作联合 80% 保证。

## 2. 分歧随机化和完整人群估计

原共同代理 gate `min(score_A,score_B)>=0.90`、模型与完整精度阈值保持冻结。
在 gate 内仅对两政策审批不同的申请随机化；标签不决定准入、方向、分层、比例或盐值。
随机分组前已可观测的 `risk_stratum × 审批分歧方向` 用于分析分层，1:1 稳定 SHA256 Bernoulli 分流。
这不是精确区组平衡；组内实际样本数及 SRM 必须核查。未来需要持久借款人 ID、首个申请和独立 pilot。

完整人群效应估计为 `Σ_h (N_h/N_eligible) × (mean_B,h−mean_A,h)`，独立组方差为
`Σ_h weight_h² × (var_A,h/n_A,h + var_B,h/n_B,h)`，收益用 Satterthwaite 自由度。
权重来自同时期全体合格申请的分组前协变量普查，分层内所有随机申请都进入分母；未用历史配对差值方差规划随机组样本。
规划权重和方差仅取 policy_development；评价/测试协变量漂移单独报告，不按评价收益重新选样本或参数。

两策略决策一致的申请继续 Champion。在本项目固定条款/费用/现金流假设下，其政策效应逐申请严格为零，
故排除这些申请不丢失政策收益对照信号。若真实模型改变额度、价格、条款、服务或共同申请行为，这个恒等前提失效，必须重新设计完整随机化。
分歧样本上的较大收益差不能直接当作全体收益；上述权重把问题保持在原完整共同合格范围。
新方案约改变 {chosen.changed_decision_fraction_of_eligible:.2%} 合格申请的审批；原协议实际决策改变暴露约 {candidates.iloc[0].changed_decision_fraction_of_eligible:.2%}。
全部分歧进入实验并不表示风险已获验证，其决策改变暴露高于原正式流量假设，仍需 shadow/pilot 安全审核。

审批率差由两冻结政策在完整协变量普查上的决定直接计算，没有必要用随机分组噪声检验已知决定。
违约差使用全人群加权独立组估计。完整 B 获批违约率上界由一致批准组合和 B 分歧批准单元的
Clopper–Pearson 单侧上界按 Bonferroni 联合后加权，再除以已知完整 B 批准比例；同时纳入一致组合的风险不确定性。
这是完整组合的原风险上限，不能套用分歧批准子集比率冒充完整组合。局部高风险仍须业务审核，真实 DPD 保持 NULL。

## 3. 开发阶段规划及流量

候选比较与选中方案：

{f.business.markdown_table(candidates,digits=2)}

开发方差、完整权重和必要的最小单元数在 planning_strata.csv 与 design_protocol.json。
正式入组固定 N={protocol['required_randomized_disagreements']:,}；每单元每组至少 {config['minimum_per_arm_per_stratum']} 个成熟有效申请作为分析质量门槛。
比例与分层漂移或质量不达标时，保留 Champion，重新规划独立的新实验；不得看收益后继续加样本。
样本量同时考虑收益、相对违约护栏和最小单元；绝对风险与所有门槛的联合功效由下表实测，不作边际功效替代。

原全人群独立组正态比较需约 372 万，是相同 5 元精度间隔的参照；原成功规则与本轮严格业务边界检验不同。
原审批/模型/现金流和冻结产物未改。流量、合格占比和成熟期均为模拟假设，测试没有标签；
测试协变量只用于覆盖与流量敏感性，不提供未来收益或安全结论。

{f.business.markdown_table(coverage,rates=('disagreement_fraction_of_all','disagreement_fraction_of_eligible'),digits=2)}

全部流量情景见 experiment_duration.csv，包括低流量不可行、测试分歧比例敏感性及完整成熟周期。

## 4. 修改统计实现后的匹配校准

每个校准场景 {config['calibration_repetitions']:,} 次：当前有限总体、当前规模 IID、规划规模 IID 三个同 Champion A/A；
另有 Δ收益=5 和 Δ违约发生率=0.002 的统计边界校准。边界场景调整 B 批准单元的二项违约概率，
是明确标识的合成压力测试，不是同策略 A/A，也不是真实政策表现。
同策略 A/A 校准收益和违约的双侧零差；审批决定在分层内相同，只作恒等检查，不对零方差制造 p 值。
收益业务边界、非劣效边界则分别校准单侧 2.5% 的错误成功概率，修补上一轮尚未完成的边界验证。
全部 20 项同时 Clopper–Pearson 区间落入预先配置的容许范围才通过，不挑盐值或剔除失败场景。

{f.business.markdown_table(calibration,rates=('alpha','false_positive_rate','simultaneous_ci_low','simultaneous_ci_high','acceptance_low','acceptance_high'),digits=4)}

另有 {config['hash_repetitions']} 次真实 SHA256 Champion/Champion 分流，逐行重新核验。
实际哈希重复频次精度有限，与十万次压缩单元统计校准分别记录。原 A/A 发现的 1:9 小样本未认证限制仍保存，未用来证明本方案。

![可行性与匹配校准](figures/feasibility_and_calibration.png)

## 5. 固定盐值历史 A/B 演练

{f.business.markdown_table(diagnostics,digits=5)}

完整合格申请收益差 **{est['delta']:+.3f} 元**，95% 区间 **[{est['ci_low']:.3f}, {est['ci_high']:.3f}]**。
下界>0：{est['positive_zero_ci_in_simulation']}；下界>5：{est['business_value_demonstrated_in_simulation']}。
不能把零差显著或更精确的区间当作已满足收益业务门槛。

独立分层 Bootstrap：

{f.business.markdown_table(boot,digits=3)}

总体及单元 SRM：

{f.business.markdown_table(quality,digits=6)}

收益真值取评价经验分布与预先列出的合成备择；均为独立组有放回未来样本模拟。
合成 +10 等用于检验设计能力，不能写成 Challenger 的收益。历史分布的大规模重采样不会创造新增客户证据。

{f.business.markdown_table(simulation[['case_id','n_disagreements','true_profit','business_value_rate','business_value_mc_low','business_value_mc_high','positive_zero_rate','default_ni_rate','approval_ni_rate','absolute_risk_rate','joint_success_rate','joint_success_mc_low','joint_success_mc_high']],rates=('business_value_rate','business_value_mc_low','business_value_mc_high','positive_zero_rate','default_ni_rate','approval_ni_rate','absolute_risk_rate','joint_success_rate','joint_success_mc_low','joint_success_mc_high'),digits=3)}

![A/B 收益与业务证据](figures/ab_business_evidence.png)

## 6. 执行与责任边界

先 shadow 验证目标协变量覆盖、分歧方向、共同安全准入及持久客户去重，再进行独立 pilot 的安全暂停演练。
满足安全条件后开启固定样本正式实验；正式样本不混入 pilot，分歧曝光按锁定计划执行，其他申请沿用 Champion 并保留实际成熟风险账本。
只允许固定终点、全体结果成熟、质量通过后的主分析。样本不足、单元不足、SRM 未解决、现金流未成熟或 DPD 缺失都阻断生产晋级；
不得根据未经调整的中期收益显著结果提前成功或追样本。新增停止/估计器/准入变更需重新匹配校准。

开发/评价 OOF 均经过既有模型开发及前轮分析，本轮本地设计锁定不构成新的独立模型验证或外部预注册。
所有收益依赖固定历史标签和模拟条款，不能识别真实政策引起的借款行为变化、未获批人群选择偏差或宏观漂移。
真实流量、成熟结果、局部风险容忍度及客户相关性待前瞻确认；production_rollout_ready 始终 false。

复现说明见 docs/ab_feasibility_v2.md；数据库 data/interim/{output.run_name}.duckdb。
保留预分析设计锁定、真实分流、匹配校准/AB/Bootstrap 重复 Parquet、SQL 表、报告、图表和最终冻结哈希。
"""
    (output.reports/"ab_feasibility_report.md").write_text(text,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="ab_feasibility_v2")
    args = parser.parse_args()
    output,config = f.paths(args.run_name),f.load_config()
    try:
        from .verify_ab_feasibility import verify
        from .verify_ab_test_design import verify as verify_parent
    except ImportError:
        from verify_ab_feasibility import verify
        from verify_ab_test_design import verify as verify_parent
    start = time.perf_counter()
    progress("Checking frozen parent A/B and A/A evidence, read-only")
    verify_parent(f.paths(config["parent_run_name"]))
    f.aa.check_freeze(f.paths(config["aa_parent_run_name"]),f.aa.load_config())
    prior_audit = json.loads((f.ROOT/f"reports/{config['aa_parent_run_name']}/artifact_verification.json").read_text(encoding="utf-8"))
    if not prior_audit["deep_hash_and_calibration_reproduction"]: raise AssertionError("Need completed deep A/A audit")
    if f.check_manifest(output,config):
        progress(json.dumps(verify(output,deep=True,progress=progress),indent=2)); return
    output.artifacts.mkdir(parents=True,exist_ok=True)
    output.reports.mkdir(parents=True,exist_ok=True)
    (output.reports/"tables").mkdir(exist_ok=True)
    source = pd.read_parquet(f.ROOT/f"artifacts/{config['parent_run_name']}/experiment_population.parquet")
    population = f.enrich(source)
    parent_config = f.legacy.load_config()
    strata,meta,candidates,protocol,old_power = f.plan(population,config,parent_config)
    protocol["selected_policies"] = json.loads((f.ROOT/f"artifacts/{config['parent_run_name']}/experiment_protocol.json").read_text(encoding="utf-8"))["selected_policies"]
    lock_design(output,config,protocol,strata,candidates)
    f.business.write_json(output.reports/"run_config.json",{"status":"running","design_locked_before_new_AB_replay":True})
    progress(f"Locked plan: {protocol['required_randomized_disagreements']:,} disagreements; exposure={protocol['selected_disagreement_exposure']:.0%}")
    cal,caldraws = f.calibrate(population,protocol,config,progress)
    if not cal.passed.all(): raise AssertionError("Matched calibration failed; do not run Challenger simulation")
    hashes = f.hash_replays(population,config,progress)
    assignments = f.assign(population,config)
    f.validate_assignments(population,assignments,config)
    target = f.cohort(population,"policy_evaluation")
    est,cells = f.estimate(assignments,target,config)
    q = f.quality(assignments,config)
    if not q.passed.all(): raise AssertionError("Fixed snapshot SRM failed; retain Champion, no seed search")
    boot,bootdraws = f.bootstrap(assignments,target,config)
    sim,simdraws = f.simulations(population,protocol,config,progress)
    duration = f.duration(population,protocol,config)
    temporary = output.database.with_name(f".{output.run_name}_{time.time_ns()}.duckdb")
    output.database.parent.mkdir(parents=True,exist_ok=True)
    with duckdb.connect(str(temporary)) as con:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='1GB'")
        con.register("_population",population)
        con.register("_assignments",assignments)
        con.execute(f.SQL.read_text(encoding="utf-8"))
        tables = {"planning_strata":strata,"design_candidates":candidates,"matched_calibration":cal,
            "matched_aa_hash_replays":hashes,"primary_stratum_estimates":cells,"assignment_quality":q,
            "ab_simulation_results":sim,"bootstrap_results":boot,"experiment_duration":duration,"original_power_reference":old_power}
        for table,frame in tables.items():
            con.register("_frame",frame)
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM _frame")
            con.unregister("_frame")
        coverage = con.table("eligibility_summary").fetchdf().sort_values(["dataset","policy_cohort"])
        csv_names = ["eligibility_summary","experiment_metrics","observed_stratum_metrics","target_stratum_weights","sql_primary_estimate",*tables]
        for table in csv_names:
            con.execute(f"COPY {table} TO ? (FORMAT CSV,HEADER)",[str(output.reports/"tables"/f"{table}.csv")])
        con.execute("CHECKPOINT")
    temporary.replace(output.database)
    for frame,name in ((assignments,"assignments"),(caldraws,"calibration_draws"),(simdraws,"ab_draws"),(bootdraws,"bootstrap_draws")):
        frame.to_parquet(output.artifacts/f"{name}.parquet",index=False,compression="zstd")
    plots(output,candidates,cal,sim,simdraws)
    report(output,config,protocol,candidates,cal,sim,est,q,coverage,boot,duration)
    progress("Independently verifying SQL, weights, locked plan and all repeated samples")
    audit = verify(output,require_freeze=False,deep=True,progress=progress)
    files = [output.database,*sorted(output.artifacts.glob("*.parquet")),
        output.artifacts/"design_protocol.json",output.artifacts/"preanalysis_design_lock.json",output.artifacts/"model_governance_decision.json",
        output.reports/"planning_strata.csv",output.reports/"design_candidates.csv",output.reports/"summary.json",output.reports/"ab_feasibility_report.md",
        *sorted((output.reports/"tables").glob("*.csv")),*sorted((output.reports/"figures").glob("*.png"))]
    manifest = {"input_sha256":f.input_hashes(config),"protocol_sha256":f.protocol_hashes(),
        "output_sha256":{f.business.relative(p,f.ROOT):f.business.sha256(p) for p in files},
        "champion_retained":True,"challenger_promoted":False,"new_live_experiment_completed":False}
    f.business.write_json(output.artifacts/"feasibility_freeze_manifest.json",manifest)
    audit["frozen_checksums_verified"] = bool(f.check_manifest(output,config))
    f.business.write_json(output.reports/"artifact_verification.json",audit)
    f.business.write_json(output.reports/"run_config.json",{"status":"completed","elapsed_seconds":time.perf_counter()-start,
        "design_locked_before_new_AB_replay":True,"manifest_sha256":f.business.sha256(output.artifacts/"feasibility_freeze_manifest.json")})
    progress(f"Completed and frozen: {output.reports/'ab_feasibility_report.md'}")


if __name__=="__main__": main()

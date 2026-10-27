"""Evidence-preserving aggregate data contract for the local Power BI project."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"dashboard/powerbi"
SOURCES = {}
VERSION = "loan_risk_dashboard_v1"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(relative):
    path = ROOT/relative
    SOURCES[relative] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(stage,name):
    relative = f"reports/{stage}/tables/{name}.csv"
    path = ROOT/relative
    SOURCES[relative] = sha(path)
    return pd.read_csv(path)


def select(frame,mapping):
    return frame[list(mapping)].rename(columns=mapping).copy()


def label(frame,kind,basis,population):
    frame["EvidenceCategory"],frame["EvidenceBasis"],frame["PopulationScope"] = kind,basis,population
    return frame


def collect():
    SOURCES.clear()
    model_summary = read_json("reports/sql_business_analysis/summary.json")
    policy_summary = read_json("reports/approval_policy_simulation/summary.json")
    aa = read_json("reports/aa_validation/summary.json")
    ab = read_json("reports/ab_feasibility_v2/summary.json")
    old = read_json("reports/ab_test_design/summary.json")
    config = read_json("configs/ab_feasibility_v2.json")
    for stage in ("sql_business_analysis","approval_policy_simulation","aa_validation","ab_feasibility_v2"):
        audit = read_json(f"reports/{stage}/artifact_verification.json")
        if audit["status"]!="passed": raise AssertionError("An audited source is required")
    for stage,filename in (("aa_validation","aa_freeze_manifest.json"),("ab_feasibility_v2","feasibility_freeze_manifest.json")):
        manifest = read_json(f"artifacts/{stage}/{filename}")
        for name,expected in manifest["output_sha256"].items():
            # Dashboards reuse only reports and aggregate evidence, not row data.
            if name.startswith(f"reports/{stage}/") and sha(ROOT/name)!=expected:
                raise AssertionError(f"Frozen source changed: {name}")
    model = select(read_csv("sql_business_analysis","model_performance_summary"),{
        "model_version":"ModelVersion","model_role":"ModelRole","repay_roc_auc":"ROCAUC",
        "default_pr_auc_ap":"DefaultPRAUC","ks":"KS","row_count":"Rows","observed_default_rate":"HistoricalDefaultRate",
        "log_loss":"LogLoss","brier_score":"BrierScore"})
    model["Model"] = model.ModelRole.map({"champion":"Champion","challenger":"Challenger"})
    label(model,"历史结果","复用开发 OOF；未获得独立未来标签","688,998 个训练申请")
    tables = {"ModelMetrics":model}
    tables["Models"] = model[["ModelVersion","ModelRole","Model"]].copy()
    risk = select(read_csv("sql_business_analysis","risk_band_summary"),{
        "dataset":"Dataset","model_version":"ModelVersion","risk_decile":"RiskDecile","row_count":"Rows",
        "labeled_rows":"LabeledRows","default_rows":"DefaultRows","observed_default_rate":"HistoricalDefaultRate",
        "population_share":"PopulationShare","mean_model_p_default":"MeanModelDefaultScore"})
    label(risk,"历史结果","训练为历史标签；测试仅模型分数/构成，无标签","训练/无标签测试的风险分箱")
    tables["RiskBands"] = risk
    tables["ScoreDrift"] = label(select(read_csv("sql_business_analysis","prediction_drift_summary"),{
        "model_version":"ModelVersion","score_psi":"ScorePSI","decile_total_variation":"TotalVariation"}),
        "历史结果","模型分数分布漂移诊断；不是已观测标签漂移","训练 OOF 对测试预测")
    curves = select(read_csv("approval_policy_simulation","approval_economic_results"),{
        "model_version":"ModelVersion","policy_id":"PolicyId","dataset":"Dataset","policy_cohort":"Cohort",
        "scenario_id":"ScenarioId","target_approval_rate":"ApprovalBudget","threshold":"Threshold",
        "total_applications":"Applications","approved_applications":"Approved","approved_default_rows":"DefaultRows",
        "approval_rate":"ApprovalRate","observed_default_rate_approved":"HistoricalApprovedDefaultRate",
        "historical_profit_per_application":"HistoricalSimulatedProfitPerApplication"})
    label(curves,"模拟分析","冻结策略 + 不变历史标签 + 假设现金流；测试收益为空","策略开发/全评价/无标签测试，分别保留")
    tables["PolicyCurves"] = curves
    selected = read_csv("approval_policy_simulation","selected_policy_results")
    selected = selected.loc[selected.prospective_arm.isin(["A","B"])]
    chosen = select(selected,{
        "prospective_arm":"Arm","model_version":"ModelVersion","dataset":"Dataset","policy_cohort":"Cohort",
        "scenario_id":"ScenarioId","threshold":"Threshold","total_applications":"Applications","approval_rate":"ApprovalRate",
        "observed_default_rate_approved":"HistoricalApprovedDefaultRate","historical_profit_per_application":"HistoricalSimulatedProfitPerApplication"})
    chosen["CohortLabel"] = chosen.Cohort.map({"policy_development":"开发子集（413,398）","policy_evaluation":"全评价子集（275,600）","unlabeled_test":"无标签测试（36,324）"})
    tables["SelectedPolicies"] = label(chosen,"模拟分析","策略结果；历史收益使用现金流假设","不等同 V2 的共同合格 ITT 范围")
    definitions = pd.DataFrame(policy_summary["selected_policies"])
    tables["PolicyDefinitions"] = select(definitions,{"prospective_arm":"Arm","model_version":"ModelVersion","threshold":"Threshold","target_approval_rate":"ApprovalBudget"})
    assumptions = select(read_csv("approval_policy_simulation","economic_assumptions"),{
        "scenario_id":"ScenarioId","principal":"Principal","interest_rate_for_horizon":"InterestRate",
        "loss_given_default":"LossGivenDefault","funding_rate_for_horizon":"FundingRate",
        "servicing_cost_per_approved":"ServicingCost","application_cost_per_application":"ApplicationCost"})
    tables["EconomicAssumptions"] = label(assumptions,"模拟假设","人民币/12个月；不是实际放款与成本账本","所有收益模拟的经济参数")
    scenarios = {"base":"基准假设","lower_interest":"较低利息收益","full_principal_loss":"违约损失100%","higher_funding":"较高资金成本","smaller_principal":"较小本金","larger_principal":"较大本金"}
    tables["Scenarios"] = pd.DataFrame([{"ScenarioId":k,"Scenario":v} for k,v in scenarios.items()])
    original_cal = read_csv("aa_validation","aa_calibration_results")
    c = select(original_cal,{"case_id":"CaseId","endpoint":"EndpointId","alpha":"Alpha","required":"Required",
        "repetitions":"Repetitions","false_positive_rate":"FalsePositiveRate","fpr_simultaneous_ci_low":"CILow",
        "fpr_simultaneous_ci_high":"CIHigh","calibration_passed":"Passed"})
    c["Pipeline"] = "原始 A/A"
    c["AcceptanceLow"] = original_cal.alpha-original_cal.fpr_tolerance
    c["AcceptanceHigh"] = original_cal.alpha+original_cal.fpr_tolerance
    matched = read_csv("ab_feasibility_v2","matched_calibration")
    c2 = select(matched,{"case_id":"CaseId","endpoint":"EndpointId","alpha":"Alpha","repetitions":"Repetitions",
        "false_positive_rate":"FalsePositiveRate","simultaneous_ci_low":"CILow","simultaneous_ci_high":"CIHigh",
        "acceptance_low":"AcceptanceLow","acceptance_high":"AcceptanceHigh","passed":"Passed"})
    c2["Pipeline"],c2["Required"] = "V2 匹配校准",True
    cal = pd.concat([c,c2],ignore_index=True)
    endpoint_labels = {"profit_itt":"收益零差","profit_equality":"收益零差","approval_rate":"审批率零差",
        "default_incidence_itt":"违约零差","default_equality":"违约零差","business_boundary":"收益业务边界","default_ni_boundary":"非劣效边界"}
    cal["Endpoint"] = cal.EndpointId.map(endpoint_labels)
    cal["Case"] = cal.Pipeline+" / "+cal.CaseId
    cal["Status"] = cal.Passed.map({True:"已认证（离线范围内）",False:"未认证"})
    cal["Interval"] = cal.apply(lambda r:f"[{r.CILow:.3%}, {r.CIHigh:.3%}]",axis=1)
    tables["Calibration"] = label(cal,"模拟分析","同 Champion 零差校准；业务/NI 边界另列为合成场景","有限历史总体/IID；不证明未来安全")
    peek = select(read_csv("aa_validation","aa_optional_stopping"),{
        "look_fraction":"LookFraction","n_total":"SamplesAtLook","single_look_two_sided_fpr":"FixedLookFPR",
        "naive_any_look_two_sided_fpr":"AnyLookFPR"})
    tables["Peeking"] = label(peek,"模拟分析","未经调整提前查看的错误演示；终点100,000","不是原正式实验的372万样本终点")
    duration = select(read_csv("ab_feasibility_v2","experiment_duration"),{
        "coverage_source":"CoverageSource","daily_applications":"DailyApplications","eligible_fraction":"EligibleFraction",
        "disagreement_exposure":"Exposure","enrollment_days":"EnrollmentDays","maturity_days":"MaturityDays",
        "grace_days":"GraceDays","final_analysis_days":"FinalDays","feasible_enrollment":"Feasible"})
    tables["DurationGrid"] = label(duration,"模拟假设","固定历史分歧率和假设流量/成熟期","不提供未来收益预测")
    cand = read_csv("ab_feasibility_v2","design_candidates")
    design = select(cand,{"design":"DesignId","disagreement_exposure":"Exposure","required_randomized":"RandomizedRequired",
        "enrollment_days":"EnrollmentDays","enrollment_feasible":"Feasible","changed_decision_fraction_of_eligible":"ChangedDecisionFraction"})
    design["Design"] = ["原全合格独立分组","V2 分歧50%入组","V2 分歧100%入组"]
    tables["DesignComparison"] = label(design,"模拟假设","冻结原流量；不同随机范围，样本数不能直接视为同种申请","原入组7444天 / V2入组577或289天")
    sim = select(read_csv("ab_feasibility_v2","ab_simulation_results"),{
        "case_id":"CaseId","n_disagreements":"DisagreementSamples","true_profit":"AssumedUnderlyingProfit",
        "synthetic_alternative":"Synthetic","business_value_rate":"BusinessSuccessRate","positive_zero_rate":"PositiveZeroRate",
        "joint_success_rate":"JointSuccessRate","joint_success_mc_low":"JointMCLow","joint_success_mc_high":"JointMCHigh"})
    sim["Scenario"] = sim.CaseId.map({"historical_current":"历史经验 / 当前规模","historical_at_planned":"历史经验 / 规划规模",
        "synthetic_true_gain_0":"合成备择 +0元","synthetic_true_gain_5":"合成边界 +5元",
        "synthetic_true_gain_10":"合成备择 +10元","synthetic_true_gain_20":"合成备择 +20元"})
    sim["Basis"] = sim.Synthetic.map({False:"历史经验分布的IID模拟",True:"合成备择；非 Challenger 经验收益"})
    tables["ABSimulations"] = label(sim,"模拟分析","条件模拟成功频率；不是收益超过门槛的后验概率","完整共同合格申请的收益差")
    r,p = ab["replay"],ab["protocol"]
    state = {"Key":1,"ProfitDelta":r["delta"],"ProfitLow":r["ci_low"],"ProfitHigh":r["ci_high"],"BusinessThreshold":5.,
        "DefaultDeltaPP":100*r["default_delta"],"DefaultUpperPP":100*r["default_upper"],"ApprovalDeltaPP":100*r["approval_delta"],
        "BApprovedRiskUpper":r["full_portfolio_approved_default_upper"],"CurrentSamples":12614,
        "RequiredSamples":p["required_randomized_disagreements"],"TargetEligible":210856,"DisagreementFraction":p["planning_disagreement_fraction"],
        "MaturityDays":config["outcome_maturity_days_assumed"],"GraceDays":config["data_grace_days"],
        "ProfitPassed":int(r["business_value_demonstrated_in_simulation"]),"GuardrailsPassed":int(ab["all_risk_guards_passed_in_simulation"]),
        "CalibrationPassed":int(ab["matched_aa_and_boundary_calibration_passed"]),"QualityPassed":int(ab["quality_passed"]),
        "SampleReached":int(ab["required_fixed_sample_reached"]),"LiveMaturityValidated":int(ab["live_outcome_maturity_validated"]),
        "ProductionReady":int(ab["production_rollout_ready"]),"ChampionRetained":int(ab["champion_retained"]),
        "ChampionAUC":float(model.loc[model.ModelRole.eq("champion"),"ROCAUC"].iloc[0]),
        "ChallengerAUC":float(model.loc[model.ModelRole.eq("challenger"),"ROCAUC"].iloc[0]),
        "AAStressFailures":int((~c.Passed).sum()),"MatchedCalibrationChecks":len(c2),
        "AAFixedTerminalFPR":float(peek.iloc[-1].FixedLookFPR),"AAPeekingTerminalFPR":float(peek.iloc[-1].AnyLookFPR),
        "LiveDefaultRate":None,"LiveProfitDelta":None,"LiveDPD30Rate":None}
    tables["DashboardState"] = label(pd.DataFrame([state]),"模拟分析","冻结 V2 历史分层演练；真实指标保持 NULL","共同合格评价210,856；随机分歧12,614")
    intervals = pd.DataFrame([
        {"Version":"原 A/B 独立组演练","Estimate":old["offline_replay_diagnostics"]["welch_offline_replay"]["delta"],
         "Low":old["offline_replay_diagnostics"]["welch_offline_replay"]["ci_low"],"High":old["offline_replay_diagnostics"]["welch_offline_replay"]["ci_high"],"RandomizedSamples":210856},
        {"Version":"V2 分层独立组演练","Estimate":r["delta"],"Low":r["ci_low"],"High":r["ci_high"],"RandomizedSamples":12614}])
    intervals["Interval"] = intervals.apply(lambda x:f"[{x.Low:+.3f}, {x.High:+.3f}]",axis=1)
    tables["ProfitIntervals"] = label(intervals,"模拟分析","独立随机组的历史模拟；不是配对估计或线上因果结果","两行均换算为完整共同合格申请，随机范围不同")
    gate_rows = [
        (1,"收益业务门槛",f"95%下界 {r['ci_low']:.3f} 元","> 5 元","未达到",0,"模拟分析"),
        (2,"ITT违约非劣效",f"上界 {r['default_upper']*100:+.4f} 个百分点","< +0.2000 个百分点","离线通过",1,"模拟分析"),
        (3,"审批率非劣效",f"差 {r['approval_delta']*100:+.4f} 个百分点","> -2.0000 个百分点","离线通过",1,"历史结果"),
        (4,"完整B批准组合风险",f"上界 {r['full_portfolio_approved_default_upper']:.3%}","<= 5.000%","离线通过",1,"模拟分析"),
        (5,"匹配校准与分流质量","20/20 校准；当前分流通过","限定范围校准与质量通过","离线通过",1,"模拟分析"),
        (6,"固定样本量","12,614 / 17,141","固定样本达到；每层每组>=100","未达到",0,"历史结果"),
        (7,"真实结果成熟","没有真实成熟结果","全部现金流和DPD结果完结","待验证",None,"真实业务待验证"),
        (8,"业务安全与客户去重","代理准入；缺持久借款人ID","真实安全准入、首申请去重","待验证",None,"真实业务待验证")]
    tables["PromotionGates"] = pd.DataFrame(gate_rows,columns=["Order","Condition","CurrentEvidence","Criterion","Status","PassFlag","EvidenceCategory"])
    missing = [
        ("身份与去重","borrower_id、首个申请、关联重复借款人","确认真实随机单位和相关性"),
        ("分流与执行","入组时间、pilot/main、概率、模型/阈值版本、实际审批日志","验证ITT、分流一致性和样本比例"),
        ("完整人群","含一致/拒绝申请的分组前分数、安全准入、流量与分歧构成","建立完整目标权重、监测漂移及暴露"),
        ("真实现金流","本金、利率、期限、实收利息、核销、回收、各项成本","确认净收益；本金回收不是收益"),
        ("成熟与逾期","到期/完结日期、违约、DPD30/60/90、结果完整性","明确成熟窗口；未知结果不填零"),
        ("业务安全","分歧局部风险、影子运行及独立pilot记录","分歧全量入组使决策改变暴露约翻倍")]
    tables["MissingData"] = label(pd.DataFrame(missing,columns=["DataGroup","RequiredFields","Purpose"]),"真实业务待验证","尚未获得真实业务观测","真实实施阻断项")
    tables["DailyTraffic"] = pd.DataFrame({"DailyApplications":[1000,5000,20000]})
    tables["Coverage"] = pd.DataFrame({"EligibleFraction":[.2,.75]})
    tables["Exposure"] = pd.DataFrame({"ExposureFraction":[.5,1.]})
    tables["AlphaLevels"] = pd.DataFrame({"Alpha":[.01,.05,.1]})
    tables["Phase"] = pd.DataFrame({"Plan":["当前选定流量假设"]})
    for frame in tables.values():
        for col in frame.columns:
            if pd.api.types.is_bool_dtype(frame[col]): frame[col] = frame[col].astype("int64")
    return tables,{"ab":ab,"aa":aa,"config":config,"state":state,"source_sha256":dict(SOURCES),"version":VERSION}


def reference_forecast(state,daily=5000,coverage=.2,exposure=1.):
    days = math.ceil(state["RequiredSamples"]/(daily*coverage*state["DisagreementFraction"]*exposure))
    return {"enrollment_days":days,"final_days":days+state["MaturityDays"]+state["GraceDays"],
        "feasible":days<=365,"changed_decision_fraction":state["DisagreementFraction"]*exposure*.5}


def assert_contract(tables,evidence):
    s = evidence["state"]
    assert abs(s["ProfitDelta"]-3.6724026857895886)<1e-12
    assert abs(s["ProfitLow"]-.11925712572745484)<1e-12
    assert s["CurrentSamples"]<s["RequiredSamples"] and not s["ProductionReady"]
    assert not s["ProfitPassed"] and s["GuardrailsPassed"]
    assert all(s[name] is None for name in ("LiveDefaultRate","LiveProfitDelta","LiveDPD30Rate"))
    for name in ("PolicyCurves","SelectedPolicies"):
        test = tables[name].loc[tables[name].Dataset.eq("test")]
        assert test.HistoricalApprovedDefaultRate.isna().all()
        assert test.HistoricalSimulatedProfitPerApplication.isna().all()
    risk_test = tables["RiskBands"].loc[tables["RiskBands"].Dataset.eq("test")]
    assert risk_test.HistoricalDefaultRate.isna().all()
    assert reference_forecast(s)["enrollment_days"]==289
    assert reference_forecast(s)["final_days"]==684
    assert reference_forecast(s,exposure=.5)["enrollment_days"]==577
    assert int((tables["Calibration"].Passed==0).sum())==6
    assert (tables["PromotionGates"].loc[tables["PromotionGates"].Status.eq("待验证"),"PassFlag"].isna()).all()


def write_data(tables):
    (OUT/"data").mkdir(parents=True,exist_ok=True)
    dictionary = {}
    for name,frame in tables.items():
        frame.to_csv(OUT/f"data/{name}.csv",index=False,encoding="utf-8-sig",float_format="%.17g")
        columns = []
        for column in frame.columns:
            dtype = frame[column].dtype
            kind = "int64" if pd.api.types.is_integer_dtype(dtype) else "double" if pd.api.types.is_numeric_dtype(dtype) else "string"
            # All-null live metrics are numeric and must remain NULL on import.
            if column in ("LiveDefaultRate","LiveProfitDelta","LiveDPD30Rate","PassFlag"): kind="double"
            columns.append({"name":column,"dataType":kind,"null_count":int(frame[column].isna().sum())})
        dictionary[name] = {"rows":len(frame),"columns":columns,"file":f"data/{name}.csv"}
    (OUT/"data_dictionary.json").write_text(json.dumps(dictionary,ensure_ascii=False,indent=2),encoding="utf-8")
    return dictionary

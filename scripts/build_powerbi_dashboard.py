"""Generate a native PBIP/PBIR report and import semantic model from audited aggregates."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

try:
    from . import powerbi_data as d
except ImportError:
    import powerbi_data as d

OUT = d.OUT
PROJECT = OUT/"LoanRisk.pbip"
REPORT = OUT/"LoanRisk.Report"
MODEL = OUT/"LoanRisk.SemanticModel"
BASE = "https://developer.microsoft.com/json-schemas/fabric/"
MEASURES = {}
PAGES = []


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def measure(table,name,expression,fmt=None,description=None):
    row = {"name":name,"expression":expression}
    if fmt: row["formatString"] = fmt
    if description: row["description"] = description
    MEASURES.setdefault(table,[]).append(row)
    return (table,name,"measure")


def add_measures():
    MEASURES.clear()
    percent,num = "0.00%","#,0"
    for name,column,fmt in (("还款 ROC-AUC","ROCAUC","0.0000"),("违约 PR-AUC","DefaultPRAUC","0.0000"),
        ("历史 KS","KS","0.0000"),("历史样本数","Rows",num)):
        measure("ModelMetrics",name,f"MAX(ModelMetrics[{column}])",fmt)
    measure("RiskBands","训练历史分箱坏账率",'CALCULATE(DIVIDE(SUM(RiskBands[DefaultRows]), SUM(RiskBands[LabeledRows])), RiskBands[Dataset] = "train")',percent)
    measure("RiskBands","测试分箱人群占比",'CALCULATE(DIVIDE(SUM(RiskBands[Rows]), CALCULATE(SUM(RiskBands[Rows]), REMOVEFILTERS(RiskBands[RiskDecile]))), RiskBands[Dataset] = "test")',percent)
    measure("ScoreDrift","分数 PSI","MAX(ScoreDrift[ScorePSI])","0.000")
    scenario = 'VAR Chosen = SELECTEDVALUE(Scenarios[ScenarioId], "base") RETURN '
    for table,name,column,fmt in (("PolicyCurves","评价模拟净收益","HistoricalSimulatedProfitPerApplication","0.00"),
        ("PolicyCurves","评价审批率","ApprovalRate",percent),("PolicyCurves","评价获批历史坏账率","HistoricalApprovedDefaultRate",percent)):
        measure(table,name,scenario+f'CALCULATE(MAX({table}[{column}]), REMOVEFILTERS(Scenarios), {table}[ScenarioId] = Chosen, {table}[Cohort] = "policy_evaluation")',fmt)
    for name,column,fmt in (("选中策略模拟收益","HistoricalSimulatedProfitPerApplication","0.00"),
        ("选中策略审批率","ApprovalRate",percent),("选中策略历史坏账率","HistoricalApprovedDefaultRate",percent)):
        measure("SelectedPolicies",name,scenario+f'CALCULATE(MAX(SelectedPolicies[{column}]), REMOVEFILTERS(Scenarios), SelectedPolicies[ScenarioId] = Chosen)',fmt)
    for name,col,fmt in (("假设本金","Principal",num),("假设利息率","InterestRate",percent),("假设违约损失率","LossGivenDefault",percent)):
        measure("EconomicAssumptions",name,scenario+f'CALCULATE(MAX(EconomicAssumptions[{col}]), REMOVEFILTERS(Scenarios), EconomicAssumptions[ScenarioId] = Chosen)',fmt)
    measure("EconomicAssumptions","当前经济假设",'SELECTEDVALUE(Scenarios[Scenario], "基准假设（默认）")')
    for name,col,fmt in (("冻结收益差","ProfitDelta","+0.000;-0.000;0.000"),("冻结收益下界","ProfitLow","+0.000;-0.000;0.000"),
        ("冻结收益上界","ProfitHigh","+0.000;-0.000;0.000"),("业务收益门槛","BusinessThreshold","0.000"),
        ("冻结分歧样本","CurrentSamples",num),("所需分歧样本","RequiredSamples",num),
        ("完整B模拟风险上界","BApprovedRiskUpper",percent)):
        measure("DashboardState",name,f"CALCULATE(MAX(DashboardState[{col}]), REMOVEFILTERS())",fmt,
            "固定V2证据，不随页面选择变成新实验结果")
    measure("DashboardState","冻结95%收益区间",'IF(OR(ISBLANK([冻结收益下界]), ISBLANK([冻结收益上界])), BLANK(), FORMAT([冻结收益下界], "0.000") & " ~ " & FORMAT([冻结收益上界], "0.000"))')
    measure("DashboardState","样本完成情况",'IF(OR(ISBLANK([冻结分歧样本]), ISBLANK([所需分歧样本])), BLANK(), FORMAT([冻结分歧样本], "#,0") & " / " & FORMAT([所需分歧样本], "#,0"))')
    measure("DashboardState","当前晋级决定",'VAR Ready = CALCULATE(MAX(DashboardState[ProductionReady]), REMOVEFILTERS()) RETURN IF(ISBLANK(Ready), "等待数据刷新", IF(Ready = 1, "需重新审核晋级证据", "保留 Champion / 暂不晋级"))',
        description="依据固定审计状态；筛选器不能绕过成熟、安全、样本或收益门槛")
    measure("DashboardState","真实业务状态",'"未验证：真实收益、坏账和DPD保持空值"')
    measure("DashboardState","V2校准通过数",'CALCULATE(MAX(DashboardState[MatchedCalibrationChecks]), REMOVEFILTERS())',num)
    measure("DashboardState","原A/A压力未认证项",'CALCULATE(MAX(DashboardState[AAStressFailures]), REMOVEFILTERS())',num)
    measure("DashboardState","假设入组天数",'VAR Daily = SELECTEDVALUE(DailyTraffic[DailyApplications], 5000) VAR Cov = SELECTEDVALUE(Coverage[EligibleFraction], 0.20) VAR Exp = SELECTEDVALUE(Exposure[ExposureFraction], 1.0) VAR D = CALCULATE(MAX(DashboardState[DisagreementFraction]), REMOVEFILTERS()) RETURN ROUNDUP(DIVIDE([所需分歧样本], Daily * Cov * Exp * D), 0)',num)
    measure("DashboardState","假设成熟天数",'CALCULATE(MAX(DashboardState[MaturityDays]), REMOVEFILTERS())',num)
    measure("DashboardState","假设宽限天数",'CALCULATE(MAX(DashboardState[GraceDays]), REMOVEFILTERS())',num)
    measure("DashboardState","假设最终分析天数",'[假设入组天数] + [假设成熟天数] + [假设宽限天数]',num)
    measure("DashboardState","假设入组可行性",'IF([假设入组天数] <= 365, "满足365天入组约束（假设）", "超出365天入组约束（假设）")')
    for endpoint,name in (("收益零差","收益零差误报率"),("违约零差","违约零差误报率")):
        measure("Calibration",name,f'VAR A = SELECTEDVALUE(AlphaLevels[Alpha], 0.05) RETURN CALCULATE(MAX(Calibration[FalsePositiveRate]), Calibration[Alpha] = A, Calibration[Endpoint] = "{endpoint}")',percent)
    measure("Peeking","固定查看双侧误报","MAX(Peeking[FixedLookFPR])",percent)
    measure("Peeking","任一次查看双侧误报","MAX(Peeking[AnyLookFPR])",percent)
    measure("ABSimulations","业务门槛成功频率","MAX(ABSimulations[BusinessSuccessRate])",percent)
    measure("ABSimulations","零差正向显著频率","MAX(ABSimulations[PositiveZeroRate])",percent)
    measure("ABSimulations","联合成功频率","MAX(ABSimulations[JointSuccessRate])",percent)
    measure("DesignComparison","冻结方案入组天数","MAX(DesignComparison[EnrollmentDays])",num)


def build_model(dictionary):
    add_measures()
    tables = []
    for name,info in dictionary.items():
        cols = [{"name":c["name"],"dataType":c["dataType"],"sourceColumn":c["name"],"summarizeBy":"none"} for c in info["columns"]]
        for col in cols:
            if any(token in col["name"] for token in ("Rate","Fraction","Alpha","PopulationShare","LossGivenDefault")) and col["dataType"]=="double": col["formatString"]="0.00%"
            if col["name"] in ("ROCAUC","DefaultPRAUC","KS"): col["formatString"]="0.0000"
            if col["name"] in ("ApprovalBudget","Exposure"): col["formatString"]="0.00%"
            if col["name"]=="Threshold": col["formatString"]="0.0000000000"
        m_types = {"int64":"Int64.Type","double":"type number","string":"type text"}
        pairs = ", ".join('{"'+c["name"]+'", '+m_types[c["dataType"]]+'}' for c in cols)
        numeric = ", ".join('"'+c["name"]+'"' for c in cols if c["dataType"]!="string")
        expression = ["let",
            f'    Source = Csv.Document(File.Contents(DataFolder & "\\{name}.csv"), [Delimiter=",", Columns={len(cols)}, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
            '    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),',
            f'    Nulls = Table.ReplaceValue(Headers, "", null, Replacer.ReplaceValue, {{{numeric}}}),',
            f'    Typed = Table.TransformColumnTypes(Nulls, {{{pairs}}}, "en-US")',"in","    Typed"]
        table = {"name":name,"columns":cols,"partitions":[{"name":name,"mode":"import","source":{"type":"m","expression":expression}}]}
        if name in MEASURES: table["measures"] = MEASURES[name]
        if name in ("PromotionGates","MissingData","DashboardState"): table["description"]="真实未知结果保持NULL；所有晋级状态均为固定证据，不允许用交互假设替代。"
        tables.append(table)
    relationships = []
    for fact in ("ModelMetrics","RiskBands","ScoreDrift","PolicyCurves","SelectedPolicies","PolicyDefinitions"):
        relationships.append({"name":"Models_"+fact,"fromTable":fact,"fromColumn":"ModelVersion","toTable":"Models","toColumn":"ModelVersion","crossFilteringBehavior":"oneDirection"})
    for fact in ("PolicyCurves","SelectedPolicies","EconomicAssumptions"):
        relationships.append({"name":"Scenarios_"+fact,"fromTable":fact,"fromColumn":"ScenarioId","toTable":"Scenarios","toColumn":"ScenarioId","crossFilteringBehavior":"oneDirection"})
    model = {"name":"LoanRisk","compatibilityLevel":1601,"model":{"culture":"zh-CN",
        "defaultPowerBIDataSourceVersion":"powerBI_V3",
        "dataAccessOptions":{"legacyRedirects":True,"returnErrorValuesAsNull":True},"tables":tables,"relationships":relationships,
        "expressions":[{"name":"DataFolder","kind":"m","expression":'"'+str(OUT/"data")+'" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]'}],
        "annotations":[{"name":"PBI_QueryOrder","value":json.dumps(list(dictionary))},
            {"name":"EvidenceBoundary","value":"No live experiment; historical OOF / simulated assumptions / pending real business validation separated"}]}}
    write(MODEL/"model.bim",model)
    write(MODEL/"definition.pbism",{"$schema":BASE+"item/semanticModel/definitionProperties/1.0.0/schema.json","version":"1.0","settings":{}})
    return model


def literal(value):
    if isinstance(value,bool): text="true" if value else "false"
    elif isinstance(value,(int,float)): text=str(value)+"D"
    else: text="'"+str(value).replace("'","''")+"'"
    return {"expr":{"Literal":{"Value":text}}}


def color(value): return {"solid":{"color":literal(value)}}


def field(table,name,kind="column",display=None):
    expr = {"Measure" if kind=="measure" else "Column":{"Expression":{"SourceRef":{"Entity":table}},"Property":name}}
    return {"field":expr,"queryRef":table+"."+name,"nativeQueryRef":display or name,"displayName":display or name}


def C(table,name,display=None): return field(table,name,"column",display)
def M(table,name,display=None): return field(table,name,"measure",display)


def container(title,background="#FFFFFF"):
    return {"title":[{"properties":{"show":literal(bool(title)),"text":literal(title or ""),"fontSize":literal(13),"fontColor":color("#17384B"),"alignment":literal("left")}}],
        "background":[{"properties":{"show":literal(True),"color":color(background),"transparency":literal(0)}}]}


def visual(page,identifier,kind,x,y,w,h,title=None,roles=None,objects=None):
    spec = {"$schema":BASE+"item/report/definition/visualContainer/2.0.0/schema.json","name":identifier,
        "position":{"x":x,"y":y,"width":w,"height":h,"z":len(page["visuals"]),"tabOrder":len(page["visuals"])},
        "visual":{"visualType":kind,"visualContainerObjects":container(title),"drillFilterOtherVisuals":False}}
    if roles: spec["visual"]["query"]={"queryState":{name:{"projections":values} for name,values in roles.items()}}
    if objects: spec["visual"]["objects"]=objects
    page["visuals"].append(spec)
    return spec


def text(page,identifier,value,x,y,w,h,size=14,bg="#FFFFFF",foreground="#17384B",bold=False):
    objects = {"general":[{"properties":{"paragraphs":[{"textRuns":[{"value":value,"textStyle":{"fontFamily":"Microsoft YaHei","fontSize":f"{size}pt","fontWeight":"bold" if bold else "normal","color":foreground}}]}]}}]}
    spec = visual(page,identifier,"textbox",x,y,w,h,objects=objects)
    spec["visual"]["visualContainerObjects"] = container(None,bg)
    return spec


def card(page,identifier,name,x,y,w=326,h=100,title=None,font_size=26):
    return visual(page,identifier,"card",x,y,w,h,title or name,{"Values":[M("DashboardState",name)]},
        {"labels":[{"properties":{"fontSize":literal(font_size),"labelDisplayUnits":literal(0),"color":color("#17384B")}}],
         "categoryLabels":[{"properties":{"show":literal(False)}}]})


def slicer(page,identifier,table,column,title,x,y,w=230,h=80):
    return visual(page,identifier,"slicer",x,y,w,h,title,{"Values":[C(table,column)]},
        {"data":[{"properties":{"mode":literal("Dropdown")}}],"selection":[{"properties":{"singleSelect":literal(True)}}]})


def chart(page,identifier,kind,title,x,y,w,h,category,values,series=None):
    roles = {"Category":[category],"Y":values}
    if series: roles["Series"]=[series]
    spec = visual(page,identifier,kind,x,y,w,h,title,roles,
        {"legend":[{"properties":{"show":literal(True),"position":literal("Top"),"fontSize":literal(10)}}],
         "categoryAxis":[{"properties":{"fontSize":literal(10)}}],"valueAxis":[{"properties":{"fontSize":literal(10)}}]})
    spec["visual"]["query"]["sortDefinition"]={"sort":[{"field":category["field"],"direction":"Ascending"}],"isDefaultSort":False}
    return spec


def table(page,identifier,title,x,y,w,h,fields):
    return visual(page,identifier,"tableEx",x,y,w,h,title,{"Values":fields},
        {"columnHeaders":[{"properties":{"fontSize":literal(10),"fontColor":color("#17384B"),"wordWrap":literal(True)}}],
         "values":[{"properties":{"fontSize":literal(10),"wordWrap":literal(True)}}],
         "grid":[{"properties":{"rowPadding":literal(7)}}]})


def page(name,title,subtitle):
    p = {"name":name,"title":title,"visuals":[]}
    text(p,"heading",title,24,18,1392,44,size=26,bg="#17384B",foreground="#FFFFFF",bold=True)
    text(p,"scope",subtitle,24,70,1392,48,size=12,bg="#E8F2EF")
    text(p,"footer","历史：复用OOF/标签  |  模拟：明确现金流与流量假设  |  真实业务：未验证，未知结果不填零  |  固定V2证据：保留Champion",24,858,1392,28,size=10,bg="#F1F5F9")
    PAGES.append(p)
    return p


def interval_image(state):
    os.environ["MPLCONFIGDIR"]=str(OUT/"matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"]="Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"]=False
    fig,ax=plt.subplots(figsize=(10.7,2.3),layout="constrained")
    ax.errorbar([state["ProfitDelta"]],[0],xerr=[[state["ProfitDelta"]-state["ProfitLow"]],[state["ProfitHigh"]-state["ProfitDelta"]]],fmt="o",capsize=10,color="#227B73",markersize=10)
    ax.axvline(0,color="#86949C",linestyle=":",label="零差参考")
    ax.axvline(5,color="#D28A36",linestyle="--",label="业务门槛 +5 元")
    ax.set(yticks=[0],yticklabels=["V2 历史分层演练"],xlabel="每完整合格申请收益差（假设人民币）",xlim=(-1,9),ylim=(-.8,.8))
    ax.set_title(f"+{state['ProfitDelta']:.3f} 元；95%区间 [{state['ProfitLow']:.3f}, {state['ProfitHigh']:.3f}]，下界未超过5元",fontsize=12)
    ax.legend(loc="lower right",fontsize=10); ax.grid(axis="x",alpha=.2)
    path=REPORT/"StaticResources/RegisteredResources/v2_profit_interval.png"
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160);plt.close(fig)


def build_pages(state):
    PAGES.clear()
    p=page("decision","01 决策总览","结项范围：离线设计与模拟；当前收益未达到业务门槛、固定样本未达规划量、真实成熟与安全数据缺失。")
    card(p,"decision","当前晋级决定",24,132,674,104)
    card(p,"range","冻结95%收益区间",718,132,698,104,"95%收益区间（元/共同合格申请）")
    card(p,"profit","冻结收益差",24,254,326,100,"V2历史模拟收益差（元）")
    card(p,"samples","样本完成情况",370,254,326,100,"分歧样本：实际 / 规划")
    card(p,"risk","完整B模拟风险上界",716,254,326,100)
    card(p,"calibration","V2校准通过数",1062,254,354,100,"匹配校准通过数（离线）")
    table(p,"gate_table","晋级条件：隐藏某项不会绕过固定决策",24,372,1392,300,
        [C("PromotionGates","Condition","条件"),C("PromotionGates","CurrentEvidence","当前证据"),C("PromotionGates","Criterion","门槛"),C("PromotionGates","Status","状态"),C("PromotionGates","EvidenceCategory","证据类型")])
    text(p,"history_band","历史结果：模型AUC为复用OOF；全评价策略收益和V2共同准入后的收益估计使用不同人群，不能直接当作同一实验。",24,690,1392,62,size=13,bg="#E8EEF6")
    text(p,"pending_band","真实待验证：借款人去重、真实安全准入、成熟现金流、DPD与执行日志；离线护栏通过不代表获准上线。",24,766,1392,62,size=13,bg="#FFF0DB")

    p=page("models","02 模型历史表现","历史结果 | 688,998个训练申请、复用开发OOF；不是新独立留出或Kaggle测试AUC。测试集只展示分数构成。")
    slicer(p,"model_selector","Models","Model","模型选择（默认展示两者）",24,132,420,88)
    table(p,"model_table","历史模型指标与分数漂移",464,132,952,150,[C("Models","Model","模型"),M("ModelMetrics","还款 ROC-AUC"),M("ModelMetrics","违约 PR-AUC"),M("ModelMetrics","历史 KS","KS"),M("ModelMetrics","历史样本数"),M("ScoreDrift","分数 PSI")])
    chart(p,"auc","clusteredBarChart","还款ROC-AUC（历史OOF）",24,300,674,230,C("Models","Model"),[M("ModelMetrics","还款 ROC-AUC")])
    chart(p,"pr","clusteredBarChart","违约PR-AUC（AP，历史OOF）",718,300,698,230,C("Models","Model"),[M("ModelMetrics","违约 PR-AUC")])
    chart(p,"bands","lineChart","训练风险分箱历史坏账率（10=最高风险）",24,548,674,240,C("RiskBands","RiskDecile","风险分箱"),[M("RiskBands","训练历史分箱坏账率")],C("Models","Model"))
    chart(p,"testmix","clusteredColumnChart","无标签测试：风险分箱构成（不是坏账率）",718,548,698,240,C("RiskBands","RiskDecile","风险分箱"),[M("RiskBands","测试分箱人群占比")],C("Models","Model"))
    text(p,"drift_note","测试标签缺失；冻结阈值在测试上的审批率约17%，明显低于训练评价约70%。分数分布漂移需在真实shadow中重查。",24,802,1392,38,size=11,bg="#FFF0DB")

    p=page("policy","03 审批策略与经济假设","模拟分析 | 阈值由开发子集冻结；曲线固定使用275,600个全评价申请。假设变化只重展示既有情景，不重新选择策略。")
    slicer(p,"scenario","Scenarios","Scenario","经济情景（默认基准假设）",24,132,350,90)
    slicer(p,"model","Models","Model","模型选择",394,132,350,90)
    visual(p,"principal","card",764,132,200,90,"假设本金（元）",{"Values":[M("EconomicAssumptions","假设本金")]})
    visual(p,"interest","card",984,132,200,90,"假设利息率",{"Values":[M("EconomicAssumptions","假设利息率")]})
    visual(p,"lgd","card",1204,132,212,90,"假设违约损失率",{"Values":[M("EconomicAssumptions","假设违约损失率")]})
    chart(p,"profit_curve","lineChart","审批预算与全评价模拟净收益",24,244,674,240,C("PolicyCurves","ApprovalBudget","开发审批预算"),[M("PolicyCurves","评价模拟净收益")],C("Models","Model"))
    chart(p,"risk_curve","lineChart","审批预算与全评价获批历史坏账率",718,244,698,240,C("PolicyCurves","ApprovalBudget","开发审批预算"),[M("PolicyCurves","评价获批历史坏账率")],C("Models","Model"))
    table(p,"selected","冻结的70%开发预算策略：人群范围分别展示",24,502,1392,252,[C("Models","Model","模型"),C("SelectedPolicies","CohortLabel","人群"),C("SelectedPolicies","Threshold","冻结阈值"),M("SelectedPolicies","选中策略审批率"),M("SelectedPolicies","选中策略历史坏账率"),M("SelectedPolicies","选中策略模拟收益")])
    text(p,"scope_note","测试行的历史坏账率和模拟收益保持空值；不把约17%的审批率当作风险已达标。收益含拒绝成本，本金回收不计为收益。分数未校准。",24,772,1392,66,size=12,bg="#FFF0DB")

    p=page("aa","04 A/A 校验与停止规则","模拟分析 | 原A/A必检通过但小样本1:9压力场景未认证；V2匹配A/A和业务/非劣效边界共20项通过，认证范围受限。")
    slicer(p,"alpha","AlphaLevels","Alpha","零差检验水平（默认5%）",24,132,340,84)
    card(p,"aa_fail","原A/A压力未认证项",384,132,496,84,"原1:9压力未认证项（全部α）")
    card(p,"aa_pass","V2校准通过数",900,132,516,84)
    chart(p,"fpr","clusteredBarChart","同Champion零差误报率：案例前缀区分原始/V2",24,234,830,284,C("Calibration","Case","校准场景"),[M("Calibration","收益零差误报率"),M("Calibration","违约零差误报率")])
    chart(p,"peek","lineChart","未经调整提前查看：固定终点10万申请的演示",874,234,542,284,C("Peeking","LookFraction","入组进度"),[M("Peeking","固定查看双侧误报"),M("Peeking","任一次查看双侧误报")])
    table(p,"aa_details","完整校准清单（不限上方选定α；业务与NI边界单侧2.5%另列）",24,536,1392,230,[C("Calibration","Pipeline","管线"),C("Calibration","CaseId","场景"),C("Calibration","Endpoint","端点"),C("Calibration","Alpha","检验水平"),C("Calibration","FalsePositiveRate","误报率"),C("Calibration","Interval","同时区间"),C("Calibration","Status","认证")])
    text(p,"aa_scope","固定终点双侧误报约4.934%，五次任一显著即停止约14.20%。一次p>0.05不是管线认证；小规模真实哈希抽查与10万次统计模拟分别记录。",24,784,1392,54,size=12,bg="#FFF0DB")

    p=page("feasibility","05 A/B 可行性与周期假设","模拟假设 | 默认每日5,000申请、共同合格20%、分歧全部入组。控件只改变流量与周期；不重算未来收益或晋级决定。")
    slicer(p,"daily","DailyTraffic","DailyApplications","每日申请量",24,132,440,84)
    slicer(p,"coverage","Coverage","EligibleFraction","共同合格占比",484,132,440,84)
    slicer(p,"exposure","Exposure","ExposureFraction","分歧人群入组比例",944,132,472,84)
    card(p,"nreq","所需分歧样本",24,234,326,94)
    card(p,"days","假设入组天数",370,234,326,94)
    card(p,"finaldays","假设最终分析天数",716,234,326,94)
    card(p,"feasible","假设入组可行性",1062,234,354,94,font_size=17)
    chart(p,"timeline","barChart","当前流量假设：入组 / 成熟 / 宽限",24,346,674,230,C("Phase","Plan"),[M("DashboardState","假设入组天数"),M("DashboardState","假设成熟天数"),M("DashboardState","假设宽限天数")])
    chart(p,"original","clusteredBarChart","冻结方案入组比较（不随控件变化）",718,346,698,230,C("DesignComparison","Design","方案"),[M("DesignComparison","冻结方案入组天数")])
    table(p,"compare","随机范围与审批改变暴露的区别",24,594,1392,160,[C("DesignComparison","Design","方案"),C("DesignComparison","RandomizedRequired","所需随机样本"),C("DesignComparison","Exposure","分歧入组比例"),C("DesignComparison","ChangedDecisionFraction","合格申请决策改变暴露"),C("DesignComparison","EnrollmentDays","冻结入组天数")])
    text(p,"horizon","V2仅在审批分歧人群随机分流，再按冻结分层权重还原完整合格ITT，减少一致决策噪声。默认289天入组 + 365天成熟 + 30天宽限 = 684天；50%暴露入组577天。全量入组使审批改变暴露约翻倍，需真实安全审核；12,614未达规划17,141。",24,772,1392,66,size=12,bg="#FFF0DB")

    p=page("value","06 收益区间、晋级与真实数据缺口","模拟分析 + 真实待验证 | 保留Champion：95%收益下界0.119未超过5元；固定样本不足，真实成熟与安全证据缺失。")
    card(p,"mean","冻结收益差",24,132,326,94,"每共同合格申请收益差（元）")
    card(p,"low","冻结收益下界",370,132,326,94,"95%下界（元）")
    card(p,"high","冻结收益上界",716,132,326,94,"95%上界（元）")
    card(p,"gate","业务收益门槛",1062,132,354,94,"业务门槛（元，不可调）")
    spec=visual(p,"income_interval","image",24,244,744,188,"冻结V2收益区间图（不随选择变化）")
    spec["visual"]["objects"]={"general":[{"properties":{"imageUrl":{"expr":{"ResourcePackageItem":{"PackageName":"RegisteredResources","PackageType":1,"ItemName":"v2_profit_interval.png"}}}}}]}
    chart(p,"sim","clusteredBarChart","合成备择与历史经验：两类成功频率需区分",788,244,628,188,C("ABSimulations","Scenario","模拟情景"),[M("ABSimulations","业务门槛成功频率"),M("ABSimulations","零差正向显著频率")])
    table(p,"intervals","原方案 / V2独立组收益区间（完整共同合格目标；随机范围不同）",24,450,1392,154,[C("ProfitIntervals","Version","方案"),C("ProfitIntervals","Estimate","估计（元）"),C("ProfitIntervals","Interval","95%区间（元）"),C("ProfitIntervals","RandomizedSamples","随机样本")])
    table(p,"missing","真实实施尚缺数据：必须补齐后才能重新决策",24,622,1392,178,[C("MissingData","DataGroup","数据组"),C("MissingData","RequiredFields","缺失字段/证据"),C("MissingData","Purpose","用途")])
    text(p,"restart","顺序：shadow核数据/准入，再独立pilot，随后锁定固定样本主实验。合成+10元联合成功约90.53%是设计演练；历史经验规划规模约0.55%，均非后验概率或真实因果证据。",24,806,1392,36,size=10,bg="#FFF0DB")
    return PAGES


def write_report(state):
    pages=build_pages(state)
    image_name="v2_profit_interval.png"
    theme_name="loan_risk_theme.json"
    theme={"name":"Loan Risk Evidence","dataColors":["#227B73","#D28A36","#3B71A3","#B34A55"],
        "background":"#F1F5F9","foreground":"#17384B","tableAccent":"#227B73",
        "visualStyles":{"*":{"*":{"title":[{"fontFamily":"Microsoft YaHei","fontSize":13}],"labels":[{"fontFamily":"Microsoft YaHei"}]}}}}
    write(REPORT/f"StaticResources/RegisteredResources/{theme_name}",theme)
    write(REPORT/"definition.pbir",{"$schema":BASE+"item/report/definitionProperties/2.0.0/schema.json","version":"4.0",
        "datasetReference":{"byPath":{"path":"../LoanRisk.SemanticModel"}}})
    write(REPORT/"definition/version.json",{"$schema":BASE+"item/report/definition/versionMetadata/1.0.0/schema.json",
        "version":"2.0.0"})
    write(REPORT/"definition/report.json",{"$schema":BASE+"item/report/definition/report/2.0.0/schema.json",
        "themeCollection":{"customTheme":{"name":theme_name,"type":"RegisteredResources","reportVersionAtImport":"5.59"}},
        "resourcePackages":[{"name":"RegisteredResources","type":"RegisteredResources","items":[
            {"name":theme_name,"path":theme_name,"type":"CustomTheme"},{"name":image_name,"path":image_name,"type":"Image"}]}],
        "settings":{"defaultFilterActionIsDataFilter":True,"useStylableVisualContainerHeader":True}})
    write(REPORT/"definition/pages/pages.json",{"$schema":BASE+"item/report/definition/pagesMetadata/1.0.0/schema.json","pageOrder":[p["name"] for p in pages],"activePageName":"decision"})
    for p in pages:
        folder=REPORT/"definition/pages"/p["name"]
        write(folder/"page.json",{"$schema":BASE+"item/report/definition/page/2.0.0/schema.json","name":p["name"],"displayName":p["title"],
            "displayOption":"FitToPage","width":1440,"height":900,
            "objects":{"background":[{"properties":{"color":color("#F1F5F9"),"transparency":literal(0)}}]}})
        for v in p["visuals"]: write(folder/"visuals"/v["name"]/"visual.json",v)
    write(PROJECT,{"$schema":BASE+"pbip/pbipProperties/1.0.0/schema.json","version":"1.0",
        "artifacts":[{"report":{"path":"LoanRisk.Report"}}],"settings":{"enableAutoRecovery":True}})
    write(OUT/"page_manifest.json",[{"name":p["name"],"title":p["title"],"visuals":len(p["visuals"]),"rendered_in_powerbi":False} for p in pages])
    (OUT/"measures.dax").write_text("\n\n".join(f"-- Table: {table}\n{m['name']} =\n{m['expression']}" for table,measures in MEASURES.items() for m in measures),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild",action="store_true",help="Explicitly regenerate existing report/semantic files; back up Desktop edits first")
    args=parser.parse_args()
    if PROJECT.exists() and not args.rebuild: raise SystemExit("Project already exists; use verifier or explicit --rebuild after preserving Desktop edits")
    tables,evidence=d.collect();d.assert_contract(tables,evidence)
    dictionary=d.write_data(tables)
    model=build_model(dictionary)
    interval_image(evidence["state"])
    write_report(evidence["state"])
    write(OUT/"source_manifest.json",{"version":d.VERSION,"source_sha256":evidence["source_sha256"],
        "data_sha256":{name:d.sha(OUT/f"data/{name}.csv") for name in tables},
        "historical_simulation_only":True,"live_metrics_not_imputed":True,"production_ready":False,
        "desktop_open_verified":False,"dax_engine_execution_verified":False,"no_external_publish":True})
    try:
        from .verify_powerbi_dashboard import verify
    except ImportError:
        from verify_powerbi_dashboard import verify
    result=verify()
    write(OUT/"artifact_verification.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()

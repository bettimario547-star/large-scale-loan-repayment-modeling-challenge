"""Render review-only layout/data previews; never label them as Desktop screenshots."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pandas as pd

try:
    from . import powerbi_data as d
    from .verify_powerbi_dashboard import read
except ImportError:
    import powerbi_data as d
    from verify_powerbi_dashboard import read

os.environ["MPLCONFIGDIR"] = str(d.OUT/"matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":9})

INK = "#17384B"
COLORS = ["#227B73","#D28A36","#3B71A3"]
FONT = Path("C:/Windows/Fonts/msyh.ttc")


def font(size): return ImageFont.truetype(str(FONT),max(9,int(size)))


def wrapped(draw,text,width,size):
    lines,line = [],""
    for char in str(text):
        if char=="\n": lines.append(line);line="";continue
        if draw.textlength(line+char,font=font(size))>width and line:
            lines.append(line);line=char
        else: line += char
    if line: lines.append(line)
    return lines


def block(draw,text,box,size=17,color=INK,max_lines=None):
    x,y,w,h = box
    lines = wrapped(draw,text,w,size)
    limit = min(len(lines),max_lines or max(1,int(h/(size*1.4))))
    for i,line in enumerate(lines[:limit]):
        if i==limit-1 and len(lines)>limit: line = line[:-2]+"…"
        draw.text((x,y+i*size*1.4),line,font=font(size),fill=color)


def lit(obj):
    value = obj["expr"]["Literal"]["Value"]
    if value.startswith("'"): return value[1:-1].replace("''", "'")
    if value in ("true","false"): return value=="true"
    return float(value.rstrip("D"))


def preview_tables(t):
    model = t["ModelMetrics"].merge(t["ScoreDrift"][["ModelVersion","ScorePSI"]],on="ModelVersion")
    selected = t["SelectedPolicies"].query("ScenarioId == 'base'").merge(t["Models"],on="ModelVersion")
    return {
        "gate_table":t["PromotionGates"].sort_values("Order")[["Condition","CurrentEvidence","Criterion","Status","EvidenceCategory"]],
        "model_table":model[["Model","ROCAUC","DefaultPRAUC","KS","Rows","ScorePSI"]],
        "selected":selected[["Model","CohortLabel","Threshold","ApprovalRate","HistoricalApprovedDefaultRate","HistoricalSimulatedProfitPerApplication"]],
        "aa_details":t["Calibration"][["Pipeline","CaseId","Endpoint","Alpha","FalsePositiveRate","Interval","Status"]],
        "compare":t["DesignComparison"][["Design","RandomizedRequired","Exposure","ChangedDecisionFraction","EnrollmentDays"]],
        "intervals":t["ProfitIntervals"][["Version","Estimate","Interval","RandomizedSamples"]],
        "missing":t["MissingData"][["DataGroup","RequiredFields","Purpose"]]
    }


def draw_table(draw,frame,labels,x,y,w,h):
    widths = [w/len(labels)]*len(labels)
    if len(labels)==3: widths = [w*.15,w*.52,w*.33]
    if len(labels)==5 and "Condition" in frame: widths = [w*.19,w*.29,w*.25,w*.11,w*.16]
    left = x
    for label,width in zip(labels,widths):
        draw.rectangle((left,y,left+width,y+35),fill="#E8F2EF")
        block(draw,label,(left+5,y+5,width-10,28),12)
        left += width
    yy = y+36
    row_height = 27
    max_rows = max(1,int((h-36)/row_height))
    if len(frame)>max_rows: max_rows=max(1,int((h-54)/row_height))
    for _,row in frame.head(max_rows).iterrows():
        left = x
        draw.line((x,yy+row_height-1,x+w,yy+row_height-1),fill="#DFE7EA")
        for name,width in zip(frame.columns,widths):
            value = row[name]
            if pd.isna(value): value = "—（未知）"
            elif isinstance(value,float):
                if any(s in name for s in ("Rate","Alpha","Fraction","Exposure")): value=f"{value:.2%}"
                elif name=="Threshold": value=f"{value:.6f}"
                else: value=f"{value:,.4f}"
            block(draw,value,(left+5,yy+4,width-10,row_height-5),11,max_lines=1)
            left += width
        yy += row_height
    if len(frame)>max_rows:
        draw.text((x+5,y+h-18),f"布局预览只画前 {max_rows} 行；原生表格有 {len(frame)} 行可滚动查看",font=font(11),fill="#6B7E89")


def chart_png(page,name,t,w,h):
    fig,ax = plt.subplots(figsize=(w/120,h/120),dpi=120,layout="constrained")
    if page=="models" and name in ("auc","pr"):
        frame = t["ModelMetrics"]
        col = "ROCAUC" if name=="auc" else "DefaultPRAUC"
        ax.barh(frame.Model,frame[col],color=COLORS[:2]);ax.set_xlim(0,1)
        for i,value in enumerate(frame[col]): ax.text(value+.015,i,f"{value:.4f}",va="center",fontsize=9)
    elif page=="models":
        frame = t["RiskBands"].query("Dataset == 'train'" if name=="bands" else "Dataset == 'test'").merge(t["Models"],on="ModelVersion")
        for i,(model,g) in enumerate(frame.groupby("Model",sort=False)):
            g = g.sort_values("RiskDecile")
            value = g.HistoricalDefaultRate if name=="bands" else g.Rows/g.Rows.sum()
            if name=="bands": ax.plot(g.RiskDecile,value*100,label=model,color=COLORS[i],marker="o",markersize=3)
            else: ax.bar(g.RiskDecile+(i-.5)*.35,value*100,width=.35,label=model,color=COLORS[i])
        ax.set_xlabel("风险十分位（10=最高风险）");ax.set_ylabel("百分比");ax.legend(fontsize=8)
    elif page=="policy":
        frame = t["PolicyCurves"].query("ScenarioId == 'base' and Cohort == 'policy_evaluation'").merge(t["Models"],on="ModelVersion")
        for i,(model,g) in enumerate(frame.groupby("Model",sort=False)):
            g = g.sort_values("ApprovalBudget")
            value = g.HistoricalSimulatedProfitPerApplication if name=="profit_curve" else g.HistoricalApprovedDefaultRate*100
            ax.plot(g.ApprovalBudget*100,value,label=model,color=COLORS[i])
        ax.set_xlabel("开发审批预算（%）");ax.set_ylabel("元/全评价申请" if name=="profit_curve" else "历史获批坏账率（%）");ax.legend(fontsize=8)
    elif name=="peek":
        f = t["Peeking"]
        ax.plot(f.LookFraction*100,f.FixedLookFPR*100,label="固定查看",color=COLORS[0],marker="o")
        ax.plot(f.LookFraction*100,f.AnyLookFPR*100,label="任一次显著",color=COLORS[1],marker="o")
        ax.set(xlabel="入组进度（%）",ylabel="双侧误报率（%）");ax.legend(fontsize=8)
    elif name=="fpr":
        f = t["Calibration"].loc[t["Calibration"].Alpha.eq(.05)&t["Calibration"].Endpoint.isin(["收益零差","违约零差"])]
        pivot = f.pivot(index="Case",columns="Endpoint",values="FalsePositiveRate")
        for i,col in enumerate(pivot): ax.barh([j+(i-.5)*.35 for j in range(len(pivot))],pivot[col]*100,height=.35,label=col,color=COLORS[i])
        labels = [s.replace("原始 A/A / ","A/A ").replace("V2 匹配校准 / ","V2 ") for s in pivot.index]
        ax.set_yticks(range(len(pivot)),labels,fontsize=6);ax.axvline(5,color="#8F9DA5",ls=":");ax.set_xlabel("误报率（%）");ax.legend(fontsize=7)
    elif name=="timeline":
        start = 0
        for label,value,c in zip(("入组","成熟","宽限"),(289,365,30),COLORS):
            ax.barh(["默认假设"],[value],left=start,color=c,label=f"{label} {value}天")
            start += value
        ax.set(xlabel="天",xlim=(0,750));ax.legend(fontsize=8)
    elif name=="original":
        f=t["DesignComparison"]
        ax.barh(f.Design,f.EnrollmentDays,color=COLORS)
        for i,v in enumerate(f.EnrollmentDays): ax.text(v+100,i,f"{int(v):,}天",va="center",fontsize=8)
        ax.set_xlim(0,9000)
    elif name=="sim":
        f=t["ABSimulations"]
        for i,(col,label) in enumerate((("BusinessSuccessRate","业务门槛成功"),("PositiveZeroRate","零差正向显著"))):
            ax.barh([j+(i-.5)*.35 for j in range(len(f))],f[col]*100,height=.35,label=label,color=COLORS[i])
        ax.set_yticks(range(len(f)),f.Scenario,fontsize=7);ax.set_xlim(0,110);ax.legend(fontsize=7)
    ax.grid(axis="x",alpha=.15);ax.spines[["top","right"]].set_visible(False)
    stream=io.BytesIO();fig.savefig(stream,format="png");plt.close(fig);stream.seek(0)
    return Image.open(stream).convert("RGB").resize((int(w),int(h)))


def render():
    tables,evidence = d.collect();data = preview_tables(tables);s=evidence["state"]
    cards={"当前晋级决定":"保留 Champion / 暂不晋级","冻结95%收益区间":f"{s['ProfitLow']:.3f} ~ {s['ProfitHigh']:.3f}",
        "冻结收益差":f"+{s['ProfitDelta']:.3f}","冻结收益下界":f"+{s['ProfitLow']:.3f}","冻结收益上界":f"+{s['ProfitHigh']:.3f}",
        "业务收益门槛":"5.000","样本完成情况":"12,614 / 17,141","完整B模拟风险上界":"3.22%","V2校准通过数":"20",
        "原A/A压力未认证项":"6","所需分歧样本":"17,141","假设入组天数":"289","假设最终分析天数":"684",
        "假设入组可行性":"满足365天入组约束（假设）","假设本金":"10,000","假设利息率":"12.00%","假设违约损失率":"80.00%"}
    selectors={"model_selector":"全部（Champion / Challenger）","model":"全部","scenario":"基准假设（无选择时回退）","alpha":"5%（无选择时回退）",
        "daily":"5,000（无选择时回退）","coverage":"20%（无选择时回退）","exposure":"100%（无选择时回退）"}
    pages=read(d.OUT/"page_manifest.json")
    folder=d.OUT/"layout_previews";folder.mkdir(exist_ok=True)
    images=[]
    for page in pages:
        canvas=Image.new("RGB",(1440,932),"#F1F5F9");draw=ImageDraw.Draw(canvas)
        draw.rectangle((0,0,1440,31),fill="#D28A36")
        draw.text((24,4),"代码布局与默认数据预览 · 不是 Power BI Desktop 截图 · 未执行 DAX · 表格示意只显示部分行",font=font(16),fill="#FFFFFF")
        paths=sorted((d.OUT/"LoanRisk.Report/definition/pages"/page["name"]/"visuals").glob("*/visual.json"))
        for path in paths:
            v=read(path);p=v["position"];spec=v["visual"];kind=spec["visualType"]
            x,y,w,h=p["x"],p["y"]+32,p["width"],p["height"]
            bg=lit(spec["visualContainerObjects"]["background"][0]["properties"]["color"]["solid"]["color"])
            draw.rectangle((x,y,x+w,y+h),fill=bg)
            if kind=="textbox":
                run=spec["objects"]["general"][0]["properties"]["paragraphs"][0]["textRuns"][0]
                style=run["textStyle"]
                block(draw,run["value"],(x+10,y+4,w-20,h-8),int(float(style["fontSize"].rstrip("pt"))),style["color"])
                continue
            title=lit(spec["visualContainerObjects"]["title"][0]["properties"]["text"])
            block(draw,title,(x+10,y+8,w-20,27),16)
            if kind=="card":
                measure=spec["query"]["queryState"]["Values"]["projections"][0]["field"]["Measure"]["Property"]
                size=30 if w>450 else 26
                if measure=="假设入组可行性": size=18
                block(draw,cards[measure],(x+12,y+42,w-24,h-44),size)
            elif kind=="slicer":
                block(draw,selectors[v["name"]]+"  ▾",(x+12,y+40,w-24,h-42),17)
            elif kind=="tableEx":
                labels=[item["nativeQueryRef"] for item in spec["query"]["queryState"]["Values"]["projections"]]
                draw_table(draw,data[v["name"]],labels,x+8,y+37,w-16,h-43)
            elif kind=="image":
                img=Image.open(d.OUT/"LoanRisk.Report/StaticResources/RegisteredResources/v2_profit_interval.png").convert("RGB")
                canvas.paste(img.resize((int(w-16),int(h-43))),(int(x+8),int(y+37)))
            else: canvas.paste(chart_png(page["name"],v["name"],tables,w-16,h-43),(int(x+8),int(y+37)))
        canvas.save(folder/f"{page['name']}.png");images.append(canvas)
    contact=Image.new("RGB",(1440,1398),"#DFE7EA")
    for i,img in enumerate(images): contact.paste(img.resize((720,466)),((i%2)*720,(i//2)*466))
    contact.save(folder/"overview.png")
    (folder/"preview_manifest.json").write_text(json.dumps({"render_source":"Pillow + matplotlib, PBIR coordinates and audited aggregates",
        "is_powerbi_screenshot":False,"dax_executed":False,"pages":[p["name"] for p in pages]},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Rendered {len(images)} review previews at {folder}")


if __name__=="__main__": render()

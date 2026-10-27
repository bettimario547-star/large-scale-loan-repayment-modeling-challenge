"""Build an editable decision memo and a verified one-page PDF from frozen data.

Uses reportlab and pypdf; both are in the Codex bundled workspace runtime.
No model, protocol, prediction, or frozen experimental output is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
STAGE = "ab_feasibility_v2"
REPORTS = ROOT / "reports" / STAGE
MARKDOWN = ROOT / "docs/experiment_closeout_decision_memo.md"
PDF = ROOT / "output/pdf/experiment_closeout_decision_memo.pdf"
PROVENANCE = ROOT / "reports/experiment_closeout/closeout_provenance.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8",newline="") as handle:
        return list(csv.DictReader(handle))


def load_evidence():
    manifest_path = ROOT / f"artifacts/{STAGE}/feasibility_freeze_manifest.json"
    manifest = read_json(manifest_path)
    sources = [REPORTS/"summary.json",REPORTS/"design_candidates.csv",
        REPORTS/"tables/eligibility_summary.csv",REPORTS/"tables/matched_calibration.csv",
        REPORTS/"tables/ab_simulation_results.csv",REPORTS/"artifact_verification.json",REPORTS/"run_config.json",
        manifest_path,ROOT/"configs/ab_feasibility_v2.json"]
    for path in sources:
        relative = path.relative_to(ROOT).as_posix()
        recorded = manifest["output_sha256"].get(relative,manifest["input_sha256"].get(relative))
        if recorded and digest(path)!=recorded:
            raise AssertionError(f"Frozen evidence changed: {relative}")
    summary,audit,run,config = (read_json(REPORTS/name) for name in ("summary.json","artifact_verification.json","run_config.json","../../configs/ab_feasibility_v2.json"))
    if run["status"]!="completed" or run["manifest_sha256"]!=digest(manifest_path): raise AssertionError("Incomplete run")
    if audit["status"]!="passed" or not audit["all_repeated_samples_reproduced"] or not audit["frozen_checksums_verified"]:
        raise AssertionError("Completed independent audit is required")
    if not summary["champion_retained"] or summary["challenger_promoted"] or summary["live_experiment_completed"]:
        raise AssertionError("Unsupported promotion or live experiment claim")
    calibration = csv_rows(REPORTS/"tables/matched_calibration.csv")
    if len(calibration)!=20 or not all(r["passed"].lower()=="true" for r in calibration): raise AssertionError("Uncertified calibration")
    candidates = csv_rows(REPORTS/"design_candidates.csv")
    original = next(r for r in candidates if r["design"]=="original_all_eligible")
    half = next(r for r in candidates if r["design"]=="disagreement_stratified" and float(r["disagreement_exposure"])==.5)
    selected = next(r for r in candidates if r["selected"].lower()=="true")
    if float(selected["disagreement_exposure"])!=1: raise AssertionError("Memo text requires the frozen full-D design")
    simulation = {r["case_id"]:r for r in csv_rows(REPORTS/"tables/ab_simulation_results.csv")}
    return summary,audit,config,original,half,selected,simulation,sources


def contents(summary,audit,config,original,half,selected,simulation):
    protocol,replay = summary["protocol"],summary["replay"]
    n = protocol["required_randomized_disagreements"]
    base = (
        "在共同合格贷款申请中，Challenger 能否使每申请净收益比 Champion 提升超过 5 元，并满足风险约束？"
        "主指标含拒绝申请成本；成功须收益差的 95% 置信下界 > 5 元，且全部护栏、固定样本、成熟与质量条件同时满足。"
    )
    design = (
        f"原全人群分组需 {int(original['required_randomized']):,} 个随机申请、入组 {int(original['enrollment_days']):,} 天；"
        f"仅约 {protocol['planning_disagreement_fraction']:.2%} 的合格申请存在审批分歧。V2 仅对分歧申请 1:1 随机化，"
        "按风险层和分歧方向三个单元估计，再加权回全体合格申请。"
        "排除一致申请以两策略对其条款、费用和现金流完全相同为前提；真实额度或定价改变时须重设方案。"
    )
    sample = (
        f"开发子集规划 {n:,} 个分歧申请，每单元每组至少 {config['minimum_per_arm_per_stratum']} 个成熟有效申请；"
        "规划备择 +10 元、业务边界 +5 元，边际功效 80%、方差安全系数 1.25。"
        f"沿用每日 {config['primary_daily_applications']:,} 申请、合格占比 {config['primary_eligible_fraction']:.0%}，分歧申请全部入组："
        f"{summary['primary_assumed_enrollment_days']} 天入组 + {config['outcome_maturity_days_assumed']} 天成熟 + {config['data_grace_days']} 天宽限 = "
        f"{summary['primary_assumed_final_analysis_days']} 天。仅 50% 分歧入组需 {int(half['enrollment_days'])} 天，超 365 天入组约束；"
        "全量分歧入组的审批改变暴露约为原方案两倍，须先通过安全审核。以上流量和成熟期均为假设。"
    )
    metrics = [
        ["收益差（元/合格申请）",f"{replay['delta']:+.3f}；95% 区间\n[{replay['ci_low']:.3f}, {replay['ci_high']:.3f}]","置信下界 > +5 元","未达到"],
        ["ITT 违约差（百分点）",f"{replay['default_delta']*100:+.4f}；单侧上界\n{replay['default_upper']*100:+.4f}","上界 < +0.2000","模拟通过"],
        ["审批率差（百分点）",f"{replay['approval_delta']*100:+.4f}\n全体策略决定普查","> -2.0000","模拟通过"],
        ["完整 B 获批组合违约率",f"单侧上界 {replay['full_portfolio_approved_default_upper']:.3%}","上界 <= 5.000%","模拟通过"],
    ]
    reliability = (
        "匹配 A/A 及收益、非劣效边界共 20 项校准通过，独立复算通过。"
        f"在历史经验分布的规划规模模拟中，业务与护栏联合成功频率约 {float(simulation['historical_at_planned']['joint_success_rate']):.2%}；"
        f"合成 +10 元备择约 {float(simulation['synthetic_true_gain_10']['joint_success_rate']):.2%}，后者只检验设计能力，不是 Challenger 的经验收益。"
    )
    decision = (
        f"保留 Champion，暂不晋级 Challenger。收益区间虽排除零，下界仍低于 5 元；"
        f"实际离线分歧样本仅 {audit['randomized_disagreements']:,}/{n:,}，未达固定规划量。"
        "护栏通过仅适用于模拟；评价 OOF 已被复用，且没有真实随机实验与成熟现金流，不能据此声明因果收益或生产安全。"
    )
    missing = [
        "身份与执行：持久借款人 ID、首个申请与入组时间；分流/阶段/概率、规则版本及实际决策执行日志。",
        "完整人群与安全：含一致和拒绝申请的分组前分数/准入普查；真实流量、分歧构成、前瞻漂移与重复客户相关性。",
        "贷款与结果：本金、利率/期限；实收利息、核销、回收及资金/服务/申请成本；到期/完结日期、违约、DPD30/60/90 和结果完整性。",
    ]
    next_gate = "真实实施顺序：shadow 核验数据与安全准入，再做独立 pilot；正式方案锁定后，固定样本且全部结果成熟、质量与业务/风险门槛均通过才重新决策。未知结果不得填零。"
    return base,design,sample,metrics,reliability,decision,missing,next_gate


def write_markdown(data):
    base,design,sample,metrics,reliability,decision,missing,next_gate = data
    rows = ["| 指标 | 离线演练结果 | 晋级界值 | 结论 |","| --- | --- | --- | --- |"]
    rows += ["| "+" | ".join(value.replace("\n","<br>") for value in row)+" |" for row in metrics]
    evidence = [
        ("冻结结果汇总","../reports/ab_feasibility_v2/summary.json"),
        ("候选方案与周期","../reports/ab_feasibility_v2/design_candidates.csv"),
        ("匹配校准","../reports/ab_feasibility_v2/tables/matched_calibration.csv"),
        ("模拟成功频率","../reports/ab_feasibility_v2/tables/ab_simulation_results.csv"),
        ("独立核查","../reports/ab_feasibility_v2/artifact_verification.json"),
        ("完整方案和复现","ab_feasibility_v2.md"),
    ]
    document = f"""# 实验结项报告与决策备忘录

贷款风险策略项目 | V2 离线设计与模拟结项 | **保留 Champion，暂不晋级 Challenger**

## 1. 业务问题

{base}

## 2. 方案为何改变

{design}

## 3. 样本量与周期

{sample}

## 4. 收益与护栏结果

{chr(10).join(rows)}

{reliability}

## 5. 保留 Champion 的理由

{decision}

## 6. 真实实施的数据缺口与重启条件

{chr(10).join('- '+line for line in missing)}

{next_gate}

证据：{' · '.join(f'[{label}]({path})' for label,path in evidence)}。本文件对应一页 PDF；结项范围为离线实验设计与模拟。
"""
    MARKDOWN.write_text(document,encoding="utf-8")


def build_pdf(data):
    base,design,sample,metrics,reliability,decision,missing,next_gate = data
    font_directory = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("MemoCJK",str(font_directory/"msyh.ttc"),subfontIndex=0))
    pdfmetrics.registerFont(TTFont("MemoCJKBold",str(font_directory/"msyhbd.ttc"),subfontIndex=0))
    navy,teal,muted = colors.HexColor("#17384B"),colors.HexColor("#186D65"),colors.HexColor("#52636F")
    body = ParagraphStyle("Body",fontName="MemoCJK",fontSize=9.1,leading=13.2,wordWrap="CJK",textColor=navy,alignment=TA_LEFT)
    heading = ParagraphStyle("Heading",parent=body,fontName="MemoCJKBold",fontSize=10.1,leading=14,spaceBefore=6,spaceAfter=3)
    title = ParagraphStyle("Title",parent=heading,fontSize=18,leading=25,spaceBefore=0,spaceAfter=4)
    subtitle = ParagraphStyle("Subtitle",parent=body,fontSize=8,leading=12,textColor=muted)
    note = ParagraphStyle("Note",parent=body,fontSize=8,leading=11.5,textColor=muted)
    small = ParagraphStyle("TableBody",parent=body,fontSize=8.5,leading=12)
    table_head = ParagraphStyle("TableHead",parent=small,fontName="MemoCJKBold",textColor=colors.white)
    banner_style = ParagraphStyle("Decision",parent=body,fontName="MemoCJKBold",fontSize=11,leading=16,textColor=teal)
    width = A4[0]-64
    story = []
    def para(text,style=body): return Paragraph(html.escape(text).replace("\n","<br/>"),style)
    def section(label,text):
        story.append(para(label,heading)); story.append(para(text))
    story.append(para("实验结项报告与决策备忘录",title))
    story.append(para("贷款风险策略项目 | V2 | 离线设计与模拟结项",subtitle))
    story.append(Spacer(1,7))
    banner = Table([[para("结项决定：保留 Champion，暂不晋级 Challenger",banner_style)]],colWidths=[width])
    banner.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#EAF4F1")),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story.append(banner)
    section("1  业务问题",base)
    section("2  方案为何改变",design)
    section("3  样本量与周期",sample)
    story.append(para("4  收益与护栏结果",heading))
    table_rows = [[para(s,table_head) for s in ["指标","离线演练结果","晋级界值","结论"]]]
    table_rows += [[para(value,small) for value in row] for row in metrics]
    table = Table(table_rows,colWidths=[width*.25,width*.32,width*.26,width*.17],hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),navy),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F2F5F7"),colors.white]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#CCD7DD"))]))
    story.append(table); story.append(Spacer(1,5)); story.append(para(reliability,note))
    section("5  保留 Champion 的理由",decision)
    story.append(para("6  真实实施的数据缺口与重启条件",heading))
    for index,line in enumerate(missing,1): story.append(para(f"{index}) {line}"))
    story.append(Spacer(1,3)); story.append(para(next_gate))
    story.append(Spacer(1,6))
    story.append(para("证据：ab_feasibility_v2 冻结汇总、候选方案、匹配校准、模拟及独立核查记录。详见同名 Markdown 的证据链接。",note))
    doc = SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=32,rightMargin=32,topMargin=27,bottomMargin=30,
        title="实验结项报告与决策备忘录",author="Loan repayment modeling portfolio",pageCompression=1)
    def footer(canvas,document):
        canvas.saveState(); canvas.setFont("MemoCJK",7); canvas.setFillColor(muted)
        canvas.drawString(32,16,"离线模拟结项 | 固定历史标签与现金流假设 | 未完成真实随机实验")
        canvas.drawRightString(A4[0]-32,16,"1 / 1")
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader = PdfReader(PDF)
    if len(reader.pages)!=1: raise AssertionError(f"Memo must fit one page, got {len(reader.pages)}")
    text = reader.pages[0].extract_text()
    required = ["业务问题","方案为何改变","样本量与周期","收益与护栏结果","保留 Champion","真实实施","17,141","12,614","684","3.672","0.119","7.226"]
    for token in required:
        if token not in text: raise AssertionError(f"Missing PDF content: {token}")
    page_box = reader.pages[0].mediabox
    if abs(float(page_box.width)-A4[0])>.5 or abs(float(page_box.height)-A4[1])>.5: raise AssertionError("Expected A4")
    return len(reader.pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-visual-review",action="store_true",help="Use only after inspecting the latest rendered page")
    args = parser.parse_args()
    if args.finalize_visual_review:
        provenance = read_json(PROVENANCE)
        for name,checksum in {**provenance["input_sha256"],**provenance["output_sha256"]}.items():
            if digest(ROOT/name)!=checksum: raise AssertionError("Source or reviewed output changed")
        preview = ROOT/"tmp/pdfs/experiment_closeout_memo.png"
        saved_preview = PROVENANCE.parent/"memo_preview.png"
        if preview.exists(): preview.replace(saved_preview)
        if not saved_preview.exists(): raise AssertionError("Missing rendered page")
        provenance.update(status="passed",visual_review_passed=True,
            visual_review_checks=["one_A4_page","legible_Chinese_glyphs","no_clipping_or_overlap","all_six_sections_visible"],
            render_engine="Poppler pdftoppm 160 dpi",
            preview_sha256=digest(saved_preview),generator_sha256=digest(Path(__file__)))
        PROVENANCE.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"status":"passed","pages":provenance["page_count"],"visual_review":True},indent=2))
        return
    summary,audit,config,original,half,selected,simulation,sources = load_evidence()
    data = contents(summary,audit,config,original,half,selected,simulation)
    MARKDOWN.parent.mkdir(parents=True,exist_ok=True)
    PDF.parent.mkdir(parents=True,exist_ok=True)
    PROVENANCE.parent.mkdir(parents=True,exist_ok=True)
    write_markdown(data)
    pages = build_pdf(data)
    provenance = {"status":"generated_and_content_checked_visual_review_pending","page_count":pages,"format":"A4",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"source_stage":STAGE,
        "input_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in sources},
        "output_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in (MARKDOWN,PDF)},
        "champion_retained":True,"challenger_promoted":False,"real_experiment_completed":False,
        "frozen_experiment_outputs_modified":False}
    PROVENANCE.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"markdown":str(MARKDOWN),"pdf":str(PDF),"pages":pages,"evidence_verified":True},ensure_ascii=False,indent=2))


if __name__=="__main__": main()

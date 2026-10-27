"""Render a native SVG and a matching PNG of the portfolio batch architecture."""
from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1600, 1100
INK = "#183149"
COLORS = {"historical": ("#edf4fb", "#35668f"), "simulation": ("#fff5e4", "#a77428"),
          "decision": ("#eaf5ef", "#397454"), "future": ("#f3f5f6", "#7a838c")}


def scene():
    items = []
    def label(x, y, value, size=20, bold=False, color=INK):
        items.append(("text", x, y, value, size, bold, color))
    def box(x, y, width, height, kind, title, lines):
        fill, stroke = COLORS[kind]
        items.append(("rect", x, y, width, height, fill, stroke, kind == "future"))
        label(x+20, y+20, title, 23, True)
        for index,line in enumerate(lines):
            label(x+20, y+65+index*28, line, 18)
    def arrow(points, dashed=False):
        items.append(("line", points, dashed))
    label(56, 35, "Loan repayment modeling & experiment decisions", 38, True)
    label(56, 89, "Local batch pipeline | Frozen historical evidence | No live trial or production promotion", 21)
    for x,kind,text_value in ((56,"historical","Historical analysis"),(386,"simulation","Offline simulation"),
                               (716,"decision","Portfolio decision"),(1046,"future","Proposed live work")):
        fill,stroke = COLORS[kind]
        items.append(("rect",x,132,24,24,fill,stroke,kind=="future"))
        label(x+36, 133, text_value, 18)
    label(56, 177, "01  DATA, VALIDATION & FROZEN MODELS", 22, True)
    box(56,214,348,180,"historical","Kaggle data",[
        "688,998 train / 36,324 test", "1,105 anonymous numeric features", "1 = repayment; 0 = default", "id is a join/submission key"])
    box(436,214,348,180,"historical","DuckDB + Parquet",[
        "SQL quality / EDA / shift checks", "Fixed 5-fold split; seed 42", "Inner early stopping", "Row order not verified as time"])
    box(816,214,348,180,"historical","Frozen business models",[
        "A: LightGBM, 5 refit models", "B: 5 folds x 3 seeds, 15 models", "Reused development OOF", "Raw float32 / NaN; uncalibrated"])
    box(1196,214,348,180,"historical","Prediction freeze",[
        "SHA-256 input / model records", "(dataset, id, model_version)", "Business reuses fixed predictions", "No retraining in downstream work"])
    for x in (404,784,1164):
        arrow([(x+2,304),(x+28,304)])
    arrow([(1370,394),(1370,425),(36,425),(36,584),(56,584)])
    label(56, 449, "02  SQL, POLICY ECONOMICS & OFFLINE EXPERIMENT VALIDATION", 22, True)
    box(56,492,348,185,"historical","SQL risk analysis",[
        "Tie-aware ROC-AUC / AP / KS", "Risk bands / approval / drift", "Full cohort + model identity", "Unknown test outcomes stay NULL"])
    box(436,492,348,185,"simulation","Approval simulation",[
        "413,398 dev / 275,600 evaluation", "Frozen thresholds / cash flows", "Full-application profit + risk", "Historical labels; assumed costs"])
    box(816,492,348,185,"simulation","Original A/B -> strict A/A",[
        "Common eligibility / hash / SRM", "Same Champion in both arms", "Calibration + fault injection", "6 stress checks uncertified"])
    box(1196,492,348,185,"simulation","V2 feasible A/B",[
        "Disagreements only; 1:1 arms", "3 strata -> full-eligible ITT", "12,614 / 17,141 planned sample", "20 matched calibration checks"])
    for x in (404,784,1164):
        arrow([(x+2,584),(x+28,584)])
    arrow([(1370,677),(1370,716),(570,716),(570,776)])
    label(56, 740, "03  DECISION & DELIVERY", 22, True)
    box(56,776,1060,220,"decision","Retain Champion; do not promote Challenger",[
        "V2 simulated profit/application: +CNY 3.672; 95% CI [0.119, 7.226]",
        "Business gate: lower bound >CNY 5; fixed sample + quality + mature outcomes required",
        "Assumed enrollment 289 days; final analysis 684 days (not measured business traffic)",
        "Bilingual README / model & experiment cards / one-page memo / 6-page Power BI",
        "Unified CLI: source review -> read-only audit -> staged rebuild in a clean work copy"])
    box(1160,776,384,220,"future","Future live implementation",[
        "borrower_id / safety / execution", "Actual cash flows / mature DPD", "Shadow -> pilot -> locked main", "All promotion gates required", "NOT IMPLEMENTED"])
    arrow([(1116,886),(1160,886)],True)
    label(56,1030,"V2 weights use the full common-eligible cohort. Agreement effects are zero only with identical terms and cash flows.",18)
    label(56,1060,"Historical OOF and simulated profit do not establish real-world causal gain. Source: model card and experiment card, 2026-10-25.",18)
    return items


def render(directory=ROOT/"docs/figures"):
    directory.mkdir(parents=True, exist_ok=True)
    items = scene()
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
           '<title id="title">Loan risk project architecture</title>',
           '<desc id="desc">Historical modeling, frozen SQL and offline experiments, retained Champion, and a separate proposed live implementation path.</desc>',
           '<rect width="1600" height="1100" fill="#ffffff"/>',
           '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#65798a"/></marker></defs>']
    for item in items:
        if item[0] == "rect":
            _,x,y,w,h,fill,stroke,dashed = item
            dash = ' stroke-dasharray="7 5"' if dashed else ''
            svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        elif item[0] == "text":
            _,x,y,value,size,bold,color = item
            svg.append(f'<text x="{x}" y="{y}" dominant-baseline="hanging" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')
        else:
            _,points,dashed = item
            dash = ' stroke-dasharray="7 5"' if dashed else ''
            coords = " ".join(f"{x},{y}" for x,y in points)
            svg.append(f'<polyline points="{coords}" fill="none" stroke="#65798a" stroke-width="2" marker-end="url(#arrow)"{dash}/>')
    svg.append('</svg>')
    (directory/"architecture.svg").write_text("\n".join(svg)+"\n",encoding="utf-8")
    from PIL import Image, ImageDraw, ImageFont
    picture = Image.new("RGB",(WIDTH,HEIGHT),"white")
    draw = ImageDraw.Draw(picture)
    def font(size,bold):
        for name in (f"C:/Windows/Fonts/{'arialbd' if bold else 'arial'}.ttf", "DejaVuSans.ttf"):
            try: return ImageFont.truetype(name,size)
            except OSError: pass
        return ImageFont.load_default(size=size)
    for item in items:
        if item[0] == "rect":
            _,x,y,w,h,fill,stroke,dashed = item
            draw.rounded_rectangle((x,y,x+w,y+h),radius=10,fill=fill,outline=stroke,width=2)
        elif item[0] == "text":
            _,x,y,value,size,bold,color = item
            draw.text((x,y),value,font=font(size,bold),fill=color,anchor="lt")
        else:
            _,points,dashed = item
            if dashed:
                start,end = points
                for x in range(start[0],end[0],12): draw.line((x,start[1],min(x+7,end[0]),end[1]),fill="#65798a",width=2)
            else: draw.line(points,fill="#65798a",width=2)
            (x1,y1),(x2,y2) = points[-2:]
            if x2 > x1: tip = [(x2,y2),(x2-10,y2-5),(x2-10,y2+5)]
            else: tip = [(x2,y2),(x2-5,y2-10),(x2+5,y2-10)]
            draw.polygon(tip,fill="#65798a")
    picture.save(directory/"architecture.png")
    print(f"Native architecture SVG and PNG written: {directory}")


if __name__ == "__main__":
    render()

"""Read-only input/install diagnostics; fetch official format schemas as data."""
import json
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for stage, names in {
    "sql_business_analysis":["model_performance_summary","risk_band_summary","prediction_drift_summary"],
    "approval_policy_simulation":["selected_policy_results","approval_economic_results","selected_policy_definitions","economic_assumptions"],
    "aa_validation":["aa_calibration_results","aa_optional_stopping","aa_srm_calibration"],
    "ab_feasibility_v2":["experiment_duration","primary_stratum_estimates","ab_simulation_results"]
}.items():
    for name in names:
        frame = pd.read_csv(ROOT/f"reports/{stage}/tables/{name}.csv")
        print(json.dumps({"table":f"{stage}/{name}","rows":len(frame),"columns":list(frame.columns),
            "head":frame.head(2).astype(object).where(frame.head(2).notna(),None).to_dict("records")},ensure_ascii=False))

candidates = [Path("C:/Program Files/Microsoft Power BI Desktop/bin/PBIDesktop.exe"),
    Path("C:/Program Files (x86)/Microsoft Power BI Desktop/bin/PBIDesktop.exe")]
for base in (Path("C:/Program Files/WindowsApps"),Path("C:/Users/Cathy/AppData/Local/Microsoft/WindowsApps")):
    try: candidates += list(base.glob("*PowerBI*/bin/PBIDesktop.exe"))+list(base.glob("*PowerBI*/PBIDesktop.exe"))
    except OSError: pass
print(json.dumps({"powerbi_executable_candidates":[str(p) for p in candidates if p.exists()]},ensure_ascii=False))

urls = [
 "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/2.0.0/schema.json",
 "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.0.0/schema.json",
 "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.0.0/schema.json",
 "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json",
 "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
 "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json",
 "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json"
]
folder = ROOT/"dashboard/powerbi/schemas"
folder.mkdir(parents=True,exist_ok=True)
def fetch(url):
    try:
        with urllib.request.urlopen(url,timeout=12) as response:
            raw = response.read()
        value = json.loads(raw)
        name = url.split("/fabric/",1)[1].replace("/","_")
        (folder/name).write_bytes(raw)
        return {"url":url,"downloaded":True,"title":value.get("title"),"required":value.get("required"),
                "properties":list(value.get("properties",{})),"refs":sorted({v for v in json.dumps(value).split('"') if '/schema.json' in v})}
    except Exception as error:
        return {"url":url,"error":str(error)}
with ThreadPoolExecutor(max_workers=7) as pool:
    for result in pool.map(fetch,urls): print(json.dumps(result,ensure_ascii=False))

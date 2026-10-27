"""Offline checks for PBIP structure, field bindings and evidence provenance.

These checks never claim to execute Power Query, DAX, or Desktop rendering.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import re
from pathlib import Path

import pandas as pd
from jsonschema import Draft7Validator
from referencing import Registry, Resource

try:
    from . import powerbi_data as d
except ImportError:
    import powerbi_data as d

OUT = d.OUT
REQUIRED_PROJECT_FILES = (
    "LoanRisk.pbip",
    "LoanRisk.Report/definition.pbir",
    "LoanRisk.Report/definition/version.json",
    "LoanRisk.Report/definition/report.json",
    "LoanRisk.Report/definition/pages/pages.json",
    "LoanRisk.SemanticModel/definition.pbism",
    "LoanRisk.SemanticModel/model.bim",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values(): yield from walk(child)
    elif isinstance(value, list):
        for child in value: yield from walk(child)


def cached_schemas(folder):
    registry, schemas = Registry(), {}
    for path in sorted(folder.glob("*.json")):
        if path.name=="schema_index.json": continue
        doc = read(path)
        if not doc.get("$schema") or not doc.get("$id"): continue
        uri = "https://developer.microsoft.com/json-schemas/fabric/"+path.name.replace("_","/")
        resource = Resource.from_contents(doc)
        registry = registry.with_resource(uri, resource).with_resource(doc["$id"], resource)
        schemas[uri] = schemas[doc["$id"]] = doc
    return registry, schemas


def verify_required_files(root):
    missing = [name for name in REQUIRED_PROJECT_FILES if not (Path(root)/name).is_file()]
    assert not missing, "Missing required Power BI project files: " + ", ".join(missing)
    return len(REQUIRED_PROJECT_FILES)


def schema_errors(root):
    verify_required_files(root)
    registry, schemas = cached_schemas(root/"schemas")
    paths = [root/"LoanRisk.pbip", root/"LoanRisk.SemanticModel/definition.pbism",
             root/"LoanRisk.Report/definition.pbir"]
    paths += sorted((root/"LoanRisk.Report/definition").rglob("*.json"))
    errors = []
    for path in paths:
        value = read(path)
        uri = value["$schema"]
        assert uri in schemas, f"Missing cached schema: {uri}"
        validator = Draft7Validator(schemas[uri], registry=registry)
        for error in validator.iter_errors(value):
            errors.append(f"{path.relative_to(root)} / {'/'.join(map(str,error.path))}: {error.message}")
    return len(paths), errors


def verify_bindings(model, visual, where):
    tables = {t["name"]:t for t in model["model"]["tables"]}
    count = 0
    for node in walk(visual):
        if not isinstance(node,dict): continue
        for kind, key in (("Column", "columns"), ("Measure", "measures")):
            if kind not in node: continue
            field = node[kind]
            name = field["Expression"]["SourceRef"]["Entity"]
            assert name in tables, f"Unknown table {name} at {where}"
            assert field["Property"] in {x["name"] for x in tables[name].get(key,[])}, f"Unknown {kind} {name}.{field['Property']} at {where}"
            count += 1
    return count


def verify_data(root, tables, dictionary):
    for name, expected in tables.items():
        spec = dictionary[name]
        assert spec["rows"] == len(expected)
        assert [c["name"] for c in spec["columns"]] == list(expected)
        dtype = {c["name"]:"string" for c in spec["columns"] if c["dataType"]=="string"}
        actual = pd.read_csv(root/spec["file"], dtype=dtype)
        assert list(actual) == list(expected)
        for c in expected:
            assert actual[c].isna().tolist() == expected[c].isna().tolist(), f"NULL changed: {name}.{c}"
            nonnull = expected[c].notna().to_numpy()
            if not nonnull.any(): continue
            pd.testing.assert_series_equal(actual[c].iloc[nonnull].reset_index(drop=True),expected[c].iloc[nonnull].reset_index(drop=True),
                check_dtype=False, check_names=False, rtol=1e-12, atol=1e-12, obj=f"{name}.{c}")


def verify_model_names(model):
    global_measures = set()
    for table in model["model"]["tables"]:
        columns = {c["name"].casefold() for c in table["columns"]}
        for measure in table.get("measures",[]):
            key = measure["name"].casefold()
            assert key not in columns, f"Measure/column name conflict: {table['name']}.{measure['name']}"
            assert key not in global_measures, f"Duplicate measure name: {measure['name']}"
            global_measures.add(key)


def verify_refresh_metadata(model):
    assert model["model"].get("defaultPowerBIDataSourceVersion")=="powerBI_V3", "Enhanced refresh metadata requires defaultPowerBIDataSourceVersion=powerBI_V3"


def verify_model(model, dictionary, tables, root):
    verify_refresh_metadata(model)
    verify_model_names(model)
    definition = model["model"]
    names = [t["name"] for t in definition["tables"]]
    assert len(names)==len(set(names)) and set(names)==set(dictionary)
    data_folder = next(x for x in definition["expressions"] if x["name"]=="DataFolder")["expression"]
    assert str(root/"data") in data_folder
    measures = {}
    for t in definition["tables"]:
        assert len(t["columns"])==len({c["name"] for c in t["columns"]})
        assert [c["name"] for c in t["columns"]]==[c["name"] for c in dictionary[t["name"]]["columns"]]
        for c, meta in zip(t["columns"],dictionary[t["name"]]["columns"]):
            assert c["dataType"]==meta["dataType"] and c["summarizeBy"]=="none"
        source = "\n".join(t["partitions"][0]["source"]["expression"])
        assert 'Table.ReplaceValue(Headers, "", null' in source
        assert '"en-US"' in source and f'\\{t["name"]}.csv' in source
        assert t["partitions"][0]["mode"]=="import"
        for measure in t.get("measures",[]):
            assert measure["name"] not in measures, "Unqualified DAX measure names must be unique"
            measures[measure["name"]] = measure["expression"]
    for relationship in definition["relationships"]:
        f,c,t,tc = (relationship[k] for k in ("fromTable","fromColumn","toTable","toColumn"))
        assert c in tables[f] and tc in tables[t]
        assert tables[t][tc].is_unique and tables[f][c].isin(tables[t][tc]).all()
        assert relationship["crossFilteringBehavior"]=="oneDirection"
        assert f not in ("DailyTraffic","Coverage","Exposure","AlphaLevels","PromotionGates")
        assert t not in ("DailyTraffic","Coverage","Exposure","AlphaLevels","PromotionGates")
    for name in ("冻结收益差","冻结收益下界","冻结收益上界","业务收益门槛","所需分歧样本","冻结分歧样本","当前晋级决定"):
        assert "REMOVEFILTERS()" in measures[name], f"Fixed evidence must ignore report filters: {name}"
        assert not any(x in measures[name] for x in ("DailyTraffic", "Coverage", "Exposure"))
    assert "SUM(" not in measures["还款 ROC-AUC"]
    assert 'Cohort] = "policy_evaluation"' in measures["评价模拟净收益"]
    # Static dependency checks catch typoed references; they do not execute DAX.
    table_names = set(tables)
    columns = {n:set(frame) for n,frame in tables.items()}
    for name, expression in measures.items():
        for table,column in re.findall(r"([A-Za-z][A-Za-z0-9_]*)\[([^\]]+)\]",expression):
            assert table in table_names and column in columns[table], f"DAX reference {table}[{column}] at {name}"
        for referenced in re.findall(r"(?<![\w\]])\[([^\]]+)\]",expression):
            assert referenced in measures, f"Unknown measure [{referenced}] at {name}"
    return len(measures)


def verify(root=OUT):
    root = Path(root)
    required_files = verify_required_files(root)
    tables,evidence = d.collect()
    d.assert_contract(tables,evidence)
    dictionary = read(root/"data_dictionary.json")
    manifest = read(root/"source_manifest.json")
    assert manifest["source_sha256"]==evidence["source_sha256"], "Dashboard source version changed"
    assert manifest["no_external_publish"] and not manifest["production_ready"]
    if manifest["desktop_open_verified"]:
        assert manifest.get("desktop_open_validation_basis"), "Desktop opening evidence must be recorded"
    if manifest.get("desktop_refresh_verified") or manifest.get("dax_engine_execution_verified"):
        assert manifest.get("desktop_runtime_validation_basis"), "Desktop runtime confirmation must have an evidence basis"
        assert manifest.get("dax_engine_execution_scope"), "Verified DAX scope must be stated"
    assert manifest["data_sha256"]=={n:d.sha(root/f"data/{n}.csv") for n in tables}
    verify_data(root,tables,dictionary)
    schema_files, errors = schema_errors(root)
    if errors: raise AssertionError("Microsoft JSON schema errors:\n"+"\n".join(errors))
    model = read(root/"LoanRisk.SemanticModel/model.bim")
    measures = verify_model(model,dictionary,tables,root)
    report = root/"LoanRisk.Report"
    meta = read(report/"definition/pages/pages.json")
    assert meta["pageOrder"]==["decision","models","policy","aa","feasibility","value"]
    assert meta["activePageName"]=="decision"
    assert read(root/"LoanRisk.pbip")["artifacts"][0]["report"]["path"]=="LoanRisk.Report"
    assert read(report/"definition.pbir")["datasetReference"]["byPath"]["path"]=="../LoanRisk.SemanticModel"
    visual_count,bindings = 0,0
    for page in meta["pageOrder"]:
        folder = report/"definition/pages"/page
        definition = read(folder/"page.json")
        width,height = definition["width"],definition["height"]
        assert width==1440 and height==900
        visuals = sorted(folder.glob("visuals/*/visual.json"))
        assert len(visuals)==len({read(v)["name"] for v in visuals})
        positions = []
        for path in visuals:
            v = read(path)
            pos = v["position"]
            assert 0<=pos["x"] and 0<=pos["y"] and pos["width"]>0 and pos["height"]>0
            assert pos["x"]+pos["width"]<=width and pos["y"]+pos["height"]<=height
            bindings += verify_bindings(model,v,path)
            positions.append((v["name"],pos))
        for (a,p),(b,q) in itertools.combinations(positions,2):
            overlap = min(p["x"]+p["width"],q["x"]+q["width"])-max(p["x"],q["x"])>0 and min(p["y"]+p["height"],q["y"]+q["height"])-max(p["y"],q["y"])>0
            assert not overlap, f"Overlapping containers: {page} {a}/{b}"
        visual_count += len(visuals)
    for package in read(report/"definition/report.json")["resourcePackages"]:
        for item in package["items"]:
            assert (report/"StaticResources"/package["name"]/item["path"]).is_file()
    state = evidence["state"]
    for daily,cov,exposure in itertools.product((1000,5000,20000),(.2,.75),(.5,1.)):
        result = d.reference_forecast(state,daily,cov,exposure)
        n = state["RequiredSamples"]
        rate = daily*cov*exposure*state["DisagreementFraction"]
        assert (result["enrollment_days"]-1)*rate<n<=result["enrollment_days"]*rate
        assert result["final_days"]==result["enrollment_days"]+395
    return {"status":"passed","validation_scope":"offline_files_schema_provenance_bindings_and_reference_math",
        "required_project_files_verified":required_files,
        "pages":len(meta["pageOrder"]),"visuals":visual_count,"schema_validated_files":schema_files,
        "tables":len(tables),"aggregate_rows":sum(len(t) for t in tables.values()),
        "measures_statically_checked":measures,"visual_field_bindings":bindings,"what_if_reference_cases":12,
        "source_hashes_verified":len(evidence["source_sha256"]),"unknown_live_metrics_preserved":True,
        "fixed_decision":"retain_champion","desktop_open_verified":bool(manifest["desktop_open_verified"]),
        "desktop_refresh_verified":bool(manifest.get("desktop_refresh_verified",False)),
        "dax_engine_execution_verified":bool(manifest.get("dax_engine_execution_verified",False)),
        "dax_engine_execution_scope":manifest.get("dax_engine_execution_scope","not_executed"),
        "desktop_runtime_validation_basis":manifest.get("desktop_runtime_validation_basis","not_confirmed"),
        "decision_page_render_verified":bool(manifest.get("decision_page_render_verified",False)),
        "all_pages_render_verified":bool(manifest.get("all_pages_render_verified",False)),
        "desktop_render_verified":bool(manifest.get("desktop_render_verified",False)),"production_ready":False}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    args = parser.parse_args()
    result = verify()
    if args.write: (OUT/"artifact_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

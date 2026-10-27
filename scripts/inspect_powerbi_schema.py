"""Download referenced official PBIR schemas and inspect their structural contract."""
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/"dashboard/powerbi/schemas"
registry = {}
filenames = {}
for file in FOLDER.glob("*.json"):
    if file.name=="schema_index.json": continue
    obj = json.loads(file.read_text(encoding="utf-8"))
    uri = obj.get("$id",obj.get("$schema"))
    # Microsoft schemas use their canonical URI as $id when present.
    canonical = "https://developer.microsoft.com/json-schemas/fabric/"+file.name.removesuffix(".json").replace("_","/")+".json"
    registry[obj.get("$id",canonical)] = obj
    filenames[obj.get("$id",canonical)] = file.name

def refs(obj):
    if isinstance(obj,dict):
        if "$ref" in obj: yield obj["$ref"]
        for value in obj.values(): yield from refs(value)
    elif isinstance(obj,list):
        for value in obj: yield from refs(value)

queue = list(registry)
while queue:
    uri = queue.pop(0)
    for ref in refs(registry[uri]):
        target = urllib.parse.urldefrag(urllib.parse.urljoin(uri,ref))[0]
        if not target or target in registry: continue
        if not target.startswith("https://developer.microsoft.com/json-schemas/fabric/"): continue
        with urllib.request.urlopen(target,timeout=15) as response: raw = response.read()
        obj = json.loads(raw)
        registry[target] = obj
        (FOLDER/target.split("/fabric/",1)[1].replace("/","_")).write_bytes(raw)
        filenames[target] = target.split("/fabric/",1)[1].replace("/","_")
        queue.append(target)

index = {uri:filenames[uri] for uri in registry}
(FOLDER/"schema_index.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
for uri,obj in registry.items():
    if any(term in uri for term in ("visualContainer/2.0","definitionProperties","pbipProperties","report/2.0","page/2.0")):
        keys = ["position","visual","themeCollection","datasetReference","version","displayOption","settings"]
        selected = {key:obj.get("properties",{}).get(key) for key in keys if key in obj.get("properties",{})}
        definitions = obj.get("definitions",obj.get("$defs",{}))
        print(json.dumps({"uri":uri,"properties":selected,"definitions":list(definitions)},ensure_ascii=False))
        for name,definition in definitions.items():
            if name in ("Visual","VisualQuery","QueryState","Projection","ThemeCollection","BaseTheme","ReportTheme","ResourcePackage","ResourcePackageItem","Position","VisualContainerObjects"):
                print(json.dumps({"definition":name,"schema":definition},ensure_ascii=False))
print(json.dumps({"schemas":len(registry)},ensure_ascii=False))
try:
    import jsonschema
    print("jsonschema_available")
except ImportError:
    print("jsonschema_missing")

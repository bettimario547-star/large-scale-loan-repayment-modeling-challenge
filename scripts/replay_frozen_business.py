"""Read-only business audit; never call the historical audit-writing CLI wrappers."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = {
    "sql_business_analysis": ("business_analysis_common", "verify_sql_business_analysis", "Paths"),
    "approval_policy_simulation": ("policy_simulation_common", "verify_approval_policy_simulation", "paths"),
    "ab_test_design": ("experiment_design_common", "verify_ab_test_design", "paths"),
    "aa_validation": ("aa_validation_common", "verify_aa_validation", "paths"),
    "ab_feasibility_v2": ("ab_feasibility_common", "verify_ab_feasibility", "paths"),
}


def replay(stage, *, deep=True, importer=importlib.import_module):
    common_name, verifier_name, factory_name = STAGES[stage]
    common = importer(f"scripts.{common_name}")
    verifier = importer(f"scripts.{verifier_name}")
    output = getattr(common, factory_name)()
    kwargs = {}
    if stage in ("aa_validation", "ab_feasibility_v2"):
        kwargs = {"deep": deep, "progress": lambda value: print(value, flush=True)}
    audit = verifier.verify(output, **kwargs)
    if audit.get("status") != "passed":
        raise RuntimeError(f"{stage}: verification failed")
    return {"stage": stage, "read_only": True, "scope": "full_replay" if deep else "rows_sql_exports_hashes_and_bootstrap",
            "note": "No frozen output or historical audit file was written.", "audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), help="Default: audit all stages in dependency order")
    parser.add_argument("--quick", action="store_true", help="Skip full repeated Monte Carlo/hash replays in A/A and V2")
    args = parser.parse_args()
    for stage in (args.stage,) if args.stage else STAGES:
        print(json.dumps({"starting_stage": stage, "read_only": True, "deep": not args.quick}), flush=True)
        result = replay(stage, deep=not args.quick)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

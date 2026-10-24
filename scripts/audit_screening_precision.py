"""Explain near-constant flags at model float32 precision versus original doubles."""

import json

import numpy as np
import pandas as pd

try:
    from . import run_feature_screening as screening
except ImportError:
    import run_feature_screening as screening


def audit():
    plans = json.loads((screening.REPORT / "feature_plans.json").read_text(encoding="utf-8"))
    candidates = sorted({name for plan in plans if plan["variant"] == "low_information_filter"
                         for name in plan["removed"]})
    raw = pd.read_parquet(screening.base.TRAIN, columns=["label"] + candidates)
    labels, rows = raw.label.to_numpy(), np.arange(len(raw))
    records = []
    for name in candidates:
        double = raw[name].to_numpy(dtype=np.float64)
        single = double.astype(np.float32)
        records.append({"feature": name,
                        "original_float64": screening.column_statistics(double, labels, rows),
                        "model_float32": screening.column_statistics(single, labels, rows)})
    result = {"note": "Screening uses the same float32 cache as the baseline; casting can merge very close values.",
              "records": records}
    screening.base.json_write(screening.REPORT / "precision_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    audit()

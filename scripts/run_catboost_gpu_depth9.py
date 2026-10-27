"""Fixed depth-nine challenger using the certified numerical GPU Pool cache."""
from pathlib import Path
import argparse
import json

try:
    from . import run_catboost_fast_gpu as engine
except ImportError:
    import run_catboost_fast_gpu as engine


def configure(root):
    source, base_source = Path(__file__).resolve(), Path(engine.__file__).resolve()
    legacy = engine.initialize(root)
    legacy_locations, legacy_pool = engine.locations, engine.binary_pool

    def locations(dataset_root):
        dataset_root = Path(dataset_root)
        return dataset_root / "reports/catboost_gpu_depth9", dataset_root / "artifacts/catboost_gpu_depth9"

    def initialize(dataset_root):
        engine.original.checked_config(dataset_root)
        params = dict(legacy["experiment"]["params"])
        params.update(depth=9, iterations=2200, l2_leaf_reg=12, gpu_ram_part=.60)
        exp = {"parent_training_signature": legacy["training_signature"],
               "source_sha256": engine.original.candidates.sha256(source),
               "engine_sha256": engine.original.candidates.sha256(base_source),
               "params": params, "feature_plans": legacy["experiment"]["feature_plans"],
               "data_input": "reuse identical certified fitting/early-stop/refit binary Pools",
               "catboost_version": engine.cb.__version__}
        signature = engine.original.trees.signing_hash(exp)
        path = locations(dataset_root)[0] / "training_config.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["training_signature"] != signature:
                raise RuntimeError("Depth-nine study configuration changed")
            return saved
        config = {"training_signature": signature, "experiment": exp}
        engine.original.candidates.json_write(path, config)
        return config

    def binary_pool(dataset_root, values, labels, indices, names, fold, stage, threads, signature, ref_borders=None):
        engine.locations = legacy_locations
        try:
            return legacy_pool(dataset_root, values, labels, indices, names, fold, stage, threads,
                               legacy["training_signature"], ref_borders)
        finally:
            engine.locations = locations

    engine.locations, engine.initialize, engine.binary_pool = locations, initialize, binary_pool
    engine.NAME, engine.__file__ = "catboost_gpu_depth9_s42", str(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=engine.ROOT)
    args, _ = parser.parse_known_args()
    configure(args.root)
    engine.main()

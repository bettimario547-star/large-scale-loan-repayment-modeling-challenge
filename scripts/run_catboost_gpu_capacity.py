"""A fixed depth-eight challenger reusing certified training-only binary Pools.

Configure the existing GPU engine inside this dedicated Python process. Keep
its source and original study untouched; model artifacts use a separate scope.
"""
from pathlib import Path
import argparse
import json

try:
    from . import run_catboost_fast_gpu as engine
except ImportError:
    import run_catboost_fast_gpu as engine


def configure(root):
    wrapper = Path(__file__).resolve()
    legacy_source = Path(engine.__file__).resolve()
    legacy_config = engine.initialize(root)
    legacy_locations, legacy_binary_pool = engine.locations, engine.binary_pool

    def locations(dataset_root):
        dataset_root = Path(dataset_root)
        return dataset_root / "reports/catboost_gpu_capacity", dataset_root / "artifacts/catboost_gpu_capacity"

    def initialize(dataset_root):
        engine.original.checked_config(dataset_root)
        params = dict(legacy_config["experiment"]["params"])
        params.update(depth=8, iterations=2000, l2_leaf_reg=8, gpu_ram_part=.60)
        experiment = {"parent_training_signature": legacy_config["training_signature"],
                      "source_sha256": engine.original.candidates.sha256(wrapper),
                      "engine_sha256": engine.original.candidates.sha256(legacy_source),
                      "params": params, "feature_plans": legacy_config["experiment"]["feature_plans"],
                      "data_input": "reuse identical certified fitting/early-stop/refit binary Pools",
                      "catboost_version": engine.cb.__version__}
        signature = engine.original.trees.signing_hash(experiment)
        report, _ = locations(dataset_root)
        path = report / "training_config.json"
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
            if config["training_signature"] != signature:
                raise RuntimeError("Capacity study source/configuration changed")
            return config
        config = {"training_signature": signature, "experiment": experiment}
        engine.original.candidates.json_write(path, config)
        return config

    def binary_pool(dataset_root, values, labels, indices, names, fold, stage, threads, signature, ref_borders=None):
        # This process runs one Pool operation at a time. The cache scope is
        # restored before model fitting, checkpointing, collection or spawning.
        engine.locations = legacy_locations
        try:
            return legacy_binary_pool(dataset_root, values, labels, indices, names, fold, stage,
                                      threads, legacy_config["training_signature"], ref_borders)
        finally:
            engine.locations = locations

    engine.locations = locations
    engine.initialize = initialize
    engine.binary_pool = binary_pool
    engine.NAME = "catboost_gpu_capacity_s42"
    engine.__file__ = str(wrapper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, default=engine.ROOT)
    known, _ = parser.parse_known_args()
    configure(known.root)
    engine.main()

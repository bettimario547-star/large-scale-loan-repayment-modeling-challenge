"""Resume one family of CPU folds with an explicit process/time budget.

Use five threads alongside the five-CPU GPU worker. Collection is a separate
step after both controllers stop, so they never write the same ensemble files.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

try:
    from . import run_public_auc_optimization as original
except ImportError:
    import run_public_auc_optimization as original


def train(root, minutes, family, parallel=False, threads=5):
    config = original.checked_config(root)
    families = {"seeds": "lgbm_drift_tuned", "availability": "lgbm_drift_available", "bins": "lgbm_drift_bins255"}
    variant = families[family]
    tasks = [task for task in original.task_order(False) if task[0] == variant
             and not original.completed_job(root, *task, config["training_signature"])]
    workers = 2 if parallel else 1
    if not 1 <= threads <= 10 or workers * threads > 10:
        raise ValueError("CPU workers must stay within the total ten-thread budget")
    if tasks and (psutil.cpu_count() < workers * threads or psutil.virtual_memory().available < (6 if parallel else 2) * 1024**3):
        raise RuntimeError("Insufficient CPU/available memory for this refinement mode")
    logs = Path(root) / "logs/public_auc_cpu_refinements" / str(time.time_ns())
    logs.mkdir(parents=True)
    deadline = time.monotonic() + minutes * 60
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(threads)
    active, failures, completed = {}, [], []
    started = time.monotonic()
    print(f"CPU refinement {family}: {workers} workers x {threads} threads, {minutes} minutes", flush=True)
    try:
        while (tasks or active) and time.monotonic() < deadline:
            while tasks and len(active) < workers:
                task = tasks.pop(0)
                _, seed, fold = task
                log = (logs / f"{variant}_s{seed}_fold{fold}.log").open("w", encoding="utf-8")
                command = [sys.executable, "-u", str(original.CODE_ROOT / "scripts/run_public_auc_optimization.py"),
                           "--stage", "worker", "--root", str(root), "--variant", variant,
                           "--seed", str(seed), "--fold", str(fold), "--threads", str(threads)]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                active[process.pid] = (process, log, task)
                print(f"Launched {task}, PID {process.pid}", flush=True)
            for pid, (process, log, task) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                del active[pid]
                if process.returncode:
                    failures.append({"task": task, "exit_code": process.returncode})
                else:
                    if original.completed_job(root, *task, config["training_signature"]) is None:
                        raise RuntimeError("Worker returned without a complete checkpoint")
                    completed.append(task)
                    print(f"Saved {task}", flush=True)
                original.candidates.json_write(Path(root) / "reports/public_auc_optimization/cpu_refinement_progress.json",
                    {"family": family, "saved_this_session": completed, "failures": failures})
            time.sleep(.5)
    finally:
        interrupted = [task for process, log, task in active.values() if process.poll() is None]
        for process, log, task in active.values():
            original.stop_worker_tree(process)
            log.close()
        original.candidates.json_write(Path(root) / "reports/public_auc_optimization/last_cpu_refinement_session.json",
            {"family": family, "minutes_budget": minutes, "elapsed_seconds": time.monotonic()-started,
             "workers": workers, "threads_per_worker": threads, "completed": completed, "failures": failures,
             "stopped_at_deadline": interrupted, "log_directory": str(logs)})
    print("CPU refinement ended; run collect after any simultaneous GPU controller ends", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=original.ROOT)
    parser.add_argument("--time-budget-minutes", type=float, default=45)
    parser.add_argument("--family", choices=("seeds", "availability", "bins"), default="seeds")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if not np.isfinite(args.time_budget_minutes) or args.time_budget_minutes <= 0:
        parser.error("Invalid time budget")
    train(args.root, args.time_budget_minutes, args.family, args.parallel, args.threads)


if __name__ == "__main__":
    main()

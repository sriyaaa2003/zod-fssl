"""A portable job list: run the remaining experiments on any machine (Kaggle, a laptop, Modal).

A job is {"kind": "grid", "seed": int, "spec": {tag, overrides, train, centralized}} or {"kind": "reverse", "seed": int}.
Results are the same JSON files the Modal runs write, plus a `hardware` field so every number says where it ran.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import pandas as pd
import torch

from . import config as C
from .federated import run, save_record
from .gen import run_reverse
from .grid import RunSpec, config_for

REVERSE_TAGS = ("rev_A_teacher", "rev_B_oracle", "rev_B_control", "rev_A_from_oracle", "rev_A_from_ensemble",
                "rev_A_from_control")


def load_jobs(path: str | Path) -> list[dict]:
    jobs = json.loads(Path(path).read_text())
    for j in jobs:
        if j.get("kind") not in ("grid", "reverse") or not isinstance(j.get("seed"), int):
            raise ValueError(f"bad job: {j}")
        if j["kind"] == "grid" and "tag" not in j.get("spec", {}):
            raise ValueError(f"grid job without a spec tag: {j}")
    return jobs


def job_name(job: dict) -> str:
    return f"{job['spec']['tag']}/seed{job['seed']}" if job["kind"] == "grid" else f"reverse/seed{job['seed']}"


def expected_files(job: dict) -> list[str]:
    """Result files a finished job leaves behind (interpolation runs leave one extra file per alpha)."""
    seed = job["seed"]
    if job["kind"] == "reverse":
        return [f"{t}__seed{seed}.json" for t in REVERSE_TAGS]
    tag = job["spec"]["tag"]
    alphas = job["spec"].get("overrides", {}).get("eval", {}).get("interp_alphas", [])
    return [f"{tag}__seed{seed}.json"] + [f"{tag}+interp{a}__seed{seed}.json" for a in alphas]


def is_done(job: dict, results_dir: str | Path) -> bool:
    d = Path(results_dir)
    return all((d / f).exists() and (d / f).stat().st_size > 0 for f in expected_files(job))


def pending(jobs: list[dict], results_dir: str | Path) -> list[dict]:
    return [j for j in jobs if not is_done(j, results_dir)]


def split_jobs(jobs: list[dict], n_workers: int, i: int) -> list[dict]:
    """Round-robin share of the jobs for worker i, so every worker gets a similar mix of long and short jobs."""
    if not 0 <= i < n_workers:
        raise ValueError("worker index out of range")
    return jobs[i::n_workers]


def hardware_info(device) -> dict:
    dev = torch.device(device)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" and torch.cuda.is_available() else None
    return {"provider": os.environ.get("ZODFL_PROVIDER", "unknown"), "gpu": gpu, "torch": torch.__version__,
            "platform": platform.platform()}


def run_job(job: dict, cfg: C.Config, index: pd.DataFrame, store, device, results_dir: str | Path, log=print) -> list[Path]:
    """Run one job and write its result files. Does not skip finished jobs; use `pending` for that."""
    hw = hardware_info(device)
    if job["kind"] == "reverse":
        recs = run_reverse(cfg, index, store, seed=job["seed"], device=device, log=log)
    else:
        rs = RunSpec(**job["spec"])
        rec = run(config_for(cfg, rs), index, store, seed=job["seed"], device=device, tag=rs.tag,
                  centralized=rs.centralized, train=rs.train, log=log)
        recs = [rec]
    paths = []
    for rec in recs:
        rec["hardware"] = hw
        paths.append(save_record(rec, results_dir))
    return paths


def run_all(jobs: list[dict], cfg: C.Config, index: pd.DataFrame, store, device, results_dir: str | Path,
            deadline: float | None = None, log=print) -> dict:
    """Run the pending jobs in order. Failures are recorded and do not stop the rest; no new job starts after
    `deadline` (a time.time() value), so a time-limited session can still save and exit cleanly."""
    done, failed, skipped = [], [], []
    for job in jobs:
        if is_done(job, results_dir):
            skipped.append(job_name(job))
            continue
        if deadline is not None and time.time() > deadline:
            skipped.append(job_name(job) + " (deadline)")
            continue
        t0 = time.time()
        try:
            run_job(job, cfg, index, store, device, results_dir, log=log)
            done.append(job_name(job))
            log(f"[job done] {job_name(job)} in {time.time() - t0:.0f}s")
        except Exception as err:  # noqa: BLE001  record it, keep going; never hide it
            failed.append({"job": job_name(job), "error": repr(err)})
            log(f"[job FAILED] {job_name(job)}: {err!r}")
    return {"done": done, "failed": failed, "skipped": skipped}

"""Build the small Kaggle code dataset and the kernel folder. No credentials and no ZOD data are involved.

    python -m zodfl.kaggle_bundle --username <your kaggle username>

Creates build/kaggle_code (one file, zodfl_code.tar.gz holding src/, config/ and jobs.json, plus
dataset-metadata.json: the Kaggle tool skips sub-folders, so the code travels as a single archive) and
build/kaggle_kernel (worker.py, kernel-metadata.json). Run the kaggle commands from inside build/ (the tool
mishandles a path containing a slash on Windows):
    cd build && kaggle datasets create -p kaggle_code -t          (first time)
    cd build && kaggle datasets version -p kaggle_code -m "update" -t   (later)
    cd build && kaggle kernels push -p kaggle_kernel
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARDS_SLUG, CODE_SLUG, KERNEL_SLUG = "zod-fssl-shards", "zodfl-code", "zod-fssl-remaining-runs"   # slug = title, as Kaggle derives it
ARCHIVE = "zodfl_code.tar.gz"


def valid_username(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-_]{1,60}", name))


def code_metadata(user: str) -> dict:
    return {"title": "zodfl code", "id": f"{user}/{CODE_SLUG}", "licenses": [{"name": "other"}],
            "subtitle": "Library, config and job list for the zod-fssl study (no data)"}


def kernel_metadata(user: str, gpu: bool = True) -> dict:
    return {"id": f"{user}/{KERNEL_SLUG}", "title": "zod-fssl remaining runs", "code_file": "worker.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "true" if gpu else "false", "enable_internet": "false",
            "dataset_sources": [f"{user}/{SHARDS_SLUG}", f"{user}/{CODE_SLUG}"],
            "competition_sources": [], "kernel_sources": []}


def build(user: str, out: Path | None = None, jobs_file: Path | None = None) -> dict:
    if not valid_username(user):
        raise ValueError(f"suspicious Kaggle username {user!r}")
    out = out or ROOT / "build"
    stage, code, kern = out / "_stage", out / "kaggle_code", out / "kaggle_kernel"
    for d in (stage, code, kern):
        if d.exists():
            shutil.rmtree(d)
    (stage / "src").mkdir(parents=True)
    code.mkdir(parents=True)
    kern.mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "zodfl", stage / "src" / "zodfl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "config", stage / "config")
    shutil.copy(jobs_file or ROOT / "kaggle" / "jobs.json", stage / "jobs.json")
    with tarfile.open(code / ARCHIVE, "w:gz") as tf:
        for name in ("src", "config", "jobs.json"):
            tf.add(stage / name, arcname=name)
    shutil.rmtree(stage)
    (code / "dataset-metadata.json").write_text(json.dumps(code_metadata(user), indent=1))
    shutil.copy(ROOT / "kaggle" / "worker.py", kern / "worker.py")
    (kern / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(user), indent=1))
    return {"code": code, "kernel": kern, "archive": code / ARCHIVE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    a = ap.parse_args()
    info = build(a.username.strip().lower())
    print(f"built {info['archive']} ({info['archive'].stat().st_size / 1e3:.0f} KB) and {info['kernel']}")


if __name__ == "__main__":
    main()

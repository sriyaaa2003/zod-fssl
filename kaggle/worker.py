"""Kaggle kernel entry point: runs the remaining zod-fssl jobs on the GPU(s) of one Kaggle session.

Inputs (attached as Kaggle datasets, both private):
  zod-fssl-shards  the shrunk ZOD images (uint8 shards), manifest.json, index.parquet, pretrained weights (*.pth)
  zodfl-code       src/, config/ and jobs.json (the list of jobs still to run)
Outputs: /kaggle/working/results/*.json (one file per run, same format as the Modal runs, with a `hardware` field),
         results_bundle.tar.gz, summary_*.json (what finished, failed or was skipped).
No internet is needed: the pretrained weights ship with the shards dataset.
"""
import glob
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

START = time.time()
MAX_HOURS = float(os.environ.get("ZODFL_MAX_HOURS", "8.5"))      # no new job starts after this (session cap is longer)
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def find_one(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"nothing under {INPUT} matches {pattern}; is the dataset attached?")
    return Path(hits[0])


def unpack_code(archive: Path) -> Path:
    """The code dataset is one tar.gz (the Kaggle upload tool skips sub-folders); unpack it into the working folder."""
    dest = WORK / "code"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")
    if not (dest / "src" / "zodfl" / "__init__.py").exists():
        raise FileNotFoundError(f"{archive} does not contain src/zodfl")
    return dest


def find_code() -> Path:
    """Kaggle may unpack an uploaded archive into folders; accept either form."""
    unpacked = sorted(glob.glob(f"{INPUT}/**/src/zodfl/__init__.py", recursive=True))
    if unpacked:
        return Path(unpacked[0]).parents[2]
    return unpack_code(find_one(f"{INPUT}/**/zodfl_code.tar.gz"))


def locate() -> dict:
    manifest = find_one(f"{INPUT}/**/manifest.json")
    return {
        "shards": manifest.parent,
        "index": find_one(f"{INPUT}/**/index.parquet"),
        "code": find_code(),
        "weights": sorted(glob.glob(f"{INPUT}/**/*.pth", recursive=True)),
    }


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=False)


def worker(i: int, n: int, paths: dict, deadline: float) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i)                 # this process sees exactly one GPU
    os.environ["ZODFL_PROVIDER"] = "kaggle"
    os.environ["TORCH_HOME"] = str(WORK / "torch_home")
    sys.path.insert(0, str(paths["code"] / "src"))
    import pandas as pd
    import torch

    from zodfl import config as C
    from zodfl.jobs import load_jobs, run_all, split_jobs
    from zodfl.store import ShardStore

    print(f"[worker {i}] torch {torch.__version__}, gpu {torch.cuda.get_device_name(0)}", flush=True)
    results = WORK / "results"
    results.mkdir(exist_ok=True)
    cfg = C.load(paths["code"] / "config" / "default.toml")
    cfg = C.override(cfg, data={"index_path": str(paths["index"]), "shards_dir": str(paths["shards"])},
                     run={"results_dir": str(results)})
    index = pd.read_parquet(paths["index"])
    store = ShardStore(paths["shards"])                         # memory-mapped: both workers share the page cache
    jobs = split_jobs(load_jobs(paths["code"] / "jobs.json"), n, i)
    print(f"[worker {i}] {len(jobs)} jobs assigned", flush=True)
    summary = run_all(jobs, cfg, index, store, "cuda", results, deadline=deadline)
    (WORK / f"summary_{i}.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv; free -h; df -h /kaggle/working /kaggle/input | head -n 5")
    paths = locate()
    print("inputs:", {k: (str(v) if not isinstance(v, list) else f"{len(v)} files") for k, v in paths.items()}, flush=True)
    ckpt = WORK / "torch_home" / "hub" / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    for w in paths["weights"]:
        shutil.copy(w, ckpt / Path(w).name)
    import torch

    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("No GPU visible. Turn on an accelerator in the notebook's Session options and run again.")
    deadline = START + MAX_HOURS * 3600
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(i, n, paths, deadline)) for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    results = WORK / "results"
    n_files = len(list(results.glob("*.json"))) if results.exists() else 0
    if n_files:
        shutil.make_archive(str(WORK / "results_bundle"), "gztar", results)
    print(f"finished after {(time.time() - START) / 3600:.2f} h; {n_files} result files; exit codes "
          f"{[p.exitcode for p in procs]}", flush=True)
    for f in sorted(WORK.glob("summary_*.json")):
        print(f.name, f.read_text(), flush=True)
    if any(p.exitcode for p in procs):
        raise SystemExit("a worker process failed; see the log above")


if __name__ == "__main__":
    main()

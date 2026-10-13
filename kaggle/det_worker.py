"""Kaggle kernel entry point for the detection study. One process per GPU; seeds are shared out round-robin.

Inputs (private datasets): zod-det-store (det_images.bin, det_meta.parquet, pretrained *.pth) and zodfl-code
(src/, config/, det_jobs.json = {"runs": [{"name", "seed", "params"}] or "seeds": [...], "split": "test"|"val",
"only": null|[tags], "params": {...}}).
Outputs: /kaggle/working/det_results/*.json (one file per model and seed), det_bundle.tar.gz, det_summary_*.json.
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
MAX_HOURS = float(os.environ.get("ZODFL_MAX_HOURS", "8.5"))
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")


def find_one(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"nothing under {INPUT} matches {pattern}; is the dataset attached?")
    return Path(hits[0])


def find_code() -> Path:
    unpacked = sorted(glob.glob(f"{INPUT}/**/src/zodfl/__init__.py", recursive=True))
    if unpacked:
        return Path(unpacked[0]).parents[2]
    archive = find_one(f"{INPUT}/**/zodfl_code.tar.gz")
    dest = WORK / "code"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")
    return dest


def locate() -> dict:
    meta = find_one(f"{INPUT}/**/det_meta.parquet")
    find_one(f"{INPUT}/**/det_images.bin")
    code = find_code()
    return {"store": meta.parent, "code": code, "weights": sorted(glob.glob(f"{INPUT}/**/*.pth", recursive=True)),
            "job": json.loads((code / "det_jobs.json").read_text())}


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=False)


def worker(i: int, n: int, paths: dict, deadline: float) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
    os.environ["ZODFL_PROVIDER"] = "kaggle"
    os.environ["TORCH_HOME"] = str(WORK / "torch_home")
    sys.path.insert(0, str(paths["code"] / "src"))
    import torch

    from zodfl.det_data import DetStore
    from zodfl.det_run import TAGS, result_name, run_det_seed, save_det_record
    from zodfl.jobs import hardware_info

    job = paths["job"]
    device = torch.device("cuda")
    print(f"[worker {i}] torch {torch.__version__}, gpu {torch.cuda.get_device_name(0)}", flush=True)
    results = WORK / "det_results"
    results.mkdir(exist_ok=True)
    store = DetStore(paths["store"])
    hw = hardware_info(device)
    done, failed, skipped = [], [], []
    runs = job.get("runs") or [{"name": None, "seed": sd, "params": job.get("params")} for sd in job["seeds"]]
    for run in runs[i::n]:
        seed, name = run["seed"], run.get("name")
        label = f"{name or 'base'}/seed{seed}"
        tags = tuple(job.get("only") or TAGS)
        todo = tuple(t for t in tags if not (results / result_name(t, seed, name)).exists())
        if not todo:
            skipped.append(label)
            continue
        if time.time() > deadline:
            skipped.append(f"{label} (deadline)")
            continue
        try:
            recs = run_det_seed(store, seed, device, run.get("params"), split=job.get("split", "test"),
                                log=lambda *a: print(f"[w{i} {label}]", *a, flush=True), only=todo,
                                ckpt_dir=WORK / "ckpt" / (name or "base"))
            for r in recs:
                save_det_record(r, results, hw, variant=name)
            done.append(label)
        except Exception as err:  # noqa: BLE001
            failed.append({"run": label, "error": repr(err)})
            print(f"[worker {i}] {label} FAILED: {err!r}", flush=True)
    (WORK / f"det_summary_{i}.json").write_text(json.dumps({"done": done, "failed": failed, "skipped": skipped}, indent=1))


def main() -> None:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv; free -h; nproc")
    paths = locate()
    print("job:", json.dumps(paths["job"]), flush=True)
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
    res = WORK / "det_results"
    n_files = len(list(res.glob("*.json"))) if res.exists() else 0
    if n_files:
        shutil.make_archive(str(WORK / "det_bundle"), "gztar", res)
    shutil.rmtree(WORK / "ckpt", ignore_errors=True)
    print(f"finished after {(time.time() - START) / 3600:.2f} h; {n_files} result files; exit codes "
          f"{[p.exitcode for p in procs]}", flush=True)
    for f in sorted(WORK.glob("det_summary_*.json")):
        print(f.name, f.read_text(), flush=True)
    if any(p.exitcode for p in procs):
        raise SystemExit("a worker process failed; see the log above")


if __name__ == "__main__":
    main()

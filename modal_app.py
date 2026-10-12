"""Modal entry points for zod-fssl. Heavy compute runs here; the laptop has no GPU.

    modal run modal_app.py::smoke           # GPU + imports
    modal run modal_app.py::index           # metadata -> /data/index.parquet (+ data_summary.json)
    modal run modal_app.py::prep            # shrink the 100k images into uint8 shards (2 parallel containers)
    modal run modal_app.py::pilot           # one short run end to end
    modal run --detach modal_app.py::grid   # stage 1 grid: config/grid.toml x seeds
    modal run modal_app.py::report          # results/*.json -> REPORT.md on the volume
"""
import json

import modal

app = modal.App("zod-fssl")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", "numpy", "pandas", "pyarrow", "scikit-learn",
                 "scipy", "pillow", "dataclass-wizard<1.0", "matplotlib")
    .env({"PYTHONPATH": "/root/src", "TORCH_HOME": "/data/torch_cache"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("config", "/root/config")
)

data = modal.Volume.from_name("zod-data", create_if_missing=True)
D = "/data"
CFG = "/root/config/default.toml"
GRID = "/root/config/grid.toml"


@app.function(image=image, gpu="T4", timeout=600, volumes={D: data})
def smoke():
    import torch
    import torchvision

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    from zodfl.model import SimSiamModel

    m = SimSiamModel(pretrained=True).cuda()
    x = torch.randint(0, 255, (8, 3, 144, 256), dtype=torch.uint8).cuda()
    from zodfl.augment import augment

    g = torch.Generator(device="cuda").manual_seed(0)
    print("pair loss", float(m.pair_loss(augment(x, g), augment(x, g)).mean()), "(pretrained weights loaded)")
    data.commit()


@app.function(image=image, volumes={D: data}, cpu=4, memory=16384, timeout=1800)
def make_index():
    import pandas as pd

    from zodfl import config as C
    from zodfl.index import build_index, summary

    data.reload()
    cfg = C.load(CFG)
    meta = pd.read_parquet(f"{D}/meta_raw.parquet")
    idx = build_index(meta, cfg.index, cfg.data)
    idx.to_parquet(cfg.data.index_path)
    s = summary(idx)
    import os

    os.makedirs(f"{D}/results", exist_ok=True)
    with open(f"{D}/results/data_summary.json", "w") as fh:
        json.dump(s, fh, indent=1)
    data.commit()
    print("index rows:", len(idx), "| splits:", s["split_counts"])
    print("stream by country:", s["stream_by_country"])
    print("frames by condition:", s["frames_by_cond"])
    print("probe_test by condition:", s["probe_test_by_cond"])
    return s


PARTS = [(0, 50000), (50000, 100000)]


@app.function(image=image, volumes={D: data}, cpu=16, memory=32768, timeout=6 * 3600)
def prep_part(part: int):
    import glob

    from zodfl import config as C
    from zodfl.prep import prep_tar

    data.reload()
    cfg = C.load(CFG)
    lo, hi = PARTS[part]
    hits = glob.glob(f"{D}/downloads/single_frames/images_blur_{lo:06d}_{hi - 1:06d}.tar.gz*")
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one archive for rows {lo}-{hi - 1}, found {hits}")
    out = prep_tar(hits[0], cfg.data.shards_dir, cfg.data.image_hw, cfg.data.shard_size, workers=14,
                   row_range=(lo, hi), max_corrupt_fraction=cfg.data.max_corrupt_fraction)
    data.commit()
    print(f"part {part}: decoded {out['n_decoded']}/{out['n_frames']}, bad rows: {out['bad_rows'][:10]}")
    return {k: out[k] for k in ("n_frames", "n_decoded", "bad_rows", "row_min", "row_max")}


@app.function(image=image, volumes={D: data}, cpu=2, memory=4096, timeout=900)
def finalize():
    from zodfl import config as C
    from zodfl.prep import write_manifest

    data.reload()
    cfg = C.load(CFG)
    man = write_manifest(cfg.data.shards_dir, cfg.data.image_hw, cfg.data.shard_size)
    data.commit()
    print("manifest:", {k: man[k] for k in ("n", "n_frames", "shard_size", "hw")}, "shards:", len(man["files"]),
          "bad rows:", len(man["bad_rows"]))
    return {k: man[k] for k in ("n", "n_frames")}


@app.function(image=image, gpu="T4", cpu=4, memory=32768, timeout=3 * 3600, volumes={D: data}, max_containers=6)
def run_job(spec: dict, seed: int, base_overrides: dict | None = None):
    import pandas as pd

    from zodfl import config as C
    from zodfl.federated import run, save_record
    from zodfl.grid import RunSpec, config_for
    from zodfl.store import ShardStore

    data.reload()
    cfg = C.load(CFG)
    if base_overrides:
        cfg = C.override(cfg, **base_overrides)
    rs = RunSpec(**spec)
    cfg = config_for(cfg, rs)
    index = pd.read_parquet(cfg.data.index_path)
    store = ShardStore(cfg.data.shards_dir).preload()
    rec = run(cfg, index, store, seed=seed, device="cuda", tag=rs.tag, centralized=rs.centralized, train=rs.train,
              log=print)
    path = save_record(rec, cfg.run.results_dir)
    data.commit()
    last = rec["evals"][-1]
    brief = {}
    for t, v in last["tasks"].items():
        g = v.get("by_cond") or v.get("by_light")
        if g:
            brief[t] = {"overall": round(g["overall_acc"], 4), "worst": round(g["worst_group_acc"], 4)}
    first = rec["evals"][0]
    return {"tag": rs.tag, "seed": seed, "path": str(path), "seconds": rec["totals"].get("seconds"), "brief": brief,
            "feature_std": [first["feature_std_normalized"], last["feature_std_normalized"]],
            "feature_erank": [round(first["feature_erank"], 2), round(last["feature_erank"], 2)]}


@app.function(image=image, gpu="T4", cpu=4, memory=32768, timeout=4 * 3600, volumes={D: data}, max_containers=3)
def stage2_job(seed: int):
    """Stage 2 (generation gap) for one seed: teacher A, then B variants, then C variants."""
    import pandas as pd

    from zodfl import config as C
    from zodfl.federated import save_record
    from zodfl.gen import run_stage2
    from zodfl.store import ShardStore

    data.reload()
    cfg = C.load(CFG)
    index = pd.read_parquet(cfg.data.index_path)
    store = ShardStore(cfg.data.shards_dir).preload()
    recs = run_stage2(cfg, index, store, seed=seed, device="cuda", log=print)
    for rec in recs:
        save_record(rec, cfg.run.results_dir)
    data.commit()
    out = []
    for r in recs:
        t = r["evals"][-1]["tasks"]
        first = next(iter(t))
        g = t[first].get("by_cond") or t[first].get("by_light")
        out.append({"tag": r["tag"], "task": first, "overall": round(g["overall_acc"], 4),
                    "worst": round(g["worst_group_acc"], 4), "seconds": round(r["totals"]["seconds"])})
    return {"seed": seed, "runs": out}


@app.function(image=image, gpu="T4", cpu=4, memory=32768, timeout=4 * 3600, volumes={D: data}, max_containers=3)
def reverse_job(seed: int):
    """Reverse transfer (phase 2, P6) for one seed."""
    import pandas as pd

    from zodfl import config as C
    from zodfl.federated import save_record
    from zodfl.gen import run_reverse
    from zodfl.store import ShardStore

    data.reload()
    cfg = C.load(CFG)
    index = pd.read_parquet(cfg.data.index_path)
    store = ShardStore(cfg.data.shards_dir).preload()
    recs = run_reverse(cfg, index, store, seed=seed, device="cuda", log=print)
    for rec in recs:
        save_record(rec, cfg.run.results_dir)
    data.commit()
    return {"seed": seed, "tags": [r["tag"] for r in recs]}


@app.function(image=image, volumes={D: data}, cpu=2, memory=2048, timeout=600)
def archive_results(prefix: str, dest: str):
    """Move results/<prefix>*.json to /data/<dest>/ so an abandoned attempt is kept but never reported."""
    import glob
    import os
    import shutil

    data.reload()
    os.makedirs(f"{D}/{dest}", exist_ok=True)
    moved = 0
    for f in glob.glob(f"{D}/results/{prefix}*.json"):
        shutil.move(f, f"{D}/{dest}/{os.path.basename(f)}")
        moved += 1
    data.commit()
    print(f"moved {moved} files to /{dest}")
    return moved


@app.function(image=image, gpu="T4", cpu=4, memory=32768, timeout=2 * 3600, volumes={D: data}, max_containers=4)
def s2_pilot_job(mult: float, seed: int):
    """Teacher A only, scored on the validation split (probe_train), for choosing the stage-2 recipe."""
    import pandas as pd

    from zodfl import config as C
    from zodfl.gen import pilot_teacher
    from zodfl.store import ShardStore

    data.reload()
    cfg = C.override(C.load(CFG), stage2={"backbone_lr_mult": mult})
    index = pd.read_parquet(cfg.data.index_path)
    store = ShardStore(cfg.data.shards_dir).preload()
    out = pilot_teacher(cfg, index, store, seed=seed, device="cuda", log=lambda *_: None)
    return {"mult": mult, "seed": seed, "score": round(out["score"], 4), "skipped": out["stats"]["skipped_steps"],
            "tasks": {t: {k: round(v, 4) for k, v in m.items()} for t, m in out["tasks"].items()}}


@app.function(image=image, volumes={D: data}, cpu=2, memory=8192, timeout=1800)
def render_report(base_tag: str = "fedavg_random20"):
    from zodfl.report import render

    data.reload()
    out = render(f"{D}/results", f"{D}/results/REPORT.md", base_tag=base_tag)
    data.commit()
    print(open(out).read()[:6000])


@app.local_entrypoint()
def index():
    make_index.remote()


@app.local_entrypoint()
def prep():
    for r in prep_part.map([0, 1]):
        print(r)
    print(finalize.remote())


@app.local_entrypoint()
def pilot(seed: int = 100, mults: str = "0.1", rounds: int = 6, kind: str = "random", budget: float = 0.2):
    """Short runs to check training health (loss, feature collapse) before the grid. One run per lr multiplier."""
    jobs = []
    for m in mults.split(","):
        spec = {"tag": f"pilot_m{m}", "overrides": {"selection": {"kind": kind, "budget": budget},
                                                    "federation": {"n_rounds": rounds, "backbone_lr_mult": float(m)},
                                                    "eval": {"eval_every": rounds, "bootstrap": 50}}}
        jobs.append((spec, seed))
    for res in run_job.starmap(jobs, order_outputs=False):
        print(json.dumps(res))


@app.local_entrypoint()
def grid(only: str = "", seeds: str = "", file: str = "config/grid.toml"):
    import sys

    sys.path.insert(0, "src")                      # the local launcher runs outside the Modal image
    from zodfl import config as C
    from zodfl.grid import load_grid

    base = C.load("config/default.toml")
    specs = load_grid(file)
    keep = set(only.split(",")) if only else None
    jobs = []
    for s in specs:
        if keep and s.tag not in keep:
            continue
        for seed in ([int(x) for x in seeds.split(',')] if seeds else (s.seeds or base.run.seeds)):
            jobs.append(({"tag": s.tag, "overrides": s.overrides, "train": s.train, "centralized": s.centralized},
                         seed))
    print(f"{len(jobs)} jobs")
    for res in run_job.starmap(jobs, order_outputs=False):
        print(json.dumps(res))


@app.local_entrypoint()
def report(base_tag: str = "fedavg_random20"):
    render_report.remote(base_tag)


@app.local_entrypoint()
def reverse(seeds: str = "0,1,2,3,4"):
    for res in reverse_job.map([int(x) for x in seeds.split(",")], order_outputs=False):
        print(json.dumps(res))


@app.local_entrypoint()
def archive(prefix: str, dest: str):
    print(archive_results.remote(prefix, dest))


@app.local_entrypoint()
def s2_pilot(mults: str = "1.0,0.3,0.1,0.03", seed: int = 100):
    for res in s2_pilot_job.starmap([(float(m), seed) for m in mults.split(",")], order_outputs=False):
        print(json.dumps(res))


@app.local_entrypoint()
def stage2(seeds: str = "0,1,2"):
    for res in stage2_job.map([int(x) for x in seeds.split(",")], order_outputs=False):
        print(json.dumps(res))

"""Detection generation gap: teacher (A, ResNet-18 Faster R-CNN, full camera) -> student (B, MobileNetV3 Faster R-CNN,
lower-resolution camera) on ZOD 2D boxes.

Per seed, five models are trained and scored on the same held-out test frames (all 12 countries, rare condition
groups over-represented):
  det_A_teacher     labels, generation-1 countries' frames
  det_B_oracle      labels, generation-2 countries' frames (what a labelled new fleet could do)
  det_B_floor       labels on `floor_fraction` of the generation-2 frames (the new fleet with almost no labels)
  det_B_kd_incar    no labels: the teacher's confident boxes on the generation-2 frames are the targets
  det_B_kd_floor    the floor's labelled frames plus the teacher's boxes on the rest (a mix)
Scores: COCO-style AP and AP50, overall and per condition group (road condition x light), with the worst group.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .det_data import DetStore
from .det_metrics import average_precision
from .det_models import build_detector, make_optimizer, n_params

TAGS = ("det_A_teacher", "det_B_oracle", "det_B_floor", "det_B_kd_incar", "det_B_kd_floor")
DEFAULTS = {
    "A": {"kind": "r18", "min_size": 540, "max_size": 960, "epochs": 6, "lr": 0.02, "batch": 4},
    "B": {"kind": "mnv3", "min_size": 360, "max_size": 640, "epochs": 6, "lr": 0.02, "batch": 4},
    "floor_fraction": 0.10, "pseudo_threshold": 0.5, "workers": 3, "max_skipped_fraction": 0.02,
    "clip": 10.0, "warmup": 100, "pretrained": True,
}


def merged(params: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (params or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


class FrameSet(Dataset):
    def __init__(self, store: DetStore, rows: list[int], boxes: dict[int, np.ndarray] | None = None, train: bool = False):
        self.store, self.rows, self.boxes, self.train = store, list(rows), boxes, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = torch.from_numpy(self.store.image(r).copy()).permute(2, 0, 1).float() / 255.0
        b = (self.boxes[r] if self.boxes is not None else self.store.boxes(r)).copy()
        if self.train and np.random.random() < 0.5:                      # horizontal flip
            img = img.flip(-1)
            w = img.shape[-1]
            if len(b):
                x1 = w - b[:, 2]
                x2 = w - b[:, 0]
                b[:, 0], b[:, 2] = x1, x2
        t = {"boxes": torch.from_numpy(b[:, :4]).float().reshape(-1, 4), "labels": torch.from_numpy(b[:, 4]).long()}
        return img, t, r


def _collate(batch):
    return tuple(zip(*batch))


def train_detector(model, store: DetStore, rows: list[int], boxes: dict[int, np.ndarray] | None, epochs: int, lr: float,
                   batch: int, device, seed: int, p: dict, log=print) -> dict:
    """SGD with warm-up and cosine decay, mixed precision, steps with a non-finite loss skipped (and counted)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(FrameSet(store, rows, boxes, train=True), batch_size=batch, shuffle=True, generator=g,
                    num_workers=p["workers"], collate_fn=_collate, drop_last=True, persistent_workers=p["workers"] > 0)
    model.to(device).train()
    opt = make_optimizer(model, lr)
    total = max(1, epochs * len(dl))
    warm = min(p["warmup"], total // 5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    steps = skipped = 0
    last = float("nan")
    t0 = time.time()
    for ep in range(epochs):
        for imgs, tgts, _ in dl:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = sum(model(imgs, tgts).values())
            steps += 1
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], p["clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            last = float(loss.detach())
        log(f"    epoch {ep + 1}/{epochs}  loss {last:.3f}  {time.time() - t0:.0f}s")
    if skipped > p["max_skipped_fraction"] * steps:
        raise RuntimeError(f"training unhealthy: {skipped} of {steps} steps had a non-finite loss")
    return {"steps": steps, "skipped": skipped, "final_loss": last, "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def predict(model, store: DetStore, rows: list[int], device, batch: int = 8, workers: int = 3,
            score_thresh: float = 0.05) -> list[dict]:
    model.to(device).eval()
    dl = DataLoader(FrameSet(store, rows), batch_size=batch, shuffle=False, num_workers=workers, collate_fn=_collate)
    out = []
    for imgs, _, _ in dl:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            res = model([i.to(device) for i in imgs])
        for r in res:
            keep = r["scores"] >= score_thresh
            out.append({"boxes": r["boxes"][keep].float().cpu().numpy(), "scores": r["scores"][keep].float().cpu().numpy(),
                        "labels": r["labels"][keep].cpu().numpy()})
    return out


def score(store: DetStore, rows: list[int], dets: list[dict]) -> dict:
    gt = [store.boxes(r) for r in rows]
    overall = average_precision(gt, dets)
    cond = store.meta.loc[rows, "cond"].to_numpy()
    groups = {}
    for c in sorted(set(cond)):
        ix = np.where(cond == c)[0]
        res = average_precision([gt[i] for i in ix], [dets[i] for i in ix])
        groups[c] = {"ap": res["ap"], "ap50": res["ap50"], "n": int(len(ix))}
    ok = [v["ap"] for v in groups.values() if not np.isnan(v["ap"])]
    return {"ap": overall["ap"], "ap50": overall["ap50"], "per_class": {str(k): v for k, v in overall["per_class"].items()},
            "groups": groups, "worst_group_ap": float(min(ok)) if ok else float("nan"),
            "mean_group_ap": float(np.mean(ok)) if ok else float("nan")}


def pseudo_boxes(dets: list[dict], rows: list[int], thresh: float) -> dict[int, np.ndarray]:
    out = {}
    for r, d in zip(rows, dets):
        k = d["scores"] >= thresh
        out[r] = np.concatenate([d["boxes"][k], d["labels"][k][:, None].astype(np.float32)], axis=1).astype(np.float32) \
            if k.any() else np.zeros((0, 5), dtype=np.float32)
    return out


def _rows(store: DetStore, role: str, gen: int | None = None) -> list[int]:
    m = store.meta
    sel = m[(m["role"] == role) & ((m["gen"] == gen) if gen is not None else True)]
    return sorted(int(r) for r in sel.index)


def run_det_seed(store: DetStore, seed: int, device, params: dict | None = None, split: str = "test",
                 log=print, only: tuple[str, ...] | None = None, ckpt_dir: str | Path | None = None) -> list[dict]:
    """Train and score the five models for one seed. `split` = 'test' (final) or 'val' (recipe choices)."""
    p = merged(params)
    rng = np.random.default_rng(seed)
    a_rows, b_rows = _rows(store, "train", 0), _rows(store, "train", 1)
    eval_rows = _rows(store, split)
    n_floor = max(1, int(round(p["floor_fraction"] * len(b_rows))))
    floor_rows = sorted(int(x) for x in rng.choice(b_rows, size=n_floor, replace=False))
    rest_rows = [r for r in b_rows if r not in set(floor_rows)]
    records: list[dict] = []

    def record(tag, model, stats, extra=None):
        t0 = time.time()
        dets = predict(model, store, eval_rows, device, workers=p["workers"])
        ev = score(store, eval_rows, dets)
        log(f"  {tag}: AP {ev['ap']:.3f}  AP50 {ev['ap50']:.3f}  worst-group AP {ev['worst_group_ap']:.3f}")
        records.append({"tag": tag, "seed": seed, "task": "detection", "split": split, "eval": ev, "train": stats,
                        "n_params": n_params(model), "eval_seconds": round(time.time() - t0, 1),
                        "extra": extra or {}, "params": p})

    def make(kind_key):
        s = p[kind_key]
        return build_detector(s["kind"], s["min_size"], s["max_size"], pretrained=p["pretrained"]), s

    def fit(model, s, rows, boxes=None, sd=0):
        return train_detector(model, store, rows, boxes, s["epochs"], s["lr"], s["batch"], device, seed * 100 + sd, p, log)

    want = lambda t: only is None or t in only  # noqa: E731
    log(f"seed {seed}: A frames {len(a_rows)}, B frames {len(b_rows)} (floor {len(floor_rows)}), eval frames {len(eval_rows)}")
    teacher, sa = make("A")
    ckpt = Path(ckpt_dir) / f"det_teacher_seed{seed}.pt" if ckpt_dir else None
    if ckpt and ckpt.exists():
        teacher.load_state_dict(torch.load(ckpt, map_location="cpu"))
        stats = {"loaded_from_checkpoint": True}
    else:
        stats = fit(teacher, sa, a_rows, sd=1)
        if ckpt:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(teacher.state_dict(), ckpt)
    if want("det_A_teacher"):
        record("det_A_teacher", teacher, stats)

    if want("det_B_oracle"):
        m, s = make("B")
        record("det_B_oracle", m, fit(m, s, b_rows, sd=2))
    if want("det_B_floor"):
        m, s = make("B")
        record("det_B_floor", m, fit(m, s, floor_rows, sd=3), {"n_labelled": len(floor_rows)})
    if want("det_B_kd_incar") or want("det_B_kd_floor"):
        dets = predict(teacher, store, b_rows, device, workers=p["workers"], score_thresh=p["pseudo_threshold"])
        pseudo = pseudo_boxes(dets, b_rows, p["pseudo_threshold"])
        n_true = sum(len(store.boxes(r)) for r in b_rows)
        n_ps = sum(len(v) for v in pseudo.values())
        info = {"pseudo_boxes": n_ps, "true_boxes_in_those_frames": n_true, "threshold": p["pseudo_threshold"]}
        if want("det_B_kd_incar"):
            m, s = make("B")
            record("det_B_kd_incar", m, fit(m, s, b_rows, pseudo, sd=4), info)
        if want("det_B_kd_floor"):
            fl = set(floor_rows)
            mixed = {r: (store.boxes(r) if r in fl else pseudo[r]) for r in b_rows}
            m, s = make("B")
            record("det_B_kd_floor", m, fit(m, s, b_rows, mixed, sd=5), {**info, "n_labelled": len(floor_rows)})
    return records


def result_name(tag: str, seed: int, variant: str | None = None) -> str:
    return f"{tag}{'@' + variant if variant else ''}__seed{seed}.json"


def save_det_record(rec: dict, results_dir: str | Path, hardware: dict | None = None, variant: str | None = None) -> Path:
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    if hardware:
        rec["hardware"] = hardware
    if variant:
        rec["variant"] = variant
    path = d / result_name(rec["tag"], rec["seed"], variant)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    tmp.replace(path)
    return path

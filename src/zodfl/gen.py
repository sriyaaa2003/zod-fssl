"""Stage 2, the generation gap: knowledge moves A -> B -> C across fleet generations that differ in
architecture (ResNet-18 -> MobileNetV3-small), sensor (resolution, field of view) and hardware (8-bit weights).

Nothing here shares images or weights across architectures. Knowledge crosses as predictions:
  * teacher-in-the-car: the old model runs on the new car's own frames and its predictions train the new model
  * public set: both models see a small shared frame set and only predictions are matched
"""
from __future__ import annotations

import copy
import dataclasses as dc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .augment import augment, to_input
from .config import Config, FederationCfg, Stage2Cfg
from .federated import build_world
from .metrics import group_summary
from .model import get_vector, set_vector, trainable_params

ARCHS = ("resnet18", "mobilenet_v3_small")


# ------------------------------------------------------------------ sensors and hardware
@dc.dataclass(frozen=True)
class Sensor:
    hw: tuple[int, int]
    fov: float = 1.0          # centre-crop fraction: smaller = narrower field of view
    bits: int = 0             # weight bits at deployment, 0 = float32


def sensors_from(s2: Stage2Cfg) -> list[Sensor]:
    return [Sensor(tuple(hw), fov, bits) for hw, fov, bits in zip(s2.sensor_hw, s2.sensor_fov, s2.quant_bits)]


def apply_sensor(x: torch.Tensor, s: Sensor) -> torch.Tensor:
    """x: normalized float [B,3,H,W] at the stored resolution -> what this generation's camera delivers."""
    _, _, h, w = x.shape
    if s.fov < 1.0:
        h2, w2 = max(1, round(h * s.fov)), max(1, round(w * s.fov))
        t, l = (h - h2) // 2, (w - w2) // 2
        x = x[:, :, t:t + h2, l:l + w2]
    if tuple(x.shape[-2:]) != tuple(s.hw):
        x = F.interpolate(x, size=s.hw, mode="bilinear", antialias=True, align_corners=False)
    return x


@torch.no_grad()
def fake_quantize(model: nn.Module, bits: int) -> nn.Module:
    """Deployment-time symmetric per-tensor weight quantization (simulated). Returns a copy; bits=0 returns the model."""
    if bits <= 0:
        return model
    q = copy.deepcopy(model)
    levels = 2 ** (bits - 1) - 1
    for m in q.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            scale = m.weight.abs().max().clamp_min(1e-12) / levels
            m.weight.copy_((m.weight / scale).round().clamp(-levels, levels) * scale)
    return q


# ------------------------------------------------------------------ model
class EnsembleTeacher(nn.Module):
    """Mean of several teachers' logits; each reads its own sensor view of the same augmented frame."""

    takes_base = True

    def __init__(self, models: list[nn.Module], sensors: list):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.sensors = list(sensors)

    def forward(self, base: torch.Tensor) -> dict[str, torch.Tensor]:
        outs = [m(apply_sensor(base, s)) for m, s in zip(self.models, self.sensors)]
        return {t: torch.stack([o[t] for o in outs]).mean(dim=0) for t in outs[0]}


class ClassifierNet(nn.Module):
    """Frozen early layers + frozen BatchNorm; trainable late blocks and one linear head per task."""

    def __init__(self, arch: str, n_classes: dict[str, int], pretrained: bool = True):
        super().__init__()
        if arch == "resnet18":
            r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.frozen = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
            self.body = nn.Sequential(r.layer3, r.layer4)
            feat = 512
        elif arch == "mobilenet_v3_small":
            w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            f = torchvision.models.mobilenet_v3_small(weights=w).features
            self.frozen, self.body = nn.Sequential(*f[:9]), nn.Sequential(*f[9:])
            feat = 576
        else:
            raise ValueError(f"unknown arch {arch!r}, expected one of {ARCHS}")
        self.arch = arch
        self.heads = nn.ModuleDict({t: nn.Linear(feat, n) for t, n in n_classes.items()})
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d,)):
                for p in m.parameters():
                    p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            h = self.frozen(x)
        h = self.body(h).mean(dim=(2, 3))
        return {t: head(h) for t, head in self.heads.items()}


# ------------------------------------------------------------------ tasks and labels
@dc.dataclass
class Tasks:
    specs: list[dict]
    classes: dict[str, list]

    @property
    def n_classes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.classes.items()}


def make_tasks(index: pd.DataFrame, specs: list[dict]) -> Tasks:
    if not specs:
        raise ValueError("no probe tasks configured")
    return Tasks(specs, {s["name"]: sorted(index[s["label_col"]].dropna().unique().tolist()) for s in specs})


def label_tables(index: pd.DataFrame, tasks: Tasks) -> dict[str, np.ndarray]:
    """task -> int label per store row (-1 where unknown)."""
    size = int(index["row"].max()) + 1
    out = {}
    for s in tasks.specs:
        m = {c: i for i, c in enumerate(tasks.classes[s["name"]])}
        arr = np.full(size, -1, dtype=np.int64)
        lab = index[s["label_col"]].map(m)
        ok = lab.notna().to_numpy()
        arr[index["row"].to_numpy()[ok]] = lab.to_numpy()[ok].astype(np.int64)
        out[s["name"]] = arr
    return out


# ------------------------------------------------------------------ losses
def sample_weights(cond: np.ndarray, q: np.ndarray, alpha: float, device) -> torch.Tensor | None:
    """(1/q(cond))^alpha scaled to mean 1 inside the batch; None when alpha = 0."""
    if alpha <= 0:
        return None
    w = torch.as_tensor(q[cond], dtype=torch.float32, device=device).clamp_min(1e-6) ** (-alpha)
    return w / w.mean()


def ce_loss(logits: dict[str, torch.Tensor], y: dict[str, torch.Tensor], w: torch.Tensor | None) -> torch.Tensor:
    total = 0.0
    for t, lg in logits.items():
        per = F.cross_entropy(lg, y[t].clamp_min(0), reduction="none") * (y[t] >= 0)
        total = total + ((per * w).mean() if w is not None else per.mean())
    return total


def kd_loss(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor], temp: float,
            w: torch.Tensor | None) -> torch.Tensor:
    total = 0.0
    for t, lg in student.items():
        per = F.kl_div(F.log_softmax(lg / temp, -1), F.softmax(teacher[t] / temp, -1), reduction="none").sum(-1)
        per = per * temp * temp
        total = total + ((per * w).mean() if w is not None else per.mean())
    return total


# ------------------------------------------------------------------ training
def safe_step(loss: torch.Tensor, opt: torch.optim.Optimizer, params: list, clip: float) -> bool:
    """One optimizer step that refuses non-finite losses/gradients. Returns False when the step was skipped."""
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return False
    loss.backward()
    if clip > 0:
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params, clip)):
            opt.zero_grad(set_to_none=True)
            return False
    opt.step()
    return True


def make_optimizer(model: ClassifierNet, s2: Stage2Cfg) -> torch.optim.Optimizer:
    body = [p for p in model.body.parameters() if p.requires_grad]
    heads = [p for p in model.heads.parameters() if p.requires_grad]
    return torch.optim.SGD([{"params": body, "lr": s2.lr * s2.backbone_lr_mult}, {"params": heads, "lr": s2.lr}],
                           momentum=s2.momentum, weight_decay=s2.weight_decay)


def _check_health(stats: dict, steps: int, skipped: int, s2: Stage2Cfg, what: str) -> None:
    stats["skipped_steps"] = skipped
    if steps + skipped and skipped / (steps + skipped) > s2.max_skipped_fraction:
        raise FloatingPointError(f"{what}: {skipped} of {steps + skipped} steps had a non-finite loss or gradient")


@dc.dataclass
class Ctx:
    cfg: Config
    store: object
    labels: dict[str, np.ndarray]
    cond_ids: np.ndarray
    q: np.ndarray
    device: torch.device
    tasks: Tasks
    index: pd.DataFrame


def _batch_loss(model, student_sensor, batch_rows, ctx: Ctx, mode: str, gen, teacher=None, teacher_sensor=None,
                alpha: float = 0.0):
    s2 = ctx.cfg.stage2
    x = ctx.store.get(batch_rows).to(ctx.device)
    base = augment(x, gen, scale_min=0.8, jitter=0.2)               # same augmented view feeds both models
    logits = model(apply_sensor(base, student_sensor))
    w = sample_weights(ctx.cond_ids[batch_rows], ctx.q, alpha, ctx.device)
    if mode == "labels":
        y = {t: torch.as_tensor(ctx.labels[t][batch_rows], device=ctx.device) for t in logits}
        return ce_loss(logits, y, w)
    with torch.no_grad():
        if getattr(teacher, "takes_base", False):
            tl = teacher(base)
        else:
            view = teacher_sensor if s2.teacher_sees == "old" else student_sensor
            tl = teacher(apply_sensor(base, view))
    return kd_loss(logits, tl, s2.temperature, w)


def train_generation(ctx: Ctx, model: ClassifierNet, sensor: Sensor, rows: dict[tuple[int, int], np.ndarray],
                     mode: str, seed: int, teacher=None, teacher_sensor=None, alpha: float = 0.0,
                     label_fraction: float = 1.0, log=print) -> dict:
    """Federated rounds (FedAvg weighted by frames used). mode: 'labels' or 'kd' (teacher-in-the-car)."""
    s2 = ctx.cfg.stage2
    dev = ctx.device
    rng = np.random.default_rng(seed)
    n_rounds = 1 + max(r for _, r in rows)
    clients = sorted({k for k, _ in rows})
    labeled: set[int] | None = None
    if mode == "labels" and label_fraction < 1.0:                     # fixed labelled subset, same for all rounds
        allrows = np.concatenate(list(rows.values()))
        k = max(1, int(round(label_fraction * len(allrows))))
        labeled = set(rng.choice(allrows, size=k, replace=False).tolist())
    gvec = get_vector(model).to(dev)
    used_total = steps_total = skipped_total = 0
    for r in range(n_rounds):
        deltas, ns = [], []
        for k in clients:
            rr = rows.get((k, r))
            if rr is None:
                continue
            if labeled is not None:
                rr = np.array([x for x in rr if x in labeled], dtype=np.int64)
            if len(rr) < 2:
                continue
            set_vector(model, gvec)
            model.train()
            opt = make_optimizer(model, s2)
            params = trainable_params(model)
            gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + r * 1_009 + k)
            for _ in range(s2.passes):
                order = rr[rng.permutation(len(rr))]
                for s in range(0, len(order), s2.batch):
                    b = order[s:s + s2.batch]
                    if len(b) < 2:
                        continue
                    loss = _batch_loss(model, sensor, b, ctx, mode, gen, teacher, teacher_sensor, alpha)
                    if safe_step(loss, opt, params, s2.grad_clip):
                        steps_total += 1
                    else:
                        skipped_total += 1
            deltas.append(get_vector(model) - gvec)
            ns.append(len(rr) * s2.passes)
            used_total += len(rr)
        if deltas:
            w = np.array(ns, dtype=np.float64) / np.sum(ns)
            for d, wk in zip(deltas, w):
                gvec = gvec + float(wk) * d
        log(f"    round {r + 1}/{n_rounds}: clients={len(deltas)}")
    if not torch.isfinite(gvec).all():
        raise FloatingPointError("federated training produced non-finite parameters")
    set_vector(model, gvec)
    stats = {"frames_trained_on": used_total, "sgd_steps": steps_total}
    _check_health(stats, steps_total, skipped_total, s2, f"{mode} training")
    return stats


def train_public(ctx: Ctx, model, sensor, public_rows: np.ndarray, teacher, teacher_sensor, seed: int) -> dict:
    """Server-side distillation on a small shared frame set; only predictions are matched."""
    s2, dev = ctx.cfg.stage2, ctx.device
    rng = np.random.default_rng(seed)
    model.train()
    opt = make_optimizer(model, s2)
    params = trainable_params(model)
    gen = torch.Generator(device=dev).manual_seed(seed)
    steps = skipped = 0
    for _ in range(s2.passes * 2):
        order = public_rows[rng.permutation(len(public_rows))]
        for s in range(0, len(order), s2.batch):
            b = order[s:s + s2.batch]
            if len(b) < 2:
                continue
            loss = _batch_loss(model, sensor, b, ctx, "kd", gen, teacher, teacher_sensor, 0.0)
            if safe_step(loss, opt, params, s2.grad_clip):
                steps += 1
            else:
                skipped += 1
    stats = {"frames_trained_on": len(public_rows), "sgd_steps": steps}
    _check_health(stats, steps, skipped, s2, "public distillation")
    return stats


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(ctx: Ctx, model: ClassifierNet, sensor: Sensor, rng: np.random.Generator,
             split: str = "probe_test") -> dict:
    ev = ctx.cfg.eval
    qmodel = fake_quantize(model, sensor.bits).eval()
    index = ctx.index
    te = index[index["split"] == split]
    rows = te["row"].to_numpy()
    logits: dict[str, list] = {t: [] for t in ctx.tasks.n_classes}
    for s in range(0, len(rows), 256):
        x = ctx.store.get(rows[s:s + 256]).to(ctx.device)
        out = qmodel(apply_sensor(to_input(x), sensor))
        for t, lg in out.items():
            logits[t].append(lg.argmax(-1).cpu().numpy())
    res: dict = {"tasks": {}, "n_test": len(te)}
    for spec in ctx.tasks.specs:
        t = spec["name"]
        pred = np.concatenate(logits[t])
        y = ctx.labels[t][rows]
        ok = y >= 0
        gcol = spec.get("group_col", "cond")
        r = {}
        for name, groups in (("by_" + gcol, te[gcol].to_numpy()), ("by_country", te["country"].to_numpy())):
            r[name] = group_summary(y[ok], pred[ok], groups[ok], ev.min_group_size, ev.bootstrap, rng)
        r["n_test"] = int(ok.sum())
        r["n_classes"] = ctx.tasks.n_classes[t]
        res["tasks"][t] = r
    return res


# ------------------------------------------------------------------ the experiment
VARIANTS_B = ("floor", "oracle", "kd_incar", "kd_incar_rarity", "kd_public")
VARIANTS_C = ("oracle", "kd_incar", "kd_incar_rarity")


def _record(ctx: Ctx, tag: str, seed: int, gen_idx: int, ev: dict, stats: dict, secs: float, extra: dict) -> dict:
    return {"tag": tag, "seed": seed, "stage": 2, "generation": gen_idx, "evals": [{"round": gen_idx, **ev}],
            "rounds": [], "totals": {**stats, "seconds": secs}, "config": dc.asdict(ctx.cfg), **extra}


def run_stage2(cfg: Config, index: pd.DataFrame, store, seed: int, device=None, log=print) -> list[dict]:
    s2 = cfg.stage2
    device = torch.device(device or cfg.run.device)
    rng = np.random.default_rng(seed)
    sensors = sensors_from(s2)
    tasks = make_tasks(index, cfg.eval.tasks)
    labels = label_tables(index, tasks)
    world_all = build_world(index, cfg.federation)                    # global condition ids + public frequencies q
    ctx = Ctx(cfg, store, labels, world_all.cond_ids, world_all.q, device, tasks, index)

    fgen = dc.replace(cfg.federation, n_rounds=s2.n_rounds)
    gen_rows = []
    for countries in s2.gen_countries:
        sub = index[(index["split"] == "stream") & index["country"].isin(countries)]
        gen_rows.append(build_world(pd.concat([sub, index[index["split"] != "stream"]]), fgen).rows)

    def new_model(g):
        torch.manual_seed(seed * 100 + g)
        return ClassifierNet(s2.archs[g], tasks.n_classes, s2.pretrained).to(device)

    records: list[dict] = []

    def finish(tag, g, model, stats, t0, extra):
        ev = evaluate(ctx, model, sensors[g], rng)
        records.append(_record(ctx, tag, seed, g, ev, stats, time.time() - t0, extra))
        t = ev["tasks"][tasks.specs[0]["name"]]
        log(f"  [{tag} seed={seed}] {tasks.specs[0]['name']}: overall={t['by_' + tasks.specs[0].get('group_col', 'cond')]['overall_acc']:.3f}")

    # generation 1: teacher A trained with labels (the fleet's existing perception stack provides them)
    t0 = time.time()
    A = new_model(0)
    stats = train_generation(ctx, A, sensors[0], gen_rows[0], "labels", seed, log=log)
    finish("s2_A_teacher", 0, A, stats, t0, {"variant": "teacher", "teacher": None})
    A.eval()

    # generation 2
    stream0 = index[(index["split"] == "stream") & index["country"].isin(s2.gen_countries[0])]["row"].to_numpy()
    public = rng.choice(stream0, size=min(s2.n_public, len(stream0)), replace=False)
    Bs: dict[str, ClassifierNet] = {}
    for v in VARIANTS_B:
        t0 = time.time()
        B = new_model(1)
        if v == "floor":
            stats = train_generation(ctx, B, sensors[1], gen_rows[1], "labels", seed, label_fraction=s2.floor_label_fraction, log=log)
        elif v == "oracle":
            stats = train_generation(ctx, B, sensors[1], gen_rows[1], "labels", seed, log=log)
        elif v == "kd_public":
            stats = train_public(ctx, B, sensors[1], public, A, sensors[0], seed)
        else:
            alpha = s2.rarity_alpha if v.endswith("rarity") else 0.0
            stats = train_generation(ctx, B, sensors[1], gen_rows[1], "kd", seed, teacher=A, teacher_sensor=sensors[0],
                                     alpha=alpha, log=log)
        Bs[v] = B.eval()
        finish(f"s2_B_{v}", 1, B, stats, t0, {"variant": v, "teacher": "s2_A_teacher" if v.startswith("kd") else None})

    # generation 3: the telephone game, each C learns from the matching B
    for v in VARIANTS_C:
        t0 = time.time()
        C = new_model(2)
        if v == "oracle":
            stats = train_generation(ctx, C, sensors[2], gen_rows[2], "labels", seed, log=log)
            teacher_tag = None
        else:
            alpha = s2.rarity_alpha if v.endswith("rarity") else 0.0
            stats = train_generation(ctx, C, sensors[2], gen_rows[2], "kd", seed, teacher=Bs[v], teacher_sensor=sensors[1],
                                     alpha=alpha, log=log)
            teacher_tag = f"s2_B_{v}"
        finish(f"s2_C_{v}", 2, C, stats, t0, {"variant": v, "teacher": teacher_tag})
    return records


def pilot_teacher(cfg: Config, index: pd.DataFrame, store, seed: int, device=None, log=print) -> dict:
    """Train only teacher A and score it on the VALIDATION split (probe_train), for choosing the training recipe
    without looking at probe_test."""
    s2 = cfg.stage2
    device = torch.device(device or cfg.run.device)
    rng = np.random.default_rng(seed)
    sensors = sensors_from(s2)
    tasks = make_tasks(index, cfg.eval.tasks)
    world_all = build_world(index, cfg.federation)
    ctx = Ctx(cfg, store, label_tables(index, tasks), world_all.cond_ids, world_all.q, device, tasks, index)
    fgen = dc.replace(cfg.federation, n_rounds=s2.n_rounds)
    sub_ = index[(index["split"] == "stream") & index["country"].isin(s2.gen_countries[0])]
    rows = build_world(pd.concat([sub_, index[index["split"] != "stream"]]), fgen).rows
    torch.manual_seed(seed * 100)
    A = ClassifierNet(s2.archs[0], tasks.n_classes, s2.pretrained).to(device)
    stats = train_generation(ctx, A, sensors[0], rows, "labels", seed, log=log)
    ev = evaluate(ctx, A, sensors[0], rng, split="probe_train")
    out = {"stats": stats, "tasks": {}}
    for spec in tasks.specs:
        g = ev["tasks"][spec["name"]]["by_" + spec.get("group_col", "cond")]
        out["tasks"][spec["name"]] = {"overall": g["overall_acc"], "mean_group": g["mean_group_acc"],
                                      "worst_group": g["worst_group_acc"]}
    out["score"] = float(np.mean([v["worst_group"] for v in out["tasks"].values()]))
    return out


# ------------------------------------------------------------------ reverse transfer (phase 2, P6)
REVERSE_VARIANTS = ("from_oracle", "from_ensemble", "from_control")


def run_reverse(cfg: Config, index: pd.DataFrame, store, seed: int, device=None, log=print) -> list[dict]:
    """Can the new generation's knowledge flow back to the old one?

    B_oracle holds new-market knowledge (true labels on generation-2 countries). The old model A is distilled from
    it on generation-1 cars' own frames:  A' = A + distillation from B_oracle (naive),  from the mean of A and
    B_oracle (anchored),  or from B_control, a B that only ever learned from A (so it holds no new information).
    Everything is scored on the same held-out frames; compare per country group (generation 1 = retention,
    generation 2 and 3 = the new markets)."""
    s2 = cfg.stage2
    device = torch.device(device or cfg.run.device)
    rng = np.random.default_rng(seed)
    sensors = sensors_from(s2)
    tasks = make_tasks(index, cfg.eval.tasks)
    world_all = build_world(index, cfg.federation)
    ctx = Ctx(cfg, store, label_tables(index, tasks), world_all.cond_ids, world_all.q, device, tasks, index)
    fgen = dc.replace(cfg.federation, n_rounds=s2.n_rounds)
    gen_rows = []
    for countries in s2.gen_countries:
        sub_ = index[(index["split"] == "stream") & index["country"].isin(countries)]
        gen_rows.append(build_world(pd.concat([sub_, index[index["split"] != "stream"]]), fgen).rows)

    def new_model(g):
        torch.manual_seed(seed * 100 + g)
        return ClassifierNet(s2.archs[g], tasks.n_classes, s2.pretrained).to(device)

    records: list[dict] = []

    def finish(tag, g, model, stats, t0, extra):
        ev = evaluate(ctx, model, sensors[g], rng)
        records.append(_record(ctx, tag, seed, g, ev, stats, time.time() - t0, extra))
        name = tasks.specs[0]["name"]
        log(f"  [{tag} seed={seed}] {name}: overall={ev['tasks'][name]['by_' + tasks.specs[0].get('group_col', 'cond')]['overall_acc']:.3f}")

    t0 = time.time()
    A = new_model(0)
    finish("rev_A_teacher", 0, A, train_generation(ctx, A, sensors[0], gen_rows[0], "labels", seed, log=log), t0,
           {"variant": "teacher", "teacher": None})
    A.eval()
    t0 = time.time()
    B_or = new_model(1)
    finish("rev_B_oracle", 1, B_or, train_generation(ctx, B_or, sensors[1], gen_rows[1], "labels", seed, log=log), t0,
           {"variant": "oracle", "teacher": None})
    B_or.eval()
    t0 = time.time()
    B_ctl = new_model(1)
    finish("rev_B_control", 1, B_ctl, train_generation(ctx, B_ctl, sensors[1], gen_rows[1], "kd", seed, teacher=A,
                                                       teacher_sensor=sensors[0], log=log), t0,
           {"variant": "control_source", "teacher": "rev_A_teacher"})
    B_ctl.eval()

    sources = {
        "from_oracle": (B_or, sensors[1], "rev_B_oracle"),
        "from_ensemble": (EnsembleTeacher([copy.deepcopy(A), B_or], [sensors[0], sensors[1]]).eval(), None,
                          "mean(rev_A_teacher, rev_B_oracle)"),
        "from_control": (B_ctl, sensors[1], "rev_B_control"),
    }
    for v in REVERSE_VARIANTS:
        t0 = time.time()
        teacher, tsensor, src = sources[v]
        A2 = copy.deepcopy(A)
        stats = train_generation(ctx, A2, sensors[0], gen_rows[0], "kd", seed, teacher=teacher, teacher_sensor=tsensor,
                                 log=log)
        finish(f"rev_A_{v}", 0, A2.eval(), stats, t0, {"variant": v, "teacher": src})
    return records

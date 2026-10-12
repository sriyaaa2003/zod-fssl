"""One-off patch (2026-10-12): stage-2 training hygiene after the first attempt produced a diverged teacher.
Gradient clipping, a guard against non-finite losses, a smaller learning rate for the pretrained body,
and scoring on the validation split for recipe selection. Kept in the repo as a record of the change."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sub(path: str, old: str, new: str, count: int = 1) -> None:
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    if old not in s:
        raise SystemExit(f"pattern not found in {path}: {old[:70]!r}")
    p.write_text(s.replace(old, new, count), encoding="utf-8", newline="\n")


# ---- config
sub("src/zodfl/config.py",
    "    lr: float = 0.01\n    momentum: float = 0.9\n    weight_decay: float = 1e-4\n    temperature",
    "    lr: float = 0.01\n    backbone_lr_mult: float = 0.1       # pretrained body learns at lr * this, heads at lr\n"
    "    grad_clip: float = 5.0              # max gradient norm; 0 = off\n"
    "    max_skipped_fraction: float = 0.02  # fail if more steps than this were skipped for non-finite loss\n"
    "    momentum: float = 0.9\n    weight_decay: float = 1e-4\n    temperature")
sub("config/default.toml",
    "lr = 0.01\nmomentum = 0.9\nweight_decay = 1e-4\ntemperature = 2.0",
    "lr = 0.01\nbackbone_lr_mult = 0.1                       # pretrained body learns at lr * this (BLUEPRINT ADR-16)\n"
    "grad_clip = 5.0\nmax_skipped_fraction = 0.02\nmomentum = 0.9\nweight_decay = 1e-4\ntemperature = 2.0")

# ---- gen.py
G = "src/zodfl/gen.py"
sub(G, "# ------------------------------------------------------------------ training\n@dc.dataclass\nclass Ctx:",
    '''# ------------------------------------------------------------------ training
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
class Ctx:''')

sub(G, '''            opt = torch.optim.SGD(trainable_params(model), lr=s2.lr, momentum=s2.momentum,
                                  weight_decay=s2.weight_decay)
            gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + r * 1_009 + k)''',
    '''            opt = make_optimizer(model, s2)
            params = trainable_params(model)
            gen = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + r * 1_009 + k)''')
sub(G, '''                    loss = _batch_loss(model, sensor, b, ctx, mode, gen, teacher, teacher_sensor, alpha)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    steps_total += 1''',
    '''                    loss = _batch_loss(model, sensor, b, ctx, mode, gen, teacher, teacher_sensor, alpha)
                    if safe_step(loss, opt, params, s2.grad_clip):
                        steps_total += 1
                    else:
                        skipped_total += 1''')
sub(G, "    used_total = steps_total = 0\n    for r in range(n_rounds):",
    "    used_total = steps_total = skipped_total = 0\n    for r in range(n_rounds):")
sub(G, '''    set_vector(model, gvec)
    return {"frames_trained_on": used_total, "sgd_steps": steps_total}''',
    '''    if not torch.isfinite(gvec).all():
        raise FloatingPointError("federated training produced non-finite parameters")
    set_vector(model, gvec)
    stats = {"frames_trained_on": used_total, "sgd_steps": steps_total}
    _check_health(stats, steps_total, skipped_total, s2, f"{mode} training")
    return stats''')
sub(G, '''    opt = torch.optim.SGD(trainable_params(model), lr=s2.lr, momentum=s2.momentum, weight_decay=s2.weight_decay)
    gen = torch.Generator(device=dev).manual_seed(seed)
    steps = 0''',
    '''    opt = make_optimizer(model, s2)
    params = trainable_params(model)
    gen = torch.Generator(device=dev).manual_seed(seed)
    steps = skipped = 0''')
sub(G, '''            loss = _batch_loss(model, sensor, b, ctx, "kd", gen, teacher, teacher_sensor, 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
    return {"frames_trained_on": len(public_rows), "sgd_steps": steps}''',
    '''            loss = _batch_loss(model, sensor, b, ctx, "kd", gen, teacher, teacher_sensor, 0.0)
            if safe_step(loss, opt, params, s2.grad_clip):
                steps += 1
            else:
                skipped += 1
    stats = {"frames_trained_on": len(public_rows), "sgd_steps": steps}
    _check_health(stats, steps, skipped, s2, "public distillation")
    return stats''')
sub(G, "def evaluate(ctx: Ctx, model: ClassifierNet, sensor: Sensor, rng: np.random.Generator) -> dict:",
    'def evaluate(ctx: Ctx, model: ClassifierNet, sensor: Sensor, rng: np.random.Generator,\n             split: str = "probe_test") -> dict:')
sub(G, '    te = index[index["split"] == "probe_test"]\n    rows = te["row"].to_numpy()\n    logits: dict',
    '    te = index[index["split"] == split]\n    rows = te["row"].to_numpy()\n    logits: dict')

p = ROOT / G
s = p.read_text(encoding="utf-8")
s += '''

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
'''
p.write_text(s, encoding="utf-8", newline="\n")
print("stage-2 fix applied")

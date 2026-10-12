"""Federated simulation: clients = groups of the index, rounds = slices of the timeline."""
from __future__ import annotations

import dataclasses as dc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import aggregate
from .client import client_round
from .config import Config, FederationCfg
from .model import SimSiamModel, get_vector, set_vector, trainable_params
from .probe import run_probes
from .selectors import Selector


@dc.dataclass
class World:
    clients: list[str]
    rows: dict[tuple[int, int], np.ndarray]     # (client index, round) -> store rows in arrival order
    cond_ids: np.ndarray                         # store row -> condition id (-1 where unknown)
    cond_names: list[str]
    q: np.ndarray                                # public condition frequencies over the stream


def build_world(index: pd.DataFrame, fcfg: FederationCfg, centralized: bool = False) -> World:
    """Only `split == "stream"` frames ever reach a client; probe frames never do."""
    stream = index[index["split"] == "stream"].copy()
    key = fcfg.client_key
    sizes = stream.groupby(key).size()
    keep = sizes[sizes >= fcfg.min_client_frames].index
    stream = stream[stream[key].isin(keep)]
    if stream.empty:
        raise ValueError("no client has min_client_frames stream frames")
    if centralized:
        stream[key] = "central"
    cond_names = sorted(index["cond"].astype(str).unique())
    cmap = {c: i for i, c in enumerate(cond_names)}
    cond_ids = np.full(int(index["row"].max()) + 1, -1, dtype=np.int64)
    cond_ids[index["row"].to_numpy()] = index["cond"].astype(str).map(cmap).to_numpy()

    # equal numbers of stream frames per round (the data is clustered in time, equal-width slices would be empty)
    edges = np.quantile(stream["ts"].to_numpy(), np.linspace(0, 1, fcfg.n_rounds + 1))
    stream["round"] = np.clip(np.searchsorted(edges, stream["ts"].to_numpy(), side="right") - 1, 0, fcfg.n_rounds - 1)
    clients = sorted(stream[key].astype(str).unique())
    cidx = {c: i for i, c in enumerate(clients)}
    rows: dict[tuple[int, int], np.ndarray] = {}
    for (c, r), g in stream.sort_values("ts").groupby([stream[key].astype(str), "round"]):
        rows[(cidx[c], int(r))] = g["row"].to_numpy()
    counts = np.bincount(cond_ids[stream["row"].to_numpy()], minlength=len(cond_names)).astype(float)
    return World(clients, rows, cond_ids, cond_names, counts / counts.sum())


def eval_rounds(n_rounds: int, every: int) -> list[int]:
    return sorted({0, n_rounds} | {r for r in range(every, n_rounds + 1, every)})


def run(cfg: Config, index: pd.DataFrame, store, seed: int, device=None, tag: str = "", centralized: bool = False,
        train: bool = True, log=print) -> dict:
    device = torch.device(device or cfg.run.device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fcfg, scfg = cfg.federation, cfg.selection
    model = SimSiamModel(pretrained=fcfg.pretrained, train_layers=tuple(fcfg.train_layers)).to(device)
    gvec = get_vector(model).to(device)
    gvec0 = gvec.clone()                                    # the shipped model (for the anchor and for interpolation)
    anchor = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad and n.startswith(("layer3", "layer4"))}
    layer_sizes = [p.numel() for p in trainable_params(model)]
    kind = cfg.aggregation.kind
    ema = aggregate.FedEMA(cfg.aggregation.tau) if kind == "fedema" else None
    world = build_world(index, fcfg, centralized)
    n_params = int(gvec.numel())
    checkpoints = set(eval_rounds(fcfg.n_rounds, cfg.eval.eval_every))
    rec = {"tag": tag, "seed": seed, "config": dc.asdict(cfg), "n_params": n_params,
           "clients": world.clients, "cond_names": world.cond_names, "rounds": [], "evals": [], "totals": {}}
    seen_total = kept_total = steps_total = 0
    bytes_total = 0
    t_start = time.time()

    for r in range(fcfg.n_rounds + 1):
        if r in checkpoints:
            set_vector(model, gvec)
            ev = run_probes(model, store, index, cfg, device, rng)
            ev["round"] = r
            rec["evals"].append(ev)
        if r == fcfg.n_rounds or not train:
            if not train:
                break
            continue
        done = []
        inits: dict[int, torch.Tensor] = {}
        for k in range(len(world.clients)):
            rows = world.rows.get((k, r))
            if rows is None or len(rows) == 0:
                continue
            gen = torch.Generator(device=device)
            gen.manual_seed(seed * 1_000_003 + r * 1_009 + k)
            sel = Selector(scfg.kind, scfg.budget, device, q=world.q, rarity_floor=fcfg.rarity_floor,
                           novelty_ema=scfg.novelty_ema)
            init_vec = gvec if ema is None else ema.init_for(k, gvec)[0]
            inits[k] = init_vec
            res = client_round(model, init_vec, rows, store, world.cond_ids, len(world.cond_names), sel, fcfg, device,
                               gen, anchor=anchor)
            seen_total += res.n_seen
            kept_total += res.n_kept
            steps_total += res.n_steps
            done.append((k, res))
        trained = [(k, res) for k, res in done if res.delta is not None]
        entry = {"round": r, "n_active": len(done), "n_trained": len(trained),
                 "n_seen": sum(res.n_seen for _, res in done), "n_kept": sum(res.n_kept for _, res in done)}
        if trained:
            n_kept = np.array([res.n_kept for _, res in trained])
            counts = np.stack([res.cond_counts for _, res in trained])
            deltas = [res.delta for _, res in trained]
            if kind in ("fedavg", "balanced"):
                w = aggregate.weights(kind, n_kept, counts)
                gvec = aggregate.apply(gvec, deltas, w)
            elif kind in ("ldawa", "ldawa_norm"):
                w = aggregate.weights_fedavg(n_kept)
                gvec = aggregate.apply_ldawa(gvec, deltas, n_kept, layer_sizes, normalize=(kind == "ldawa_norm"))
            else:                                           # fedema: average the clients' trained models
                w = aggregate.weights_fedavg(n_kept)
                trained_vecs = {k: inits[k] + res.delta for k, res in trained}
                new = sum(float(wk) * trained_vecs[k] for (k, _), wk in zip(trained, w))
                ema.after_round(trained_vecs, new)
                gvec = new
            bytes_total += len(trained) * n_params * 4 * 2
            losses = [x for _, res in trained for x in res.losses]
            entry.update({
                "mean_loss": float(np.mean(losses)) if losses else None,
                "weights": {world.clients[k]: float(wk) for (k, _), wk in zip(trained, w)},
                "kept_by_cond": {world.cond_names[c]: int(v) for c, v in enumerate(counts.sum(axis=0)) if v},
                "max_buffer_seen": max(res.max_buffer_seen for _, res in trained),
            })
        rec["rounds"].append(entry)
        log(f"[{tag} seed={seed}] round {r + 1}/{fcfg.n_rounds}: active={len(done)} kept={entry['n_kept']}"
            f"/{entry['n_seen']} loss={entry.get('mean_loss')}")
    if train and cfg.eval.interp_alphas:                    # shipped-to-trained weight interpolation (WiSE-FT style)
        rec["interp"] = {}
        for a in cfg.eval.interp_alphas:
            set_vector(model, (1 - a) * gvec0 + a * gvec)
            ev = run_probes(model, store, index, cfg, device, rng)
            ev["round"] = fcfg.n_rounds
            ev["alpha"] = a
            rec["interp"][str(a)] = ev
    rec["totals"] = {"frames_seen": seen_total, "frames_trained_on": kept_total, "sgd_steps": steps_total,
                     "bytes_communicated": bytes_total, "seconds": time.time() - t_start}
    return rec


def save_record(rec: dict, results_dir: str | Path) -> Path:
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rec['tag']}__seed{rec['seed']}.json"
    main = {k: v for k, v in rec.items() if k != "interp"}
    path.write_text(json.dumps(main, indent=1, default=float))
    for a, ev in (rec.get("interp") or {}).items():         # one derived record per interpolation weight
        derived = {**main, "tag": f"{rec['tag']}+interp{a}", "evals": [rec["evals"][0], ev], "rounds": [],
                   "derived_from": rec["tag"]}
        (d / f"{derived['tag']}__seed{rec['seed']}.json").write_text(json.dumps(derived, indent=1, default=float))
    return path

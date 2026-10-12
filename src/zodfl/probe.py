"""Frozen-feature probes on held-out frames (never part of any client stream)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .augment import to_input
from .config import Config
from .metrics import group_summary


def effective_rank(f: np.ndarray, max_rows: int = 4000) -> float:
    """exp(entropy of the normalized squared singular values): near 1 = collapsed, large = rich features.
    Label-free, so it can monitor training health without touching any test result."""
    f = np.asarray(f[:max_rows], dtype=np.float64)
    f = f - f.mean(axis=0, keepdims=True)
    s2 = np.linalg.svd(f, compute_uv=False) ** 2
    if s2.sum() <= 0:
        return 1.0
    p = s2 / s2.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def extract_features(model, store, rows: np.ndarray, device, batch: int = 256) -> np.ndarray:
    was = model.training
    model.eval()
    out = []
    for s in range(0, len(rows), batch):
        x = store.get(rows[s:s + batch]).to(device)
        if str(device).startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                f = model.embed(to_input(x))
        else:
            f = model.embed(to_input(x))
        out.append(f.float().cpu().numpy())
    model.train(was)
    return np.concatenate(out) if out else np.zeros((0, model.feat_dim), dtype=np.float32)


def run_probes(model, store, index: pd.DataFrame, cfg: Config, device, rng: np.random.Generator) -> dict:
    """index must hold columns: row, split, cond, country and the label/group columns named by cfg.eval.tasks."""
    tr = index[index["split"] == "probe_train"]
    te = index[index["split"] == "probe_test"]
    ftr = extract_features(model, store, tr["row"].to_numpy(), device)
    fte = extract_features(model, store, te["row"].to_numpy(), device)
    std = float(np.mean(np.std(fte / (np.linalg.norm(fte, axis=1, keepdims=True) + 1e-8), axis=0)))
    results: dict = {"feature_std_normalized": std, "feature_erank": effective_rank(fte), "n_probe_train": len(tr), "n_probe_test": len(te), "tasks": {}}
    scaler = StandardScaler().fit(ftr)
    xtr, xte = scaler.transform(ftr), scaler.transform(fte)
    for task in cfg.eval.tasks:
        col, gcol = task["label_col"], task.get("group_col", "cond")
        ok_tr, ok_te = tr[col].notna().to_numpy(), te[col].notna().to_numpy()
        ytr_all, yte = tr[col].to_numpy(), te.loc[ok_te, col].to_numpy()
        for frac in cfg.eval.probe_fractions:
            name = task["name"] if frac >= 1 else f'{task["name"]}@{round(frac * 100)}pct'
            fit_idx = np.flatnonzero(ok_tr)
            if frac < 1:                                   # fixed subset (probe_seed): identical across runs
                sub_rng = np.random.default_rng(cfg.data.probe_seed)
                fit_idx = np.sort(sub_rng.choice(fit_idx, size=max(10, int(frac * len(fit_idx))), replace=False))
            y_fit = ytr_all[fit_idx]
            seen = set(np.unique(y_fit))
            keep = np.array([v in seen for v in yte])
            if len(seen) < 2 or keep.sum() == 0:
                results["tasks"][name] = {"skipped": "fewer than 2 classes in the probe training set or no usable test rows"}
                continue
            clf = LogisticRegression(max_iter=500, C=cfg.eval.probe_c)
            clf.fit(xtr[fit_idx], y_fit)
            pred = clf.predict(xte[ok_te][keep])
            res = {}
            for gname, gseries in (("by_" + gcol, te.loc[ok_te, gcol].to_numpy()[keep]),
                                   ("by_country", te.loc[ok_te, "country"].to_numpy()[keep])):
                res[gname] = group_summary(yte[keep], pred, gseries, cfg.eval.min_group_size, cfg.eval.bootstrap, rng)
            res["n_test"] = int(keep.sum())
            res["n_classes"] = int(len(seen))
            res["n_fit"] = int(len(fit_idx))
            results["tasks"][name] = res
    return results

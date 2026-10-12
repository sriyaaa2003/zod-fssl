"""Group metrics with bootstrap intervals. A group smaller than min_n is reported as n/a, never imputed."""
from __future__ import annotations

import numpy as np


def group_accuracy(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray) -> dict[str, tuple[float, int]]:
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[str(g)] = (float((y[m] == yhat[m]).mean()), int(m.sum()))
    return out


def _summ(acc: dict[str, tuple[float, int]], min_n: int) -> tuple[float, float, str | None]:
    ok = {g: a for g, (a, n) in acc.items() if n >= min_n}
    if not ok:
        return float("nan"), float("nan"), None
    worst = min(ok, key=ok.get)
    return float(np.mean(list(ok.values()))), float(ok[worst]), worst


def group_summary(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray, min_n: int, n_boot: int,
                  rng: np.random.Generator) -> dict:
    """Overall accuracy, mean over groups, worst group (groups with >= min_n test frames only),
    per-group table, and percentile 95% bootstrap intervals for mean-group and worst-group."""
    y, yhat, groups = np.asarray(y), np.asarray(yhat), np.asarray(groups)
    acc = group_accuracy(y, yhat, groups)
    mean_g, worst_g, worst_name = _summ(acc, min_n)
    res = {
        "overall_acc": float((y == yhat).mean()),
        "mean_group_acc": mean_g,
        "worst_group_acc": worst_g,
        "worst_group": worst_name,
        "groups": {g: {"acc": a, "n": n} for g, (a, n) in acc.items()},
        "n_groups_eval": int(sum(n >= min_n for _, n in acc.values())),
        "min_n": min_n,
    }
    if n_boot > 0 and res["n_groups_eval"] > 0:
        idx_by_g = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
        means, worsts = [], []
        for _ in range(n_boot):
            b = {}
            for g, idx in idx_by_g.items():
                s = rng.choice(idx, size=len(idx), replace=True)
                b[str(g)] = (float((y[s] == yhat[s]).mean()), len(idx))
            m_, w_, _ = _summ(b, min_n)
            means.append(m_)
            worsts.append(w_)
        res["mean_group_ci95"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        res["worst_group_ci95"] = [float(np.percentile(worsts, 2.5)), float(np.percentile(worsts, 97.5))]
    return res


def forgetting(history: list[dict[str, float]]) -> dict[str, float]:
    """history = per-checkpoint {group: accuracy} in time order.
    Forgetting of a group = best accuracy at any earlier checkpoint minus the final one (>= 0 means lost)."""
    if len(history) < 2:
        return {}
    last = history[-1]
    out = {}
    for g, final in last.items():
        earlier = [h[g] for h in history[:-1] if g in h]
        if earlier:
            out[g] = float(max(earlier) - final)
    return out

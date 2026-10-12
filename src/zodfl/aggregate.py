"""Server-side aggregation of client deltas."""
from __future__ import annotations

import numpy as np
import torch


def weights_fedavg(n_kept: np.ndarray) -> np.ndarray:
    n = np.asarray(n_kept, dtype=np.float64)
    if n.sum() <= 0:
        raise ValueError("no client trained this round")
    return n / n.sum()


def weights_balanced(counts: np.ndarray) -> np.ndarray:
    """Condition-balanced weights: every condition gets equal total weight across clients.

    counts[k, c] = frames of condition c that client k trained on this round. A client's weight is
    sum_c counts[k, c] / N_c over conditions that appeared at all (N_c > 0), then normalized.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=0)
    present = total > 0
    if not present.any():
        raise ValueError("no client trained this round")
    w = (counts[:, present] / total[present]).sum(axis=1)
    return w / w.sum()


def weights(kind: str, n_kept: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if kind == "fedavg":
        return weights_fedavg(n_kept)
    if kind == "balanced":
        return weights_balanced(counts)
    raise ValueError(f"unknown aggregation {kind!r}")


def apply(global_vec: torch.Tensor, deltas: list[torch.Tensor], w: np.ndarray) -> torch.Tensor:
    if len(deltas) != len(w):
        raise ValueError("one weight per delta required")
    out = global_vec.clone()
    for d, wk in zip(deltas, w):
        out += float(wk) * d.to(out.device)
    return out


def apply_ldawa(global_vec: torch.Tensor, deltas: list[torch.Tensor], n_kept: np.ndarray, layer_sizes: list[int],
                normalize: bool = False) -> torch.Tensor:
    """L-DAWA (Rehman et al., ICCV 2023, their Eq. 5 with FedAvg sample weights), per layer l:
        w_g[l] <- sum_k (n_k / n) * cos(w_g[l], w_k[l]) * w_k[l]
    `normalize=False` is the formula as printed. Two cases the printed formula does not cover are handled as follows:
      * a layer whose global weights are all (near) zero has no direction, so its cosine is taken as 1
        (otherwise zero-initialised parameters such as BatchNorm biases could never learn);
      * as printed, a client with a negative cosine contributes with a negative weight (a reflection).
    `normalize=True` is our safer variant, not the paper's: negative cosines are clamped to 0 and the weights are
    divided by their sum (a convex combination), falling back to plain FedAvg for the layer if all weights vanish."""
    if sum(layer_sizes) != global_vec.numel():
        raise ValueError("layer sizes must cover the whole parameter vector")
    p = weights_fedavg(n_kept)
    out = torch.empty_like(global_vec)
    start = 0
    for size in layer_sizes:
        g = global_vec[start:start + size]
        gn = float(g.norm())
        acc, wsum, fed = torch.zeros_like(g), 0.0, torch.zeros_like(g)
        for d, pk in zip(deltas, p):
            wk = g + d[start:start + size].to(g.device)
            cos = 1.0 if gn < 1e-8 else float((g @ wk) / (gn * wk.norm().clamp_min(1e-12)))
            if normalize:
                cos = max(cos, 0.0)
            acc += float(pk) * cos * wk
            wsum += float(pk) * cos
            fed += float(pk) * wk
        if normalize:
            out[start:start + size] = acc / wsum if wsum > 1e-8 else fed
        else:
            out[start:start + size] = acc
        start += size
    return out


class FedEMA:
    """FedEMA-style client state (Zhuang et al., ICLR 2022), applied to our SimSiam model.

    A client that trained in the previous round starts from  mu * (its own previous model) + (1 - mu) * (global),
    with mu = min(lambda_k * ||global - local||, 1) and lambda_k = tau / ||global_new - local|| fixed the first time
    the client is sampled. Other clients start from the global model. The server then averages the trained models
    with FedAvg weights."""

    def __init__(self, tau: float = 0.7):
        self.tau = tau
        self.local: dict[int, torch.Tensor] = {}
        self.lam: dict[int, float] = {}
        self.prev: set[int] = set()

    def init_for(self, k: int, global_vec: torch.Tensor) -> tuple[torch.Tensor, float | None]:
        if k in self.prev and k in self.lam:
            mu = min(self.lam[k] * float((global_vec - self.local[k]).norm()), 1.0)
            return mu * self.local[k] + (1 - mu) * global_vec, mu
        return global_vec, None

    def after_round(self, trained: dict[int, torch.Tensor], new_global: torch.Tensor) -> None:
        for k, vec in trained.items():
            if k not in self.lam:
                self.lam[k] = self.tau / max(float((new_global - vec).norm()), 1e-12)
            self.local[k] = vec.clone()
        self.prev = set(trained)

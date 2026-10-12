"""Single-pass frame selection: choose k of the n frames that just arrived. Nothing is stored."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .augment import augment, to_input

KINDS = ("all", "random", "loss", "novelty", "rarity")


def n_to_keep(kind: str, n: int, budget: float) -> int:
    if kind == "all":
        return n
    return int(min(n, max(1, round(budget * n))))


class Selector:
    """One instance per client per round (stateless across rounds, see BLUEPRINT section 6)."""

    def __init__(self, kind: str, budget: float, device, q: np.ndarray | None = None,
                 rarity_floor: float = 1e-4, novelty_ema: float = 0.99):
        if kind not in KINDS:
            raise ValueError(f"unknown selector {kind!r}")
        if kind == "rarity" and q is None:
            raise ValueError("rarity selection needs the global condition frequencies q")
        self.kind, self.budget, self.device = kind, budget, device
        self.q = None if q is None else torch.as_tensor(q, dtype=torch.float32)
        self.floor, self.ema = rarity_floor, novelty_ema
        self._mean: torch.Tensor | None = None

    @torch.no_grad()
    def select(self, model, x_uint8: torch.Tensor, cond_ids: np.ndarray, gen: torch.Generator) -> torch.Tensor:
        """Return sorted indices (LongTensor) of the kept frames within this arrival batch."""
        n = x_uint8.shape[0]
        k = n_to_keep(self.kind, n, self.budget)
        if self.kind == "all" or k >= n:
            return torch.arange(n)
        if self.kind == "random":
            score = torch.rand(n, generator=gen, device=self.device)
        elif self.kind == "loss":
            was_training = model.training
            model.eval()
            a1, a2 = augment(x_uint8, gen), augment(x_uint8, gen)
            score = model.pair_loss(a1, a2)            # high loss = surprising
            model.train(was_training)
        elif self.kind == "novelty":
            was_training = model.training
            model.eval()
            f = F.normalize(model.embed(to_input(x_uint8)), dim=-1)
            model.train(was_training)
            if self._mean is None:
                self._mean = F.normalize(f.mean(dim=0), dim=0)
                score = 1 - f @ self._mean              # first batch: distance to its own mean
            else:
                score = 1 - f @ self._mean
                self._mean = F.normalize(self.ema * self._mean + (1 - self.ema) * f.mean(dim=0), dim=0)
        else:  # rarity: Gumbel top-k with weights 1/q(cond)
            q = self.q.to(self.device)[torch.as_tensor(cond_ids, device=self.device)].clamp_min(self.floor)
            u = torch.rand(n, generator=gen, device=self.device).clamp_min(1e-12)
            score = -torch.log(q) - torch.log(-torch.log(u))
        idx = torch.topk(score.float(), k).indices
        return torch.sort(idx.cpu()).values

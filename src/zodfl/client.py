"""One client, one round: stream frames once, select, take SimSiam steps on a bounded buffer."""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import torch

from .augment import augment
from .config import FederationCfg
from .model import SimSiamModel, get_vector, set_vector
from .selectors import Selector


@dc.dataclass
class ClientResult:
    delta: torch.Tensor | None
    n_seen: int
    n_kept: int
    n_steps: int
    cond_counts: np.ndarray          # kept frames per condition id
    losses: list[float]
    max_buffer_seen: int             # largest number of frames ever held in the buffer


def client_round(model: SimSiamModel, global_vec: torch.Tensor, rows: np.ndarray, store, cond_ids: np.ndarray,
                 n_conds: int, selector: Selector, fcfg: FederationCfg, device, gen: torch.Generator,
                 anchor: dict[str, torch.Tensor] | None = None) -> ClientResult:
    """`rows` are this client's stream frames for the round, already in arrival (time) order.
    Every frame is read once; kept frames pass through a buffer that never exceeds max_buffer."""
    set_vector(model, global_vec)
    model.train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    body = [p for n, p in named if n.startswith(("layer3", "layer4"))]
    head = [p for n, p in named if not n.startswith(("layer3", "layer4"))]
    opt = torch.optim.SGD([{"params": body, "lr": fcfg.local_lr * fcfg.backbone_lr_mult},
                           {"params": head, "lr": fcfg.local_lr}],
                          momentum=fcfg.momentum, weight_decay=fcfg.weight_decay)
    anchored = [(p, anchor[n]) for n, p in named if anchor and n in anchor] if fcfg.anchor_mu > 0 else []
    counts = np.zeros(n_conds, dtype=np.int64)
    losses: list[float] = []
    buf: torch.Tensor | None = None
    n_kept = n_steps = max_buf = 0

    def step(batch_u8: torch.Tensor) -> None:
        nonlocal n_steps
        x1, x2 = augment(batch_u8, gen), augment(batch_u8, gen)
        loss = model.pair_loss(x1, x2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p, a in anchored:                       # L2-SP style pull toward the shipped weights
            p.grad.add_(p.detach() - a, alpha=fcfg.anchor_mu)
        opt.step()
        losses.append(float(loss.detach()))
        n_steps += 1

    for start in range(0, len(rows), fcfg.arrival_batch):
        chunk = rows[start:start + fcfg.arrival_batch]
        x = store.get(chunk).to(device)
        keep = selector.select(model, x, cond_ids[chunk], gen)
        n_kept += len(keep)
        np.add.at(counts, cond_ids[chunk][keep.numpy()], 1)
        kept = x[keep.to(device)]
        buf = kept if buf is None else torch.cat([buf, kept])
        while buf.shape[0] >= fcfg.train_batch:
            step(buf[:fcfg.train_batch])
            buf = buf[fcfg.train_batch:]
        max_buf = max(max_buf, int(buf.shape[0]))
    if buf is not None and buf.shape[0] >= fcfg.min_final_batch:
        step(buf)
        max_buf = max(max_buf, int(buf.shape[0]))
    if max_buf > fcfg.max_buffer:
        raise AssertionError(f"buffer exceeded its cap: {max_buf} > {fcfg.max_buffer}")

    delta = None if n_steps == 0 else get_vector(model) - global_vec
    return ClientResult(delta, len(rows), n_kept, n_steps, counts, losses, max_buf)

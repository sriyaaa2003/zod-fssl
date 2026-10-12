"""Per-sample GPU augmentation for SimSiam views. Uses an explicit torch.Generator only."""
from __future__ import annotations

import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def normalize(x01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def to_input(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B,3,H,W] -> normalized float, no augmentation (used for features and scoring)."""
    return normalize(x_uint8.float() / 255.0)


def _u(gen: torch.Generator, n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.rand(n, generator=gen, device=device) * (hi - lo) + lo


def augment(x_uint8: torch.Tensor, gen: torch.Generator, scale_min: float = 0.45,
            jitter: float = 0.4) -> torch.Tensor:
    """Random resized crop (aspect kept) + horizontal flip + brightness/contrast/saturation."""
    dev = x_uint8.device
    x = x_uint8.float() / 255.0
    b, _, h, w = x.shape

    s = torch.sqrt(_u(gen, b, scale_min, 1.0, dev))               # side scale from area scale
    tx = (1 - s) * _u(gen, b, -1.0, 1.0, dev)
    ty = (1 - s) * _u(gen, b, -1.0, 1.0, dev)
    flip = torch.where(torch.rand(b, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = s * flip
    theta[:, 0, 2] = tx
    theta[:, 1, 1] = s
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, size=(b, 3, h, w), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    br = 1 + _u(gen, b, -jitter, jitter, dev).view(b, 1, 1, 1)
    ct = 1 + _u(gen, b, -jitter, jitter, dev).view(b, 1, 1, 1)
    sa = 1 + _u(gen, b, -jitter, jitter, dev).view(b, 1, 1, 1)
    x = x * br
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * ct + m
    g = x.mean(dim=1, keepdim=True)
    x = (x - g) * sa + g
    return normalize(x.clamp(0.0, 1.0))

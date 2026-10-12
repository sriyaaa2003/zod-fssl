"""ResNet-18 encoder with frozen BatchNorm + early layers, SimSiam head, flat parameter vectors.

Trainable: layer3, layer4, projector, predictor. Everything else is frozen, and BatchNorm stays in
eval mode, so a client holds no running statistics and the whole model state is its parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _mlp(dims: list[int], last_norm: bool, last_affine: bool = True) -> nn.Sequential:
    """Linear stack with BatchNorm1d (batch statistics only: no running buffers to leak between clients).
    SimSiam relies on BatchNorm in its heads to avoid collapse; LayerNorm collapsed in the first pilot."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=False if (i < len(dims) - 2 or last_norm) else True))
        is_last = i == len(dims) - 2
        if not is_last:
            layers.append(nn.BatchNorm1d(dims[i + 1], track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
        elif last_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1], affine=last_affine, track_running_stats=False))
    return nn.Sequential(*layers)


class SimSiamModel(nn.Module):
    feat_dim = 512

    def __init__(self, pretrained: bool = True, proj_dim: int = 512, pred_hidden: int = 128,
                 train_layers: tuple[str, ...] = ("layer3", "layer4")):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        self.layer3, self.layer4 = r.layer3, r.layer4
        self.projector = _mlp([self.feat_dim, proj_dim, proj_dim, proj_dim], last_norm=True, last_affine=False)
        self.predictor = _mlp([proj_dim, pred_hidden, proj_dim], last_norm=False)

        for p in self.stem.parameters():
            p.requires_grad_(False)
        for name in ("layer3", "layer4"):
            if name not in train_layers:
                for p in getattr(self, name).parameters():
                    p.requires_grad_(False)
        for m in self.modules():                     # freeze BatchNorm affine parameters too
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():                     # BatchNorm always uses stored statistics
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.stem(x)
        h = self.layer4(self.layer3(h))
        return h.mean(dim=(2, 3))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features for probes/novelty (no grad, eval mode is the caller's job)."""
        return self.features(x)

    def pair_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Per-sample symmetric SimSiam loss, shape [B]; lower means the two views agree."""
        z1, z2 = self.projector(self.features(x1)), self.projector(self.features(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        d1 = -F.cosine_similarity(p1, z2.detach(), dim=-1)
        d2 = -F.cosine_similarity(p2, z1.detach(), dim=-1)
        return 0.5 * (d1 + d2)


def trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def get_vector(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p.detach() for p in trainable_params(model)]).clone()


def set_vector(model: nn.Module, vec: torch.Tensor) -> None:
    """Copy values into the parameters. (vector_to_parameters would alias vec's memory and let
    local training overwrite the global vector.)"""
    i = 0
    with torch.no_grad():
        for p in trainable_params(model):
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p))
            i += n
    if i != vec.numel():
        raise ValueError(f"vector has {vec.numel()} values, model expects {i}")

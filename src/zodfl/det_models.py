"""Detectors for the generation-gap study: Faster R-CNN with a ResNet-18 FPN (generation A, full camera) and with a
MobileNetV3-small FPN (generation B, cheaper chip and lower-resolution camera).

Both use the same anchor layout (small anchors: ZOD pedestrians are only ~5 px wide at the stored 960x540 size).
The `Sensor` of a generation is its input size: torchvision rescales every image so its short side equals `min_size`
and maps the predicted boxes back to the stored-image coordinates, so evaluation is always in the same coordinates.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.ops import misc as misc_ops

ANCHOR_SIZES = ((8, 12), (16, 24), (32, 48), (64, 128), (128, 256))
ANCHOR_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)                 # height / width


def build_detector(kind: str, min_size: int, max_size: int, pretrained: bool = True, num_classes: int = 4,
                   fpn_channels: int = 128, trainable_layers: int = 3) -> FasterRCNN:
    anchors = AnchorGenerator(ANCHOR_SIZES, ANCHOR_RATIOS)
    if kind == "r18":
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=w, trainable_layers=trainable_layers,
                                       norm_layer=misc_ops.FrozenBatchNorm2d if pretrained else nn.BatchNorm2d,
                                       returned_layers=[1, 2, 3, 4])
        backbone.out_channels = 256
    elif kind == "mnv3":
        w = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        norm = misc_ops.FrozenBatchNorm2d if pretrained else nn.BatchNorm2d
        feats = mobilenet_v3_small(weights=w, norm_layer=norm).features
        n_train = max(0, min(trainable_layers + 3, len(feats)))
        for b in feats[: len(feats) - n_train]:                         # freeze the earliest blocks only
            for p in b.parameters():
                p.requires_grad = False
        backbone = BackboneWithFPN(feats, {"1": "0", "3": "1", "8": "2", "12": "3"}, [16, 24, 48, 576], fpn_channels)
    else:
        raise ValueError(f"unknown detector kind {kind!r}")
    return FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchors, min_size=min_size,
                      max_size=max_size, box_score_thresh=0.05, box_detections_per_img=150,
                      rpn_pre_nms_top_n_train=1000, rpn_post_nms_top_n_train=500,
                      rpn_pre_nms_top_n_test=800, rpn_post_nms_top_n_test=400)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_optimizer(model: nn.Module, lr: float, wd: float = 1e-4) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)

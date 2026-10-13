"""COCO-style average precision, written out so there is no dependency on pycocotools.

AP is computed per class over IoU thresholds 0.50:0.05:0.95 with 101-point interpolated precision (the COCO recipe),
then averaged over classes that have at least one ground-truth box. `ap50` is the same at IoU 0.5 only.
Detections and boxes are numpy: gt (n,4) per image per class, dets (m,4) with scores (m,).
"""
from __future__ import annotations

import numpy as np

IOUS = np.linspace(0.5, 0.95, 10)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _ap_one_class(gts: list[np.ndarray], dets: list[np.ndarray], scores: list[np.ndarray], thr: float) -> float | None:
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return None
    flat = [(s, i, j) for i, sc in enumerate(scores) for j, s in enumerate(sc)]
    if not flat:
        return 0.0
    flat.sort(key=lambda t: -t[0])
    matched = [np.zeros(len(g), dtype=bool) for g in gts]
    ious = [iou_matrix(d, g) for d, g in zip(dets, gts)]
    tp = np.zeros(len(flat))
    for k, (_, i, j) in enumerate(flat):
        if len(gts[i]) == 0:
            continue
        cand = np.where(matched[i], -1.0, ious[i][j])
        best = int(np.argmax(cand))
        if cand[best] >= thr:
            matched[i][best] = True
            tp[k] = 1.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    for k in range(len(prec) - 2, -1, -1):                    # precision envelope
        prec[k] = max(prec[k], prec[k + 1])
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, pts, side="left")
    q = np.array([prec[i] if i < len(prec) else 0.0 for i in idx])
    return float(q.mean())


def average_precision(gt: list[np.ndarray], det: list[dict], n_classes: int = 3) -> dict:
    """gt[i]: (n,5) x1 y1 x2 y2 cls (cls 1..n_classes). det[i]: {'boxes': (m,4), 'scores': (m,), 'labels': (m,)}.
    Returns {'ap': mean over IoUs and classes, 'ap50': ..., 'per_class': {cls: ap}}."""
    per_class, per_class50 = {}, {}
    for c in range(1, n_classes + 1):
        g = [x[x[:, 4] == c, :4] if len(x) else np.zeros((0, 4)) for x in gt]
        d, s = [], []
        for dd in det:
            m = dd["labels"] == c if len(dd["labels"]) else np.zeros(0, dtype=bool)
            d.append(dd["boxes"][m] if len(dd["boxes"]) else np.zeros((0, 4)))
            s.append(dd["scores"][m] if len(dd["scores"]) else np.zeros(0))
        aps = [_ap_one_class(g, d, s, t) for t in IOUS]
        if aps[0] is None:
            continue
        per_class[c] = float(np.mean(aps))
        per_class50[c] = float(aps[0])
    if not per_class:
        return {"ap": float("nan"), "ap50": float("nan"), "per_class": {}}
    return {"ap": float(np.mean(list(per_class.values()))), "ap50": float(np.mean(list(per_class50.values()))),
            "per_class": per_class}

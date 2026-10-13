"""Detection data: ZOD 2D boxes -> compact box arrays, frame subset selection, and a JPEG store.

Boxes come from ZOD's object_detection.json (a MultiPoint of 4 corners per object, in full-resolution pixels of the
3848x2168 camera image). We keep three classes and scale boxes to the stored image size.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("Vehicle", "VulnerableVehicle", "Pedestrian")           # label ids 1..3 (0 is background in torchvision)
FULL_W, FULL_H = 3848, 2168
STORE_W, STORE_H = 960, 540
_FRAME_RE = re.compile(r"single_frames/(\d+)/")


def frame_id_from_path(name: str) -> int | None:
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else None


def boxes_from_features(features: list[dict], sx: float = STORE_W / FULL_W, sy: float = STORE_H / FULL_H,
                        min_w: float = 3.0, min_h: float = 4.0) -> np.ndarray:
    """-> float32 array (n, 5): x1, y1, x2, y2 (stored-image pixels), class id 1..3. Unclear objects and other classes
    are dropped; boxes that end up smaller than min_w x min_h in the stored image are dropped (not learnable)."""
    out = []
    for f in features:
        p = f.get("properties", {})
        cls = p.get("class")
        if cls not in CLASSES or p.get("unclear"):
            continue
        g = f["geometry"]
        pts = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        x1, x2 = max(0.0, min(xs) * sx), min(STORE_W, max(xs) * sx)
        y1, y2 = max(0.0, min(ys) * sy), min(STORE_H, max(ys) * sy)
        if x2 - x1 >= min_w and y2 - y1 >= min_h:
            out.append((x1, y1, x2, y2, CLASSES.index(cls) + 1))
    return np.asarray(out, dtype=np.float32).reshape(-1, 5)


def select_frames(index: pd.DataFrame, gen_countries: list[list[str]], n_train: int, n_test_per_cond: int,
                  n_val: int, seed: int, rare_power: float = 0.5) -> pd.DataFrame:
    """Pick the frames used by the detection study. Returns rows of `index` plus `role` (train/val/test) and `gen`
    (0, 1, 2 for training frames: the generation whose countries the frame comes from; -1 for val/test).

    train: n_train per generation, from the 'stream' split of that generation's countries, sampled with weight
           (group count)^(rare_power - 1) per frame so that rare condition groups are not drowned out (0.5 = square root).
    val:   n_val frames from probe_train (all gen countries), for choosing the recipe.
    test:  up to n_test_per_cond frames per condition group from probe_test (all gen countries).
    """
    rng = np.random.default_rng(seed)
    allowed = {c for g in gen_countries for c in g}
    parts = []
    for gi, countries in enumerate(gen_countries):
        pool = index[(index["split"] == "stream") & index["country"].isin(countries)]
        if len(pool) < n_train:
            raise ValueError(f"generation {gi}: only {len(pool)} stream frames for {countries}, need {n_train}")
        cnt = pool["cond"].map(pool["cond"].value_counts()).to_numpy(dtype=float)
        w = cnt ** (rare_power - 1.0)
        pick = rng.choice(len(pool), size=n_train, replace=False, p=w / w.sum())
        sub = pool.iloc[np.sort(pick)].copy()
        sub["role"], sub["gen"] = "train", gi
        parts.append(sub)
    val_pool = index[(index["split"] == "probe_train") & index["country"].isin(allowed)]
    val = val_pool.iloc[np.sort(rng.choice(len(val_pool), size=min(n_val, len(val_pool)), replace=False))].copy()
    val["role"], val["gen"] = "val", -1
    parts.append(val)
    test_pool = index[(index["split"] == "probe_test") & index["country"].isin(allowed)]
    for _, grp in test_pool.groupby("cond"):
        sub = grp.iloc[np.sort(rng.choice(len(grp), size=min(n_test_per_cond, len(grp)), replace=False))].copy()
        sub["role"], sub["gen"] = "test", -1
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    if out["row"].duplicated().any():
        raise AssertionError("a frame was selected twice")
    return out


class DetStore:
    """Concatenated JPEG bytes in one file plus an offsets table; frames are addressed by frame id (`row`)."""

    def __init__(self, directory: str | Path):
        d = Path(directory)
        self.meta = pd.read_parquet(d / "det_meta.parquet").set_index("row")
        self._file = d / "det_images.bin"
        self._mm = np.memmap(self._file, dtype=np.uint8, mode="r")
        self._boxes = {int(r): np.asarray(json.loads(b), dtype=np.float32).reshape(-1, 5)
                       for r, b in zip(self.meta.index, self.meta["boxes"])}

    def boxes(self, row: int) -> np.ndarray:
        return self._boxes[int(row)]

    def image(self, row: int) -> np.ndarray:
        """-> uint8 array (H, W, 3)."""
        from PIL import Image
        m = self.meta.loc[int(row)]
        raw = bytes(self._mm[int(m["offset"]): int(m["offset"]) + int(m["length"])])
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def encode_jpeg(arr: np.ndarray, quality: int = 90) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def write_store(directory: str | Path, rows: list[int], jpegs: dict[int, bytes], boxes: dict[int, np.ndarray],
                selection: pd.DataFrame) -> pd.DataFrame:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    offs, lens, pos = [], [], 0
    with open(d / "det_images.bin", "wb") as f:
        for r in rows:
            b = jpegs[r]
            f.write(b)
            offs.append(pos)
            lens.append(len(b))
            pos += len(b)
    sel = selection.set_index("row").loc[rows]
    meta = pd.DataFrame({"row": rows, "offset": offs, "length": lens,
                         "boxes": [json.dumps(np.round(boxes[r], 2).tolist()) for r in rows],
                         "n_boxes": [len(boxes[r]) for r in rows],
                         "country": sel["country"].to_numpy(), "cond": sel["cond"].to_numpy(),
                         "light": sel["light"].to_numpy(), "role": sel["role"].to_numpy(), "gen": sel["gen"].to_numpy()})
    meta.to_parquet(d / "det_meta.parquet")
    return meta

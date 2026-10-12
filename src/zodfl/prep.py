"""Stream a ZOD image tar.gz once, shrink each frame, write uint8 .npy shards. No full-size image is kept."""
from __future__ import annotations

import io
import json
import re
import tarfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

FRAME_RE = re.compile(r"(?:^|/)single_frames/(\d+)/camera_front_blur/[^/]+\.jpg$")


def decode(args: tuple[int, bytes, tuple[int, int]]) -> tuple[int, np.ndarray | None]:
    """JPEG bytes -> uint8 [3,H,W]. Uses JPEG draft mode so an 8 MP image is decoded at reduced size."""
    row, raw, (h, w) = args
    try:
        im = Image.open(io.BytesIO(raw))
        im.draft("RGB", (w * 2, h * 2))
        im = im.convert("RGB").resize((w, h), Image.BILINEAR)
        return row, np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy()
    except Exception:  # noqa: BLE001  corrupt image: reported, never silently replaced
        return row, None


def _save(shard_dir: Path, sid: int, arr: np.ndarray) -> str:
    name = f"shard_{sid:04d}.npy"
    np.save(shard_dir / name, arr)
    return name


def prep_tar(tar_path: str | Path, out_dir: str | Path, hw: tuple[int, int], shard_size: int,
             workers: int = 8, row_range: tuple[int, int] | None = None, max_corrupt_fraction: float = 0.002,
             in_flight: int = 64) -> dict:
    """Process one image archive. Frame rows are int(frame_id). Shards are aligned to shard_size, so two
    archives that cover disjoint, shard-aligned row ranges can be processed in parallel containers."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = row_range if row_range else (0, None)
    if lo % shard_size:
        raise ValueError("row_range must start on a shard boundary")
    shards: dict[int, np.ndarray] = {}
    filled: dict[int, int] = {}
    bad: list[int] = []
    written: dict[int, str] = {}
    seen: set[int] = set()
    n_ok = 0

    def place(row: int, img: np.ndarray | None) -> None:
        nonlocal n_ok
        sid = row // shard_size
        if sid not in shards:
            shards[sid] = np.zeros((shard_size, 3, *hw), dtype=np.uint8)
            filled[sid] = 0
        if img is None:
            bad.append(row)
        else:
            shards[sid][row - sid * shard_size] = img
            n_ok += 1
        filled[sid] += 1
        if filled[sid] == shard_size:
            written[sid] = _save(out, sid, shards.pop(sid))

    with ProcessPoolExecutor(max_workers=workers) as pool, tarfile.open(tar_path, "r|*") as tf:
        pending: deque = deque()
        for m in tf:
            mt = FRAME_RE.search(m.name) if m.isfile() else None
            if not mt:
                continue
            row = int(mt.group(1))
            if row < lo or (hi is not None and row >= hi):
                raise ValueError(f"frame {row} outside the declared row_range {row_range}")
            if row in seen:
                raise ValueError(f"frame {row} appears twice")
            seen.add(row)
            pending.append(pool.submit(decode, (row, tf.extractfile(m).read(), hw)))
            while len(pending) >= in_flight:
                place(*pending.popleft().result())
        while pending:
            place(*pending.popleft().result())

    for sid, arr in list(shards.items()):                    # partial shards at the end of a range
        written[sid] = _save(out, sid, arr)
    n_total = len(seen)
    if n_total == 0:
        raise RuntimeError("no frames found in the archive")
    if len(bad) / n_total > max_corrupt_fraction:
        raise RuntimeError(f"{len(bad)} of {n_total} images could not be decoded (> {max_corrupt_fraction:.2%})")
    part = {"tar": str(tar_path), "n_frames": n_total, "n_decoded": n_ok, "bad_rows": sorted(bad),
            "shards": {str(k): v for k, v in sorted(written.items())}, "row_min": min(seen), "row_max": max(seen)}
    (out / f"part_{lo:06d}.json").write_text(json.dumps(part))
    return part


def write_manifest(out_dir: str | Path, hw: tuple[int, int], shard_size: int) -> dict:
    """Combine part_*.json into manifest.json once every archive has been processed."""
    out = Path(out_dir)
    parts = [json.loads(p.read_text()) for p in sorted(out.glob("part_*.json"))]
    if not parts:
        raise FileNotFoundError("no part files found")
    shard_ids = sorted({int(k) for p in parts for k in p["shards"]})
    if shard_ids != list(range(len(shard_ids))):
        raise RuntimeError(f"shards are not contiguous: {shard_ids[:5]}...{shard_ids[-5:]}")
    files = [f"shard_{i:04d}.npy" for i in shard_ids]
    n = max(p["row_max"] for p in parts) + 1
    bad = sorted(r for p in parts for r in p["bad_rows"])
    present = sorted(r for p in parts for r in range(p["row_min"], p["row_max"] + 1))  # contiguous check is advisory
    man = {"n": n, "shard_size": shard_size, "hw": list(hw), "files": files, "bad_rows": bad,
           "n_frames": sum(p["n_frames"] for p in parts), "advisory_rows_spanned": len(present)}
    (out / "manifest.json").write_text(json.dumps(man))
    return man

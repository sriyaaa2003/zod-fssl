"""Random access to pre-shrunk frames. Rows are global frame numbers 0..N-1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class ArrayStore:
    """In-memory store; used by tests and small runs."""

    def __init__(self, images: np.ndarray):
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be uint8 [N,3,H,W]")
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def get(self, rows) -> torch.Tensor:
        return torch.from_numpy(self.images[np.asarray(rows, dtype=np.int64)])


class ShardStore:
    """uint8 .npy shards on disk, memory-mapped; manifest.json written by prep."""

    def __init__(self, shards_dir: str | Path):
        self.dir = Path(shards_dir)
        man = json.loads((self.dir / "manifest.json").read_text())
        self.n: int = man["n"]
        self.shard_size: int = man["shard_size"]
        self.files: list[str] = man["files"]
        self.hw: tuple[int, int] = tuple(man["hw"])
        self._maps: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.n

    def _shard(self, i: int) -> np.ndarray:
        if i not in self._maps:
            self._maps[i] = np.load(self.dir / self.files[i], mmap_mode="r")
        return self._maps[i]

    def preload(self) -> "ShardStore":
        """Read every shard fully into RAM (about 11 GB for 100k frames at 144x256)."""
        for i in range(len(self.files)):
            self._maps[i] = np.load(self.dir / self.files[i])
        return self

    def get(self, rows) -> torch.Tensor:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), 3, *self.hw), dtype=np.uint8)
        sid = rows // self.shard_size
        for s in np.unique(sid):
            m = sid == s
            out[m] = self._shard(int(s))[rows[m] - int(s) * self.shard_size]
        return torch.from_numpy(out)

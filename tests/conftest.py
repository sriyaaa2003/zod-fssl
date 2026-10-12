"""Synthetic fixtures only (no ZOD data): tiny images, 3 'countries', 2 'conditions', 1 label."""
import numpy as np
import pandas as pd
import pytest

from zodfl import config as C
from zodfl.store import ArrayStore

H, W = 32, 64


def make_world(n_per=120, seed=0, countries=("SE", "DE", "NO"), conds=("day", "night")):
    rng = np.random.default_rng(seed)
    rows, imgs = [], []
    r = 0
    for ci, c in enumerate(countries):
        for i in range(n_per):
            cond = conds[1] if (i % 5 == 0) else conds[0]          # 20% rare 'night'
            base = 60 + 40 * ci + (-30 if cond == "night" else 30)
            img = np.clip(rng.normal(base, 25, size=(3, H, W)), 0, 255).astype(np.uint8)
            imgs.append(img)
            rows.append({"row": r, "country": c, "ts": float(i + 1000 * ci * 0 + rng.random()), "cond": cond,
                         "label": int(cond == "night")})
            r += 1
    df = pd.DataFrame(rows)
    split = np.array(["stream"] * len(df), dtype=object)
    perm = rng.permutation(len(df))
    n_probe = int(0.3 * len(df))
    split[perm[:n_probe // 2]] = "probe_train"
    split[perm[n_probe // 2:n_probe]] = "probe_test"
    df["split"] = split
    return df, ArrayStore(np.stack(imgs))


@pytest.fixture
def world():
    return make_world()


@pytest.fixture
def cfg():
    base = C.Config()
    return C.override(
        base,
        data={"image_hw": (H, W)},
        federation={"n_rounds": 3, "min_client_frames": 20, "arrival_batch": 16, "train_batch": 8,
                    "min_final_batch": 4, "max_buffer": 15, "pretrained": False, "local_lr": 0.02},
        selection={"kind": "random", "budget": 0.5},
        eval={"eval_every": 3, "bootstrap": 20, "min_group_size": 5,
              "tasks": [{"name": "night", "label_col": "label"}]},
        run={"device": "cpu"},
    )

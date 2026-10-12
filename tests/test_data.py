import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from zodfl import config as C
from zodfl.index import build_index, summary
from zodfl.prep import prep_tar, write_manifest
from zodfl.store import ShardStore

HW = (18, 32)


def _jpeg(color, size=(128, 72)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_tar(path, ids, corrupt=(), extra_members=True):
    with tarfile.open(path, "w:gz") as tf:
        for i in ids:
            raw = b"not a jpeg" if i in corrupt else _jpeg((i * 5 % 255, 80, 160))
            ti = tarfile.TarInfo(f"single_frames/{i:06d}/camera_front_blur/{i:06d}_india_2021-04-19T10:23:10Z.jpg")
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
        if extra_members:
            raw = b"{}"
            ti = tarfile.TarInfo("single_frames/000000/other.json")
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))


def test_prep_writes_shards_and_manifest(tmp_path):
    make_tar(tmp_path / "a.tar.gz", range(40))
    out = tmp_path / "shards"
    part = prep_tar(tmp_path / "a.tar.gz", out, HW, shard_size=16, workers=2)
    assert part["n_frames"] == 40 and part["n_decoded"] == 40 and part["bad_rows"] == []
    man = write_manifest(out, HW, 16)
    assert man["n"] == 40 and man["files"] == ["shard_0000.npy", "shard_0001.npy", "shard_0002.npy"]
    store = ShardStore(out)
    x = store.get([0, 7, 39]).numpy()
    assert x.shape == (3, 3, *HW) and x.dtype == np.uint8
    # solid red channel i*5 survives shrinking + JPEG within tolerance
    for row, k in zip([0, 7, 39], range(3)):
        assert abs(float(x[k, 0].mean()) - (row * 5 % 255)) < 8
    assert np.array_equal(store.preload().get([7]).numpy(), x[1:2])


def test_prep_reports_corrupt_images_and_enforces_threshold(tmp_path):
    make_tar(tmp_path / "a.tar.gz", range(40), corrupt={3})
    part = prep_tar(tmp_path / "a.tar.gz", tmp_path / "ok", HW, 16, workers=2, max_corrupt_fraction=0.1)
    assert part["bad_rows"] == [3] and part["n_decoded"] == 39
    with pytest.raises(RuntimeError, match="could not be decoded"):
        prep_tar(tmp_path / "a.tar.gz", tmp_path / "strict", HW, 16, workers=2, max_corrupt_fraction=0.001)


def test_prep_rejects_rows_outside_declared_range(tmp_path):
    make_tar(tmp_path / "a.tar.gz", range(40))
    with pytest.raises(ValueError, match="outside"):
        prep_tar(tmp_path / "a.tar.gz", tmp_path / "o", HW, 16, workers=2, row_range=(0, 32))


def test_two_disjoint_archives_combine(tmp_path):
    make_tar(tmp_path / "a.tar.gz", range(0, 32))
    make_tar(tmp_path / "b.tar.gz", range(32, 64))
    out = tmp_path / "s"
    prep_tar(tmp_path / "a.tar.gz", out, HW, 16, workers=2, row_range=(0, 32))
    prep_tar(tmp_path / "b.tar.gz", out, HW, 16, workers=2, row_range=(32, 64))
    man = write_manifest(out, HW, 16)
    assert man["n"] == 64 and len(man["files"]) == 4 and man["n_frames"] == 64


# ---------------------------------------------------------------- index
def fake_meta(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "frame_id": [f"{i:06d}" for i in range(n)],
        "time": pd.to_datetime("2021-01-01", utc=True) + pd.to_timedelta(rng.integers(0, 10**7, n), unit="s"),
        "country_code": rng.choice(["SE", "DE", "PL"], n, p=[0.5, 0.3, 0.2]),
        "collection_car": rng.choice(["india", "golf"], n),
        "road_type": rng.choice(["city", "highway"], n),
        "road_condition": rng.choice(["normal", "wet", "snow"], n, p=[0.7, 0.2, 0.1]),
        "time_of_day": rng.choice(["day", "night", "twilight"], n, p=[0.7, 0.2, 0.1]),
        "num_vulnerable_vehicles": rng.integers(0, 3, n), "num_pedestrians": rng.integers(0, 3, n),
        "num_vehicles": rng.integers(0, 40, n),
    }).assign(time=lambda d: d["time"].astype(str))


def test_index_columns_labels_and_splits():
    cfg = C.Config()
    idx = build_index(fake_meta(), cfg.index, cfg.data)
    assert {"row", "ts", "country", "cond", "light", "split", "label_vru", "label_road_type", "label_density",
            "label_road_condition"} <= set(idx.columns)
    assert set(idx["split"]) == {"stream", "probe_train", "probe_test"}
    assert idx["cond"].str.contains("/").all() and set(idx["light"]) == {"day", "night"}
    assert (idx["label_vru"] == ((fake_meta()[["num_vulnerable_vehicles", "num_pedestrians"]].sum(axis=1) > 0)
                                 .astype(int).to_numpy())).all()
    assert idx["label_density"].between(0, 2).all()
    big = build_index(fake_meta(6000), cfg.index, cfg.data)                 # small groups round down; use more data
    frac = (big["split"] != "stream").mean()
    assert abs(frac - cfg.data.probe_fraction) < 0.01
    assert (big["split"] == "probe_train").sum() == (big["split"] == "probe_test").sum()   # halves equal per stratum


def test_index_split_is_deterministic_and_strata_are_covered():
    cfg = C.Config()
    a = build_index(fake_meta(), cfg.index, cfg.data)
    b = build_index(fake_meta(), cfg.index, cfg.data)
    assert a["split"].tolist() == b["split"].tolist()
    big = a.groupby(["country", "cond"]).filter(lambda g: len(g) >= 20)
    cov = big.groupby(["country", "cond"])["split"].agg(lambda s: set(s))
    assert all({"probe_train", "probe_test", "stream"} <= v for v in cov)


def test_index_rejects_bad_metadata():
    cfg = C.Config()
    m = fake_meta()
    with pytest.raises(KeyError):
        build_index(m.drop(columns=["road_type"]), cfg.index, cfg.data)
    m2 = m.copy()
    m2.loc[0, "time"] = "not a time"
    with pytest.raises(ValueError, match="unparseable"):
        build_index(m2, cfg.index, cfg.data)
    with pytest.raises(ValueError, match="duplicate"):
        build_index(pd.concat([m, m.iloc[:1]]), cfg.index, cfg.data)


def test_summary_counts_add_up():
    cfg = C.Config()
    idx = build_index(fake_meta(), cfg.index, cfg.data)
    s = summary(idx)
    assert s["n_frames"] == len(idx) == sum(s["split_counts"].values())
    json.dumps(s)                                                           # must be JSON serializable

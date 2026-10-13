import numpy as np
import pandas as pd
import pytest

from zodfl.det_data import (DetStore, boxes_from_features, encode_jpeg, frame_id_from_path, select_frames,
                            write_store)
from zodfl.det_metrics import average_precision, iou_matrix


def _feat(cls, pts, unclear=False, geom="MultiPoint"):
    return {"geometry": {"type": geom, "coordinates": pts if geom == "MultiPoint" else [pts]},
            "properties": {"class": cls, "unclear": unclear}}


def test_frame_id_is_parsed_from_the_tar_member_name():
    assert frame_id_from_path("single_frames/000123/annotations/object_detection.json") == 123
    assert frame_id_from_path("something/else.json") is None


def test_boxes_are_axis_aligned_scaled_and_filtered():
    pts = [[1000, 1000], [1100, 1010], [1105, 1200], [1002, 1195]]       # a box of ~105 x 200 px at full resolution
    feats = [_feat("Vehicle", pts), _feat("Pedestrian", pts, unclear=True), _feat("PoleObject", pts),
             _feat("Pedestrian", [[10, 10], [12, 10], [12, 14], [10, 14]]),          # far too small once scaled
             _feat("VulnerableVehicle", pts, geom="Polygon")]
    b = boxes_from_features(feats)
    assert b.shape == (2, 5) and list(b[:, 4]) == [1, 2]
    s = 960 / 3848
    assert b[0, 0] == pytest.approx(1000 * s, abs=1e-3) and b[0, 2] == pytest.approx(1105 * s, abs=1e-3)
    assert boxes_from_features([]).shape == (0, 5)


def _index(n=3000):
    rng = np.random.default_rng(0)
    countries = np.array(["PL", "DE", "SE", "GB", "IT", "NO"])
    df = pd.DataFrame({"row": np.arange(n), "country": rng.choice(countries, n),
                       "cond": rng.choice(["normal/day", "wet/night", "snow/day"], n, p=[0.8, 0.1, 0.1]),
                       "light": "day", "split": rng.choice(["stream", "probe_train", "probe_test"], n, p=[0.8, 0.1, 0.1])})
    return df


def test_frame_selection_is_disjoint_stratified_and_deterministic():
    idx = _index()
    gens = [["PL", "DE"], ["SE", "GB"], ["IT", "NO"]]
    a = select_frames(idx, gens, n_train=200, n_test_per_cond=30, n_val=50, seed=1)
    b = select_frames(idx, gens, n_train=200, n_test_per_cond=30, n_val=50, seed=1)
    assert a["row"].tolist() == b["row"].tolist()
    assert not a["row"].duplicated().any()
    tr = a[a["role"] == "train"]
    assert sorted(tr["gen"].unique()) == [0, 1, 2] and (tr.groupby("gen").size() == 200).all()
    for g, countries in enumerate(gens):
        assert set(tr[tr["gen"] == g]["country"]) <= set(countries)
    assert set(tr.merge(idx, on="row", suffixes=("", "_i"))["split_i"]) == {"stream"}
    assert set(a[a["role"] == "test"].merge(idx, on="row", suffixes=("", "_i"))["split_i"]) == {"probe_test"}
    assert (a[a["role"] == "test"].groupby("cond").size() <= 30).all()
    # sqrt re-weighting raises the share of the rare group above its raw share
    raw = (idx[idx["split"] == "stream"]["cond"] == "wet/night").mean()
    assert (tr["cond"] == "wet/night").mean() > raw
    with pytest.raises(ValueError):
        select_frames(idx, gens, n_train=10 ** 6, n_test_per_cond=3, n_val=5, seed=0)


def test_store_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    imgs = {r: rng.integers(0, 255, (54, 96, 3), dtype=np.uint8) for r in (5, 9)}
    boxes = {5: np.array([[1, 2, 30, 40, 1]], dtype=np.float32), 9: np.zeros((0, 5), dtype=np.float32)}
    sel = pd.DataFrame({"row": [5, 9], "country": ["PL", "SE"], "cond": ["a", "b"], "light": ["day", "night"],
                        "role": ["train", "test"], "gen": [0, -1]})
    write_store(tmp_path, [5, 9], {r: encode_jpeg(a) for r, a in imgs.items()}, boxes, sel)
    st = DetStore(tmp_path)
    assert st.image(5).shape == (54, 96, 3) and st.image(9).shape == (54, 96, 3)
    assert st.boxes(5).tolist() == [[1, 2, 30, 40, 1]] and st.boxes(9).shape == (0, 5)
    assert st.meta.loc[9, "role"] == "test"


def test_iou():
    a = np.array([[0, 0, 10, 10.0]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[5, 0, 15, 10.0]]))[0, 0] == pytest.approx(1 / 3)
    assert iou_matrix(a, np.array([[20, 20, 30, 30.0]]))[0, 0] == 0


def _det(boxes, scores, labels):
    return {"boxes": np.array(boxes, dtype=float).reshape(-1, 4), "scores": np.array(scores, dtype=float),
            "labels": np.array(labels, dtype=int)}


def test_ap_perfect_none_and_ranking():
    gt = [np.array([[0, 0, 10, 10, 1], [20, 20, 40, 40, 2]], dtype=float), np.array([[5, 5, 15, 15, 1]], dtype=float)]
    perfect = [_det([[0, 0, 10, 10], [20, 20, 40, 40]], [0.9, 0.8], [1, 2]), _det([[5, 5, 15, 15]], [0.7], [1])]
    r = average_precision(gt, perfect, n_classes=3)
    assert r["ap"] == pytest.approx(1.0) and r["ap50"] == pytest.approx(1.0) and set(r["per_class"]) == {1, 2}
    none = [_det([], [], []), _det([], [], [])]
    assert average_precision(gt, none)["ap"] == 0.0
    # a confident false positive ranked above the true positive lowers AP; ranking it below costs less
    gt1 = [np.array([[0, 0, 10, 10, 1]], dtype=float)]
    fp_first = [_det([[50, 50, 60, 60], [0, 0, 10, 10]], [0.9, 0.5], [1, 1])]
    fp_last = [_det([[50, 50, 60, 60], [0, 0, 10, 10]], [0.5, 0.9], [1, 1])]
    assert average_precision(gt1, fp_first)["ap"] == pytest.approx(0.5, abs=0.02)
    assert average_precision(gt1, fp_last)["ap"] == pytest.approx(1.0)


def test_ap_is_lower_at_stricter_iou():
    gt = [np.array([[0, 0, 100, 100, 1]], dtype=float)]
    shifted = [_det([[8, 0, 108, 100]], [0.9], [1])]            # IoU ~0.85
    r = average_precision(gt, shifted)
    assert r["ap50"] == pytest.approx(1.0) and 0.3 < r["ap"] < 0.9


def _tiny_store(tmp_path, n_per=8):
    """Synthetic frames with a bright rectangle per box so a detector has something it could learn."""
    rng = np.random.default_rng(0)
    rows, imgs, boxes, recs = [], {}, {}, []
    rid = 0
    for role, gen in [("train", 0), ("train", 1), ("train", 2), ("val", -1), ("test", -1)]:
        for _ in range(n_per):
            img = rng.integers(0, 40, (108, 192, 3), dtype=np.uint8)
            x, y = int(rng.integers(10, 150)), int(rng.integers(10, 70))
            img[y:y + 25, x:x + 30] = 220
            imgs[rid] = img
            boxes[rid] = np.array([[x, y, x + 30, y + 25, 1]], dtype=np.float32)
            recs.append({"row": rid, "country": "PL", "cond": ["normal/day", "wet/night"][rid % 2], "light": "day",
                         "role": role, "gen": gen})
            rows.append(rid)
            rid += 1
    write_store(tmp_path, rows, {r: encode_jpeg(a) for r, a in imgs.items()}, boxes, pd.DataFrame(recs))
    return DetStore(tmp_path)


def test_detection_pipeline_runs_end_to_end_on_tiny_data(tmp_path):
    import torch

    from zodfl.det_run import run_det_seed, save_det_record
    store = _tiny_store(tmp_path / "s")
    p = {"A": {"min_size": 108, "max_size": 192, "epochs": 1, "batch": 2, "lr": 0.01},
         "B": {"min_size": 72, "max_size": 128, "epochs": 1, "batch": 2, "lr": 0.01},
         "workers": 0, "pretrained": False, "floor_fraction": 0.25, "warmup": 2}
    recs = run_det_seed(store, 0, torch.device("cpu"), p, split="test", log=lambda *_: None,
                        ckpt_dir=tmp_path / "ck")
    assert [r["tag"] for r in recs] == ["det_A_teacher", "det_B_oracle", "det_B_floor", "det_B_kd_incar", "det_B_kd_floor"]
    for r in recs:
        ev = r["eval"]
        assert 0.0 <= ev["ap"] <= 1.0 and set(ev["groups"]) == {"normal/day", "wet/night"}
        assert r["train"]["skipped"] == 0
    assert recs[2]["extra"]["n_labelled"] == 2
    assert recs[3]["extra"]["pseudo_boxes"] >= 0
    path = save_det_record(recs[0], tmp_path / "out", {"gpu": None})
    assert path.name == "det_A_teacher__seed0.json" and path.stat().st_size > 0
    # a second call reuses the saved teacher instead of retraining it
    again = run_det_seed(store, 0, torch.device("cpu"), p, split="test", log=lambda *_: None, only=("det_A_teacher",),
                         ckpt_dir=tmp_path / "ck")
    assert again[0]["train"] == {"loaded_from_checkpoint": True}


def _det_rec(tag, seed, ap, worst, split="test"):
    return {"tag": tag, "seed": seed, "split": split, "task": "detection",
            "eval": {"ap": ap, "ap50": ap * 2, "per_class": {"1": ap, "2": ap / 2, "3": ap / 3},
                     "groups": {"normal/day": {"ap": ap, "ap50": ap * 2, "n": 100},
                                "wet/night": {"ap": worst, "ap50": worst * 2, "n": 12}},
                     "worst_group_ap": worst, "mean_group_ap": (ap + worst) / 2},
            "extra": {"pseudo_boxes": 10, "true_boxes_in_those_frames": 20, "threshold": 0.5} if "kd" in tag else {}}


def test_detection_report_tables_and_pairs(tmp_path):
    import json

    from zodfl.det_report import det_block, gap_closed_det, load_det
    d = tmp_path / "detection"
    d.mkdir()
    vals = {"det_A_teacher": 0.30, "det_B_floor": 0.05, "det_B_kd_incar": 0.10, "det_B_kd_floor": 0.12, "det_B_oracle": 0.20}
    for seed in (0, 1, 2):
        for tag, ap in vals.items():
            r = _det_rec(tag, seed, ap + 0.01 * seed, ap / 2 + 0.01 * seed)
            (d / f"{tag}__seed{seed}.json").write_text(json.dumps(r))
    (d / "det_A_teacher@v1__seed100.json").write_text(json.dumps({**_det_rec("det_A_teacher", 100, 0.9, 0.9, split="val"), "variant": "v1"}))
    recs = load_det(tmp_path)
    assert len(recs) == 15, "validation-split and variant runs must not enter the report"
    block = det_block(tmp_path)
    assert "det_B_kd_incar" in block and "worst-group AP" in block and "wet/night (n=12)" in block
    assert "`det_B_kd_incar` minus `det_B_floor`: +0.050 (95% CI" in block
    gap = gap_closed_det(recs)
    assert "det_B_kd_incar: 0.333" in gap and "det_B_kd_floor: 0.467" in gap
    assert det_block(tmp_path / "nowhere") == ""

"""Build the frame index (one row per ZOD frame) from the per-frame metadata.json files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataCfg, IndexCfg


def build_index(meta: pd.DataFrame, icfg: IndexCfg, dcfg: DataCfg) -> pd.DataFrame:
    """meta: rows of ZOD metadata.json (frame_id, time, country_code, road_condition, time_of_day, ...).

    `row` is int(frame_id) and addresses the image shards. `split` is fixed by dcfg.probe_seed and
    stratified by country x condition, so every group is represented in the probe sets.
    """
    need = {"frame_id", "time", "country_code", "road_type", "time_of_day", *icfg.vru_columns,
            icfg.density_column, *[c for c in icfg.condition_columns if c != "light"]}
    missing = need - set(meta.columns)
    if missing:
        raise KeyError(f"metadata is missing columns: {sorted(missing)}")
    df = meta.copy()
    df["row"] = df["frame_id"].astype(int)
    if df["row"].duplicated().any():
        raise ValueError("duplicate frame ids in metadata")
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if t.isna().any():
        raise ValueError(f"{int(t.isna().sum())} frames have an unparseable time")
    df["ts"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    df["country"] = df["country_code"].astype(str)
    df["light"] = np.where(df["time_of_day"].isin(icfg.night_values), "night", "day")
    df["cond"] = df[icfg.condition_columns].astype(str).agg("/".join, axis=1)

    df["label_vru"] = (df[list(icfg.vru_columns)].sum(axis=1) > 0).astype(int)
    df["label_road_type"] = df["road_type"].astype(str)
    df["label_density"] = pd.qcut(df[icfg.density_column], icfg.density_bins, labels=False, duplicates="drop").astype(int)
    if "road_condition" in df:
        df["label_road_condition"] = df["road_condition"].astype(str)

    if dcfg.max_frames and dcfg.max_frames < len(df):
        keep = np.random.default_rng(dcfg.probe_seed).choice(len(df), size=dcfg.max_frames, replace=False)
        df = df.iloc[np.sort(keep)].copy()

    rng = np.random.default_rng(dcfg.probe_seed)
    split = np.full(len(df), "stream", dtype=object)
    pos = {i: p for p, i in enumerate(df.index)}
    for _, g in df.groupby(["country", "cond"]):
        idx = np.array([pos[i] for i in g.index])
        idx = idx[rng.permutation(len(idx))]
        n_probe = int(round(dcfg.probe_fraction * len(idx)))
        if len(idx) >= 6:
            n_probe = max(n_probe, 2)
        n_probe -= n_probe % 2
        split[idx[: n_probe // 2]] = "probe_train"
        split[idx[n_probe // 2: n_probe]] = "probe_test"
    df["split"] = split
    keep_cols = ["row", "frame_id", "country", "collection_car", "ts", "light", "cond", "label_vru", "label_road_type",
                 "label_density", "label_road_condition", "split"]
    return df[[c for c in keep_cols if c in df.columns]].sort_values("row").reset_index(drop=True)


def summary(idx: pd.DataFrame) -> dict:
    """Counts used in the report and to sanity-check groups; written to results/data_summary.json."""
    stream = idx[idx["split"] == "stream"]
    return {
        "n_frames": int(len(idx)),
        "split_counts": idx["split"].value_counts().to_dict(),
        "stream_by_country": stream["country"].value_counts().to_dict(),
        "frames_by_cond": idx["cond"].value_counts().to_dict(),
        "probe_test_by_cond": idx[idx["split"] == "probe_test"]["cond"].value_counts().to_dict(),
        "stream_cond_by_country": {c: g["cond"].value_counts().to_dict() for c, g in stream.groupby("country")},
        "time_min": float(idx["ts"].min()), "time_max": float(idx["ts"].max()),
    }

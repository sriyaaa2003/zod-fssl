"""Explore ZOD metadata/annotations on the Modal volume to pick condition groups and probe tasks (ADR-6).

    modal run modal_index.py::explore
"""
import modal

app = modal.App("zod-fssl-index")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "pyarrow")
data = modal.Volume.from_name("zod-data", create_if_missing=True)
DATA_DIR = "/data"
DL = f"{DATA_DIR}/downloads/single_frames"


def _find(prefix: str) -> str:
    import glob

    hits = glob.glob(f"{DL}/{prefix}*")
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one archive for {prefix!r}, found {hits}")
    return hits[0]


@app.function(image=image, volumes={DATA_DIR: data}, timeout=3 * 3600, cpu=4, memory=16384)
def explore(od_frames: int = 3000):
    import collections
    import json
    import tarfile

    import pandas as pd

    data.reload()
    rows = []
    with tarfile.open(_find("infos.tar.gz"), "r|*") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith("/metadata.json"):
                rows.append(json.loads(tf.extractfile(m).read()))
    meta = pd.DataFrame(rows)
    meta.to_parquet(f"{DATA_DIR}/meta_raw.parquet")
    data.commit()
    print("metadata rows:", len(meta))
    print("columns/dtypes:\n", meta.dtypes.to_string())
    print("\nFULL sample metadata.json:\n", json.dumps(rows[0], indent=1)[:3000])
    meta["time"] = pd.to_datetime(meta["time"], utc=True, errors="coerce")
    print("\ntime range:", meta["time"].min(), "->", meta["time"].max(), "| null times:", meta["time"].isna().sum())
    for col in ("country_code", "collection_car", "time_of_day", "scraped_weather", "road_type", "road_condition"):
        if col in meta:
            vc = meta[col].value_counts(dropna=False)
            print(f"\n== {col} ({len(vc)} values)\n{vc.head(40).to_string()}")
    if {"country_code", "time_of_day"} <= set(meta.columns):
        print("\n== country x time_of_day\n", pd.crosstab(meta["country_code"], meta["time_of_day"]).to_string())
    if {"country_code", "scraped_weather"} <= set(meta.columns):
        print("\n== country x weather (top)\n", pd.crosstab(meta["country_code"], meta["scraped_weather"]).to_string())
    if {"collection_car", "country_code"} <= set(meta.columns):
        cc = meta.groupby("collection_car")["country_code"].agg(["size", "nunique"])
        print("\n== frames per car and #countries per car\n", cc.sort_values("size", ascending=False).to_string())

    # annotations: road_condition keys across all frames, one object_detection sample + property keys
    rc_keys, rc_vals, od_samples, od_keys, od_classes = collections.Counter(), collections.Counter(), [], collections.Counter(), collections.Counter()
    n_od = 0
    with tarfile.open(_find("annotations.tar.gz"), "r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            if name.endswith("/road_condition.json"):
                d = json.loads(tf.extractfile(m).read())
                for k, v in d.items():
                    rc_keys[k] += 1
                    rc_vals[(k, str(v))] += 1
            elif name.endswith("/object_detection.json") and n_od < od_frames:
                raw = tf.extractfile(m).read()
                d = json.loads(raw)
                n_od += 1
                if len(od_samples) < 1:
                    od_samples.append(raw[:2500].decode("utf-8", "replace"))
                for feat in d:
                    props = feat.get("properties", {})
                    for k in props:
                        od_keys[k] += 1
                    od_classes[str(props.get("class"))] += 1
    print("\n== road_condition keys:", dict(rc_keys))
    print("== road_condition value counts:", dict(rc_vals))
    print("\n== object_detection sample (head):\n", od_samples[0] if od_samples else None)
    print("\n== object_detection property keys (first", n_od, "frames):", dict(od_keys.most_common(30)))
    print("== object classes:", dict(od_classes.most_common(40)))

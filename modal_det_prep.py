"""Build the detection image store (960x540 JPEG) and the box labels on the Modal volume. CPU only.

    modal run modal_det_prep.py::select            # choose the frames, parse their boxes -> /det/selection.parquet
    modal run modal_det_prep.py::parts             # read the two image archives in parallel -> /det/parts/*.pkl
    modal run modal_det_prep.py::finalize          # merge -> /det/store/{det_images.bin,det_meta.parquet}
Config: config/default.toml (stage2.gen_countries, data.probe_seed) and the DET dict below.
"""
import modal

app = modal.App("zod-det-prep")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "pyarrow", "pillow", "dataclass-wizard<1.0")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir("src", "/root/src")
    .add_local_dir("config", "/root/config")
)
data = modal.Volume.from_name("zod-data", create_if_missing=True)
D = "/data"
DL = f"{D}/downloads/single_frames"
OUT = f"{D}/det"
DET = {"n_train_per_gen": 3000, "n_test_per_cond": 400, "n_val": 500, "seed": 7}
ARCHIVES = ["images_blur_000000_049999.tar.gz", "images_blur_050000_099999.tar.gz"]


@app.function(image=image, volumes={D: data}, cpu=4, memory=16384, timeout=3600)
def select():
    import glob
    import json
    import os
    import tarfile

    import numpy as np
    import pandas as pd

    from zodfl import config as C
    from zodfl.det_data import boxes_from_features, frame_id_from_path, select_frames

    data.reload()
    cfg = C.load("/root/config/default.toml")
    index = pd.read_parquet(cfg.data.index_path)
    sel = select_frames(index, cfg.stage2.gen_countries, DET["n_train_per_gen"], DET["n_test_per_cond"], DET["n_val"],
                        DET["seed"])
    want = set(int(r) for r in sel["row"])
    ann = glob.glob(f"{DL}/annotations.tar.gz*")[0]
    boxes = {}
    with tarfile.open(ann, "r|*") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith("/object_detection.json")):
                continue
            fid = frame_id_from_path(m.name)
            if fid in want:
                boxes[fid] = boxes_from_features(json.loads(tf.extractfile(m).read()))
    missing = want - set(boxes)
    print(f"selected {len(want)} frames; annotations found for {len(boxes)}; missing {len(missing)}")
    sel = sel[sel["row"].isin(boxes)].reset_index(drop=True)          # a frame without an annotation file is unusable
    sel["n_boxes"] = [len(boxes[int(r)]) for r in sel["row"]]
    os.makedirs(OUT, exist_ok=True)
    sel.to_parquet(f"{OUT}/selection.parquet")
    pd.to_pickle({int(k): v for k, v in boxes.items()}, f"{OUT}/boxes.pkl")
    print(sel.groupby(["role", "gen"]).size().to_string())
    print("boxes per frame (mean):", float(sel["n_boxes"].mean()), "| frames with no box:", int((sel["n_boxes"] == 0).sum()))
    allb = np.concatenate([b for b in boxes.values() if len(b)])
    print("class counts:", {int(c): int((allb[:, 4] == c).sum()) for c in (1, 2, 3)})
    data.commit()


@app.function(image=image, volumes={D: data}, cpu=4, memory=16384, timeout=3 * 3600)
def part(i: int):
    import io
    import tarfile

    import numpy as np
    import pandas as pd
    from PIL import Image

    from zodfl.det_data import STORE_H, STORE_W, encode_jpeg, frame_id_from_path

    data.reload()
    sel = pd.read_parquet(f"{OUT}/selection.parquet")
    want = set(int(r) for r in sel["row"])
    import glob
    path = glob.glob(f"{DL}/{ARCHIVES[i]}*")[0]
    jpegs = {}
    with tarfile.open(path, "r|*") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".jpg")):
                continue
            fid = frame_id_from_path(m.name)
            if fid not in want:
                continue
            im = Image.open(io.BytesIO(tf.extractfile(m).read()))
            im.draft("RGB", (im.width // 2, im.height // 2))
            im = im.convert("RGB").resize((STORE_W, STORE_H), Image.BILINEAR)
            jpegs[fid] = encode_jpeg(np.asarray(im))
    import os
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    pd.to_pickle(jpegs, f"{OUT}/parts/part{i}.pkl")
    data.commit()
    print(f"part {i}: {len(jpegs)} images, {sum(len(v) for v in jpegs.values()) / 1e6:.0f} MB")
    return len(jpegs)


@app.function(image=image, volumes={D: data}, cpu=2, memory=16384, timeout=1800)
def finalize():
    import pandas as pd

    from zodfl.det_data import write_store

    data.reload()
    sel = pd.read_parquet(f"{OUT}/selection.parquet")
    boxes = pd.read_pickle(f"{OUT}/boxes.pkl")
    jpegs = {}
    for i in range(len(ARCHIVES)):
        jpegs.update(pd.read_pickle(f"{OUT}/parts/part{i}.pkl"))
    sel = sel[sel["row"].isin(jpegs)].reset_index(drop=True)
    rows = [int(r) for r in sel["row"]]
    meta = write_store(f"{OUT}/store", rows, jpegs, boxes, sel)
    data.commit()
    print(f"store: {len(meta)} frames, {meta['length'].sum() / 1e9:.2f} GB; roles {meta['role'].value_counts().to_dict()}")


@app.local_entrypoint()
def parts():
    print(list(part.map(range(len(ARCHIVES)))))

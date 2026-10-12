"""Look at ZOD's 2D object annotations and the original image size, to design the detection stage (ADR-6 style:
write the parser against the real format, not a guess).

    modal run modal_det_inspect.py::inspect
"""
import modal

app = modal.App("zod-det-inspect")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pillow", "numpy")
data = modal.Volume.from_name("zod-data", create_if_missing=True)
D = "/data"
DL = f"{D}/downloads/single_frames"


@app.function(image=image, volumes={D: data}, timeout=1800, cpu=2, memory=8192)
def inspect(n_frames: int = 1500):
    import collections
    import glob
    import io
    import json
    import tarfile

    from PIL import Image

    data.reload()
    ann = glob.glob(f"{DL}/annotations.tar.gz*")[0]
    geom, ncoord, classes, prop_keys, ex = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), []
    unclear, occl, sizes, n = collections.Counter(), [], [], 0
    with tarfile.open(ann, "r|*") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith("/object_detection.json")):
                continue
            feats = json.loads(tf.extractfile(m).read())
            n += 1
            for f in feats:
                g = f["geometry"]
                geom[g["type"]] += 1
                c = g["coordinates"]
                flat = c[0] if g["type"] == "Polygon" else c
                ncoord[len(flat)] += 1
                p = f["properties"]
                classes[p.get("class")] += 1
                unclear[str(p.get("unclear"))] += 1
                occl.append(str(p.get("occlusion_ratio")))
                for k in p:
                    prop_keys[k] += 1
                xs, ys = [q[0] for q in flat], [q[1] for q in flat]
                sizes.append((max(xs) - min(xs), max(ys) - min(ys), p.get("class")))
                if len(ex) < 2:
                    ex.append(f)
            if n >= n_frames:
                break
    print("frames read:", n, "| objects:", sum(classes.values()))
    print("geometry types:", dict(geom), "| coords per geometry:", dict(ncoord))
    print("classes:", dict(classes.most_common()))
    print("unclear:", dict(unclear), "| occlusion_ratio values:", dict(collections.Counter(occl).most_common(8)))
    print("property keys:", dict(prop_keys.most_common(14)))
    print("example features:", json.dumps(ex, indent=1)[:1800])
    import numpy as np

    for cls in ("Vehicle", "VulnerableVehicle", "Pedestrian"):
        s = np.array([(w, h) for w, h, c in sizes if c == cls])
        if len(s):
            print(f"{cls}: n={len(s)}, width px median={np.median(s[:, 0]):.0f}, p10={np.percentile(s[:, 0], 10):.0f}, "
                  f"height median={np.median(s[:, 1]):.0f}; share with width>=80px (8px at 384 wide)="
                  f"{(s[:, 0] >= 80).mean():.2f}")

    img = glob.glob(f"{DL}/images_blur_000000_049999.tar.gz*")[0]
    with tarfile.open(img, "r|*") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".jpg"):
                im = Image.open(io.BytesIO(tf.extractfile(m).read()))
                print("original image size (W,H):", im.size, "| name:", m.name)
                break

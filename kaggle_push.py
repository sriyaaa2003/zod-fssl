"""Push the shrunk ZOD images from the Modal volume to a PRIVATE Kaggle dataset (option B: no data touches the laptop).

    modal run kaggle_push.py::plan          # list what would be uploaded (no credentials needed, no upload)
    modal run kaggle_push.py::push_cli      # create (or version) the private dataset <you>/zod-fssl-shards

Needs the Modal secret `kaggle-credentials` holding KAGGLE_USERNAME and either KAGGLE_KEY (kaggle.json style) or
KAGGLE_API_TOKEN. Create it yourself with scripts/set_kaggle_secret.ps1 so the key never passes through anyone else.
The dataset is private by default. The images are resized, face/plate-blurred ZOD frames (CC BY-SA 4.0); keep the
dataset private, or comply with attribution and share-alike if you ever publish it.
"""
import modal

app = modal.App("zod-fssl-kaggle-push")
image = modal.Image.debian_slim(python_version="3.11").pip_install("kaggle")
data = modal.Volume.from_name("zod-data")
D = "/data"
SLUG = "zod-fssl-shards"
WEIGHTS = ("resnet18-f37072fd.pth", "mobilenet_v3_small-047dcff4.pth")


def dataset_meta(user: str) -> dict:
    """Kaggle limits: title 6-50 characters, subtitle 20-80 characters."""
    return {"title": "ZOD Frames shrunk for zod-fssl", "id": f"{user}/{SLUG}", "licenses": [{"name": "CC-BY-SA-4.0"}],
            "subtitle": "ZOD frames shrunk to 144x256 (uint8) with index and pretrained weights",
            "description": "Derived from the Zenseact Open Dataset (CC BY-SA 4.0), face/plate-blurred images, "
                           "resized. Private; used by the zod-fssl study."}


def _files() -> list[tuple[str, str]]:
    """(name in the dataset, path on the volume). Flat on purpose: the Kaggle CLI skips sub-directories."""
    import glob
    import os

    out = [(os.path.basename(p), p) for p in sorted(glob.glob(f"{D}/shards/shard_*.npy"))]
    out += [("manifest.json", f"{D}/shards/manifest.json"), ("index.parquet", f"{D}/index.parquet")]
    out += [(w, f"{D}/torch_cache/hub/checkpoints/{w}") for w in WEIGHTS]
    missing = [n for n, p in out if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing on the volume: {missing}")
    return out


def _authenticate() -> str:
    """Make the kaggle CLI inside the container accept the secret, trying each credential style in turn.
    Returns a label for the style that worked. Never prints a value."""
    import json
    import os
    import pathlib
    import subprocess

    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    cfg_dir = pathlib.Path.home() / ".kaggle"
    cfg_dir.mkdir(exist_ok=True)
    cfg_file = cfg_dir / "kaggle.json"
    attempts = []                                   # the token first: uploading files is refused with the old key
    if tok:
        attempts.append(("KAGGLE_API_TOKEN", "token", tok))
    if key:
        attempts.append(("KAGGLE_KEY used as an access token", "token", key))
        attempts.append(("legacy kaggle.json from KAGGLE_USERNAME + KAGGLE_KEY (listing works, uploads may be refused)",
                         "file", key))
    for label, how, value in attempts:
        os.environ.pop("KAGGLE_API_TOKEN", None)
        if cfg_file.exists():
            cfg_file.unlink()
        if how == "file":
            cfg_file.write_text(json.dumps({"username": user, "key": value}))
            cfg_file.chmod(0o600)
        else:
            os.environ["KAGGLE_API_TOKEN"] = value
        r = subprocess.run(["kaggle", "datasets", "list", "--mine"], capture_output=True, text=True)
        text = r.stdout + r.stderr
        print(f"auth attempt '{label}': exit {r.returncode}, accepted={r.returncode == 0 and 'Authentication required' not in text}")
        if r.returncode == 0 and "Authentication required" not in text:
            return label
    raise RuntimeError("Kaggle rejected every credential style stored in the secret kaggle-credentials; "
                       "recreate it with scripts/set_kaggle_secret.ps1 (check the username and the key/token)")


@app.function(image=image, volumes={D: data}, timeout=600)
def plan():
    import os

    data.reload()
    files = _files()
    total = sum(os.path.getsize(p) for _, p in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB")
    for n, p in files[:3] + files[-4:]:
        print(f"  {n:45s} {os.path.getsize(p) / 1e6:10.1f} MB")
    return {"files": len(files), "gb": round(total / 1e9, 2)}


@app.function(image=image, volumes={D: data}, secrets=[modal.Secret.from_name("kaggle-credentials")],
              cpu=2, memory=4096, timeout=4 * 3600)
def push(link: bool = True):
    import json
    import os
    import shutil
    import subprocess
    import tempfile

    data.reload()
    user = os.environ.get("KAGGLE_USERNAME", "").strip().lower()
    if not user:
        raise RuntimeError("the kaggle-credentials secret must contain KAGGLE_USERNAME")
    if not (os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")):
        raise RuntimeError("the kaggle-credentials secret must contain KAGGLE_KEY or KAGGLE_API_TOKEN")
    print("authenticated with:", _authenticate())
    bundle = tempfile.mkdtemp(prefix="bundle_")
    for name, src in _files():
        dst = os.path.join(bundle, name)
        if link:
            os.symlink(src, dst)
        else:
            shutil.copy(src, dst)
    meta = dataset_meta(user)
    with open(os.path.join(bundle, "dataset-metadata.json"), "w") as fh:
        json.dump(meta, fh)
    cmd = ["kaggle", "datasets", "create", "-p", bundle, "-t", "-r", "skip"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode != 0 and "already exist" in (r.stdout + r.stderr).lower():
        cmd = ["kaggle", "datasets", "version", "-p", bundle, "-m", "update", "-t", "-r", "skip"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode != 0:
        raise RuntimeError("kaggle CLI failed (see output above); if it mentions symlinks, rerun with --no-link")
    return f"{user}/{SLUG}"


@app.local_entrypoint()
def push_cli(no_link: bool = False):
    print(push.remote(link=not no_link))

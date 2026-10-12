"""Fetch one ZOD archive from Zenseact's official Dropbox share straight into the 'zod-data' Modal volume.

    modal run modal_dropbox.py::fetch --url "<dropbox file link>" --name sequences_mini.tar.gz

Nothing is written to the local machine. Approved by the user: sequences_mini.tar.gz (1.33 GB).
"""
import modal

app = modal.App("zod-dropbox")
image = modal.Image.debian_slim(python_version="3.11").apt_install("curl")
data = modal.Volume.from_name("zod-data", create_if_missing=True)
DATA_DIR = "/data"


@app.function(image=image, volumes={DATA_DIR: data}, timeout=3 * 3600, cpu=2)
def fetch(url: str, name: str = "sequences_mini.tar.gz"):
    import os
    import subprocess

    out_dir = f"{DATA_DIR}/zod_mini"
    os.makedirs(out_dir, exist_ok=True)
    dest = f"{out_dir}/{name}"
    url = url.replace("dl=0", "dl=1")
    subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", dest, url], check=True)
    size = os.path.getsize(dest)
    print("downloaded", dest, size / 1e6, "MB")
    subprocess.run(["tar", "tzf", dest], check=True, stdout=subprocess.DEVNULL)  # integrity check
    print("archive listing OK")
    data.commit()

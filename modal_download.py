"""Fetch ZOD into the 'zod-data' Modal volume.

    modal run modal_download.py::cli_help      # print the zod download options
    modal run modal_download.py::download      # needs the zod-download secret (ZOD_URL)

Self-contained on purpose: Modal only uploads the file you run.
"""
import modal

app = modal.App("zod-fssl-download")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "dataclass-wizard<1.0",  # zod 0.8.0 imports JSONSerializable, removed in 1.0
    "click",  # zod.cli imports click.exceptions; recent typer no longer pulls it in
    "zod[cli]",
)
data = modal.Volume.from_name("zod-data", create_if_missing=True)
DATA_DIR = "/data"


@app.function(image=image, timeout=300)
def cli_help():
    import subprocess

    subprocess.run(["zod", "download", "--help"], check=False)


@app.function(
    image=image,
    volumes={DATA_DIR: data},
    secrets=[modal.Secret.from_name("zod-download")],
    timeout=24 * 3600,
)
def download(subset: str = "frames", version: str = "mini", dry_run: bool = False):
    """Camera (blur) + annotations + infos only: no LiDAR, no OXTS, no DNAT copy."""
    import os
    import subprocess

    url = os.environ["ZOD_URL"]
    cmd = [
        "zod", "download", "-y", f"--url={url}", f"--output-dir={DATA_DIR}",
        f"--subset={subset}", f"--version={version}",
        "--images", "--blur", "--no-dnat", "--annotations", "--infos",
        "--no-lidar", "--no-oxts",
    ]
    if dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if not dry_run:
        data.commit()
    print("done:", subset, version, "dry_run =", dry_run)

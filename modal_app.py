"""Modal entry points for zod-fssl.

    modal run modal_app.py::smoke        # check GPU + imports (a few seconds on a T4)

Downloading ZOD lives in modal_download.py so this file runs without the download secret.
"""
import modal

app = modal.App("zod-fssl")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.1",
    "torchvision==0.19.1",
    "numpy",
    "pillow",
    "dataclass-wizard<1.0",  # zod 0.8.0 imports JSONSerializable, removed in 1.0
    "zod[cli]",
)

data = modal.Volume.from_name("zod-data", create_if_missing=True)
DATA_DIR = "/data"


@app.function(image=image, gpu="T4", timeout=300)
def smoke():
    import torch
    import torchvision

    print("torch", torch.__version__, "torchvision", torchvision.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        x = torch.randn(1024, 1024, device="cuda")
        print("matmul ok:", float((x @ x).abs().mean()))
    import zod  # noqa: F401

    print("zod devkit import ok")

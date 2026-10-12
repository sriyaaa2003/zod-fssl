"""Look inside the downloaded ZOD archives without unpacking them (streams the tar.gz).

    modal run modal_inspect.py::inspect

Prints archive sizes, the first member names, directory/suffix patterns, and a few small JSON files,
so the index/prep code is written against the real layout (ADR-6), not a guess.
"""
import modal

app = modal.App("zod-fssl-inspect")
image = modal.Image.debian_slim(python_version="3.11")
data = modal.Volume.from_name("zod-data", create_if_missing=True)
DATA_DIR = "/data"


@app.function(image=image, volumes={DATA_DIR: data}, timeout=3600, cpu=2, memory=4096)
def inspect(max_names: int = 25, json_samples: int = 4):
    import collections
    import glob
    import os
    import tarfile

    data.reload()
    archives = sorted(f for f in glob.glob(f"{DATA_DIR}/downloads/**/*", recursive=True) if os.path.isfile(f))
    print("archives:")
    for a in archives:
        print(f"  {os.path.getsize(a) / 1e9:8.2f} GB  {a}")

    for a in archives:
        print("\n" + "=" * 100 + f"\n{a}")
        small = os.path.getsize(a) < 3e9
        names, suffix, depth, samples = [], collections.Counter(), collections.Counter(), []
        n = 0
        with tarfile.open(a, "r|*") as tf:
            for m in tf:
                n += 1
                if len(names) < max_names:
                    names.append(m.name)
                if small:
                    suffix[os.path.splitext(m.name)[1]] += 1
                    depth[m.name.count("/")] += 1
                    if m.isfile() and m.name.endswith(".json") and len(samples) < json_samples and m.size < 20000:
                        samples.append((m.name, tf.extractfile(m).read(1500).decode("utf-8", "replace")))
                elif n >= max_names:  # big image archives: only the head
                    break
        print("first member names:")
        for nm in names:
            print("  ", nm)
        if small:
            print("members:", n, "| suffixes:", dict(suffix.most_common(8)), "| depth:", dict(depth))
            for nm, txt in samples:
                print(f"--- {nm}\n{txt}")

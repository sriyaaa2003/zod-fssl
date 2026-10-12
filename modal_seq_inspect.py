import modal
app = modal.App("zod-seq-inspect")
image = modal.Image.debian_slim(python_version="3.11")
data = modal.Volume.from_name("zod-data")

@app.function(image=image, volumes={"/data": data}, timeout=1800)
def ls():
    import tarfile, collections
    data.reload()
    c = collections.Counter(); ex = {}
    seqs = set()
    with tarfile.open("/data/zod_mini/sequences_mini.tar.gz", "r|gz") as tf:
        for m in tf:
            if not m.isfile(): continue
            parts = m.name.split("/")
            key = "/".join(p if not p.isdigit() else "N" for p in parts[:-1]) + "/*." + parts[-1].rsplit(".",1)[-1]
            c[key] += 1; ex.setdefault(key, m.name)
            if len(parts) > 2: seqs.add(parts[1])
    print("sequences:", len(seqs))
    for k, v in sorted(c.items()): print(v, k, "e.g.", ex[k])

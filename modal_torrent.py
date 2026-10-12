"""Fetch the ZOD camera images + annotations + infos from the official Academic Torrents
release (listed in Zenseact's access email) straight into the 'zod-data' Modal volume.
Nothing is downloaded to the local machine.

Method follows https://academictorrents.com/docs/downloading.html : aria2c on the .torrent file
(not the magnet link). The .torrent can carry "Backup URLs (WebSeeds)", plain HTTP sources that
aria2 uses automatically when BitTorrent peers are unavailable; a magnet link carries none.

    modal run modal_torrent.py::probe            # diagnose tracker, read .torrent, list files + webseeds
    modal run --detach modal_torrent.py::fetch   # download the 3 permitted files (~50 GB)

Permitted files (user approved 2026-10-10): images_front_blur.tar.gz, annotations.tar.gz, infos.tar.gz.
LiDAR, OXTS and the DNAT image copy are deliberately NOT downloaded.
"""
import modal

app = modal.App("zod-fssl-torrent")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("aria2", "curl")
    .pip_install("bencodepy", "requests")
)
data = modal.Volume.from_name("zod-data", create_if_missing=True)
DATA_DIR = "/data"
META_DIR = f"{DATA_DIR}/torrent_meta"
RAW_DIR = f"{DATA_DIR}/zod_raw"

INFOHASH = "329ae74426e067ef06b82241b5906400ca3caf03"
TORRENT_URL = f"https://academictorrents.com/download/{INFOHASH}.torrent"
MAGNET = (
    f"magnet:?xt=urn:btih:{INFOHASH}"
    "&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php"
    "&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
)
WANTED = ("images_front_blur.tar.gz", "annotations.tar.gz", "infos.tar.gz")

# HTTP(S) trackers first: UDP may be blocked from the container; the https opentrackr announce
# returned live peers in the tracker() diagnostic.
EXTRA_TRACKERS = ",".join([
    "https://tracker.opentrackr.org:443/announce",
    "http://tracker.opentrackr.org:1337/announce",
    "https://tracker.tamersunion.org:443/announce",
    "https://tracker.gbitt.info:443/announce",
    "https://tracker1.520.jp:443/announce",
    "http://tracker.files.fm:6969/announce",
    "http://open.acgnxtracker.com:80/announce",
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://open.stealth.si:80/announce",
])
NET_FLAGS = [
    "--enable-dht=true", "--enable-dht6=false", "--enable-peer-exchange=true",
    "--bt-enable-lpd=false", f"--bt-tracker={EXTRA_TRACKERS}", "--bt-max-peers=200",
]


def _dec(v):
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def _download_torrent_file() -> str:
    """Documented route: GET https://academictorrents.com/download/<hash>.torrent"""
    import os

    import requests

    os.makedirs(META_DIR, exist_ok=True)
    path = f"{META_DIR}/{INFOHASH}.torrent"
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        r = requests.get(TORRENT_URL, timeout=60)
        print("GET .torrent ->", r.status_code, r.headers.get("content-type"), len(r.content), "bytes")
        r.raise_for_status()
        if not r.content.startswith(b"d"):  # bencoded dict; an HTML page means a terms gate or error
            raise RuntimeError("response is not a .torrent file: " + r.content[:200].decode("utf-8", "replace"))
        with open(path, "wb") as fh:
            fh.write(r.content)
    return path


def _inspect(path: str) -> dict:
    import bencodepy

    meta = bencodepy.decode(open(path, "rb").read())
    info = meta[b"info"]
    files = []
    if b"files" in info:
        for i, f in enumerate(info[b"files"], start=1):  # aria2 indices are 1-based, in file order
            files.append((i, "/".join(_dec(p) for p in f[b"path"]), int(f[b"length"])))
    else:
        files.append((1, _dec(info[b"name"]), int(info[b"length"])))
    return {
        "name": _dec(info.get(b"name", b"")),
        "announce": _dec(meta.get(b"announce", b"")),
        "announce_list": [[_dec(t) for t in tier] for tier in meta.get(b"announce-list", [])],
        "url_list": [_dec(u) for u in (meta.get(b"url-list") if isinstance(meta.get(b"url-list"), list)
                                       else [meta[b"url-list"]] if b"url-list" in meta else [])],
        "httpseeds": [_dec(u) for u in meta.get(b"httpseeds", [])],
        "piece_length": int(info[b"piece length"]),
        "files": files,
    }


@app.function(image=image, timeout=300)
def tracker():
    """Ask the trackers directly: scrape (seeders/leechers) and announce (peer list)."""
    import os
    import urllib.parse

    import bencodepy
    import requests

    raw = bytes.fromhex(INFOHASH)
    q = urllib.parse.quote_from_bytes(raw)
    pid = urllib.parse.quote_from_bytes(b"-qB4650-" + os.urandom(6).hex().encode()[:12])
    for base in ("https://academictorrents.com", "https://tracker.opentrackr.org:443"):
        for kind, path in (("scrape", "/scrape.php" if "academic" in base else "/scrape"),
                           ("announce", "/announce.php" if "academic" in base else "/announce")):
            if kind == "scrape":
                url = f"{base}{path}?info_hash={q}"
            else:
                url = (f"{base}{path}?info_hash={q}&peer_id={pid}&port=51413&uploaded=0&downloaded=0"
                       f"&left=50000000000&compact=1&numwant=50&event=started")
            try:
                r = requests.get(url, timeout=30)
                body = r.content
                try:
                    d = bencodepy.decode(body)
                    show = {}
                    for k, v in d.items():
                        k = k.decode()
                        if k == b"peers".decode() and isinstance(v, bytes):
                            show[k] = f"{len(v) // 6} compact peers"
                        elif k == "files":
                            show[k] = {fk.hex(): {sk.decode(): sv for sk, sv in fv.items()} for fk, fv in v.items()}
                        else:
                            show[k] = v.decode("utf-8", "replace") if isinstance(v, bytes) else v
                    print(f"{base} {kind}: HTTP {r.status_code} ->", show)
                except Exception:  # noqa: BLE001
                    print(f"{base} {kind}: HTTP {r.status_code} raw ->", body[:200])
            except Exception as err:  # noqa: BLE001
                print(f"{base} {kind}: ERROR {err!r}")


@app.function(image=image, timeout=300)
def tracker_variants():
    """Does the Academic Torrents tracker hand out peers under different announce styles?"""
    import os
    import urllib.parse

    import bencodepy
    import requests

    q = urllib.parse.quote_from_bytes(bytes.fromhex(INFOHASH))
    ext = requests.get("https://api.ipify.org", timeout=20).text
    print("container public ip:", ext)
    variants = {
        "compact1_started": "compact=1&event=started&left=50000000000&numwant=200",
        "compact0_started": "compact=0&event=started&left=50000000000&numwant=200",
        "no_event": "compact=1&left=50000000000&numwant=200",
        "as_seeder": "compact=1&event=started&left=0&numwant=200",
        "no_compact_param": "event=started&left=50000000000&numwant=200",
    }
    for prefix in ("-qB4650-", "-TR4000-", "-lt0D80-", "-AR1750-"):
        for name, extra in variants.items():
            pid = urllib.parse.quote_from_bytes(prefix.encode() + os.urandom(6).hex().encode()[:12])
            url = (f"https://academictorrents.com/announce.php?info_hash={q}&peer_id={pid}"
                   f"&port=51413&uploaded=0&downloaded=0&{extra}")
            try:
                r = requests.get(url, timeout=30, headers={"User-Agent": prefix})
                d = bencodepy.decode(r.content)
                peers = d.get(b"peers")
                n = len(peers) // 6 if isinstance(peers, bytes) else len(peers or [])
                extra_keys = {k.decode(): (v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
                              for k, v in d.items() if k != b"peers"}
                print(f"{prefix} {name}: peers={n} {extra_keys}")
            except Exception as err:  # noqa: BLE001
                print(f"{prefix} {name}: ERROR {err!r}")


@app.function(image=image, volumes={DATA_DIR: data}, timeout=1800)
def probe():
    import subprocess

    print("== tracker test (FAQ: expect an 'Invalid info_hash' failure message if reachable)")
    subprocess.run(["curl", "-sS", "-m", "30", "https://academictorrents.com/announce.php"], check=False)
    print()
    print("== .torrent file")
    path = _download_torrent_file()
    data.commit()
    t = _inspect(path)
    print("name:", t["name"], "| piece length:", t["piece_length"], "| files:", len(t["files"]))
    print("announce:", t["announce"])
    print("announce-list:", t["announce_list"])
    print("WEBSEEDS (url-list):", t["url_list"])
    print("httpseeds:", t["httpseeds"])
    total = 0
    for idx, name, size in t["files"]:
        flag = "WANTED" if name.split("/")[-1] in WANTED else ""
        total += size if flag else 0
        print(f"{idx:>3} {size / 1e9:>10.2f} GB  {name}  {flag}")
    print(f"wanted total: {total / 1e9:.2f} GB")


@app.function(image=image, volumes={DATA_DIR: data}, timeout=24 * 3600, cpu=4, memory=8192)
def fetch():
    import os
    import subprocess
    import threading

    path = _download_torrent_file()
    t = _inspect(path)
    picked = [i for i, name, _ in t["files"] if name.split("/")[-1] in WANTED]
    if len(picked) != len(WANTED):
        raise RuntimeError(f"expected {len(WANTED)} files, matched {picked}")
    print("selected file indices:", picked, "| webseeds:", t["url_list"], flush=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    stop = threading.Event()

    def committer():  # persist progress so a restart can resume
        while not stop.wait(300):
            try:
                data.commit()
            except Exception as err:  # noqa: BLE001
                print("commit warning:", err, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    try:
        subprocess.run(
            ["aria2c", f"--select-file={','.join(map(str, picked))}", f"--dir={RAW_DIR}",
             "--seed-time=0", "--continue=true", "--file-allocation=none", "--check-integrity=true",
             "--summary-interval=120", "--console-log-level=notice", "--bt-stop-timeout=1800",
             "--max-connection-per-server=8", "--split=8",
             *NET_FLAGS, path],
            check=True,
        )
    finally:
        stop.set()
        data.commit()
    print("done. contents:", flush=True)
    subprocess.run(["bash", "-c", f"du -ah {RAW_DIR} | tail -n 20"], check=False)

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

from zodfl.jobs import job_name, load_jobs
from zodfl.kaggle_bundle import ARCHIVE, build, code_metadata, kernel_metadata, valid_username


def _import_worker():
    spec = importlib.util.spec_from_file_location("kaggle_worker", Path("kaggle/worker.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_usernames_are_validated():
    assert valid_username("sriya-c") and valid_username("a1_b2") and valid_username("saf2003")
    assert not valid_username("Bad Name!") and not valid_username("") and not valid_username("x")


def test_kernel_and_dataset_metadata():
    km = kernel_metadata("saf2003")
    assert km["dataset_sources"] == ["saf2003/zod-fssl-shards", "saf2003/zodfl-code"]
    assert km["enable_internet"] == "false", "the kernel must not need internet access"
    assert km["is_private"] == "true" and km["enable_gpu"] == "true" and km["kernel_type"] == "script"
    assert code_metadata("saf2003")["id"] == "saf2003/zodfl-code"


def test_code_dataset_is_one_archive_with_the_code_and_jobs_but_no_data_or_bytecode(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"kind": "reverse", "seed": 0}]))
    info = build("saf2003", out=tmp_path / "b", jobs_file=jobs)
    code, kern, archive = info["code"], info["kernel"], info["archive"]
    # the Kaggle tool skips sub-folders, so the dataset folder itself must contain only flat files
    assert sorted(p.name for p in code.iterdir()) == ["dataset-metadata.json", ARCHIVE]
    assert not _dirs_in(code)
    with tarfile.open(archive) as tf:
        names = tf.getnames()
    assert "jobs.json" in names and "config/default.toml" in names and "src/zodfl/jobs.py" in names
    assert not [n for n in names if "__pycache__" in n or n.endswith((".pyc", ".parquet", ".npy", ".pth"))]
    assert (kern / "worker.py").exists()
    meta = json.loads((kern / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "worker.py"
    with pytest.raises(ValueError):
        build("not a user!", out=tmp_path / "c", jobs_file=jobs)


def _dirs_in(folder: Path):
    return [p for p in folder.iterdir() if p.is_dir()]


def test_worker_unpacks_the_code_archive_and_finds_the_other_inputs(tmp_path):
    w = _import_worker()
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"kind": "reverse", "seed": 0}]))
    info = build("saf2003", out=tmp_path / "b", jobs_file=jobs)
    inp = tmp_path / "input"
    (inp / "shards").mkdir(parents=True)
    for name in ("manifest.json", "index.parquet", "resnet18-f37072fd.pth"):
        (inp / "shards" / name).write_text("x")
    (inp / "code").mkdir()
    (inp / "code" / ARCHIVE).write_bytes(info["archive"].read_bytes())
    w.INPUT, w.WORK = inp, tmp_path / "working"
    w.WORK.mkdir()
    p = w.locate()
    assert p["shards"] == inp / "shards" and len(p["weights"]) == 1
    assert (p["code"] / "src" / "zodfl" / "__init__.py").exists() and (p["code"] / "jobs.json").exists()
    w.INPUT = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="attached"):
        w.locate()


def test_remaining_job_list_in_the_repo_is_valid_and_unique():
    jobs = load_jobs("kaggle/jobs.json")
    assert len(jobs) == 31 and sum(j["kind"] == "reverse" for j in jobs) == 5
    names = [job_name(j) for j in jobs]
    assert len(set(names)) == len(names)


def test_modal_push_script_never_hardcodes_credentials_and_uses_a_private_default():
    src = Path("kaggle_push.py").read_text()
    assert "KAGGLE_KEY=" not in src
    assert '"-u"' not in src and "--public" not in src, "the dataset must never be created public"
    assert "kaggle-credentials" in src and "-t" in src


def test_dataset_metadata_respects_kaggle_length_limits():
    import kaggle_push
    for meta in (kaggle_push.dataset_meta("saf2003"), code_metadata("saf2003")):
        assert 6 <= len(meta["title"]) <= 50, meta["title"]
        assert 20 <= len(meta["subtitle"]) <= 80, (len(meta["subtitle"]), meta["subtitle"])
        assert meta["id"].split("/")[0] == "saf2003"
    assert kaggle_push.dataset_meta("saf2003")["licenses"] == [{"name": "CC-BY-SA-4.0"}]


def test_worker_also_accepts_the_code_already_unpacked_by_kaggle(tmp_path):
    w = _import_worker()
    inp = tmp_path / "input"
    (inp / "shards").mkdir(parents=True)
    for name in ("manifest.json", "index.parquet"):
        (inp / "shards" / name).write_text("x")
    (inp / "zodfl-code" / "src" / "zodfl").mkdir(parents=True)
    (inp / "zodfl-code" / "src" / "zodfl" / "__init__.py").write_text("")
    (inp / "zodfl-code" / "jobs.json").write_text("[]")
    w.INPUT, w.WORK = inp, tmp_path / "working"
    w.WORK.mkdir()
    p = w.locate()
    assert p["code"] == inp / "zodfl-code" and not (w.WORK / "code").exists()


def test_kernel_slug_matches_the_slug_kaggle_derives_from_the_title():
    import re
    km = kernel_metadata("saf2003")
    slug_from_title = re.sub(r"[^a-z0-9]+", "-", km["title"].lower()).strip("-")
    assert km["id"] == f"saf2003/{slug_from_title}"


def test_detection_kernel_slug_and_inputs():
    import re

    from zodfl.kaggle_bundle import det_kernel_metadata
    km = det_kernel_metadata("saf2003")
    assert km["id"] == "saf2003/" + re.sub(r"[^a-z0-9]+", "-", km["title"].lower()).strip("-")
    assert km["dataset_sources"] == ["saf2003/zod-det-store", "saf2003/zodfl-code"]
    assert km["enable_internet"] == "false" and km["is_private"] == "true" and km["code_file"] == "det_worker.py"


def test_build_adds_detection_job_and_kernel(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps([{"kind": "reverse", "seed": 0}]))
    dj = tmp_path / "det_jobs.json"
    dj.write_text(json.dumps({"seeds": [0], "split": "val"}))
    info = build("saf2003", out=tmp_path / "b", jobs_file=jobs, det_jobs_file=dj)
    with tarfile.open(info["archive"]) as tf:
        assert "det_jobs.json" in tf.getnames()
    assert (info["det_kernel"] / "det_worker.py").exists() and (info["det_kernel"] / "kernel-metadata.json").exists()

import json
import time

import pytest

from zodfl import config as C
from zodfl.jobs import (expected_files, hardware_info, is_done, job_name, load_jobs, pending, run_all, run_job,
                        split_jobs)
from zodfl.report import load_records
from conftest import H, W

GRID = {"kind": "grid", "seed": 0, "spec": {"tag": "tiny", "overrides": {"selection": {"kind": "random", "budget": 0.5}},
                                            "train": True, "centralized": False}}
TAX = {"kind": "grid", "seed": 1, "spec": {"tag": "tax", "overrides": {"eval": {"interp_alphas": [0.25, 0.5]}}}}
REV = {"kind": "reverse", "seed": 0}


def test_expected_files_cover_interpolation_and_reverse():
    assert expected_files(GRID) == ["tiny__seed0.json"]
    assert expected_files(TAX) == ["tax__seed1.json", "tax+interp0.25__seed1.json", "tax+interp0.5__seed1.json"]
    assert len(expected_files(REV)) == 6 and "rev_A_from_ensemble__seed0.json" in expected_files(REV)
    assert job_name(GRID) == "tiny/seed0" and job_name(REV) == "reverse/seed0"


def test_pending_ignores_finished_jobs_and_empty_files(tmp_path):
    (tmp_path / "tiny__seed0.json").write_text("{}")
    assert is_done(GRID, tmp_path) and not is_done(TAX, tmp_path)
    (tmp_path / "tax__seed1.json").write_text("")                      # a truncated file does not count as done
    assert [job_name(j) for j in pending([GRID, TAX, REV], tmp_path)] == ["tax/seed1", "reverse/seed0"]


def test_split_jobs_is_a_partition():
    jobs = [{"kind": "reverse", "seed": i} for i in range(7)]
    parts = [split_jobs(jobs, 3, i) for i in range(3)]
    assert sorted(j["seed"] for p in parts for j in p) == list(range(7))
    with pytest.raises(ValueError):
        split_jobs(jobs, 2, 2)


def test_load_jobs_validates(tmp_path):
    p = tmp_path / "j.json"
    p.write_text(json.dumps([GRID, REV]))
    assert len(load_jobs(p)) == 2
    p.write_text(json.dumps([{"kind": "other", "seed": 0}]))
    with pytest.raises(ValueError):
        load_jobs(p)
    p.write_text(json.dumps([{"kind": "grid", "seed": 0, "spec": {}}]))
    with pytest.raises(ValueError):
        load_jobs(p)


def test_run_all_runs_grid_and_interpolation_jobs_and_records_hardware(world, cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("ZODFL_PROVIDER", "unit-test")
    df, store = world
    small = C.override(cfg, eval={"probe_fractions": [1.0]})
    out = run_all([GRID, TAX], small, df, store, "cpu", tmp_path, log=lambda *_: None)
    assert out["done"] == ["tiny/seed0", "tax/seed1"] and not out["failed"]
    recs = load_records(tmp_path)
    assert sorted(r["tag"] for r in recs) == ["tax", "tax+interp0.25", "tax+interp0.5", "tiny"]
    assert all(r["hardware"]["provider"] == "unit-test" and r["hardware"]["gpu"] is None for r in recs)
    again = run_all([GRID, TAX], small, df, store, "cpu", tmp_path, log=lambda *_: None)   # idempotent
    assert again["done"] == [] and len(again["skipped"]) == 2


def test_failures_are_recorded_and_do_not_stop_other_jobs(world, cfg, tmp_path):
    df, store = world
    bad = {"kind": "grid", "seed": 0, "spec": {"tag": "bad", "overrides": {"selection": {"kind": "nonsense"}}}}
    out = run_all([bad, GRID], C.override(cfg, eval={"probe_fractions": [1.0]}), df, store, "cpu", tmp_path,
                  log=lambda *_: None)
    assert out["done"] == ["tiny/seed0"] and out["failed"][0]["job"] == "bad/seed0"


def test_deadline_stops_new_jobs_but_not_finished_ones(world, cfg, tmp_path):
    df, store = world
    out = run_all([GRID], C.override(cfg, eval={"probe_fractions": [1.0]}), df, store, "cpu", tmp_path,
                  deadline=time.time() - 1, log=lambda *_: None)
    assert out["done"] == [] and out["skipped"] == ["tiny/seed0 (deadline)"]


def test_reverse_job_through_the_portable_runner(world, cfg, tmp_path):
    df, store = world
    g = C.override(
        cfg,
        stage2={"gen_countries": [["SE"], ["DE"], ["NO"]], "archs": ["resnet18", "mobilenet_v3_small", "mobilenet_v3_small"],
                "sensor_hw": [[H, W], [24, 48], [16, 32]], "sensor_fov": [1.0, 0.9, 0.8], "quant_bits": [0, 0, 8],
                "n_rounds": 2, "passes": 1, "batch": 16, "pretrained": False},
    )
    paths = run_job(REV, g, df, store, "cpu", tmp_path, log=lambda *_: None)
    assert len(paths) == 6 and is_done(REV, tmp_path)


def test_hardware_info_without_gpu():
    info = hardware_info("cpu")
    assert info["gpu"] is None and "torch" in info

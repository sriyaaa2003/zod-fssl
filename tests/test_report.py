from zodfl import config as C
from zodfl.federated import run, save_record
from zodfl.report import load_records, render, summarize


def test_report_renders_from_json_only(world, cfg, tmp_path):
    df, store = world
    for tag, kind in (("base", "random"), ("alt", "loss")):
        c = C.override(cfg, selection={"kind": kind, "budget": 0.5})
        for seed in (0, 1):
            save_record(run(c, df, store, seed=seed, device="cpu", tag=tag, log=lambda *_: None), tmp_path)
    recs = load_records(tmp_path)
    assert len(recs) == 4
    s = summarize(recs[0], "night", "by_cond")
    assert 0 <= s["overall"] <= 1
    out = render(tmp_path, tmp_path / "REPORT.md", base_tag="base")
    text = out.read_text()
    assert "| base | 2 |" in text and "| alt | 2 |" in text and "Paired differences" in text
    assert "95% CI" in text


def test_readme_block_is_generated_and_inserted(world, cfg, tmp_path):
    from zodfl.readme_block import END, START, build_block, insert

    df, store = world
    c0 = C.override(cfg, eval={"tasks": [{"name": "vru_present", "label_col": "label"}], "probe_fractions": [1.0, 0.1]})
    for tag, kind in (("fedavg_random20", "random"), ("fedavg_loss20", "loss")):
        c = C.override(c0, selection={"kind": kind, "budget": 0.5})
        for seed in (0, 1):
            save_record(run(c, df, store, seed=seed, device="cpu", tag=tag, log=lambda *_: None), tmp_path)
    block = build_block(tmp_path)
    assert "vru_present@10pct" in block and "fedavg_loss20" in block and "95% CI" in block
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold\n{END}\noutro\n")
    insert(readme, block)
    text = readme.read_text()
    assert text.startswith("intro\n") and text.rstrip().endswith("outro") and "old" not in text and "worst-group" in text


def test_pilot_runs_are_excluded_from_results(world, cfg, tmp_path):
    df, store = world
    save_record(run(cfg, df, store, seed=100, device="cpu", tag="pilot_m0.1", log=lambda *_: None), tmp_path)
    save_record(run(cfg, df, store, seed=0, device="cpu", tag="real", log=lambda *_: None), tmp_path)
    assert [r["tag"] for r in load_records(tmp_path)] == ["real"]

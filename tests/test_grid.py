import pytest

from zodfl import config as C
from zodfl.grid import RunSpec, config_for, load_grid


def test_default_grid_loads_and_every_run_builds_a_valid_config():
    base = C.load("config/default.toml")
    specs = load_grid("config/grid.toml")
    assert len(specs) >= 10 and len({s.tag for s in specs}) == len(specs)
    for s in specs:
        cfg = config_for(base, s)
        if s.overrides.get("selection", {}).get("kind") == "all":
            assert cfg.selection.budget == 1.0


def test_grid_contains_the_references_and_the_full_selector_aggregator_cross():
    tags = {s.tag for s in load_grid("config/grid.toml")}
    assert {"init", "central_all", "fedavg_all", "balanced_all"} <= tags
    for sel in ("random", "loss", "novelty", "rarity"):
        for agg in ("fedavg", "balanced"):
            assert f"{agg}_{sel}20" in tags


def test_unknown_override_section_raises():
    with pytest.raises(ValueError):
        config_for(C.Config(), RunSpec(tag="x", overrides={"nonsense": {"a": 1}}))


def test_duplicate_tags_rejected(tmp_path):
    p = tmp_path / "g.toml"
    p.write_text('[[run]]\ntag="a"\n[[run]]\ntag="a"\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_grid(p)


def test_phase2_grid_builds_valid_configs_and_has_no_overlap_with_stage1_tags():
    base = C.load("config/default.toml")
    p2 = load_grid("config/grid_phase2.toml")
    p1_tags = {s.tag for s in load_grid("config/grid.toml")}
    assert len(p2) == 16 and not ({s.tag for s in p2} & p1_tags)
    for s in p2:
        cfg = config_for(base, s)
        if s.tag.startswith("tax_"):
            assert cfg.eval.interp_alphas == [0.25, 0.5, 0.75]
        if s.tag.startswith("layer4only"):
            assert cfg.federation.train_layers == ["layer4"]
        if s.tag.startswith("anchor"):
            assert cfg.federation.anchor_mu > 0
        if s.tag.startswith(("ldawa", "fedema")):
            assert cfg.aggregation.kind in ("ldawa", "ldawa_norm", "fedema")

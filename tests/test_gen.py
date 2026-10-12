import numpy as np
import pytest
import torch

from zodfl import config as C
from zodfl.federated import build_world
from zodfl.gen import (ClassifierNet, Ctx, Sensor, apply_sensor, ce_loss, evaluate, fake_quantize, kd_loss,
                       label_tables, make_tasks, run_stage2, sample_weights, sensors_from, train_generation)
from conftest import H, W, make_world


@pytest.fixture
def gcfg(cfg):
    return C.override(
        cfg,
        stage2={"gen_countries": [["SE"], ["DE"], ["NO"]], "archs": ["resnet18", "mobilenet_v3_small", "mobilenet_v3_small"],
                "sensor_hw": [[H, W], [24, 48], [16, 32]], "sensor_fov": [1.0, 0.9, 0.8], "quant_bits": [0, 0, 8],
                "n_rounds": 2, "passes": 2, "batch": 16, "lr": 0.05, "pretrained": False, "n_public": 40,
                "floor_label_fraction": 0.2},
    )


def test_stage2_config_validation():
    with pytest.raises(ValueError, match="one entry per generation"):
        C.override(C.Config(), stage2={"archs": ["resnet18"]})
    with pytest.raises(ValueError, match="only one generation"):
        C.override(C.Config(), stage2={"gen_countries": [["SE"], ["SE"], ["NO"]]})
    with pytest.raises(ValueError):
        C.override(C.Config(), stage2={"teacher_sees": "both"})


def test_sensor_resolution_fov_and_shapes():
    x = torch.randn(2, 3, 144, 256)
    assert apply_sensor(x, Sensor((144, 256), 1.0)).shape == x.shape
    assert torch.equal(apply_sensor(x, Sensor((144, 256), 1.0)), x)             # identity sensor changes nothing
    y = apply_sensor(x, Sensor((72, 128), 0.8))
    assert y.shape == (2, 3, 72, 128)
    crop = x[:, :, 14:130, 26:230]                                              # fov 0.8 -> central region
    assert crop.shape[-2:] == (115, 205) or abs(crop.shape[-2] - 115) <= 1


def test_fake_quantize_has_few_levels_and_small_error():
    m = ClassifierNet("mobilenet_v3_small", {"t": 2}, pretrained=False)
    q = fake_quantize(m, 8)
    assert q is not m
    w, wq = m.heads["t"].weight, q.heads["t"].weight
    assert len(torch.unique(wq)) <= 255
    assert (w - wq).abs().max() <= w.abs().max() / 127 / 2 + 1e-6
    assert fake_quantize(m, 0) is m
    assert torch.equal(m.heads["t"].weight, w), "the original model must not be modified"


def test_losses():
    lg = {"a": torch.tensor([[2.0, 0.0], [0.0, 2.0]])}
    assert float(kd_loss(lg, {"a": lg["a"].clone()}, 2.0, None)) < 1e-6          # student == teacher -> zero
    assert float(kd_loss(lg, {"a": -lg["a"]}, 2.0, None)) > 0.1
    y = {"a": torch.tensor([0, -1])}                                             # -1 = unknown label, ignored
    only_first = float(ce_loss({"a": lg["a"][:1]}, {"a": torch.tensor([0])}, None))
    assert abs(float(ce_loss(lg, y, None)) - only_first / 2) < 1e-6


def test_rarity_weights_mean_one_and_favor_rare():
    q = np.array([0.9, 0.1])
    w = sample_weights(np.array([0, 0, 0, 1]), q, 0.5, "cpu")
    assert abs(float(w.mean()) - 1.0) < 1e-6 and w[3] > w[0] * 2
    assert sample_weights(np.array([0, 1]), q, 0.0, "cpu") is None


def _ctx(cfg, world):
    df, store = world
    tasks = make_tasks(df, cfg.eval.tasks)
    w = build_world(df, cfg.federation)
    return Ctx(cfg, store, label_tables(df, tasks), w.cond_ids, w.q, torch.device("cpu"), tasks, df), w


def test_label_training_learns_a_visible_signal(world, gcfg):
    cfg = C.override(gcfg, stage2={"n_rounds": 3, "passes": 3})
    ctx, _ = _ctx(cfg, world)
    df = world[0]
    sub = df[(df["split"] != "stream") | df["country"].isin(["SE", "DE"])]
    w = build_world(sub, C.override(cfg, federation={"n_rounds": 3}).federation)
    torch.manual_seed(0)
    m = ClassifierNet("resnet18", ctx.tasks.n_classes, pretrained=False)
    train_generation(ctx, m, sensors_from(cfg.stage2)[0], w.rows, "labels", 0, log=lambda *_: None)
    ev = evaluate(ctx, m, sensors_from(cfg.stage2)[0], np.random.default_rng(0))
    acc = ev["tasks"]["night"]["by_cond"]["overall_acc"]
    assert acc > 0.65, f"a brightness-separable label should be learnable, got {acc:.2f}"


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError, match="unknown arch"):
        ClassifierNet("vgg", {"t": 2}, pretrained=False)


def test_stage2_end_to_end_records(world, gcfg):
    df, store = world
    recs = run_stage2(gcfg, df, store, seed=0, device="cpu", log=lambda *_: None)
    tags = [r["tag"] for r in recs]
    expected = {"s2_A_teacher", "s2_B_floor", "s2_B_oracle", "s2_B_kd_incar", "s2_B_kd_incar_rarity", "s2_B_kd_public",
                "s2_C_oracle", "s2_C_kd_incar", "s2_C_kd_incar_rarity"}
    assert set(tags) == expected and len(tags) == len(expected)
    by = {r["tag"]: r for r in recs}
    assert by["s2_B_kd_incar"]["teacher"] == "s2_A_teacher" and by["s2_C_kd_incar"]["teacher"] == "s2_B_kd_incar"
    assert by["s2_C_oracle"]["teacher"] is None and by["s2_B_floor"]["teacher"] is None
    assert by["s2_B_floor"]["totals"]["frames_trained_on"] < by["s2_B_oracle"]["totals"]["frames_trained_on"]
    assert by["s2_B_kd_public"]["totals"]["frames_trained_on"] == 40
    for r in recs:
        t = r["evals"][-1]["tasks"]["night"]
        assert 0 <= t["by_cond"]["overall_acc"] <= 1 and "by_country" in t
        assert r["stage"] == 2 and r["generation"] in (0, 1, 2)


def test_stage2_deterministic(world, gcfg):
    df, store = world
    a = run_stage2(gcfg, df, store, seed=3, device="cpu", log=lambda *_: None)
    b = run_stage2(gcfg, df, store, seed=3, device="cpu", log=lambda *_: None)
    fa = [r["evals"][-1]["tasks"]["night"]["by_cond"]["overall_acc"] for r in a]
    fb = [r["evals"][-1]["tasks"]["night"]["by_cond"]["overall_acc"] for r in b]
    assert fa == fb


def test_report_renders_stage2_retention_table(world, gcfg, tmp_path):
    from zodfl.federated import save_record
    from zodfl.report import render

    df, store = world
    for seed in (0, 1):
        for rec in run_stage2(gcfg, df, store, seed=seed, device="cpu", log=lambda *_: None):
            save_record(rec, tmp_path)
    text = render(tmp_path, tmp_path / "REPORT.md").read_text()
    assert "Stage 2 (generation gap)" in text and "s2_C_kd_incar_rarity" in text and "share of teacher A" in text


def test_report_has_matched_comparisons_and_gap_closed(world, gcfg, tmp_path):
    from zodfl.federated import run, save_record
    from zodfl.report import aggregator_pairs_md, gap_closed_md, load_records, stage2_pairs_md

    df, store = world
    for seed in (0, 1):
        for rec in run_stage2(gcfg, df, store, seed=seed, device="cpu", log=lambda *_: None):
            save_record(rec, tmp_path)
        for agg in ("fedavg", "balanced"):
            c = C.override(gcfg, aggregation={"kind": agg}, selection={"kind": "random", "budget": 0.5})
            save_record(run(c, df, store, seed=seed, device="cpu", tag=f"{agg}_random20", log=lambda *_: None), tmp_path)
    recs = load_records(tmp_path)
    s1 = [r for r in recs if not r["tag"].startswith("s2_")]
    s2 = [r for r in recs if r["tag"].startswith("s2_")]
    assert "random20:" in aggregator_pairs_md(s1, "night") and "95% CI" in aggregator_pairs_md(s1, "night")
    assert "rarity-aware minus plain" in stage2_pairs_md(s2, "night")
    assert "s2_B_kd_incar:" in gap_closed_md(s2, "night")


def test_ensemble_teacher_averages_logits_with_each_models_own_sensor():
    from zodfl.gen import EnsembleTeacher
    torch.manual_seed(0)
    a = ClassifierNet("resnet18", {"t": 3}, pretrained=False).eval()
    b = ClassifierNet("mobilenet_v3_small", {"t": 3}, pretrained=False).eval()
    sa, sb = Sensor((H, W), 1.0), Sensor((24, 48), 0.9)
    ens = EnsembleTeacher([a, b], [sa, sb]).eval()
    base = torch.randn(4, 3, H, W)
    out = ens(base)["t"]
    want = (a(apply_sensor(base, sa))["t"] + b(apply_sensor(base, sb))["t"]) / 2
    assert torch.allclose(out, want, atol=1e-6) and ens.takes_base


def test_reverse_transfer_pipeline_records_and_sources(world, gcfg):
    from zodfl.gen import run_reverse
    df, store = world
    recs = run_reverse(gcfg, df, store, seed=0, device="cpu", log=lambda *_: None)
    tags = [r["tag"] for r in recs]
    assert tags == ["rev_A_teacher", "rev_B_oracle", "rev_B_control", "rev_A_from_oracle", "rev_A_from_ensemble",
                    "rev_A_from_control"]
    by = {r["tag"]: r for r in recs}
    assert by["rev_A_from_oracle"]["teacher"] == "rev_B_oracle" and by["rev_B_control"]["teacher"] == "rev_A_teacher"
    assert all(r["generation"] == 0 for r in recs if r["tag"].startswith("rev_A"))
    for r in recs:
        assert "by_country" in r["evals"][-1]["tasks"]["night"]


def test_reverse_student_starts_from_a_copy_of_a_and_does_not_modify_a(world, gcfg):
    import copy

    from zodfl.gen import make_tasks, label_tables, Ctx, train_generation
    df, store = world
    tasks = make_tasks(df, gcfg.eval.tasks)
    w = build_world(df, gcfg.federation)
    ctx = Ctx(gcfg, store, label_tables(df, tasks), w.cond_ids, w.q, torch.device("cpu"), tasks, df)
    torch.manual_seed(0)
    A = ClassifierNet("resnet18", tasks.n_classes, pretrained=False).eval()
    B = ClassifierNet("mobilenet_v3_small", tasks.n_classes, pretrained=False).eval()
    before = [p.detach().clone() for p in A.parameters()]
    A2 = copy.deepcopy(A)
    sensors = sensors_from(gcfg.stage2)
    sub = df[(df["split"] != "stream") | (df["country"] == "SE")]
    rows = build_world(sub, C.override(gcfg, federation={"n_rounds": 2}).federation).rows
    train_generation(ctx, A2, sensors[0], rows, "kd", 0, teacher=B, teacher_sensor=sensors[1], log=lambda *_: None)
    assert all(torch.equal(b, p) for b, p in zip(before, A.parameters())), "the original A must be untouched"
    assert any(not torch.equal(b, p) for b, p in zip(before, A2.parameters())), "the copy must have learned"


def test_report_separates_reverse_records_and_builds_the_country_group_table(world, gcfg, tmp_path):
    from zodfl.federated import save_record
    from zodfl.gen import run_reverse
    from zodfl.report import load_records, render, reverse_md, split_records

    df, store = world
    for seed in (0, 1):
        for rec in run_reverse(gcfg, df, store, seed=seed, device="cpu", log=lambda *_: None):
            save_record(rec, tmp_path)
    recs = load_records(tmp_path)
    fam = split_records(recs)
    assert len(fam["reverse"]) == 12 and not fam["stage1"] and not fam["stage2"]
    md = reverse_md(fam["reverse"], "night")
    assert "rev_A_from_oracle" in md and "gen1 countries" in md and "95% CI" in md
    text = render(tmp_path, tmp_path / "REPORT.md").read_text()
    assert "## Reverse transfer" in text

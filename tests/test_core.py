import numpy as np
import pytest
import torch

from zodfl import aggregate, config as C, metrics
from zodfl.augment import augment
from zodfl.client import client_round
from zodfl.federated import build_world, run
from zodfl.model import SimSiamModel, get_vector, set_vector
from zodfl.selectors import Selector, n_to_keep
from conftest import H, W, make_world


# ---------------------------------------------------------------- config
def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        C.override(C.Config(), selection={"kind": "magic"})
    with pytest.raises(ValueError):
        C.override(C.Config(), selection={"kind": "all", "budget": 0.5})
    with pytest.raises(ValueError):
        C.override(C.Config(), federation={"train_batch": 32, "max_buffer": 64})
    with pytest.raises(ValueError):
        C.from_dict({"nope": {}})


def test_default_toml_loads():
    cfg = C.load("config/default.toml")
    assert cfg.selection.kind == "random" and cfg.federation.n_rounds == 24


# ---------------------------------------------------------------- augment / model
def test_augment_shape_range_and_determinism():
    x = torch.randint(0, 255, (6, 3, H, W), dtype=torch.uint8)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    a, b = augment(x, g1), augment(x, g2)
    assert a.shape == (6, 3, H, W) and torch.allclose(a, b)
    c = augment(x, torch.Generator().manual_seed(2))
    assert not torch.allclose(a, c)


def test_set_vector_does_not_alias():
    m = SimSiamModel(pretrained=False)
    v = get_vector(m)
    ref = v.clone()
    set_vector(m, v)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    x = torch.randint(0, 255, (4, 3, H, W), dtype=torch.uint8)
    g = torch.Generator().manual_seed(0)
    m.train()
    loss = m.pair_loss(augment(x, g), augment(x, g)).mean()
    loss.backward()
    opt.step()
    assert torch.equal(v, ref), "local training must not modify the global vector"
    assert not torch.equal(get_vector(m), ref)


def test_batchnorm_stays_eval_and_frozen():
    m = SimSiamModel(pretrained=False)
    m.train()
    bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    assert bns and all(not b.training for b in bns)
    assert all(not p.requires_grad for b in bns for p in b.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert get_vector(m).numel() == n_train


# ---------------------------------------------------------------- selectors
@pytest.mark.parametrize("kind", ["all", "random", "loss", "novelty", "rarity"])
def test_selector_budget_and_sorted(kind):
    m = SimSiamModel(pretrained=False)
    x = torch.randint(0, 255, (40, 3, H, W), dtype=torch.uint8)
    cond = np.zeros(40, dtype=np.int64)
    sel = Selector(kind, 0.25, "cpu", q=np.array([0.9, 0.1]))
    idx = sel.select(m, x, cond, torch.Generator().manual_seed(0))
    assert len(idx) == n_to_keep(kind, 40, 0.25)
    assert torch.equal(idx, torch.sort(idx).values) and len(set(idx.tolist())) == len(idx)


def test_loss_selector_picks_the_highest_loss_frames():
    torch.manual_seed(0)
    m = SimSiamModel(pretrained=False).eval()
    x = torch.randint(0, 255, (32, 3, H, W), dtype=torch.uint8)
    sel = Selector("loss", 0.25, "cpu")
    idx = sel.select(m, x, np.zeros(32, dtype=np.int64), torch.Generator().manual_seed(5))
    with torch.no_grad():
        g = torch.Generator().manual_seed(5)
        score = m.pair_loss(augment(x, g), augment(x, g))
    assert set(idx.tolist()) == set(torch.topk(score, 8).indices.tolist())


def test_rarity_selector_prefers_rare_conditions():
    m = SimSiamModel(pretrained=False)
    n = 400
    cond = np.array([0] * 380 + [1] * 20)                     # condition 1 is rare in the batch
    x = torch.randint(0, 255, (n, 3, H, W), dtype=torch.uint8)
    q = np.array([0.95, 0.05])
    hits = 0
    for s in range(5):
        idx = Selector("rarity", 0.1, "cpu", q=q).select(m, x, cond, torch.Generator().manual_seed(s))
        hits += int((cond[idx.numpy()] == 1).sum())
    random_expected = 5 * 40 * (20 / 400)
    assert hits > 2 * random_expected


def test_random_selector_is_reproducible():
    m = SimSiamModel(pretrained=False)
    x = torch.randint(0, 255, (30, 3, H, W), dtype=torch.uint8)
    a = Selector("random", 0.3, "cpu").select(m, x, np.zeros(30, dtype=np.int64), torch.Generator().manual_seed(3))
    b = Selector("random", 0.3, "cpu").select(m, x, np.zeros(30, dtype=np.int64), torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


# ---------------------------------------------------------------- aggregation
def test_fedavg_weights():
    w = aggregate.weights_fedavg(np.array([10, 30, 60]))
    assert np.allclose(w, [0.1, 0.3, 0.6]) and np.isclose(w.sum(), 1)
    with pytest.raises(ValueError):
        aggregate.weights_fedavg(np.array([0, 0]))


def test_balanced_gives_each_condition_equal_total_weight_when_clients_are_homogeneous():
    # clients 0,1 only saw condition 0 (common); client 2 only saw condition 1 (rare, few frames)
    counts = np.array([[500, 0], [300, 0], [0, 20]])
    w = aggregate.weights_balanced(counts)
    assert np.isclose(w.sum(), 1)
    assert np.isclose(w[0] + w[1], 0.5) and np.isclose(w[2], 0.5)       # equal mass per condition
    fed = aggregate.weights_fedavg(counts.sum(axis=1))
    assert fed[2] < 0.03                                                 # FedAvg drowns the rare client


def test_balanced_ignores_conditions_nobody_trained_on():
    w = aggregate.weights_balanced(np.array([[5, 0, 0], [5, 0, 0]]))
    assert np.allclose(w, [0.5, 0.5])


def test_apply_is_weighted_sum_and_does_not_mutate_input():
    g = torch.zeros(4)
    out = aggregate.apply(g, [torch.ones(4), 3 * torch.ones(4)], np.array([0.25, 0.75]))
    assert torch.allclose(out, torch.full((4,), 2.5)) and torch.equal(g, torch.zeros(4))


# ---------------------------------------------------------------- world + client
def test_stream_and_probe_frames_are_disjoint(world, cfg):
    df, _ = world
    w = build_world(df, cfg.federation)
    stream_rows = set(np.concatenate(list(w.rows.values())).tolist())
    probe_rows = set(df[df["split"] != "stream"]["row"].tolist())
    assert stream_rows and probe_rows and not (stream_rows & probe_rows)


def test_rounds_follow_time_order(world, cfg):
    df, _ = world
    w = build_world(df, cfg.federation)
    ts = df.set_index("row")["ts"]
    for (_, _r), rows in w.rows.items():
        assert np.all(np.diff(ts.loc[rows].to_numpy()) >= 0)


class CountingStore:
    def __init__(self, inner):
        self.inner, self.reads = inner, {}

    def get(self, rows):
        for r in np.asarray(rows).tolist():
            self.reads[r] = self.reads.get(r, 0) + 1
        return self.inner.get(rows)


@pytest.mark.parametrize("kind,budget", [("all", 1.0), ("random", 0.3), ("loss", 0.3)])
def test_client_single_pass_bounded_buffer(world, cfg, kind, budget):
    df, store = world
    cfg = C.override(cfg, selection={"kind": kind, "budget": budget})
    w = build_world(df, cfg.federation)
    rows = np.concatenate([v for (k, _), v in w.rows.items() if k == 0])
    cs = CountingStore(store)
    m = SimSiamModel(pretrained=False)
    res = client_round(m, get_vector(m), rows, cs, w.cond_ids, len(w.cond_names),
                       Selector(kind, budget, "cpu", q=w.q), cfg.federation, "cpu", torch.Generator().manual_seed(0))
    assert set(cs.reads) == set(rows.tolist()) and max(cs.reads.values()) == 1          # each frame read once
    assert res.max_buffer_seen <= cfg.federation.max_buffer
    assert res.n_seen == len(rows) and 0 < res.n_kept <= len(rows)
    assert res.cond_counts.sum() == res.n_kept
    assert res.delta is not None and res.delta.shape == get_vector(m).shape
    assert np.isfinite(res.losses).all()


def test_client_with_too_few_kept_frames_returns_no_delta(world, cfg):
    df, store = world
    cfg = C.override(cfg, selection={"kind": "random", "budget": 0.05})
    w = build_world(df, cfg.federation)
    rows = w.rows[(0, 0)][:16]                                   # one arrival batch -> 1 kept frame
    m = SimSiamModel(pretrained=False)
    res = client_round(m, get_vector(m), rows, store, w.cond_ids, len(w.cond_names),
                       Selector("random", 0.05, "cpu"), cfg.federation, "cpu", torch.Generator().manual_seed(0))
    assert res.n_steps == 0 and res.delta is None


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("agg", ["fedavg", "balanced"])
def test_end_to_end_run_produces_complete_record(world, cfg, agg):
    df, store = world
    cfg = C.override(cfg, aggregation={"kind": agg})
    rec = run(cfg, df, store, seed=0, device="cpu", tag="t", log=lambda *_: None)
    assert [e["round"] for e in rec["evals"]] == [0, 3]
    assert len(rec["rounds"]) == 3
    assert rec["totals"]["frames_trained_on"] > 0 and rec["totals"]["bytes_communicated"] > 0
    t = rec["evals"][-1]["tasks"]["night"]
    assert "by_cond" in t and "by_country" in t and 0 <= t["by_cond"]["overall_acc"] <= 1
    assert rec["evals"][-1]["feature_std_normalized"] > 0
    for rd in rec["rounds"]:
        assert abs(sum(rd["weights"].values()) - 1) < 1e-9


def test_run_is_deterministic_for_a_seed(world, cfg):
    df, store = world
    a = run(cfg, df, store, seed=4, device="cpu", tag="a", log=lambda *_: None)
    b = run(cfg, df, store, seed=4, device="cpu", tag="a", log=lambda *_: None)
    assert [r["mean_loss"] for r in a["rounds"]] == [r["mean_loss"] for r in b["rounds"]]
    assert a["evals"][-1]["tasks"]["night"]["by_cond"]["overall_acc"] == b["evals"][-1]["tasks"]["night"]["by_cond"]["overall_acc"]


def test_reference_without_training_only_evaluates(world, cfg):
    df, store = world
    rec = run(cfg, df, store, seed=0, device="cpu", tag="init", train=False, log=lambda *_: None)
    assert len(rec["evals"]) == 1 and rec["evals"][0]["round"] == 0 and rec["rounds"] == []


def test_centralized_run_has_one_client(world, cfg):
    df, store = world
    rec = run(cfg, df, store, seed=0, device="cpu", tag="c", centralized=True, log=lambda *_: None)
    assert rec["clients"] == ["central"]


# ---------------------------------------------------------------- metrics
def test_group_summary_worst_group_and_min_n():
    y = np.array([1] * 10 + [0] * 10 + [1] * 2)
    yhat = np.array([1] * 10 + [1] * 10 + [0] * 2)             # group a perfect, b all wrong, c tiny and wrong
    g = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 2)
    s = metrics.group_summary(y, yhat, g, min_n=5, n_boot=50, rng=np.random.default_rng(0))
    assert s["worst_group"] == "b" and s["worst_group_acc"] == 0.0 and s["n_groups_eval"] == 2
    assert s["mean_group_acc"] == 0.5 and s["groups"]["c"]["n"] == 2
    lo, hi = s["worst_group_ci95"]
    assert lo <= 0.0 <= hi


def test_group_summary_all_groups_too_small():
    s = metrics.group_summary(np.array([1, 0]), np.array([1, 0]), np.array(["a", "b"]), 5, 10, np.random.default_rng(0))
    assert np.isnan(s["worst_group_acc"]) and s["worst_group"] is None


def test_forgetting_is_best_earlier_minus_final():
    hist = [{"snow": 0.5, "day": 0.9}, {"snow": 0.8, "day": 0.9}, {"snow": 0.6, "day": 0.95}]
    f = metrics.forgetting(hist)
    assert np.isclose(f["snow"], 0.2) and np.isclose(f["day"], 0.0 - 0.05)
    assert metrics.forgetting(hist[:1]) == {}


def test_probe_on_random_labels_is_near_chance(cfg):
    df, store = make_world(n_per=400, seed=3)
    rng = np.random.default_rng(0)
    df["label"] = rng.integers(0, 2, size=len(df))
    from zodfl.probe import run_probes
    m = SimSiamModel(pretrained=False)
    res = run_probes(m, store, df, cfg, "cpu", np.random.default_rng(0))
    acc = res["tasks"]["night"]["by_cond"]["overall_acc"]
    assert 0.35 < acc < 0.65


def test_effective_rank_detects_collapse():
    from zodfl.probe import effective_rank
    rng = np.random.default_rng(0)
    rich = rng.normal(size=(500, 64))
    collapsed = np.tile(rng.normal(size=(1, 64)), (500, 1)) + 1e-6 * rng.normal(size=(500, 64))
    rank_one = rng.normal(size=(500, 1)) @ rng.normal(size=(1, 64))
    assert effective_rank(rich) > 40 and effective_rank(rank_one) < 1.5
    assert effective_rank(collapsed) > 0          # defined even for nearly constant features


def test_simsiam_heads_use_batchnorm_without_running_buffers():
    m = SimSiamModel(pretrained=False)
    bn1d = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm1d)]
    assert bn1d and all(not b.track_running_stats for b in bn1d)
    m.eval()                                       # scoring in eval mode must work on a batch of any size >= 2
    x = torch.randint(0, 255, (3, 3, H, W), dtype=torch.uint8)
    g = torch.Generator().manual_seed(0)
    assert torch.isfinite(m.pair_loss(augment(x, g), augment(x, g))).all()


def test_backbone_lr_multiplier_zero_freezes_the_pretrained_blocks(world, cfg):
    df, store = world
    cfg0 = C.override(cfg, federation={"backbone_lr_mult": 0.0}, selection={"kind": "all", "budget": 1.0})
    w = build_world(df, cfg0.federation)
    rows = np.concatenate([v for (k, _), v in w.rows.items() if k == 0])
    torch.manual_seed(0)
    m = SimSiamModel(pretrained=False)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if n.startswith(("layer3", "layer4"))}
    head_before = torch.cat([p.detach().flatten().clone() for n, p in m.named_parameters() if n.startswith("projector")])
    res = client_round(m, get_vector(m), rows, store, w.cond_ids, len(w.cond_names), Selector("all", 1.0, "cpu"),
                       cfg0.federation, "cpu", torch.Generator().manual_seed(0))
    assert res.n_steps > 0
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters() if n in before)
    head_after = torch.cat([p.detach().flatten() for n, p in m.named_parameters() if n.startswith("projector")])
    assert not torch.equal(head_before, head_after), "heads must still learn when the backbone multiplier is 0"


def test_low_label_probe_is_reported_with_a_fixed_subset(world, cfg):
    from zodfl.probe import run_probes
    df, store = world
    c = C.override(cfg, eval={"probe_fractions": [1.0, 0.25]})
    m = SimSiamModel(pretrained=False)
    r1 = run_probes(m, store, df, c, "cpu", np.random.default_rng(0))
    r2 = run_probes(m, store, df, c, "cpu", np.random.default_rng(1))
    assert {"night", "night@25pct"} <= set(r1["tasks"])
    assert r1["tasks"]["night@25pct"]["n_fit"] < r1["tasks"]["night"]["n_fit"]
    assert r1["tasks"]["night@25pct"]["n_fit"] == r2["tasks"]["night@25pct"]["n_fit"]
    a = r1["tasks"]["night@25pct"]["by_cond"]["overall_acc"]
    assert a == r2["tasks"]["night@25pct"]["by_cond"]["overall_acc"], "the subset must not depend on the run rng"

import numpy as np
import pytest
import torch

from zodfl import aggregate, config as C
from zodfl.client import client_round
from zodfl.federated import build_world, run, save_record
from zodfl.model import SimSiamModel, get_vector, trainable_params
from zodfl.report import load_records
from zodfl.selectors import Selector


# ---------------------------------------------------------------- config validation
def test_new_config_fields_are_validated():
    C.override(C.Config(), aggregation={"kind": "ldawa"}, federation={"train_layers": ["layer4"], "anchor_mu": 0.5},
               eval={"interp_alphas": [0.25, 0.5]})
    with pytest.raises(ValueError):
        C.override(C.Config(), federation={"train_layers": []})
    with pytest.raises(ValueError):
        C.override(C.Config(), federation={"train_layers": ["layer1"]})
    with pytest.raises(ValueError):
        C.override(C.Config(), federation={"anchor_mu": -1.0})
    with pytest.raises(ValueError):
        C.override(C.Config(), eval={"interp_alphas": [1.5]})
    with pytest.raises(ValueError):
        C.override(C.Config(), aggregation={"kind": "median"})


# ---------------------------------------------------------------- L-DAWA
def test_ldawa_equals_fedavg_when_clients_do_not_move():
    g = torch.randn(30)
    out = aggregate.apply_ldawa(g, [torch.zeros(30), torch.zeros(30)], np.array([10, 30]), [10, 20])
    assert torch.allclose(out, g, atol=1e-6)


def test_ldawa_normalized_is_a_convex_combination_and_unnormalized_shrinks():
    torch.manual_seed(0)
    g = torch.randn(40) * 3
    d1, d2 = torch.randn(40), torch.randn(40)
    n = np.array([20, 80])
    fed = aggregate.apply(g, [d1, d2], aggregate.weights_fedavg(n))
    nrm = aggregate.apply_ldawa(g, [d1, d2], n, [40], normalize=True)
    raw = aggregate.apply_ldawa(g, [d1, d2], n, [40], normalize=False)
    # with cosines near 1 the normalized variant is close to FedAvg; the raw variant is slightly smaller in norm
    assert (nrm - fed).norm() / fed.norm() < 0.02
    assert raw.norm() < nrm.norm() + 1e-6


def test_ldawa_as_printed_reflects_a_client_with_negative_cosine_and_the_normalized_variant_drops_it():
    g = torch.ones(20)
    good = torch.zeros(20)
    bad = torch.zeros(20)
    bad[10:] = -2.0 * g[10:]                         # second layer of the bad client is flipped (cosine -1)
    n = np.array([50, 50])
    raw = aggregate.apply_ldawa(g, [good, bad], n, [10, 10], normalize=False)
    assert torch.allclose(raw[:10], g[:10], atol=1e-6)                        # first layer: both clients agree
    assert torch.allclose(raw[10:], g[10:], atol=1e-6)                        # as printed: 0.5*g + 0.5*(-1)*(-g)
    nrm = aggregate.apply_ldawa(g, [good, bad], n, [10, 10], normalize=True)
    assert torch.allclose(nrm[10:], g[10:], atol=1e-6)                        # flipped client gets weight 0, good one stays
    assert torch.isfinite(nrm).all()
    with pytest.raises(ValueError):
        aggregate.apply_ldawa(g, [good], np.array([1]), [5, 5])               # layer sizes must cover the vector


@pytest.mark.parametrize("normalize", [False, True])
def test_ldawa_zero_global_layer_is_treated_as_cosine_one(normalize):
    g = torch.zeros(6)                                # e.g. BatchNorm biases at initialisation
    d1, d2 = torch.full((6,), 1.0), torch.full((6,), 3.0)
    out = aggregate.apply_ldawa(g, [d1, d2], np.array([50, 50]), [6], normalize=normalize)
    assert torch.allclose(out, aggregate.apply(g, [d1, d2], aggregate.weights_fedavg(np.array([50, 50]))))
    assert torch.allclose(out, torch.full((6,), 2.0))


# ---------------------------------------------------------------- FedEMA
def test_fedema_state_machine_matches_the_published_rule():
    ema = aggregate.FedEMA(tau=0.7)
    g0 = torch.zeros(4)
    init, mu = ema.init_for(0, g0)
    assert mu is None and torch.equal(init, g0)                       # first time: starts from the global model
    trained = {0: torch.full((4,), 2.0)}
    new_global = torch.ones(4)
    ema.after_round(trained, new_global)
    expected_lambda = 0.7 / float((new_global - trained[0]).norm())
    assert ema.lam[0] == pytest.approx(expected_lambda)
    g1 = torch.full((4,), 0.5)
    init, mu = ema.init_for(0, g1)
    want_mu = min(ema.lam[0] * float((g1 - trained[0]).norm()), 1.0)
    assert mu == pytest.approx(want_mu)
    assert torch.allclose(init, want_mu * trained[0] + (1 - want_mu) * g1)
    ema.after_round({}, g1)                                           # client skipped a round -> reset to global
    init, mu = ema.init_for(0, g1)
    assert mu is None and torch.equal(init, g1)


def test_fedema_mu_is_capped_at_one():
    ema = aggregate.FedEMA(tau=0.7)
    ema.lam[3] = 1e6
    ema.local[3] = torch.ones(5)
    ema.prev = {3}
    init, mu = ema.init_for(3, torch.zeros(5))
    assert mu == 1.0 and torch.equal(init, torch.ones(5))


# ---------------------------------------------------------------- freezing and anchor
def test_train_layers_freezes_layer3(world, cfg):
    full = SimSiamModel(pretrained=False)
    l4 = SimSiamModel(pretrained=False, train_layers=("layer4",))
    assert get_vector(l4).numel() < get_vector(full).numel()
    df, store = world
    c = C.override(cfg, federation={"train_layers": ["layer4"], "backbone_lr_mult": 1.0}, selection={"kind": "all", "budget": 1.0})
    w = build_world(df, c.federation)
    rows = np.concatenate([v for (k, _), v in w.rows.items() if k == 0])
    before = {n: p.detach().clone() for n, p in l4.named_parameters() if n.startswith("layer3")}
    res = client_round(l4, get_vector(l4), rows, store, w.cond_ids, len(w.cond_names), Selector("all", 1.0, "cpu"),
                       c.federation, "cpu", torch.Generator().manual_seed(0))
    assert res.n_steps > 0 and all(torch.equal(before[n], p) for n, p in l4.named_parameters() if n in before)


def test_anchor_penalty_keeps_pretrained_layers_closer_to_the_shipped_weights(world, cfg):
    df, store = world
    base = C.override(cfg, federation={"backbone_lr_mult": 1.0, "local_lr": 0.05}, selection={"kind": "all", "budget": 1.0})
    w = build_world(df, base.federation)
    rows = np.concatenate([v for (k, _), v in w.rows.items() if k == 0])
    dist = {}
    for mu in (0.0, 5.0):
        c = C.override(base, federation={"anchor_mu": mu})
        torch.manual_seed(0)
        m = SimSiamModel(pretrained=False)
        anchor = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad and n.startswith(("layer3", "layer4"))}
        client_round(m, get_vector(m), rows, store, w.cond_ids, len(w.cond_names), Selector("all", 1.0, "cpu"),
                     c.federation, "cpu", torch.Generator().manual_seed(1), anchor=anchor)
        dist[mu] = float(sum(((p.detach() - anchor[n]) ** 2).sum() for n, p in m.named_parameters() if n in anchor))
    assert dist[5.0] < dist[0.0]


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("kind", ["ldawa", "ldawa_norm", "fedema"])
def test_new_aggregators_run_end_to_end(world, cfg, kind):
    df, store = world
    c = C.override(cfg, aggregation={"kind": kind}, selection={"kind": "all", "budget": 1.0})
    rec = run(c, df, store, seed=0, device="cpu", tag=kind, log=lambda *_: None)
    assert len(rec["rounds"]) == 3 and rec["totals"]["frames_trained_on"] > 0
    assert all(np.isfinite(r["mean_loss"]) for r in rec["rounds"] if r.get("mean_loss") is not None)


def test_interpolation_alpha_zero_reproduces_the_untrained_evaluation(world, cfg, tmp_path):
    df, store = world
    c = C.override(cfg, eval={"interp_alphas": [0.0, 0.5, 1.0]}, selection={"kind": "all", "budget": 1.0})
    rec = run(c, df, store, seed=0, device="cpu", tag="tax", log=lambda *_: None)
    init_acc = rec["evals"][0]["tasks"]["night"]["by_cond"]["overall_acc"]
    assert rec["interp"]["0.0"]["tasks"]["night"]["by_cond"]["overall_acc"] == init_acc
    final_acc = rec["evals"][-1]["tasks"]["night"]["by_cond"]["overall_acc"]
    assert rec["interp"]["1.0"]["tasks"]["night"]["by_cond"]["overall_acc"] == final_acc
    save_record(rec, tmp_path)
    tags = sorted(r["tag"] for r in load_records(tmp_path))
    assert tags == ["tax", "tax+interp0.0", "tax+interp0.5", "tax+interp1.0"]
    derived = [r for r in load_records(tmp_path) if r["tag"] == "tax+interp0.5"][0]
    assert derived["derived_from"] == "tax" and len(derived["evals"]) == 2 and "interp" not in derived

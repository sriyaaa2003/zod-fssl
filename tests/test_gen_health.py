import pytest
import torch

from zodfl import config as C
from zodfl.gen import ClassifierNet, _check_health, make_optimizer, pilot_teacher, safe_step
from conftest import H, W


@pytest.fixture
def gcfg(cfg):
    return C.override(
        cfg,
        stage2={"gen_countries": [["SE"], ["DE"], ["NO"]], "archs": ["resnet18", "mobilenet_v3_small", "mobilenet_v3_small"],
                "sensor_hw": [[H, W], [24, 48], [16, 32]], "sensor_fov": [1.0, 0.9, 0.8], "quant_bits": [0, 0, 8],
                "n_rounds": 2, "passes": 1, "batch": 16, "lr": 0.05, "pretrained": False, "n_public": 40},
    )


def test_safe_step_skips_non_finite_loss_and_leaves_parameters_alone():
    lin = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    before = [p.detach().clone() for p in lin.parameters()]
    bad = lin(torch.ones(2, 3)).sum() * float("nan")
    assert safe_step(bad, opt, list(lin.parameters()), 5.0) is False
    assert all(torch.equal(b, p) for b, p in zip(before, lin.parameters()))
    good = lin(torch.ones(2, 3)).sum()
    assert safe_step(good, opt, list(lin.parameters()), 5.0) is True
    assert not all(torch.equal(b, p) for b, p in zip(before, lin.parameters()))


def test_safe_step_clips_gradient_norm():
    lin = torch.nn.Linear(3, 1, bias=False)
    torch.nn.init.zeros_(lin.weight)
    opt = torch.optim.SGD(lin.parameters(), lr=1.0)
    loss = lin(torch.full((1, 3), 1e4)).sum()                       # gradient norm about 1.7e4
    assert safe_step(loss, opt, list(lin.parameters()), 2.0)
    assert float(lin.weight.norm()) <= 2.0 + 1e-5


def test_training_fails_loudly_when_too_many_steps_are_non_finite(gcfg):
    with pytest.raises(FloatingPointError):
        _check_health({}, steps=10, skipped=5, s2=gcfg.stage2, what="x")
    stats = {}
    _check_health(stats, steps=100, skipped=1, s2=gcfg.stage2, what="x")
    assert stats["skipped_steps"] == 1


def test_body_gets_the_smaller_learning_rate(gcfg):
    m = ClassifierNet("resnet18", {"t": 2}, pretrained=False)
    opt = make_optimizer(m, gcfg.stage2)
    assert opt.param_groups[0]["lr"] == pytest.approx(gcfg.stage2.lr * gcfg.stage2.backbone_lr_mult)
    assert opt.param_groups[1]["lr"] == pytest.approx(gcfg.stage2.lr)


def test_validation_scoring_uses_probe_train_and_reports_no_skipped_steps(world, gcfg):
    df, store = world
    out = pilot_teacher(gcfg, df, store, seed=0, device="cpu", log=lambda *_: None)
    assert set(out["tasks"]) == {"night"} and 0 <= out["score"] <= 1 and out["stats"]["skipped_steps"] == 0

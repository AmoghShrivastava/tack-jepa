"""JEPA loss, VICReg, collapse canary, EMA — behavioral unit tests (PRD §8)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.jepa_loss import collapse_canary, jepa_latent_loss, vicreg_regularizer  # noqa: E402
from training.ema import ema_momentum, ema_update, make_target  # noqa: E402


def test_jepa_loss_stops_target_gradient():
    pred = torch.randn(8, 16, requires_grad=True)
    tgt = torch.randn(8, 16, requires_grad=True)
    jepa_latent_loss(pred, tgt).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert tgt.grad is None  # detached: no gradient flows into the target


def test_vicreg_penalizes_collapse():
    gen = torch.Generator().manual_seed(0)
    healthy = torch.randn(64, 32, generator=gen)
    collapsed = torch.ones(64, 32) + 1e-3 * torch.randn(64, 32, generator=gen)
    # every dim a copy of one signal -> maximal off-diagonal covariance
    correlated = torch.randn(64, 1, generator=gen).expand(64, 32).contiguous()
    v_h, c_h = vicreg_regularizer(healthy)
    v_c, _ = vicreg_regularizer(collapsed)
    _, c_corr = vicreg_regularizer(correlated)
    assert v_c > v_h  # collapsed batch pays a much higher variance penalty
    assert v_c > 0.9  # std ~1e-3 vs gamma=1 hinge
    # decorrelated dims pay far less covariance penalty than correlated ones
    assert c_h < 0.1 * c_corr


def test_collapse_canary_detects():
    collapsed = torch.ones(32, 16) + 1e-4 * torch.randn(32, 16)
    healthy = torch.randn(32, 16)
    assert collapse_canary(collapsed) > 0.99
    assert abs(collapse_canary(healthy)) < 0.5


def test_ema_momentum_schedule():
    assert ema_momentum(0, 1000) == pytest.approx(0.996)
    assert ema_momentum(999, 1000) == pytest.approx(0.9999)
    assert 0.996 < ema_momentum(500, 1000) < 0.9999


def test_ema_update_moves_toward_online():
    online = torch.nn.Linear(4, 4)
    target = make_target(online)
    assert not any(p.requires_grad for p in target.parameters())
    # perturb online, EMA-update target many times -> target converges to online
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    for _ in range(2000):
        ema_update(target, online, momentum=0.99)
    for pt, po in zip(target.parameters(), online.parameters(), strict=True):
        assert np.allclose(pt.detach().numpy(), po.detach().numpy(), atol=1e-6)

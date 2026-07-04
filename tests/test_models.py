"""Encoder/predictor/probe shape + behavior tests (PRD §8 testing row)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from models.encoder import TaxelGraphEncoder  # noqa: E402
from models.predictor import ActionConditionedPredictor  # noqa: E402
from models.probes import PhysicsProbes  # noqa: E402


def tiny_batch(n_per_graph=(30, 40), seed=0):
    """Two small graphs batched PyG-style (concatenated nodes + batch vector)."""
    rng = np.random.default_rng(seed)
    forces, links, poss, nrms, batches, edges = [], [], [], [], [], []
    offset = 0
    for gi, n in enumerate(n_per_graph):
        forces.append(rng.normal(size=(n, 3)))
        links.append(rng.integers(0, 21, size=n))
        poss.append(rng.normal(size=(n, 3)))
        nrms.append(rng.normal(size=(n, 3)))
        batches.append(np.full(n, gi))
        e = rng.integers(0, n, size=(2, n * 4)) + offset
        edges.append(e)
        offset += n
    qpos = rng.normal(size=(len(n_per_graph), 22))
    t = lambda x, dt: torch.as_tensor(np.concatenate(x), dtype=dt)  # noqa: E731
    return dict(
        force=t(forces, torch.float32),
        link_index=t(links, torch.long),
        edge_index=torch.as_tensor(np.concatenate(edges, axis=1), dtype=torch.long),
        batch=t(batches, torch.long),
        pos=t(poss, torch.float32),
        normal=t(nrms, torch.float32),
        qpos=torch.as_tensor(qpos, dtype=torch.float32),
    )


@pytest.mark.parametrize("use_geometry", [True, False])
def test_encoder_shapes(use_geometry):
    enc = TaxelGraphEncoder(
        hidden=64, n_layers=2, heads=4, node_out=32, global_dim=48, use_geometry=use_geometry
    )
    b = tiny_batch()
    node, glob = enc(**b)
    assert node.shape == (70, 32)
    assert glob.shape == (2, 48)
    assert torch.isfinite(node).all() and torch.isfinite(glob).all()


def test_no_fk_ablation_ignores_geometry():
    """The No-FK encoder's output must be invariant to pos/normal inputs."""
    enc = TaxelGraphEncoder(hidden=64, n_layers=2, heads=4, use_geometry=False)
    enc.eval()
    b = tiny_batch()
    _, g1 = enc(**b)
    b2 = dict(b)
    b2["pos"] = b["pos"] + 100.0
    b2["normal"] = -b["normal"]
    _, g2 = enc(**b2)
    assert torch.allclose(g1, g2)
    # ...while the baseline encoder must NOT be invariant to geometry
    enc_geo = TaxelGraphEncoder(hidden=64, n_layers=2, heads=4, use_geometry=True)
    enc_geo.eval()
    _, g3 = enc_geo(**b)
    _, g4 = enc_geo(**b2)
    assert not torch.allclose(g3, g4)


def test_predictor_shapes_and_horizon():
    pred = ActionConditionedPredictor(dim=48, n_layers=2, heads=4, context_len=8, max_horizon=8)
    ctx = torch.randn(3, 8, 48)
    actions = torch.randn(3, 8 + 4, 22)
    out = pred(ctx, actions, horizon=4)
    assert out.shape == (3, 48)
    # different horizons give different predictions (horizon embedding works)
    out1 = pred(ctx, actions, horizon=1)
    assert not torch.allclose(out, out1)
    with pytest.raises(ValueError):
        pred(ctx, actions, horizon=99)


def test_probes():
    probes = PhysicsProbes(node_dim=32, global_dim=48, hidden=16)
    node = torch.randn(70, 32)
    glob = torch.randn(2, 48)
    out = probes(node, glob)
    assert out["force_mag"].shape == (70,)
    assert out["slip_logit"].shape == (70,)
    assert out["contact_area"].shape == (2,)
    losses = PhysicsProbes.losses(
        out,
        force_mag=torch.rand(70),
        slip=torch.randint(0, 2, (70,)).float(),
        contact_area=torch.rand(2) * 100,
    )
    for v in losses.values():
        assert torch.isfinite(v)
        v_ = v  # all differentiable
    sum(losses.values()).backward()

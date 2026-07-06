"""Encoder/predictor/probe shape + behavior tests (PRD §8 testing row)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from models.ablations.image_native import TactileImageEncoder  # noqa: E402
from models.encoder import TaxelGraphEncoder  # noqa: E402
from models.predictor import ActionConditionedPredictor  # noqa: E402
from models.probes import PhysicsProbes  # noqa: E402
from sim.taxel_layout import TaxelLayout  # noqa: E402


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
    sum(losses.values()).backward()  # all differentiable


def test_contact_area_scale_normalizes_loss_magnitude():
    """contact_area's raw-count scale (~2244) must not dominate the other
    probe losses — the fix for the mid-run spike found during Phase 6."""
    probes = PhysicsProbes(node_dim=32, global_dim=48, hidden=16)
    node = torch.randn(70, 32)
    glob = torch.randn(2, 48)
    out = probes(node, glob)
    large_area = torch.tensor([2000.0, 1800.0])  # near the full 2244-taxel scale

    unscaled = PhysicsProbes.losses(
        out, force_mag=torch.rand(70), slip=torch.zeros(70), contact_area=large_area
    )
    scaled = PhysicsProbes.losses(
        out,
        force_mag=torch.rand(70),
        slip=torch.zeros(70),
        contact_area=large_area,
        contact_area_scale=2244.0,
    )
    # normalizing brings the contact_area term down to a comparable
    # magnitude to force_mag's, instead of dwarfing it
    assert scaled["contact_area"] < unscaled["contact_area"]
    assert scaled["contact_area"] < 10 * scaled["force_mag"]


def test_rasterize_occupancy_reflects_contact_not_constant():
    """Regression test for the Phase 6 collapse bug: occupancy must be a
    per-taxel CONTACT indicator, not a hardcoded constant (torch.ones_like)
    that would dominate the image's magnitude with zero information."""
    layout = TaxelLayout.load()
    enc = TactileImageEncoder(
        layout=layout, dim=16, n_layers=1, heads=2, patch=4, node_out=8, global_dim=8
    )
    n = layout.n_taxels
    batch = torch.zeros(n, dtype=torch.long)

    img_zero = enc.rasterize(torch.zeros(n, 3), batch)
    assert torch.all(img_zero[:, 2] == 0), "occupancy must be all-zero when nothing is touched"

    touched = torch.arange(0, 50)
    force = torch.zeros(n, 3)
    force[touched, 0] = 0.05
    img_touch = enc.rasterize(force, batch)
    assert img_touch[:, 2].sum().item() == pytest.approx(len(touched))
    assert img_touch[:, 2].max() > 0


def test_image_native_encoder_responds_to_different_contact_patterns():
    """The encoder must produce different global latents for genuinely
    different contact patterns (not the collapsed behavior found in Phase 6,
    where the occupancy-channel bug swamped this signal to near-zero)."""
    layout = TaxelLayout.load()
    torch.manual_seed(0)
    enc = TactileImageEncoder(
        layout=layout, dim=16, n_layers=1, heads=2, patch=4, node_out=8, global_dim=8
    )
    enc.eval()
    n = layout.n_taxels
    batch = torch.zeros(n, dtype=torch.long)
    link_index = torch.zeros(n, dtype=torch.long)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    pos = torch.zeros(n, 3)
    normal = torch.zeros(n, 3)
    qpos = torch.zeros(1, 22)

    def make_force(touched):
        f = torch.zeros(n, 3)
        f[touched, 0] = 0.05
        f[touched, 1] = 0.03
        return f

    force_a = make_force(torch.arange(0, 50))
    force_b = make_force(torch.arange(500, 550))
    with torch.no_grad():
        _, glob_a = enc(force_a, link_index, edge_index, batch, pos, normal, qpos)
        _, glob_b = enc(force_b, link_index, edge_index, batch, pos, normal, qpos)
    assert not torch.allclose(glob_a, glob_b)

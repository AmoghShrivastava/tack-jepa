"""Phase 3 exit criterion: DataLoader yields correctly-shaped graph batches.

Builds a tiny synthetic 2-episode shard set in tmp_path (no Genesis needed),
then drives the full WebDataset -> window -> PyG Batch pipeline.
"""

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
wds = pytest.importorskip("webdataset")

from data.dataset import make_loader, shard_urls  # noqa: E402
from data.shard_writer import local_wds_url  # noqa: E402
from sim.forward_kinematics import KinematicChain  # noqa: E402
from sim.hand_env import ALLEGRO_URDF, PALM_UP_QUAT  # noqa: E402
from sim.taxel_layout import TaxelLayout  # noqa: E402

S = 24  # steps per synthetic episode


def synthetic_episode(layout: TaxelLayout, seed: int) -> dict:
    """Kinematically-valid episode: FK poses from random smooth joint motion."""
    rng = np.random.default_rng(seed)
    chain = KinematicChain.from_urdf(ALLEGRO_URDF)
    T = layout.n_taxels
    q0 = rng.uniform(0.0, 0.3, size=16)
    dq = rng.uniform(-0.01, 0.03, size=16)
    link_pos = np.zeros((S, len(layout.link_names), 3), dtype=np.float32)
    link_quat = np.zeros((S, len(layout.link_names), 4), dtype=np.float32)
    qpos22 = np.zeros((S, 22), dtype=np.float32)
    action22 = np.zeros((S, 22), dtype=np.float32)
    for s in range(S):
        q = q0 + s * dq
        poses = chain.fk(q, base_pos=(0, 0, 0.25), base_quat=PALM_UP_QUAT)
        for li, ln in enumerate(layout.link_names):
            link_pos[s, li] = poses[ln].pos
            link_quat[s, li] = poses[ln].quat
        qpos22[s, :16] = q
        action22[s, :16] = q + dq
    f_normal = rng.exponential(0.05, size=(S, T)).astype(np.float16)
    f_shear = rng.normal(0, 0.02, size=(S, T, 2)).astype(np.float16)
    force_mag = np.sqrt(
        f_normal.astype(np.float32) ** 2 + (f_shear.astype(np.float32) ** 2).sum(-1)
    ).astype(np.float16)
    return {
        "f_normal": f_normal,
        "f_shear": f_shear,
        "force_mag": force_mag,
        "slip": (rng.random((S, T)) < 0.01).astype(np.uint8),
        "link_pos": link_pos,
        "link_quat": link_quat,
        "qpos22": qpos22,
        "action22": action22,
        "obj_pos": np.zeros((S, 3), dtype=np.float32),
    }


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    layout = TaxelLayout.load()
    d = tmp_path_factory.mktemp("shards")
    with wds.ShardWriter(local_wds_url(d / "train-%04d.tar"), maxcount=8, verbose=0) as sink:
        for i in range(2):
            buf = io.BytesIO()
            np.savez_compressed(buf, **synthetic_episode(layout, seed=i))
            sink.write({"__key__": f"ep_{i}", "npz": buf.getvalue()})
    return d


def test_loader_yields_correct_batches(shard_dir):
    layout = TaxelLayout.load()
    T = layout.n_taxels
    B, N, k = 2, 4, 2
    loader = make_loader(
        shard_urls(shard_dir, "train"),
        batch_size=B,
        context_len=N,
        horizon=k,
        stride=6,
    )
    n_batches = 0
    for batch in loader:
        ctx, tgt = batch["context_batch"], batch["target_batch"]
        assert ctx.num_graphs == B * N
        assert tgt.num_graphs == B
        assert ctx.pos.shape == (B * N * T, 3)
        assert ctx.force.shape == (B * N * T, 3)
        assert ctx.normal.shape == (B * N * T, 3)
        assert ctx.qpos.shape == (B * N, 22)
        assert tgt.qpos.shape == (B, 22)
        assert batch["actions"].shape == (B, N + k, 22)
        # edges reference valid node ids and grow with batching
        assert ctx.edge_index.max() < B * N * T
        assert ctx.edge_index.shape[0] == 2
        # probe labels ride along on each graph
        assert ctx.force_mag.shape == (B * N * T,)
        assert ctx.slip.shape == (B * N * T,)
        assert ctx.contact_area.shape == (B * N,)
        # batch vector maps nodes to graphs
        assert int(ctx.batch.max()) == B * N - 1
        n_batches += 1
        if n_batches >= 2:
            break
    assert n_batches >= 1

"""WebDataset shard reader -> JEPA sequence samples -> PyG batches (PRD §8).

A training sample is a window from one episode: N context timesteps and one
target timestep k steps ahead, plus the actions spanning both. Graphs (FK
world positions, radius edges) are built per window at load time — the radius
graph must be recomputed per timestep anyway (§5.5), and windows touch ~9
steps, keeping CPU cost per sample modest. Flagged scale-up option: precompute
edges into shards if the loader becomes the bottleneck on a training cluster.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from torch_geometric.data import Batch, Data

from data.graph_construction import GraphBuilder
from data.shard_writer import local_wds_url
from sim.forward_kinematics import quat_to_matrix
from sim.taxel_layout import TaxelLayout


def _episode_from_sample(sample: dict) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(sample["npz"])) as d:
        return {k: d[k] for k in d.files}


class TaxelSequenceDataset(IterableDataset):
    """Iterates (context graphs, target graph, actions, labels) windows."""

    def __init__(
        self,
        shard_pattern: str | list[str],
        layout: TaxelLayout | None = None,
        context_len: int = 8,
        horizon: int = 1,
        stride: int = 4,
        shuffle: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        self.layout = layout or TaxelLayout.load()
        self.builder = GraphBuilder(self.layout)
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride
        urls = (
            [str(p) for p in shard_pattern]
            if isinstance(shard_pattern, list)
            else str(shard_pattern)
        )
        self.pipeline = wds.WebDataset(
            urls, shardshuffle=shuffle if shuffle else False, seed=seed, empty_check=False
        )
        self.shuffle = shuffle

    def _graph_at(self, ep: dict, s: int) -> Data:
        link_quat = ep["link_quat"][s]
        link_rot = np.stack([quat_to_matrix(q) for q in link_quat])
        g = self.builder.build(
            link_pos=ep["link_pos"][s].astype(np.float64),
            link_rot=link_rot,
            f_normal=ep["f_normal"][s].astype(np.float64),
            f_shear=ep["f_shear"][s].astype(np.float64),
            qpos=ep["qpos22"][s],
            action=ep["action22"][s],
            slip=ep["slip"][s].astype(np.float64),
        )
        t = torch.as_tensor
        # NB: named link_id, not link_index — PyG's Batch increments any
        # attribute containing 'index' by num_nodes per graph when collating
        return Data(
            pos=t(g.pos),
            normal=t(g.normal),
            force=t(g.force),
            link_id=t(g.link_index),
            edge_index=t(g.edge_index),
            qpos=t(g.qpos).unsqueeze(0),          # (1, 22) per graph
            force_mag=t(g.force_mag),
            slip=t(g.slip),
            contact_area=t([g.contact_area], dtype=torch.float32),
            num_nodes=g.pos.shape[0],
        )

    def __iter__(self):
        N, k = self.context_len, self.horizon
        for sample in self.pipeline:
            ep = _episode_from_sample(sample)
            S = ep["qpos22"].shape[0]
            starts = list(range(0, S - (N + k), self.stride))
            if self.shuffle:
                np.random.default_rng().shuffle(starts)
            for s0 in starts:
                t_ctx = list(range(s0, s0 + N))
                t_tgt = s0 + N - 1 + k
                yield {
                    "context": [self._graph_at(ep, s) for s in t_ctx],
                    "target": self._graph_at(ep, t_tgt),
                    # actions for steps t-N+1 .. t+k (N + k entries)
                    "actions": torch.as_tensor(
                        ep["action22"][s0 : s0 + N + k], dtype=torch.float32
                    ),
                    "horizon": k,
                    "episode": sample.get("__key__", ""),
                    "t0": s0,
                }


def collate_sequences(samples: list[dict]) -> dict:
    """B window samples -> one training batch.

    context_batch: PyG Batch of B*N graphs ordered [b0t0..b0tN-1, b1t0, ...]
    target_batch:  PyG Batch of B graphs (encoded by the EMA target encoder)
    """
    B = len(samples)
    N = len(samples[0]["context"])
    ctx_graphs = [g for s in samples for g in s["context"]]
    return {
        "context_batch": Batch.from_data_list(ctx_graphs),
        "target_batch": Batch.from_data_list([s["target"] for s in samples]),
        "actions": torch.stack([s["actions"] for s in samples]),  # (B, N+k, 22)
        "horizon": samples[0]["horizon"],
        "B": B,
        "N": N,
    }


def make_loader(
    shard_pattern,
    batch_size: int = 8,
    num_workers: int = 0,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    ds = TaxelSequenceDataset(shard_pattern, **dataset_kwargs)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_sequences,
        num_workers=num_workers,
        drop_last=True,
    )


def shard_urls(shard_dir: str | Path, split: str) -> list[str]:
    return [local_wds_url(p) for p in sorted(Path(shard_dir).glob(f"{split}-*.tar"))]

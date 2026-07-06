"""Episode dumps -> WebDataset tar shards (PRD §8 data format).

Each shard sample is one processed episode: synthesized per-taxel forces
(float16 — they're sparse and small), link poses in taxel-layout order, the
22-dim state/action convention, and per-taxel slip labels derived from
contact tangential speeds (§5.9).

Split discipline: the validation split is OBJECT-DISJOINT (PRD §7.3) — whole
object variants are held out, not random episodes.

    python -m data.shard_writer --episodes datasets/stage_a --out datasets/shards
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import webdataset as wds

from sim.episode_processing import (
    _layout_order_maps,
    hand_contacts_at_step,
    load_episode,
    synthesize_episode,
)
from sim.taxel_layout import TaxelLayout


def local_wds_url(path: str | Path) -> str:
    """Windows-safe local path for webdataset.

    A bare `C:\\...` path parses as URL scheme 'c' and fails, and webdataset's
    reader opens `parsed.path` verbatim (so file:///C:/... breaks too). A
    scheme-less RELATIVE path with forward slashes survives both its writer
    (gopen) and reader (StreamingOpen) code paths on every platform.
    """
    import os

    rel = os.path.relpath(Path(path).resolve(), Path.cwd())
    return rel.replace("\\", "/")


SLIP_SPEED_THRESHOLD = 0.005  # m/s of tangential relative velocity
SLIP_FORCE_THRESHOLD = 0.02   # N on the taxel for slip to be meaningful
VAL_VARIANTS = (2, 4)         # ('sphere', 0.034) and ('box', (0.055, 0.045, 0.035))


def slip_labels(
    dump: dict, layout: TaxelLayout, force_mag: np.ndarray
) -> np.ndarray:
    """(S, T) uint8: taxel is in contact AND near a slipping contact point.

    A contact is slipping when its tangential relative speed exceeds
    SLIP_SPEED_THRESHOLD; the label lands on taxels of the same link within
    that link's kernel bandwidth of the contact point.
    """
    _, global_to_layout = _layout_order_maps(dump, layout)
    from sim.episode_processing import link_poses_layout_order

    link_pos, link_rot = link_poses_layout_order(dump, layout)
    S, T = force_mag.shape
    labels = np.zeros((S, T), dtype=np.uint8)
    tan = dump.get("contact_tangential_speed")
    if tan is None:
        return labels
    sigma = 1.5 * layout.spacing
    for s in np.unique(dump["contact_step"]):
        sel = np.flatnonzero(dump["contact_step"] == s)
        c_pos, _, c_link = hand_contacts_at_step(dump, int(s), global_to_layout)
        # hand_contacts_at_step yields one record per HAND side of each raw
        # contact; replicate each raw contact's tangential speed to match
        speeds = []
        for i in sel:
            n_sides = sum(
                int(dump[f"contact_link_{side}"][i]) in global_to_layout
                for side in ("a", "b")
            )
            speeds.extend([tan[i]] * n_sides)
        speeds = np.asarray(speeds)
        assert len(speeds) == len(c_pos)
        for ci in np.flatnonzero(speeds > SLIP_SPEED_THRESHOLD):
            li = int(c_link[ci])
            tidx = np.flatnonzero(layout.link_index == li)
            local = (c_pos[ci] - link_pos[s, li]) @ link_rot[s, li]
            d = np.linalg.norm(layout.positions[tidx] - local, axis=1)
            near = tidx[d < sigma[li]]
            labels[s, near] = 1
    # slip only where there is actual load on the taxel
    labels &= (force_mag > SLIP_FORCE_THRESHOLD).astype(np.uint8)
    return labels


def process_episode(path: Path, layout: TaxelLayout) -> dict[str, np.ndarray]:
    dump = load_episode(path)
    data = synthesize_episode(dump, layout)
    force_mag = data.magnitude
    perm, _ = _layout_order_maps(dump, layout)
    return {
        "f_normal": data.f_normal.astype(np.float16),
        "f_shear": data.f_shear.astype(np.float16),
        "force_mag": force_mag.astype(np.float16),
        "slip": slip_labels(dump, layout, force_mag),
        "link_pos": dump["link_pos"][:, perm].astype(np.float32),
        "link_quat": dump["link_quat"][:, perm].astype(np.float32),
        # qpos22/action22 already in PRD S5.2 order [finger16, wrist_pos3,
        # wrist_rotvec3] — built by the episode generator from Genesis's
        # native (wrist-first) dof layout via genesis_to_prd_order().
        "qpos22": dump["qpos22"].astype(np.float32),
        "action22": dump["action22"].astype(np.float32),
        "obj_pos": dump["obj_pos"].astype(np.float32),
    }


def write_shards(
    episode_dir: Path,
    out_dir: Path,
    shard_size: int = 32,
    val_variants: tuple = VAL_VARIANTS,
):
    layout = TaxelLayout.load()
    episodes = sorted(episode_dir.glob("ep_*.npz"))
    if not episodes:
        raise FileNotFoundError(f"no episodes in {episode_dir}")
    splits: dict[str, list[Path]] = {"train": [], "val": []}
    for p in episodes:
        variant = int(p.stem.split("_")[1])
        splits["val" if variant in val_variants else "train"].append(p)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, paths in splits.items():
        pattern = local_wds_url(out_dir / f"{split}-%04d.tar")
        with wds.ShardWriter(pattern, maxcount=shard_size, verbose=0) as sink:
            for p in paths:
                arrays = process_episode(p, layout)
                buf = io.BytesIO()
                np.savez_compressed(buf, **arrays)
                sink.write({"__key__": p.stem, "npz": buf.getvalue()})
        counts[split] = len(paths)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "val_variants": list(val_variants),
                "n_taxels": layout.n_taxels,
                "steps_per_episode": None,
                "notes": "object-disjoint split per PRD 7.3",
            },
            indent=2,
        )
    )
    print(f"wrote shards to {out_dir}: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="e.g. datasets/stage_a")
    parser.add_argument("--out", type=Path, required=True, help="e.g. datasets/shards_a")
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument(
        "--val-variants", type=int, nargs="+", default=None,
        help="Object-variant indices to hold out entirely for val (object-disjoint, PRD §7.3). "
             f"Default: {VAL_VARIANTS} (Stage A/B's 6-variant pool).",
    )
    args = parser.parse_args()
    val_variants = tuple(args.val_variants) if args.val_variants is not None else VAL_VARIANTS
    write_shards(args.episodes, args.out, args.shard_size, val_variants=val_variants)


if __name__ == "__main__":
    main()

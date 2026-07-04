"""Plain-YAML config loading (decided Phase 0; PRD §8 left Hydra vs YAML open).

base.yaml holds the full default config; a variant YAML (stage or §7.2
ablation) contains only the keys it overrides; CLI `key.sub=value` overrides
win last. The fully-resolved config is dumped next to every run's outputs so
results are reproducible from config + seed alone (PRD §7.6.3).
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(raw: str):
    try:
        return yaml.safe_load(raw)  # ints, floats, bools, null, lists
    except yaml.YAMLError:
        return raw


def apply_dotted_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.sub=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(variant: str | None = None, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    if variant and variant != "baseline":  # base.yaml IS the baseline
        vpath = CONFIG_DIR / f"{variant}.yaml"
        cfg = deep_merge(cfg, yaml.safe_load(vpath.read_text()) or {})
    if overrides:
        cfg = apply_dotted_overrides(cfg, overrides)
    return cfg


def dump_config(cfg: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

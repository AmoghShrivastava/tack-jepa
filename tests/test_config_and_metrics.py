"""Config composition and eval metric correctness (PRD §8)."""

import numpy as np
import pytest

from eval.metrics import auroc, binary_metrics, mae, r2
from training.config import apply_dotted_overrides, deep_merge, load_config


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    assert deep_merge(base, {"a": {"y": 5}}) == {"a": {"x": 1, "y": 5}, "b": 3}


def test_dotted_overrides_types():
    cfg = {"train": {"lr": 0.1}}
    apply_dotted_overrides(cfg, ["train.lr=3e-4", "train.steps=10", "run_name=x"])
    assert cfg["train"]["lr"] == pytest.approx(3e-4)
    assert cfg["train"]["steps"] == 10
    assert cfg["run_name"] == "x"


def test_load_config_variants_compose():
    cfg = load_config("no_fk,phase4_cpu", overrides=["train.steps=7"])
    assert cfg["model"]["variant"] == "no_fk"      # from no_fk.yaml
    assert cfg["model"]["hidden"] == 96             # from phase4_cpu.yaml
    assert cfg["train"]["steps"] == 7               # CLI wins last
    base = load_config("baseline")
    assert base["model"]["variant"] == "baseline"
    assert base["model"]["hidden"] == 256           # base.yaml untouched


def test_every_ablation_config_loads():
    for v in ("no_fk", "image_native", "reconstruction", "no_vicreg"):
        cfg = load_config(v)
        assert cfg["model"]["variant"] == v
    assert load_config("no_vicreg")["train"]["vicreg_var_weight"] == 0.0


def test_regression_metrics():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(t, t) == 0.0
    assert r2(t, t) == 1.0
    assert r2(np.full(4, t.mean()), t) == pytest.approx(0.0)


def test_auroc_known_cases():
    # perfect separation
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    # perfectly wrong
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    # random-ish scores -> ~0.5 on balanced labels
    rng = np.random.default_rng(0)
    s = rng.random(2000)
    y = rng.integers(0, 2, 2000)
    assert abs(auroc(s, y) - 0.5) < 0.05
    # degenerate labels -> nan
    assert np.isnan(auroc(s, np.zeros(2000)))


def test_binary_metrics_shapes():
    m = binary_metrics(np.array([-1.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["positives"] == 1

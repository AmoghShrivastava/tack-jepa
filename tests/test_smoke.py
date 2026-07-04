"""Phase 0 smoke test: the package skeleton imports and pytest runs."""

import importlib

import pytest

PACKAGES = ["sim", "data", "models", "models.ablations", "training", "eval"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    assert importlib.import_module(name) is not None

"""The wandb stand-in imports cleanly and refuses to be used."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub():
    path = os.path.join(ROOT, "scripts", "vint", "stubs", "wandb", "__init__.py")
    spec = importlib.util.spec_from_file_location("wandb_stub_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_stub_imports_and_any_use_raises():
    w = _stub()
    with pytest.raises(RuntimeError, match="wandb stub"):
        w.log({})
    with pytest.raises(RuntimeError):
        w.Image

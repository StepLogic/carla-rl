"""CARLA's `agents` package (BasicAgent) is vendored from the 0.9.16 container
into dependencies/carla_agents, because the cluster's client env has none and
expert_oracle.py imported it from a hard-coded local 0.9.15 path (2026-11-04,
imitation-first plan, Task 1)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav import expert_oracle  # noqa: E402


def test_agents_path_defaults_to_vendored_copy(monkeypatch):
    monkeypatch.delenv("CARLA_AGENTS_PATH", raising=False)
    p = expert_oracle.carla_agents_path()
    assert p.endswith(os.path.join("dependencies", "carla_agents"))
    assert os.path.isfile(os.path.join(p, "agents", "navigation", "basic_agent.py"))


def test_agents_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CARLA_AGENTS_PATH", str(tmp_path))
    assert expert_oracle.carla_agents_path() == str(tmp_path)


def test_no_hard_coded_local_carla_path_remains():
    src = open(expert_oracle.__file__).read()
    assert "/home/robotlab/Apps/CARLA_0.9.15" not in src

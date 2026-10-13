"""The cluster arm's env builder: the config decisions, and the wrapper order.

User 2026-10-12: move the recurrent SAC policy to the cluster. The builder exists
because train_carla_tqc imports jax and the cluster environment has none, so the
config assembly had to be restated -- and a restatement is exactly where a
hard-won detail gets dropped. Every test here is one of those details, and each
one names the run it cost.

No CARLA and no GPU: the config is a pure function and the env factory is
injected.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("gymnasium")

from torch_rl.carla_env_build import (build_env, check_cluster_env,  # noqa: E402
                                      env_config)

BASE = {
    "town": "Town01",
    "experiment": {"hero": {"blueprint": ["vehicle.micro.microlino"]}},
    "env_config": {"carla": {"town": "Town01", "port": 2000, "timeout": 50.0,
                             "quality_level": "Low", "start_server": True}},
}


@pytest.fixture(autouse=True)
def _gated(monkeypatch):
    """Every test runs with the arm's own flags; check_cluster_env refuses
    otherwise, which is itself tested below."""
    monkeypatch.setenv("GATED", "1")
    monkeypatch.delenv("CARLA_START_SERVER", raising=False)
    monkeypatch.setenv("PERCEPTION", "debit")


# ------------------------------------------------------------ the config dict --
def test_the_town_is_set_on_the_CARLA_BLOCK():
    """THE BUG THAT INVALIDATED EVERY RESULT BEFORE 2026-09-10 EVENING.
    CarlaCore loads the world from env_config.carla.town; the base config
    hardcodes Town01 there. Setting only the top-level key left every env on
    Town01 while the banner printed the town asked for."""
    cfg = env_config("Town05", 2000, base=BASE)
    assert cfg["env_config"]["carla"]["town"] == "Town05"


def test_the_top_level_town_is_set_too_but_is_not_what_matters():
    cfg = env_config("Town05", 2000, base=BASE)
    assert cfg["town"] == "Town05"


def test_the_core_is_forbidden_from_STARTING_a_server():
    """CarlaCore's default spawns a server per env on a random port, and its
    close() SIGKILLs every process with "carla" in its name anywhere on the
    machine. On a shared cluster node that kills other people's jobs."""
    cfg = env_config("Town05", 2000, base=BASE)
    assert cfg["env_config"]["carla"]["start_server"] is False


def test_the_port_is_the_one_asked_for():
    assert env_config("Town05", 2010, base=BASE)["env_config"]["carla"]["port"] == 2010


def test_short_horizon_is_on_by_default():
    """The long-horizon sampler draws goals 120-240 m away, unreachable inside
    the step cap -- the reason an early lineage scored zero arrivals in every
    configuration."""
    cfg = env_config("Town05", 2000, base=BASE)
    assert cfg["experiment"]["hero"]["is_short_horizon"] is True


def test_short_horizon_can_be_turned_off_explicitly(monkeypatch):
    monkeypatch.setenv("SHORT_HORIZON", "0")
    cfg = env_config("Town05", 2000, base=BASE)
    assert cfg["experiment"]["hero"]["is_short_horizon"] is False


def test_the_client_timeout_is_overridable(monkeypatch):
    monkeypatch.setenv("CARLA_TIMEOUT", "300")
    cfg = env_config("Town12", 2000, base=BASE)
    assert cfg["env_config"]["carla"]["timeout"] == 300.0


def test_the_base_config_is_not_mutated():
    """The module-level config dict is shared with every other caller in the
    process; mutating it would leak a town into the next env built."""
    before = BASE["env_config"]["carla"]["town"]
    env_config("Town07", 2010, base=BASE)
    assert BASE["env_config"]["carla"]["town"] == before
    assert BASE["env_config"]["carla"]["start_server"] is True


# ----------------------------------------------------------------- the guards --
def test_starting_a_server_from_python_is_REFUSED_here(monkeypatch):
    """Locally the trainer starts CARLA and owns it. In a Slurm job the batch
    script does, inside apptainer; honouring this flag would put two simulators
    on one GPU."""
    monkeypatch.setenv("CARLA_START_SERVER", "1")
    with pytest.raises(SystemExit, match="CARLA_START_SERVER"):
        check_cluster_env()


@pytest.mark.parametrize("backend", ["segvae", "vae", "vc1"])
def test_the_jax_perception_backends_are_refused(monkeypatch, backend):
    """There is no jaxlib in the cluster environment, so these cannot even load.
    This arm's perception IS the DEBiT encoder."""
    monkeypatch.setenv("PERCEPTION", backend)
    with pytest.raises(SystemExit, match="PERCEPTION"):
        check_cluster_env()


def test_a_non_gated_run_is_refused(monkeypatch):
    monkeypatch.setenv("GATED", "0")
    with pytest.raises(SystemExit, match="GATED"):
        check_cluster_env()


def test_the_arms_own_settings_pass():
    check_cluster_env()


# ------------------------------------------------------------ the wrapper order --
import gymnasium  # noqa: E402


class _FakeCarlaEnv(gymnasium.Env):
    """Stands in for CarlaEnv: records the config it was built with and carries
    the spaces the wrappers above it read. Subclasses gymnasium.Env because
    gymnasium.Wrapper asserts on the type."""

    instances = []

    def __init__(self, config):
        import numpy as np
        from gymnasium import spaces
        super().__init__()
        _FakeCarlaEnv.instances.append(config)
        self.config = config
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
        self.action_space = spaces.Box(np.array([-1.0, 0.0], np.float32),
                                       np.array([1.0, 1.0], np.float32))


def test_the_wrapper_order_is_v52s_with_the_goal_wrapper_substituted():
    """The stack must be the one v52 trained on, so a result differs in the goal
    representation and the learner -- not in the task. Checked by walking the
    chain rather than by trusting the source order."""
    _FakeCarlaEnv.instances.clear()
    try:
        env = build_env("Town05", seed=0, encoder=None, port=2000,
                        env_factory=_FakeCarlaEnv, base=BASE)
    except ImportError as e:
        pytest.skip(f"a wrapper dependency is missing here: {e}")
    chain = []
    node = env
    while node is not None:
        chain.append(type(node).__name__)
        node = getattr(node, "env", None)
    assert chain[0] == "TimeLimit"
    assert "DebitGoalWrapper" in chain
    assert "GatedReward" in chain
    assert "LaneFollowReward" in chain
    assert "CoreStateWrapper" in chain
    assert chain.index("DebitGoalWrapper") < chain.index("GatedReward")
    assert chain.index("GatedReward") < chain.index("LaneFollowReward")
    assert chain.index("LaneFollowReward") < chain.index("CoreStateWrapper")
    assert "VAEGoalWrapper" not in chain


def test_the_env_is_built_from_the_config_this_module_assembles():
    _FakeCarlaEnv.instances.clear()
    try:
        build_env("Town07", seed=1, port=2010, env_factory=_FakeCarlaEnv, base=BASE)
    except ImportError as e:
        pytest.skip(f"a wrapper dependency is missing here: {e}")
    cfg = _FakeCarlaEnv.instances[-1]
    assert cfg["env_config"]["carla"]["town"] == "Town07"
    assert cfg["env_config"]["carla"]["port"] == 2010
    assert cfg["env_config"]["carla"]["start_server"] is False

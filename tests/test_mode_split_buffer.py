"""ModeSplitBuffer: the batch composition must not follow the insert ratio.

THIS FILE EXISTS BECAUSE A MANIFEST CLAIMED IT DID. The buffer was checked with
an ad-hoc heredoc that was never saved, and the evidence pack then described it
as "TESTED" -- a claim a reader could not verify and which a peer session caught.
An assertion that lives only in a shell history is not a test.

WHAT IS BEING PROTECTED. The peer's gated agent lost a run to buffer starvation:
one mode produced ~90% of env steps, a uniformly sampled buffer gave the other
mode's critic 10-37% of each batch, and that mode's evaluation collapsed from
40-65% to 15%. The defence is that batch composition is FIXED on sample rather
than inherited from whatever the environment happened to produce.

THE IMBALANCE RUNS THE OTHER WAY IN CARLA, which is why the test covers both
directions. In Habitat, explore episodes were the long ones. Here the goal arm's
cap is MAX_EPISODE_STEPS=2000 (run_lanefollow.sh) against 500 for the goal-free
arm (run_lanefollow_nogoal.sh), so GOAL rows dominate instead. A design that only
survived one direction would be an accident.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

gym = pytest.importorskip("gymnasium")
pytest.importorskip("jaxrl2")

from gymnasium import spaces  # noqa: E402

from vae_nav.mode_split_buffer import ModeSplitBuffer  # noqa: E402

GOAL, EXPLORE = 1, 0


def _spaces():
    return (spaces.Dict({"latent": spaces.Box(-1, 1, (4,), np.float32),
                         "vector": spaces.Box(-1, 1, (3,), np.float32)}),
            spaces.Box(-1, 1, (2,), np.float32))


def _tr(mode, tag):
    """One transition whose OBSERVATION carries its mode as a value, so a batch
    can be attributed back to its source buffer without trusting the buffer."""
    o = {"latent": np.full(4, tag, np.float32),
         "vector": np.full(3, tag, np.float32)}
    return dict(observations=o, actions=np.zeros(2, np.float32),
                rewards=np.float32(tag), masks=np.float32(1),
                dones=np.float32(0), next_observations=o, mode=mode)


def _fill(buf, n_goal, n_explore):
    for _ in range(n_explore):
        buf.insert(_tr(EXPLORE, -1.0))
    for _ in range(n_goal):
        buf.insert(_tr(GOAL, +1.0))


@pytest.mark.parametrize("n_goal,n_explore", [(20, 180), (180, 20)])
def test_composition_is_fixed_regardless_of_insert_ratio(n_goal, n_explore):
    """90/10 either way must still yield 50/50 in the batch."""
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400, goal_frac=0.5)
    _fill(buf, n_goal, n_explore)
    assert buf.counts() == {"goal_rows": n_goal, "explore_rows": n_explore}

    tags = np.asarray(buf.sample(64)["observations"]["latent"])[:, 0]
    assert (tags > 0).sum() == 32, "goal half of the batch is not 50%"
    assert (tags < 0).sum() == 32, "explore half of the batch is not 50%"


def test_shapes_survive_the_concatenation():
    """The join is leaf-wise over a FrozenDict, which is a Mapping and NOT a
    dict -- an isinstance(x, dict) test silently fails on it and hands
    np.concatenate the whole observation mapping, producing an object array
    rather than an error."""
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400)
    _fill(buf, 50, 50)
    b = buf.sample(32)
    assert np.asarray(b["observations"]["latent"]).shape == (32, 4)
    assert np.asarray(b["observations"]["vector"]).shape == (32, 3)
    assert np.asarray(b["actions"]).shape == (32, 2)
    assert np.asarray(b["rewards"]).shape == (32,)


def test_single_mode_does_not_stall_the_run():
    """At the start of training only one mode has been visited. Refusing to
    sample would stall; the whole batch comes from the populated side."""
    obs_sp, act_sp = _spaces()
    for present, tag in ((GOAL, +1.0), (EXPLORE, -1.0)):
        buf = ModeSplitBuffer(obs_sp, act_sp, 400)
        for _ in range(10):
            buf.insert(_tr(present, tag))
        tags = np.asarray(buf.sample(16)["observations"]["latent"])[:, 0]
        assert len(tags) == 16
        assert np.all(np.sign(tags) == np.sign(tag))


def test_empty_buffer_raises_rather_than_returning_garbage():
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400)
    with pytest.raises(ValueError):
        buf.sample(8)


def test_mode_key_is_stripped_before_insert():
    """`mode` routes the row; it must not reach the learner as a feature."""
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400)
    _fill(buf, 20, 20)
    assert "mode" not in buf.sample(8).keys()


def test_goal_frac_is_honoured_when_not_one_half():
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400, goal_frac=0.25)
    _fill(buf, 100, 100)
    tags = np.asarray(buf.sample(64)["observations"]["latent"])[:, 0]
    assert (tags > 0).sum() == 16
    assert (tags < 0).sum() == 48


def test_goal_rows_come_first_in_the_batch():
    """Ordering is part of the contract, not an accident of implementation.

    _concat puts the goal block ahead of the explore block, and a gated learner
    that slices the batch by mode (rather than masking it) depends on that.
    Reshuffling the join would keep every other assertion in this file passing
    while silently handing each critic the other mode's rows, so the order is
    pinned here. Suggested by session 5ecc2b15, which is building against it.
    """
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400, goal_frac=0.5)
    _fill(buf, 100, 100)
    tags = np.asarray(buf.sample(64)["observations"]["latent"])[:, 0]
    assert np.all(tags[:32] > 0), "first half of the batch is not the goal block"
    assert np.all(tags[32:] < 0), "second half of the batch is not the explore block"


def test_every_key_keeps_its_own_width():
    """Per-key widths, not just a batch-dimension check.

    The FrozenDict join fails in a specific way: a Mapping that is not a dict
    slips past the recursion and np.concatenate is handed the whole observation
    mapping, yielding an OBJECT array rather than raising. That survives a
    len() check and dies much later. Asserting each key's own width catches the
    collapse at the join. Uses the real gated observation widths.
    """
    obs_sp = spaces.Dict({
        "goal": spaces.Box(-np.inf, np.inf, (12,), np.float32),
        "latent": spaces.Box(-np.inf, np.inf, (128,), np.float32),
        "vector": spaces.Box(-5.1, 5.1, (4,), np.float32),
        "mode": spaces.Box(0, 1, (1,), np.float32)})
    act_sp = spaces.Box(-1, 1, (2,), np.float32)
    buf = ModeSplitBuffer(obs_sp, act_sp, 200)
    for mode, tag in ((GOAL, +1.0), (EXPLORE, -1.0)):
        o = {k: np.full(sp.shape, tag, np.float32)
             for k, sp in obs_sp.spaces.items()}
        for _ in range(40):
            buf.insert(dict(observations=o, actions=np.zeros(2, np.float32),
                            rewards=np.float32(tag), masks=np.float32(1),
                            dones=np.float32(0), next_observations=o, mode=mode))
    b = buf.sample(32)
    for key, width in (("goal", 12), ("latent", 128), ("vector", 4), ("mode", 1)):
        arr = np.asarray(b["observations"][key])
        assert arr.dtype != object, f"{key} collapsed to an object array at the join"
        assert arr.shape == (32, width), f"{key} has shape {arr.shape}, want (32, {width})"

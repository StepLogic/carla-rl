"""The image-goal observation wrapper: latched goal frame, pixels, wrong-goal gate.

User 2026-10-12: DEBiT image goal, 5-25 m first, and (agreed with the peer) the
paired wrong-goal gate from the FIRST eval rather than bolted on later.

Each test names a failure that would otherwise be invisible in a training curve:
  * a goal frame that re-renders every step (the task quietly becomes "chase a
    moving image" instead of "reach the view you were shown")
  * a missing goal camera read as a black image (trains against a blank goal)
  * a float frame encoded as near-black by DEBiT's uint8 preprocessing
  * a "wrong" goal metres from the true one, so the gate reports no gap for a
    policy that IS reading the image
  * pixels dropped from info, which silently makes the adapters untrainable
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

from vae_nav.debit_goal_wrapper import (BINOC_DIM, CUR_KEY,  # noqa: E402
                                        GOAL_KEY, DebitGoalWrapper,
                                        WrongGoalBank)


def _frame(fill, size=112):
    return np.full((size, size, 3), int(fill), np.uint8)


import gymnasium  # noqa: E402


class _Env(gymnasium.Env):
    """A CARLA stand-in publishing exactly what the real stack publishes:
    info["rgb"], info["goal_rgb"], info["goal_pos"]. Subclasses gymnasium.Env
    because gymnasium.Wrapper asserts on the type."""

    def __init__(self, goal_pos=(0.0, 0.0), goal_fill=200, drift=False):
        from gymnasium import spaces
        super().__init__()
        # THE REAL ENV'S SPACES, as the pilot run revealed them: "state" is
        # the env's own 5400-d MobileNet features, "vector" is the 4-d
        # proprioceptive block, and there is NO mode key -- GatedReward
        # publishes the mode in info. The first version of this fake invented a
        # 9-d "state" and a "mode" key, so it tested a shape the env never
        # produces and the trainer failed on real frames with KeyError: mode.
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (5400,), np.float32),
            "goal": spaces.Box(-np.inf, np.inf, (5400,), np.float32),
            "vector": spaces.Box(-5.1, 5.1, (4,), np.float32)})
        self.goal_pos = np.asarray(goal_pos, np.float64)
        self.goal_fill = int(goal_fill)
        self.drift = bool(drift)
        self.mode_value = 1          # GatedReward's per-episode draw
        self.t = 0

    def _obs_info(self):
        obs = {"state": np.zeros(5400, np.float32),
               "goal": np.zeros(5400, np.float32),
               "vector": np.arange(4, dtype=np.float32)}
        # drift: the goal camera re-renders slightly differently each step, as
        # the real one does under moving light.
        gf = self.goal_fill + (self.t if self.drift else 0)
        info = {"rgb": _frame(10 + self.t, size=120), "goal_rgb": _frame(gf),
                "goal_pos": self.goal_pos, "mode": self.mode_value}
        return obs, info

    def reset(self, **kw):
        self.t = 0
        return self._obs_info()

    def step(self, action):
        self.t += 1
        obs, info = self._obs_info()
        return obs, 0.0, False, False, info


class _Enc:
    """Records what it was asked to encode, so the test can check the PAIR."""

    def __init__(self):
        self.calls = []

    def encode(self, cur, goal):
        self.calls.append((cur.copy(), goal.copy()))
        return np.full(BINOC_DIM, float(goal[0, 0, 0]), np.float32)


# ------------------------------------------------------------- the goal frame --
def test_the_goal_frame_is_latched_at_reset_and_never_re_rendered():
    """THE TASK DEFINITION. The goal camera re-renders every step; scoring
    against the latest render would make the target drift under the policy."""
    env = DebitGoalWrapper(_Env(drift=True), encoder=_Enc())
    _obs, info = env.reset()
    latched = info[GOAL_KEY].copy()
    for _ in range(4):
        _o, _r, _t, _tr, info = env.step(np.zeros(2, np.float32))
        np.testing.assert_array_equal(info[GOAL_KEY], latched)


def test_the_current_frame_does_change_every_step():
    env = DebitGoalWrapper(_Env(), encoder=_Enc())
    _obs, info = env.reset()
    first = info[CUR_KEY].copy()
    _o, _r, _t, _tr, info = env.step(np.zeros(2, np.float32))
    assert not np.array_equal(info[CUR_KEY], first)


def test_a_missing_goal_camera_is_refused_not_zeroed():
    """A zero frame is a VALID image: it would train the policy against a black
    goal and nothing downstream would look wrong."""
    class _NoGoal(_Env):
        def reset(self, **kw):
            obs, info = super().reset(**kw)
            info.pop("goal_rgb")
            return obs, info

    with pytest.raises(KeyError, match="goal_rgb"):
        DebitGoalWrapper(_NoGoal()).reset()


def test_a_float_frame_is_refused():
    """DEBiT's preprocessing does its own /255, so a pre-normalised frame would
    encode as near black."""
    class _Float(_Env):
        def reset(self, **kw):
            obs, info = super().reset(**kw)
            info["goal_rgb"] = info["goal_rgb"].astype(np.float32) / 255.0
            return obs, info

    with pytest.raises(ValueError, match="uint8"):
        DebitGoalWrapper(_Float()).reset()


# ------------------------------------------------------------ what it exposes --
def test_both_frames_reach_info_as_pixels():
    """The adapters are trained from these. Drop them and the arm silently
    becomes a frozen-encoder run."""
    env = DebitGoalWrapper(_Env(), encoder=_Enc())
    _obs, info = env.reset()
    for key in (CUR_KEY, GOAL_KEY):
        assert info[key].dtype == np.uint8 and info[key].ndim == 3


def test_the_encoder_is_given_the_pair_in_the_right_order():
    enc = _Enc()
    env = DebitGoalWrapper(_Env(goal_fill=200), encoder=enc)
    env.reset()
    cur, goal = enc.calls[-1]
    assert int(cur[0, 0, 0]) == 10 and int(goal[0, 0, 0]) == 200


def test_the_observation_replaces_the_goal_block_with_binoc():
    env = DebitGoalWrapper(_Env(), encoder=_Enc())
    obs, _info = env.reset()
    assert set(obs) == {"vector", "mode", "binoc"}
    assert obs["binoc"].shape == (BINOC_DIM,)
    assert "goal" not in env.observation_space.spaces
    assert "state" not in env.observation_space.spaces, \
        "the env's 5400-d MobileNet features must not travel as proprioception"
    assert env.observation_space.spaces["mode"].shape == (1,)


def test_without_an_encoder_the_pixels_still_flow_and_the_flag_says_so():
    """A collection run needs no weights; what it must not do is pretend the
    zero vector is an encoding."""
    env = DebitGoalWrapper(_Env(), encoder=None)
    obs, info = env.reset()
    assert info["binoc_encoded"] is False
    assert not obs["binoc"].any()
    assert info[GOAL_KEY].any()


# ------------------------------------------------------- the wrong-goal gate --
def test_a_wrong_goal_must_be_FAR_from_the_true_one():
    """The distance floor is the whole gate. A distractor metres away overlaps
    the true view, and the measured gap goes to zero for a policy that is in
    fact reading the goal."""
    bank = WrongGoalBank(min_m=5.0)
    bank.add(_frame(1), (0.0, 0.0))
    bank.add(_frame(2), (1.0, 0.0))          # 1 m away: too close
    rng = np.random.default_rng(0)
    assert bank.pick((0.0, 0.0), rng) is None
    bank.add(_frame(3), (50.0, 0.0))
    picked = bank.pick((0.0, 0.0), rng)
    assert picked is not None and int(picked[0, 0, 0]) == 3


def test_an_empty_bank_serves_the_TRUE_goal_rather_than_a_near_distractor():
    """The first episodes of a wrong-goal run have nothing to swap in. Serving a
    close frame would corrupt the gate; serving the true goal is visible in the
    flag and simply excluded from the pairing."""
    env = DebitGoalWrapper(_Env(goal_fill=200), encoder=_Enc(), wrong_goal=True)
    _obs, info = env.reset()
    assert info["served_wrong_goal"] is False
    assert int(info[GOAL_KEY][0, 0, 0]) == 200


def test_once_the_bank_has_a_far_frame_the_wrong_goal_is_served():
    env = DebitGoalWrapper(_Env(goal_pos=(0.0, 0.0), goal_fill=200),
                           encoder=_Enc(), wrong_goal=True)
    env.reset()                                   # banks the (0, 0) goal
    env.env.goal_pos = np.array([100.0, 0.0])     # next episode's goal, far away
    env.env.goal_fill = 150
    _obs, info = env.reset()
    assert info["served_wrong_goal"] is True
    assert int(info[GOAL_KEY][0, 0, 0]) == 200    # the OTHER episode's goal view


def test_a_wrong_goal_run_still_banks_the_TRUE_frames():
    """Otherwise the bank stops growing after the first swap and every later
    episode gets the same distractor."""
    env = DebitGoalWrapper(_Env(goal_fill=200), encoder=_Enc(), wrong_goal=True)
    env.reset()
    env.env.goal_pos = np.array([100.0, 0.0])
    env.env.goal_fill = 150
    env.reset()
    fills = sorted(int(f[0, 0, 0]) for f, _ in env.bank.frames)
    assert fills == [150, 200]


def test_the_default_run_never_serves_a_wrong_goal():
    env = DebitGoalWrapper(_Env(), encoder=_Enc())
    for _ in range(3):
        _obs, info = env.reset()
        assert info["served_wrong_goal"] is False


def test_the_distance_floor_comes_from_the_environment_when_unset(monkeypatch):
    monkeypatch.setenv("WRONG_GOAL_MIN_M", "12.5")
    env = DebitGoalWrapper(_Env(), encoder=_Enc(), wrong_goal=True)
    assert env.bank.min_m == 12.5


# ------------------------------------------- what the pilot run taught us --
def test_the_mode_comes_from_INFO_and_reaches_the_observation():
    """GatedReward draws the mode per episode and publishes it in info; this
    env's observation has no mode key at all. The learner's state block must end
    with it, so the wrapper adds it."""
    env = DebitGoalWrapper(_Env(), encoder=_Enc())
    obs, _info = env.reset()
    assert obs["mode"].shape == (1,) and obs["mode"][0] == 1.0
    env.env.mode_value = 0
    obs, _r, _t, _tr, _i = env.step(np.zeros(2, np.float32))
    assert obs["mode"][0] == 0.0


def test_a_missing_mode_in_info_is_refused():
    """Without GatedReward below, the gated learner cannot tell the arms apart
    -- and a default would quietly train everything as one mode."""
    class _NoMode(_Env):
        def reset(self, **kw):
            obs, info = super().reset(**kw)
            info.pop("mode")
            return obs, info

    with pytest.raises(KeyError, match="mode"):
        DebitGoalWrapper(_NoMode(), encoder=_Enc()).reset()


def test_a_missing_state_key_is_refused_not_silently_dropped():
    """The first version copied only the keys that happened to exist, so a
    renamed block produced a SHORT state and the learner's mode_index landed on
    the wrong column -- with no error."""
    class _NoVector(_Env):
        def reset(self, **kw):
            obs, info = super().reset(**kw)
            obs.pop("vector")
            return obs, info

    with pytest.raises(KeyError, match="vector"):
        DebitGoalWrapper(_NoVector(), encoder=_Enc()).reset()


def test_the_proprioceptive_block_carries_no_goal_bearing():
    """THE ARM'S CENTRAL QUESTION DEPENDS ON THIS. The env's vector is
    [steer, throttle, speed, mask flag]; it computes a goal heading for its own
    termination check and does not publish it. If a bearing were in here, the
    wrong-goal gate would show no gap however blind the policy was."""
    env = DebitGoalWrapper(_Env(), encoder=_Enc())
    obs, _info = env.reset()
    assert obs["vector"].shape == (4,)

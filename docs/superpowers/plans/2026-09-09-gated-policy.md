# Gated Goal/Explore Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one CARLA policy that serves both goal-reaching and goal-free driving, switched by a commanded mode bit, so the three-way ablation of gated against goal-only against goal-free becomes possible.

**Architecture:** A per-episode mode bit enters the observation as its own key, selects one of two critics with separate discounts and temperatures through masked losses, and stratifies replay at a fixed batch composition. The learner is a port of the Habitat `GatedTQCLearner`, which subclasses the same vendored jaxrl2 `TQCLearner` this repo already uses.

**Tech Stack:** JAX, flax, optax, vendored jaxrl2 (TQC with quantile critics), gymnasium wrappers, CARLA 0.9.x, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-gated-policy-design.md`

## Global Constraints

- Perception is the **frozen** autoencoder at `checkpoints/lane_vae_120/best.pkl`, latent dim **128**. Do not gate against the explorer arm, whose backend is a 384-d MobileNet.
- Observation for the gated arm: `{goal: (12,), latent: (128,), vector: (4,), mode: (1,)}`. Total 145.
- Action space is unchanged and already shared: `Box([-0.5, 0.0], [0.5, 0.6])` with `INCREMENTAL_ACTIONS=0`, `MAX_STEER=0.5`, `MAX_THROTTLE=0.6`.
- Mode encoding is **`GOAL = 1`, `EXPLORE = 0`**, matching `src/vae_nav/gated_reward.py:44`. Never invert this.
- Both lane arms run at **`DISCOUNT=0.99`**. The Habitat rationale of 1.0 against 0.99 does **not** apply here. Two critics are justified by **return scale**, order 100 against order 10, and by the kappa derived from it. Say so in comments; do not inherit the Habitat reason.
- `dependencies/jaxrl2` is a **nested git repository**. Commits to it are separate from carla-rl commits and must be made from inside that directory.
- Tests run by path; there is no `pytest.ini` and no `conftest.py`. Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/<file> -v`.
- Never gate a launch on an exact process name. The CARLA binary name is 23 characters and the kernel truncates the comm field to 15, so exact-name matching can never succeed. Use `pgrep -f '[C]arlaUE4-Linux-Shipping'`, and never place that pattern unbracketed inside a long-running command.
- Gate GPU launches on about 6000 MiB free, polled, reusing the existing pattern. Do not gate on process names.
- **Take the CARLA port lock.** Seventeen run scripts already hold `/tmp/carla_rl_port2000.lock` for their lifetime through a non-blocking `flock` on file descriptor 200, refusing to start if another job holds it. The gated run script must do the same. Any two CARLA jobs on this machine are mutually exclusive, so a gated run and any lane run cannot overlap.
- **Three sessions share one card.** Our floor is about 5.1 GB, and another user's Habitat jobs take about 4.8 GB per stage on the same 12.3 GB card. Only one holder at a time once their training is up. Treat a long GPU task as something to schedule, not something to launch.

---

### Task 1: Per-sample quantile Huber loss

The existing loss reduces over the batch with `.mean()`, which averages away the per-sample structure a mode mask needs. This adds a sibling that stops one reduction short.

**Files:**
- Modify: `dependencies/jaxrl2/jaxrl2/agents/tqc/quantiles.py` (append after `quantile_huber_loss`, which ends at line 74)
- Test: `dependencies/jaxrl2/tests/test_quantile_per_sample.py`

**Interfaces:**
- Consumes: `taus(n_quantiles)` from the same module.
- Produces: `quantile_huber_loss_per_sample(current_quantiles, target_quantiles, kappa=1.0) -> jnp.ndarray` of shape `(batch,)`.

- [ ] **Step 1: Write the failing test**

```python
# dependencies/jaxrl2/tests/test_quantile_per_sample.py
import jax.numpy as jnp
import numpy as np

from jaxrl2.agents.tqc.quantiles import (quantile_huber_loss,
                                         quantile_huber_loss_per_sample)


def _fixture(batch=4, n_critics=2, n_quantiles=5, kept=3, seed=0):
    rng = np.random.default_rng(seed)
    cur = jnp.asarray(rng.normal(size=(n_critics, batch, n_quantiles)), jnp.float32)
    tgt = jnp.asarray(rng.normal(size=(batch, kept)), jnp.float32)
    return cur, tgt


def test_per_sample_returns_one_value_per_batch_row():
    cur, tgt = _fixture()
    per = quantile_huber_loss_per_sample(cur, tgt)
    assert per.shape == (4,)


def test_mean_of_per_sample_equals_the_reduced_loss():
    # The per-sample form must be the SAME loss, one reduction short, or the
    # gated learner silently trains on a different objective.
    cur, tgt = _fixture()
    assert np.allclose(np.asarray(per_mean := quantile_huber_loss_per_sample(cur, tgt).mean()),
                       np.asarray(quantile_huber_loss(cur, tgt)), atol=1e-5), per_mean


def test_masking_a_row_removes_its_contribution():
    cur, tgt = _fixture()
    per = quantile_huber_loss_per_sample(cur, tgt)
    mask = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    masked = (per * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    assert np.allclose(np.asarray(masked), np.asarray(per[:2].mean()), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dependencies/jaxrl2 && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_quantile_per_sample.py -v`
Expected: FAIL with `ImportError: cannot import name 'quantile_huber_loss_per_sample'`

- [ ] **Step 3: Write minimal implementation**

Append to `dependencies/jaxrl2/jaxrl2/agents/tqc/quantiles.py`:

```python
def quantile_huber_loss_per_sample(current_quantiles: jnp.ndarray,
                                   target_quantiles: jnp.ndarray,
                                   kappa: float = 1.0) -> jnp.ndarray:
    """quantile_huber_loss, reduced to (batch,) instead of a scalar.

    WHY THIS EXISTS. The gated learner trains two critics on disjoint halves of
    one batch, selected by the mode bit. Masking requires a per-row loss; the
    pairwise form averages the batch axis away, so a mask applied to its output
    would scale the whole loss rather than drop rows. Everything else -- the
    tau weighting, the Huber kappa, the sum over the current-quantile axis --
    is identical, so mean() of this is exactly quantile_huber_loss.
    """
    n_quantiles = current_quantiles.shape[-1]
    t = taus(n_quantiles)
    delta = target_quantiles[None, :, None, :] - current_quantiles[..., None]
    abs_delta = jnp.abs(delta)
    huber = jnp.where(abs_delta <= kappa,
                      0.5 * delta ** 2,
                      kappa * (abs_delta - 0.5 * kappa))
    weight = jnp.abs(t[None, None, :, None] - (delta < 0).astype(jnp.float32))
    # SUM over the current-quantile axis as the reduced form does, then mean
    # over critics and targets only -- the batch axis survives.
    return (weight * huber / kappa).sum(axis=2).mean(axis=(0, 2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dependencies/jaxrl2 && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_quantile_per_sample.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit (in the NESTED repo)**

```bash
cd dependencies/jaxrl2
git add jaxrl2/agents/tqc/quantiles.py tests/test_quantile_per_sample.py
git commit -m "Add per-sample quantile Huber loss for mode-masked critics"
```

---

### Task 2: Finish pinning the mode-split replay buffer

`src/vae_nav/mode_split_buffer.py` was documented as tested when no test existed.

**Since this plan was written, the peer session has written that test** in the
shared checkout: `tests/test_mode_split_buffer.py`, seven cases, passing, and
they corrected the evidence manifest rather than quietly fixing it. Start by
reading theirs. Keep it, and add only the cases below that it does not already
cover. The two most likely additions are the goal-rows-come-first ordering
assertion, which the learner's slicing depends on, and the explicit shape check
per observation key. Do not write a second competing file.

**Files:**
- Test: `tests/test_mode_split_buffer.py`
- Modify (only if a test fails): `src/vae_nav/mode_split_buffer.py`

**Interfaces:**
- Consumes: `ModeSplitBuffer(observation_space, action_space, capacity, goal_frac=0.5, mode_key="mode")` with `insert(tr, mode=None)`, `sample(batch_size)`, `counts()`, `__len__`.
- Produces: a verified drop-in for jaxrl2's `ReplayBuffer` at the trainer's single insert site.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mode_split_buffer.py
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("jaxrl2")

from gymnasium import spaces
from vae_nav.mode_split_buffer import ModeSplitBuffer

OBS = spaces.Dict({"goal": spaces.Box(-np.inf, np.inf, (12,), np.float32),
                   "latent": spaces.Box(-np.inf, np.inf, (128,), np.float32),
                   "vector": spaces.Box(-5.1, 5.1, (4,), np.float32),
                   "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
ACT = spaces.Box(np.array([-0.5, 0.0], np.float32),
                 np.array([0.5, 0.6], np.float32))


def _obs(mode):
    return {"goal": np.zeros(12, np.float32), "latent": np.zeros(128, np.float32),
            "vector": np.zeros(4, np.float32),
            "mode": np.array([mode], np.float32)}


def _tr(mode):
    return dict(observations=_obs(mode), actions=np.zeros(2, np.float32),
                rewards=np.float32(0.0), masks=np.float32(1.0),
                dones=np.float32(0.0), next_observations=_obs(mode), mode=mode)


def _fill(buf, n_goal, n_explore):
    for _ in range(n_goal):
        buf.insert(_tr(1))
    for _ in range(n_explore):
        buf.insert(_tr(0))


def test_fixed_half_and_half_under_explore_heavy_imbalance():
    # The Habitat failure: 90% of steps were explore and the goal critic starved.
    buf = ModeSplitBuffer(OBS, ACT, capacity=2000)
    _fill(buf, n_goal=100, n_explore=900)
    batch = buf.sample(64)
    frac_goal = float(np.asarray(batch["observations"]["mode"]).ravel().mean())
    assert abs(frac_goal - 0.5) < 1e-6, frac_goal


def test_fixed_half_and_half_under_goal_heavy_imbalance():
    # CARLA INVERTS IT: the goal cap is 2000 steps and the goal-free cap is 500,
    # so goal rows dominate here. The split must hold in both directions.
    buf = ModeSplitBuffer(OBS, ACT, capacity=2000)
    _fill(buf, n_goal=900, n_explore=100)
    batch = buf.sample(64)
    frac_goal = float(np.asarray(batch["observations"]["mode"]).ravel().mean())
    assert abs(frac_goal - 0.5) < 1e-6, frac_goal


def test_goal_rows_come_first_in_the_batch():
    # The learner's utd slicing depends on this ordering; assert it explicitly
    # so a future reshuffle cannot break the learner silently.
    buf = ModeSplitBuffer(OBS, ACT, capacity=2000)
    _fill(buf, n_goal=50, n_explore=50)
    modes = np.asarray(buf.sample(8)["observations"]["mode"]).ravel()
    assert modes[:4].tolist() == [1.0] * 4
    assert modes[4:].tolist() == [0.0] * 4


def test_one_empty_side_draws_the_whole_batch_from_the_other():
    buf = ModeSplitBuffer(OBS, ACT, capacity=2000)
    _fill(buf, n_goal=10, n_explore=0)
    modes = np.asarray(buf.sample(16)["observations"]["mode"]).ravel()
    assert modes.tolist() == [1.0] * 16


def test_empty_buffer_raises():
    buf = ModeSplitBuffer(OBS, ACT, capacity=2000)
    with pytest.raises(ValueError):
        buf.sample(8)


def test_nested_observation_mapping_is_joined_leafwise():
    # THE SUBTLE ONE. jaxrl2 returns flax FrozenDicts, which implement Mapping
    # but do NOT subclass dict. An isinstance(x, dict) test fails silently, the
    # recursion never fires, np.concatenate is handed the whole mapping, and the
    # result is an object array that only breaks much later.
    buf = ModeSplitBuffer(OBS, ACT, capacity=2000)
    _fill(buf, n_goal=20, n_explore=20)
    obs = buf.sample(8)["observations"]
    for key, dim in (("goal", 12), ("latent", 128), ("vector", 4), ("mode", 1)):
        arr = np.asarray(obs[key])
        assert arr.dtype != object, f"{key} collapsed to an object array"
        assert arr.shape == (8, dim), (key, arr.shape)


def test_mode_key_is_stripped_from_stored_transitions():
    buf = ModeSplitBuffer(OBS, ACT, capacity=2000)
    _fill(buf, n_goal=4, n_explore=4)
    assert "mode" not in buf.sample(4)


def test_counts_track_routing():
    buf = ModeSplitBuffer(OBS, ACT, capacity=2000)
    _fill(buf, n_goal=7, n_explore=3)
    assert buf.counts() == {"goal_rows": 7, "explore_rows": 3}
    assert len(buf) == 10
```

- [ ] **Step 2: Run the tests and record which fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_mode_split_buffer.py -v`
Expected: the file is unverified, so treat any failure as a real defect. `test_nested_observation_mapping_is_joined_leafwise` and `test_goal_rows_come_first_in_the_batch` are the two most likely to fail.

- [ ] **Step 3: Fix only what the tests prove is broken**

Do not refactor. If `_concat` mishandles the mapping type, fix `_concat`. If ordering is wrong, fix `sample`. Leave everything else alone.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_mode_split_buffer.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_mode_split_buffer.py src/vae_nav/mode_split_buffer.py
git commit -m "Test the mode-split replay buffer, which was documented as tested but was not"
```

---

### Task 3: Publish the mode bit in the observation

The learner recovers the mask from the observation, not from the batch, because the buffer strips the mode key on insert. This adds a fourth observation key behind an opt-in flag.

**Files:**
- Modify: `src/vae_nav/vae_goal_wrapper.py` (module constants near line 49; space construction near line 105; observation assembly near line 189)
- Test: `tests/test_gated_obs.py`

**Interfaces:**
- Consumes: `OBS_WITH_MODE` environment variable, `GatedReward` publishing `info["mode"]`.
- Produces: observation key `mode` of shape `(1,)`, float32, value 1.0 in goal mode and 0.0 in explore mode; goal block zeroed when mode is 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gated_obs.py
import importlib
import os

import numpy as np
import pytest

pytest.importorskip("gymnasium")


def _reload(**env):
    # Module-level flags are read at import, so the layout can only be changed
    # by reloading -- the same pattern the explorer layout tests use.
    for k, v in env.items():
        os.environ[k] = v
    import vae_nav.vae_goal_wrapper as m
    return importlib.reload(m)


def test_mode_key_absent_by_default():
    m = _reload(OBS_WITH_MODE="0", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1")
    assert m.WITH_MODE is False


def test_mode_key_present_when_enabled():
    m = _reload(OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1")
    assert m.WITH_MODE is True
    assert m.MODE_DIM == 1


def test_gated_layout_requires_the_goal_block():
    # A gated run with OBS_WITH_GOAL=0 has nothing to gate: the mode bit would
    # select between a goal the actor cannot see and the same observation again.
    m = _reload(OBS_WITH_MODE="1", OBS_WITH_GOAL="0", OBS_WITH_VECTOR="1")
    with pytest.raises(ValueError, match="OBS_WITH_MODE=1 requires the goal block"):
        m.check_gated_coherence()


def test_gated_layout_requires_the_vector_block():
    # Both rewards are speed-gated, so the policy must observe its own speed.
    m = _reload(OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0")
    with pytest.raises(ValueError, match="OBS_WITH_MODE=1 requires the vector block"):
        m.check_gated_coherence()


def test_explore_mode_zeroes_the_goal_block():
    m = _reload(OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1")
    block = np.arange(m.GOAL_DIM, dtype=np.float32) + 1.0
    assert np.all(m.apply_mode(block.copy(), mode=0) == 0.0)


def test_goal_mode_leaves_the_goal_block_untouched():
    m = _reload(OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1")
    block = np.arange(m.GOAL_DIM, dtype=np.float32) + 1.0
    assert np.array_equal(m.apply_mode(block.copy(), mode=1), block)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_obs.py -v`
Expected: FAIL with `AttributeError: module 'vae_nav.vae_goal_wrapper' has no attribute 'WITH_MODE'`

- [ ] **Step 3: Write minimal implementation**

Add near the other flags in `src/vae_nav/vae_goal_wrapper.py`:

```python
# THE MODE BIT GETS ITS OWN KEY, not a fifth slot in `vector`.
#
# PixelMultiplexer gives every non-image key its own Dense -> LayerNorm -> tanh
# projection before concatenating. Appending the bit to `vector` would push a
# single 0/1 value through a LayerNorm shared with steer, throttle and speed,
# where it would be partly normalised away. Its own key gives it its own
# projection. It also leaves the 4-vector layout untouched, so goal-arm
# checkpoints stay loadable for a warm start.
WITH_MODE = os.environ.get("OBS_WITH_MODE", "0") == "1"
MODE_DIM = 1
GOAL, EXPLORE = 1, 0        # must match vae_nav.gated_reward


def check_gated_coherence():
    """Reject gated configurations that cannot learn. Called at construction."""
    if not WITH_MODE:
        return
    if not WITH_GOAL:
        raise ValueError(
            "OBS_WITH_MODE=1 requires the goal block (OBS_WITH_GOAL=1). A gated "
            "policy switches between goal-reaching and goal-free driving; with "
            "no goal in the observation the bit selects between a goal the "
            "actor cannot see and the same observation again.")
    if not WITH_VECTOR:
        raise ValueError(
            "OBS_WITH_MODE=1 requires the vector block (OBS_WITH_VECTOR=1). "
            "BOTH gated rewards are speed-gated, so a policy that cannot "
            "observe its own speed sees the gate as an unobservable coin flip.")


def apply_mode(goal_block, mode):
    """Zero the goal block in explore mode, in place, and return it.

    ZEROED, NOT OMITTED. One actor cannot accept two input shapes. This is the
    dead-input cost of gating, and measuring it against the goal-free arm is
    the reason the gated arm exists.
    """
    if int(mode) == EXPLORE:
        goal_block[:] = 0.0
    return goal_block
```

In `__init__`, call `check_gated_coherence()` before building the space, add the key:

```python
        self.with_mode = WITH_MODE
        if self.with_mode:
            space["mode"] = spaces.Box(0.0, 1.0, (MODE_DIM,), np.float32)
```

and in the observation assembly, after the goal block is built:

```python
        if self.with_mode:
            mode = int(info.get("mode", GOAL))
            goal = apply_mode(goal, mode)
            out["mode"] = np.array([float(mode)], np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_obs.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_gated_obs.py src/vae_nav/vae_goal_wrapper.py
git commit -m "Publish the mode bit as its own observation key"
```

---

### Task 4: Pin the gated reward wrapper

`src/vae_nav/gated_reward.py` is written but has never touched CARLA. Pin its contract against a stub environment before it runs against the simulator.

**Files:**
- Test: `tests/test_gated_reward.py`
- Modify (only if a test fails): `src/vae_nav/gated_reward.py`

**Interfaces:**
- Consumes: an env whose `step` returns `(obs, reward, terminated, truncated, info)` with `info` carrying `hero_pos`, `collision`, `lane_any`, `in_junction`, `off_road`, `speed_factor`, `delta_distance`, `projected_distance`.
- Produces: `info["mode"]`, `info["gated_reward"]`, and per-mode termination.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gated_reward.py
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from vae_nav.gated_reward import EXPLORE, GOAL, GatedReward


class StubEnv:
    """Minimal env: returns a scripted info and always reports done=False."""
    def __init__(self, info=None):
        self.info = info or {}
        self.observation_space = None
        self.action_space = None
        self.metadata = {}

    def reset(self, **kw):
        return np.zeros(4, np.float32), dict(self.info, hero_pos=(0.0, 0.0))

    def step(self, action):
        return np.zeros(4, np.float32), 0.0, False, False, dict(self.info)

    def close(self):
        pass


def _wrap(force_mode, info=None, max_steps=5):
    env = StubEnv(info or {"hero_pos": (0.0, 0.0), "speed_factor": 0.0})
    return GatedReward(env, force_mode=force_mode, max_steps=max_steps, seed=0)


def test_forced_goal_mode_is_reported():
    w = _wrap(GOAL)
    _, info = w.reset()
    assert info["mode"] == GOAL


def test_forced_explore_mode_is_reported():
    w = _wrap(EXPLORE)
    _, info = w.reset()
    assert info["mode"] == EXPLORE


def test_mode_is_constant_for_the_whole_episode():
    # Each critic must see a whole return from a single mode; a mid-episode
    # switch would splice two return distributions into one target.
    w = _wrap(None)
    _, info = w.reset()
    drawn = info["mode"]
    for _ in range(5):
        _, _, _, _, step_info = w.step(np.zeros(2, np.float32))
        assert step_info["mode"] == drawn


def test_explore_prob_zero_always_draws_goal():
    env = StubEnv({"hero_pos": (0.0, 0.0), "speed_factor": 0.0})
    w = GatedReward(env, explore_prob=0.0, max_steps=5, seed=0)
    assert w.reset()[1]["mode"] == GOAL


def test_explore_prob_one_always_draws_explore():
    env = StubEnv({"hero_pos": (0.0, 0.0), "speed_factor": 0.0})
    w = GatedReward(env, explore_prob=1.0, max_steps=5, seed=0)
    assert w.reset()[1]["mode"] == EXPLORE


def test_collision_terminates_in_explore_mode():
    w = _wrap(EXPLORE, {"hero_pos": (0.0, 0.0), "speed_factor": 0.0,
                        "collision": True})
    w.reset()
    _, _, terminated, _, _ = w.step(np.zeros(2, np.float32))
    assert terminated is True


def test_lane_crossing_in_a_junction_does_not_terminate_explore_mode():
    # The env's own done fires on ANY lane crossing, which would end an episode
    # every time the car turned through a junction. Explore mode re-derives it
    # with the junction exemption.
    w = _wrap(EXPLORE, {"hero_pos": (0.0, 0.0), "speed_factor": 0.0,
                        "lane_any": True, "in_junction": True, "off_road": False})
    w.reset()
    _, _, terminated, _, _ = w.step(np.zeros(2, np.float32))
    assert terminated is False


def test_explore_mode_truncates_at_max_steps():
    w = _wrap(EXPLORE, max_steps=3)
    w.reset()
    truncs = [w.step(np.zeros(2, np.float32))[3] for _ in range(3)]
    assert truncs == [False, False, True]


def test_reward_is_published_and_finite():
    w = _wrap(GOAL, {"hero_pos": (0.0, 0.0), "speed_factor": 0.0,
                     "delta_distance": 1.0, "projected_distance": 0.5})
    w.reset()
    _, reward, _, _, info = w.step(np.zeros(2, np.float32))
    assert np.isfinite(reward)
    assert info["gated_reward"] == reward
```

- [ ] **Step 2: Run the tests and record which fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_reward.py -v`
Expected: the file has never run, so treat any failure as a real defect. The displacement read from `info["hero_pos"]` is the most likely silent zero.

- [ ] **Step 3: Fix only what the tests prove is broken**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_reward.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_gated_reward.py src/vae_nav/gated_reward.py
git commit -m "Pin the gated reward wrapper's mode and termination contract"
```

---

### Task 5: Port the gated two-critic learner

The substantial task. Port `GatedTQCLearner` from `/home/robotlab/scratch/robot-learning-tools/jaxrl2/jaxrl2/agents/tqc/gated_tqc_learner.py`, adapting the mask source to this repo's dict observation.

**Files:**
- Create: `dependencies/jaxrl2/jaxrl2/agents/tqc/gated_tqc_learner.py`
- Modify: `dependencies/jaxrl2/jaxrl2/agents/tqc/__init__.py` (export `GatedTQCLearner`)
- Test: `dependencies/jaxrl2/tests/test_gated_tqc.py`

**Interfaces:**
- Consumes: `TQCLearner` and its attributes `_actor`, `_critic`, `_target_critic_params`, `_temp`; `PixelMultiplexer`; `StateActionQuantileEnsemble`; `quantile_huber_loss_per_sample` from Task 1.
- Produces: `GatedTQCLearner(..., discount_e: float, init_temperature_e: float = 1.0, explore_actor_warmup: int = 0)` with attributes `_critic_e`, `_target_critic_e_params`, `_temp_e`, and `update(batch, ...)` returning per-mode statistics including `frac_goal`.

**The one adaptation from the source.** Habitat reads the mask from a flat state
vector at a computed index: `observations["state"][:, mode_index]`. This repo's
observation is a dict with a dedicated key, so the mask is
`batch["observations"]["mode"][:, 0] > 0.5`. That is the only structural change;
everything else ports as written.

- [ ] **Step 1: Write the failing test**

```python
# dependencies/jaxrl2/tests/test_gated_tqc.py
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
from gymnasium import spaces

from jaxrl2.agents.tqc.gated_tqc_learner import GatedTQCLearner

OBS = spaces.Dict({"goal": spaces.Box(-np.inf, np.inf, (12,), np.float32),
                   "latent": spaces.Box(-np.inf, np.inf, (128,), np.float32),
                   "vector": spaces.Box(-5.1, 5.1, (4,), np.float32),
                   "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
ACT = spaces.Box(np.array([-0.5, 0.0], np.float32),
                 np.array([0.5, 0.6], np.float32))


def _agent():
    return GatedTQCLearner(seed=0, observations=OBS.sample(),
                           actions=ACT.sample(), discount=0.99, discount_e=0.99,
                           hidden_dims=(32, 32), num_qs=2, num_quantiles=5,
                           top_quantiles_to_drop_per_net=1)


def _batch(mode, n=8, seed=0):
    rng = np.random.default_rng(seed)
    def obs():
        return {"goal": rng.normal(size=(n, 12)).astype(np.float32),
                "latent": rng.normal(size=(n, 128)).astype(np.float32),
                "vector": rng.normal(size=(n, 4)).astype(np.float32),
                "mode": np.full((n, 1), float(mode), np.float32)}
    return {"observations": obs(), "next_observations": obs(),
            "actions": rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32),
            "rewards": rng.normal(size=(n,)).astype(np.float32),
            "masks": np.ones((n,), np.float32),
            "dones": np.zeros((n,), np.float32)}


def _flat(params):
    import jax
    return np.concatenate([np.asarray(x).ravel()
                           for x in jax.tree_util.tree_leaves(params)])


def test_pure_goal_batch_leaves_the_explore_critic_bit_identical():
    a = _agent()
    before = _flat(a._critic_e.params)
    a.update(_batch(mode=1))
    assert np.array_equal(before, _flat(a._critic_e.params))


def test_pure_explore_batch_leaves_the_goal_critic_bit_identical():
    a = _agent()
    before = _flat(a._critic.params)
    a.update(_batch(mode=0))
    assert np.array_equal(before, _flat(a._critic.params))


def test_pure_goal_batch_does_update_the_goal_critic():
    a = _agent()
    before = _flat(a._critic.params)
    a.update(_batch(mode=1))
    assert not np.array_equal(before, _flat(a._critic.params))


def test_pure_explore_batch_does_update_the_explore_critic():
    a = _agent()
    before = _flat(a._critic_e.params)
    a.update(_batch(mode=0))
    assert not np.array_equal(before, _flat(a._critic_e.params))


def test_frac_goal_reports_batch_composition():
    a = _agent()
    info = a.update(_batch(mode=1))
    assert abs(float(info["frac_goal"]) - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dependencies/jaxrl2 && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_tqc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxrl2.agents.tqc.gated_tqc_learner'`

- [ ] **Step 3: Port the learner**

Read `/home/robotlab/scratch/robot-learning-tools/jaxrl2/jaxrl2/agents/tqc/gated_tqc_learner.py` in full and port it. Required elements, all present in the source:

1. `class GatedTQCLearner(TQCLearner)`. In `__init__`, call `super().__init__(...)`, then build `self._critic_e` with **fresh parameters and its own optax optimiser**, and `self._target_critic_e_params` as a deep copy. Build `self._temp_e` with its own optimiser.
2. `_critic_loss_masked(...)`: compute the target for every row at that critic's own scalar discount, take `quantile_huber_loss_per_sample`, then reduce with `(per * mask).sum() / jnp.maximum(mask.sum(), 1.0)`.
3. The jitted update calls `_critic_loss_masked` **twice**, once per critic, each with its own mask, target parameters, discount and alpha.
4. Actor loss uses a per-row `jnp.where` on **both** the Q value and the temperature, against the **just-updated** critics, and multiplies explore rows by `explore_actor_w`.
5. Two temperatures updated separately. A mode absent from the batch has its measured entropy replaced by its own target so the update takes a zero step rather than dragging alpha on no data.
6. Per-slice statistics weighted by mask count. Because the buffer concatenates goal rows first, a unit-to-data ratio above one makes the last slice all-explore, and reporting it alone shows the goal critic at zero.

**Write the discount comment for CARLA, not Habitat.** The source says two
critics exist because of a 1.0 against 0.99 discount split. Here both arms run
at 0.99 and the reason is return scale, order 100 against order 10, plus the
kappa derived from it.

Then export it:

```python
# dependencies/jaxrl2/jaxrl2/agents/tqc/__init__.py
from jaxrl2.agents.tqc.gated_tqc_learner import GatedTQCLearner  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dependencies/jaxrl2 && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_tqc.py tests/test_quantile_per_sample.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit (in the NESTED repo)**

```bash
cd dependencies/jaxrl2
git add jaxrl2/agents/tqc/gated_tqc_learner.py jaxrl2/agents/tqc/__init__.py tests/test_gated_tqc.py
git commit -m "Add gated two-critic TQC learner with mode-masked losses"
```

---

### Task 6: Checkpoint round-trip for the second critic and temperature

A gated checkpoint carries three tensors a plain one does not. Saving must include them and loading must accept a plain checkpoint as a warm start.

**Files:**
- Modify: `src/vae_nav/policy_io.py`
- Test: `tests/test_gated_checkpoint.py`

**Interfaces:**
- Consumes: `GatedTQCLearner` attributes `_critic_e`, `_target_critic_e_params`, `_temp_e`.
- Produces: `save_agent_state` writing keys `critic_e`, `target_critic_e`, `temp_e`; `load_agent_state(..., reset_explore_critic: bool = False)` warm-starting actor and goal critic from a plain checkpoint.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gated_checkpoint.py
import numpy as np
import pytest

pytest.importorskip("jax")

from vae_nav.policy_io import load_agent_state, save_agent_state


def test_gated_checkpoint_round_trips_the_second_critic(tmp_path):
    from tests.test_gated_obs import _reload  # layout helper
    from jaxrl2.agents.tqc.gated_tqc_learner import GatedTQCLearner  # noqa: F401
    pytest.skip("enable once Task 5 has landed; see plan Task 6")


def test_plain_checkpoint_warm_starts_a_gated_agent(tmp_path):
    pytest.skip("enable once Task 5 has landed; see plan Task 6")
```

Replace both skips with the real assertions once Task 5 lands. The first saves a
gated agent, loads it into a fresh gated agent, and asserts the explore critic's
flattened parameters are bit-identical. The second saves a plain agent, loads it
into a gated agent with `reset_explore_critic=True`, and asserts the actor and
goal critic match while the explore critic does not.

- [ ] **Step 2: Run test to verify it fails or skips**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_checkpoint.py -v`
Expected: 2 skipped before Task 5, then written out and failing.

- [ ] **Step 3: Extend save and load**

Add the three keys to the saved dictionary. In the loader, branch on their
presence: a checkpoint without them is a plain one, and loading it into a gated
agent restores actor and goal critic only, leaving the explore critic at its
initialisation. Keep the existing loud pytree check for everything else.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_checkpoint.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_gated_checkpoint.py src/vae_nav/policy_io.py
git commit -m "Round-trip the gated learner's second critic and temperature"
```

---

### Task 7: Wire the gated arm into the trainer

**Files:**
- Modify: `train_carla_tqc.py` — `make_env` (add a `GATED=1` branch before the goal-reward branches, mirroring the early return at line 277), `build_agent` (construct `GatedTQCLearner` when gated), replay construction (near line 467), and the single insert site (line 810)
- Test: `tests/test_gated_config.py`

**Interfaces:**
- Consumes: `GatedReward`, `ModeSplitBuffer`, `GatedTQCLearner`, `VAEGoalWrapper` with `OBS_WITH_MODE=1`.
- Produces: a `GATED=1` training path.

- [ ] **Step 1: Write the failing config-coherence test**

```python
# tests/test_gated_config.py
import os

import pytest


def _env(**kw):
    base = {"GATED": "1", "OBS_WITH_MODE": "1", "OBS_WITH_GOAL": "1",
            "OBS_WITH_VECTOR": "1", "PERCEPTION": "vae", "DISCOUNT": "0.99",
            "EXPLORE_DISCOUNT": "0.99", "LANE_EXPLORE": "0",
            "EXPLORE_PROB": "0.5", "GOAL_BATCH_FRAC": "0.5"}
    base.update(kw)
    return base


def _check(env):
    from vae_nav.gated_config import check_gated_run
    return check_gated_run(env)


def test_a_coherent_gated_config_is_accepted():
    _check(_env())


def test_rejects_a_gated_run_without_the_mode_key():
    with pytest.raises(ValueError, match="OBS_WITH_MODE"):
        _check(_env(OBS_WITH_MODE="0"))


def test_rejects_a_gated_run_without_the_goal_block():
    with pytest.raises(ValueError, match="goal block"):
        _check(_env(OBS_WITH_GOAL="0"))


def test_rejects_a_gated_run_without_the_vector_block():
    with pytest.raises(ValueError, match="vector block"):
        _check(_env(OBS_WITH_VECTOR="0"))


def test_rejects_the_mobilenet_backend():
    # The gated arm unifies the two 128-d autoencoder arms. MobileNet is the
    # explorer's 384-d backend and would make every warm-start checkpoint
    # unloadable.
    with pytest.raises(ValueError, match="PERCEPTION"):
        _check(_env(PERCEPTION="mobilenet"))


def test_rejects_gating_against_the_explorer_arm():
    with pytest.raises(ValueError, match="LANE_EXPLORE"):
        _check(_env(LANE_EXPLORE="1"))


def test_rejects_an_explore_probability_outside_the_unit_interval():
    with pytest.raises(ValueError, match="EXPLORE_PROB"):
        _check(_env(EXPLORE_PROB="1.5"))


def test_rejects_a_degenerate_batch_fraction():
    # goal_frac of 0 or 1 starves one critic entirely, which is the exact
    # failure the split buffer exists to prevent.
    with pytest.raises(ValueError, match="GOAL_BATCH_FRAC"):
        _check(_env(GOAL_BATCH_FRAC="0.0"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vae_nav.gated_config'`

- [ ] **Step 3: Write `src/vae_nav/gated_config.py` and wire the trainer**

`check_gated_run(env)` raises `ValueError` with the messages the tests match.
Call it once at trainer start when `GATED=1`.

In `make_env`, add the gated branch **before** every goal-reward branch and
return early from it, mirroring the explorer branch's early return, so no goal
wrapper can be half-applied. Stack: `CarlaEnv` -> `CoreStateWrapper` ->
`LaneFollowReward` -> `GatedReward` -> `VAEGoalWrapper` -> `TimeLimit`.

At the insert site, replace `buffer.insert(tr)` with a mode-carrying insert:

```python
            else:
                if gated:
                    buffer.insert(tr, mode=int(inf.get("mode", 1)))
                else:
                    buffer.insert(tr)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gated_config.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_gated_config.py src/vae_nav/gated_config.py train_carla_tqc.py
git commit -m "Wire the gated arm into the trainer behind GATED=1"
```

---

### Task 8: Run script

**Files:**
- Create: `run_gated.sh`
- Test: covered by `tests/test_gated_config.py`; verify by a short smoke run

**Interfaces:**
- Consumes: everything above.
- Produces: `checkpoints/carla_gated/checkpoint_*.pkl`.

- [ ] **Step 1: Write the script**

Copy `run_lanefollow.sh` as the base, since the gated arm is the goal arm plus a
mode. Set `GATED=1`, `OBS_WITH_MODE=1`, `OBS_WITH_GOAL=1`, `OBS_WITH_VECTOR=1`,
`PERCEPTION=vae`, `EXPLORE_PROB=0.5`, `GOAL_BATCH_FRAC=0.5`, `DISCOUNT=0.99`,
`EXPLORE_DISCOUNT=0.99`, `POLICY_FOLDER=carla_gated`, `MAX_STEPS=350000`.

Carry over from `run_lanefollow_nogoal.sh`: `XLA_FLAGS=--xla_gpu_enable_command_buffer=`
(a free-memory check is necessary but not sufficient; JAX can still fail while
instantiating a CUDA command buffer, which no byte count predicts) and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`.

Reuse the polled VRAM gate at about 6000 MiB free. Use
`pgrep -f '[C]arlaUE4-Linux-Shipping'` for cleanup and never an exact name match.

Take the port lock exactly as the other seventeen scripts do, near the top and
before anything else:

```bash
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  exit 1
fi
```

Non-blocking on purpose. A gated run that waited on the lock would sit holding
its VRAM reservation while another job ran, which is the deadlock this
convention avoids. Refuse and exit instead.

Set the two episode caps explicitly and adjacently, with a comment that they
differ on purpose and that the goal cap being the larger one is why goal rows
dominate here, the inverse of the Habitat imbalance.

- [ ] **Step 2: Smoke run**

Run: `MAX_STEPS=2000 ./run_gated.sh`
Expected: the `[Gated]` and `[VAEGoal]` banners print with a `'mode': (1,)` key,
training reaches 2000 steps, and the log shows `frac_goal` near 0.5.

- [ ] **Step 3: Confirm the batch composition holds**

Check the logged `frac_goal` stays near 0.5 across the run even though the
episode caps differ by four to one. If it drifts, the buffer is not being used.

- [ ] **Step 4: Commit**

```bash
git add run_gated.sh
git commit -m "Add the gated arm's run script with the VRAM gate"
```

---

### Task 9: Three-way evaluation

The deliverable the whole design exists for.

**Files:**
- Create: `eval_three_way.py`
- Create: `run_three_way_eval.sh`

**Interfaces:**
- Consumes: `checkpoints/carla_gated/`, `checkpoints/carla_lanefollow/checkpoint_50000.pkl`, `checkpoints/carla_lanefollow_nogoal/checkpoint_350000.pkl`.
- Produces: a results table over shared seeds, town split and trajectories.

- [ ] **Step 1: Force the mode from the evaluator**

Goal-mode evaluation sets `force_mode=1`; explore-mode evaluation sets
`force_mode=0`. The gated arm is evaluated twice, once per mode.

- [ ] **Step 2: Report the right metrics per arm**

Goal mode: success and path efficiency. Explore mode: return, episode length,
and the termination breakdown.

**Do not report the trainer's success metric for any explore-mode arm.** It
measures arrival at a goal the policy cannot see and is not paid for. The
finished goal-free run reports 0.000 over 60 episodes for exactly this reason
and labels every evaluation video a failure. Print a refusal rather than a
number, so nobody reads a zero as a result.

- [ ] **Step 3: Share the evaluation conditions**

All three arms use the same seeds, the same frozen town split from
`configs/town_split.json`, and the same trajectories. The comparison means
nothing otherwise.

- [ ] **Step 4: Commit**

```bash
git add eval_three_way.py run_three_way_eval.sh
git commit -m "Add the three-way gated/goal-only/goal-free evaluation"
```

---

## Self-Review

**Spec coverage.** Section 3, which arms, is Task 7 and Task 8 through
`PERCEPTION` and `LANE_EXPLORE` rejection. Section 4, observation, is Task 3.
Section 5, learner, is Tasks 1, 5 and 6, including the CARLA-specific discount
comment. Section 6, replay, is Task 2 and the insert site in Task 7. Section 7,
environment side, is Task 4. Section 8, testing, is distributed across every
task and all four named test files exist. Section 9, success criteria, is Task
9. Section 10, operational constraints, is Task 8. Section 11, risks: the
warm-up damping is Task 5 element 4, the warm-start path is Task 6, the layout
is settled before the first long run by Tasks 3 and 7.

**Gap found and accepted.** Spec section 11 lists reward hacking through free
respawns on a wedge truncation. No task addresses it, because the CARLA wedge
behaviour differs from the Habitat one and inventing a countermeasure before
observing the failure would be speculative. Watch the termination breakdown in
Task 8's smoke run and Task 9's evaluation; if wedge truncations climb, add a
terminal cost then.

**Placeholder scan.** Task 6's tests ship as explicit skips with the assertions
described, because they cannot be written before Task 5 fixes the learner's
attribute names. That is a sequencing note, not a placeholder, and the step says
exactly what replaces them.

**Type consistency.** `GOAL = 1` and `EXPLORE = 0` in Tasks 3, 4 and 5.
`mode` is `(1,)` float32 in Tasks 2, 3, 5 and 7. `check_gated_coherence` is the
wrapper-level check in Task 3; `check_gated_run` is the environment-level check
in Task 7; the two names are deliberately different and both are used
consistently.

## Execution Handoff

Tasks 1 through 6 are offline and need no GPU or simulator. Task 8's smoke run
and Task 9 need both, and the GPU is shared, so gate them.

# Two-Mode Image-Goal Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One 2-mode CARLA policy on the image-goal lineage — goal mode (ReViND stop-at-viewpoint reward, now with an in-lane-gated velocity preference) and explore mode (the SAME unified reward, ending on a 100 m path arrival) — with spatial VC-1 patch-grid features, a restructured observation, and a single bounded TQC critic.

**Architecture:** A new `TwoModeImageGoalReward` wrapper (sibling of `GatedReward`, which is FROZEN and only imported) owns the per-episode mode draw, the collision latch, the stop window and both termination rules. `VAEGoalWrapper` gains the 4-d goal block and 10-d state block. `VcSource` gains a patch-grid readout (37,632-d). A new `make_env` branch (placed BEFORE the single-mode image-goal branch) wires the stack; `GATED_ONE_CRITIC=1` routes to plain `TQCLearner` + plain `ReplayBuffer` with target bounds read from env at import. The trainer's HER selection is unchanged; `ImageGoalRelabelBuffer` learns to skip explore episodes.

**Tech Stack:** CARLA 0.9.15 (started from Python), ig-venv (torch 2.5.1+cu121, jax, vc_models), jaxrl2 TQC, gymnasium wrappers, pytest.

**Spec:** This plan's design was settled in conversation 2026-09-20 (user decisions: unified reward, explore = same formula, velocity term gated on L_t, K=0.1, spatial features, state block restructure, one bounded critic, this box, EXPLORE_PROB=0.5, band 5-25 m). The table below is the binding spec:

| piece | spec |
|---|---|
| reward (BOTH modes) | `r_t = −1.0 + 0.75·L_t − 0.75·offroad − 0.75·persistent-collision − 0.1·|v−v*|·L_t` |
| L_t | on-lane AND moving-with-lane-direction AND no crossing this step AND not collided (unchanged from `image_goal_reward.py`) |
| goal mode ends | success (2 m + 15° yaw + 10-step stop window) / simulator_invalid / 1000-step cap |
| explore mode ends | 100 m cumulative path (`EXPLORE_GOAL_DIST_M`) / simulator_invalid / 1000-step cap. NO arrival bonus. Crossings and off-road are NON-terminal in both modes |
| obs | `latent` (37632,) spatial VC-1 grid · `goal` (4,) bearing+yaw, zeroed in explore · `state` (10,) = has_collided + speed/target + 4×2 action history, never mode-masked · `mode` (1,) |
| learner | plain `TQCLearner` (ONE critic), `TQC_TARGET_MIN=-260 TQC_TARGET_MAX=100`, `TEMP_LR=3e-4`, `TOP_QUANTILES_DROP=2`, `TQC_KAPPA=1.0`, `DISCOUNT=0.99` |
| HER | `ImageGoalRelabelBuffer` legs for goal-mode episodes only; explore rows insert unrelabelled |
| run | `run_gated_ig.sh`: `carla_image_goal_2mode`, Town01, 1M steps, band 5-25 m, `EXPLORE_PROB=0.5`, seed 42, THIS BOX, 50k smoke first |

## Global Constraints

- **Worktree:** all work in `/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl`. Interpreter for anything touching torch/jax/vc_models: `dependencies/ig-venv/bin/python`. Plain `python3` for pure-python tests is fine.
- **pytest:** always run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (memory rule — collection dies on ROS plugins otherwise).
- **FROZEN FILES:** `src/vae_nav/gated_reward.py` and `src/vae_nav/gated_config.py`'s existing checks may be IMPORTED from but only `check_gated_run` gains an additive vc1/2-mode exception (Task 5). No other edit to the v7 lineage's reward/logic files. Existing run scripts (`run_image_goal.sh`, `run_gated.sh`, `train_v7.sh`) are never edited.
- **v2 checkpoints become unloadable BY DESIGN:** the goal block shrinks 12-d → 4-d and the coll key becomes a 10-d `state` key. Old checkpoints fail on parameter shape mismatch. Do not add compatibility shims.
- **Readout refusal is a feature:** `VC1_READOUT=grid2` changes the backend name to `vc1_vitb_grid2`, so `assert_matches` refuses any CLS checkpoint at load. Never "fix" that refusal.
- **Reward changes are the user's spec, verbatim:** constants `BASE_STEP_COST=1.0`, `IN_LANE_BONUS=0.75`, `OFFROAD_PENALTY=0.75`, `PERSISTENT_COLLISION_PENALTY=0.75`, `IG_SPEED_PREF_K=0.1`, `GOAL_DISTANCE_THRESHOLD_M=2.0`, `GOAL_YAW_THRESHOLD_DEG=15.0`, `STOP_SPEED_MPS=0.2`, `STOP_WINDOW_STEPS=10`, `EXPLORE_GOAL_DIST_M=100.0`, `TARGET_SPEED=20.0` km/h. Each read from an env var with that default, `IG_`-prefixed where the module already does so.
- **ReViND invariant:** every step's reward < 0, success included. The velocity term must never be able to lift a step to ≥ 0 (it is a penalty, gated on L_t).
- **No CARLA in unit tests.** All wrapper tests use the stub-env pattern of `tests/test_image_goal_reward.py`. The only CARLA contact is the Task 1 probe script and the Task 8 smoke run.
- One commit per task; commits happen in the worktree.

---

### Task 1: VC-1 spatial readout (probe + `VcSource` readout flag)

**Files:**
- Create: `scripts/vc1_grid_probe.py`
- Modify: `src/vae_nav/vc1_source.py`
- Test: `tests/test_vc1_source_readout.py`

**Interfaces:**
- Consumes: `VcSource.__init__/latent/identity/assert_matches` (existing, `vc1_source.py`).
- Produces: `VcSource` with env `VC1_READOUT` in `("cls", "grid2")` (default `cls` — every existing run unchanged); when `grid2`: `latent_dim=37632`, `name="vc1_vitb_grid2"`, `identity()` records `{"readout": "grid2", "grid": (7, 7, 768)}`. Module-level pure function `pool_grid(tokens, pool=2)` — reshapes (196, 768) → (14, 14, 768) → 2×2 average over axes (0,1) → (7, 7, 768) → flatten 37632.

- [ ] **Step 1: Write the failing tests**

```python
"""VC1_READOUT: the grid readout is pure math on patch tokens; the real-model
probe (scripts/vc1_grid_probe.py) verifies the extraction against the live
ViT -- these tests only guard the wiring."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

def test_pool_grid_shape_and_values():
    from vae_nav.vc1_source import pool_grid
    tokens = np.arange(196 * 768, dtype=np.float32).reshape(196, 768)
    g = pool_grid(tokens, pool=2)
    assert g.shape == (37632,)
    grid = g.reshape(7, 7, 768)
    # 2x2 average over the token grid: cell (0,0) averages tokens (0,0),(0,1),(1,0),(1,1)
    t = tokens.reshape(14, 14, 768)
    np.testing.assert_allclose(grid[0, 0], t[:2, :2].reshape(-1, 768).mean(0))

def test_readout_env_selects_grid(monkeypatch):
    import importlib
    import vae_nav.vc1_source as m
    monkeypatch.setenv("VC1_READOUT", "grid2")
    importlib.reload(m)
    assert m.READOUT == "grid2"
    src = m.VcSource.__new__(m.VcSource)   # no torch load; attribute wiring only
    assert m.VC1_GRID_DIM == 37632
    monkeypatch.delenv("VC1_READOUT")
    importlib.reload(m)
    assert m.READOUT == "cls"

def test_grid_backend_name_and_identity(monkeypatch):
    import importlib, vae_nav.vc1_source as m
    monkeypatch.setenv("VC1_READOUT", "grid2"); importlib.reload(m)
    monkeypatch.setattr(m.VcSource, "_load_model", lambda self: None)
    src = m.VcSource.__new__(m.VcSource)
    src.name = m.backend_name()   # the name/identity functions read the env
    src.latent_dim = m.VC1_GRID_DIM
    assert src.name == "vc1_vitb_grid2"
    monkeypatch.delenv("VC1_READOUT"); importlib.reload(m)
```

- [ ] **Step 2: Run tests to verify they fail** — `cd .claude/worktrees/gated-impl && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_vc1_source_readout.py -v`. Expected: FAIL, `pool_grid`/`READOUT` undefined.
- [ ] **Step 3: Write the probe script** `scripts/vc1_grid_probe.py`, in the style of `scripts/vc1_encoder_check.py` (read that file first for the venv/model-loading pattern). It must: load `vc1_vitb` via vc_models as `VcSource` does; print `model.classifier_feature` (expected `use_cls_token`); extract tokens by mirroring `forward_features` (`vit.py:101-117`) WITHOUT `handle_outcome`:

```python
# tokens == the model's own pre-pooling sequence, verified against CLS:
B = x.shape[0]
h = model.patch_embed(x)
cls_tok = model.cls_token.expand(B, -1, -1) + model.pos_embed[:, :1, :]
h = torch.cat((cls_tok, h + model.pos_embed[:, 1:, :]), dim=1)
h = model.blocks(h)
h = model.norm(h)
tokens = h[:, 1:]            # (1, 196, 768)
assert torch.allclose(tokens[:, :1].squeeze(1), model(x), atol=1e-4)
```
  then `pool_grid(tokens.squeeze(0).cpu().numpy())` → print shape, mean/std, timing per frame (target < 10 ms), and save a few frames' pooled vectors. Encode synthetic frames (noise + gradient) — NO CARLA needed.
- [ ] **Step 4: Run the probe** with `dependencies/ig-venv/bin/python scripts/vc1_grid_probe.py`. Expected: the `allclose` assertion passes (proves tokens are the pre-pooling sequence), shape (196, 768) → (37632,), sane norms. If `classifier_feature` is not `use_cls_token`, STOP and report — the extraction path is wrong for this checkpoint.
- [ ] **Step 5: Implement** in `vc1_source.py`: module level

```python
READOUT = os.environ.get("VC1_READOUT", "cls").lower()
if READOUT not in ("cls", "grid2"):
    raise SystemExit(f"VC1_READOUT={READOUT!r} is not cls|grid2")
VC1_GRID_DIM = 37632  # 14x14x768 tokens, 2x2 average-pooled -> 7x7x768

def pool_grid(tokens, pool=2):
    """(196, 768) patch tokens -> (37632,) spatial grid, 2x2 cell average."""
    t = np.asarray(tokens, np.float32).reshape(14, 14, 768)
    g = t.reshape(14 // pool, pool, 14 // pool, pool, 768).mean(axis=(1, 3))
    return g.reshape(-1).astype(np.float32)

def backend_name():
    return VC1_MODEL_NAME if READOUT == "cls" else f"{VC1_MODEL_NAME}_grid2"
```
  In `VcSource.__init__`: `self.latent_dim = VC1_GRID_DIM if READOUT == "grid2" else VC1_EMBED_DIM`; `self.name = backend_name()`; in `latent()`: if READOUT == "grid2", extract tokens per the probe's verified recipe and `return pool_grid(tokens)`. `identity()` gains `"readout": READOUT`. The banner print names the readout.
- [ ] **Step 6: Tests pass, suite still green** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -x -q`).
- [ ] **Step 7: Commit** `feat: VC-1 patch-grid readout (VC1_READOUT=grid2, 37632-d)`

### Task 2: Unified reward — velocity preference term

**Files:**
- Modify: `src/vae_nav/image_goal_reward.py` (the pure function `image_goal_reward` + banner in `ImageGoalReward.__init__`)
- Test: `tests/test_image_goal_reward.py` (extend)

**Interfaces:**
- Produces: `image_goal_reward(info, disp, has_collided)` returns `(reward, components)` with a new key `components["speed_penalty"]` (0.0 when the term is off). New module constants `SPEED_PREF_K = float(os.environ.get("IG_SPEED_PREF_K", 0.1))` and `TARGET_SPEED_KMH = float(os.environ.get("TARGET_SPEED", 20.0))`; the term is `−SPEED_PREF_K * |v_mps − v*_mps| * in_lane`.

- [ ] **Step 1: Write the failing tests** (extend `tests/test_image_goal_reward.py`, following its existing DRIVING-info pattern):

```python
def test_velocity_term_penalises_slow_driving_in_lane():
    info = dict(DRIVING); info["speed_kmh"] = 10.0        # half of 20 km/h
    _, comp = image_goal_reward(info, disp_forward_1m(), has_collided=False)
    assert comp["speed_penalty"] == pytest.approx(
        -SPEED_PREF_K * abs(_kmh_to_mps(10.0) - _kmh_to_mps(20.0)))
    r, _ = image_goal_reward(info, disp_forward_1m(), False)
    assert r == pytest.approx(-0.25 + comp["speed_penalty"])

def test_velocity_term_zero_off_lane_or_parked_offroad():
    info = dict(DRIVING); info["speed_kmh"] = 10.0
    off = dict(info); off["off_road"] = True
    assert image_goal_reward(off, disp_forward_1m(), False)[1]["speed_penalty"] == 0.0
    crossed = dict(info); crossed["lane_invasion_any"] = True
    assert image_goal_reward(crossed, disp_forward_1m(), False)[1]["speed_penalty"] == 0.0

def test_velocity_term_at_target_is_zero():
    info = dict(DRIVING); info["speed_kmh"] = 20.0
    assert image_goal_reward(info, disp_forward_1m(), False)[1]["speed_penalty"] == 0.0

def test_invariant_with_velocity_term():
    # Every cell of the reward table, term included, stays < 0.
    for speed in (0.0, 10.0, 20.0, 40.0):
        for off in (False, True):
            for coll in (False, True):
                info = dict(DRIVING); info["speed_kmh"] = speed
                info["off_road"] = off
                r, _ = image_goal_reward(info, disp_forward_1m(), coll)
                assert r < 0.0
```
  (`disp_forward_1m()` = `(1.0, 0.0)` in the fixture style the file already uses; `_kmh_to_mps = kmh / 3.6` — import `_speed_mps`'s convention from the module rather than redefining it if one is importable.)
- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement.** In `image_goal_reward` after computing `in_lane`:

```python
v = _speed_mps(info)
speed_penalty = (-SPEED_PREF_K * abs(v - TARGET_SPEED_KMH / 3.6)
                 if in_lane else 0.0)
r += speed_penalty
```
  add `"speed_penalty": speed_penalty` to the components dict; extend the `ImageGoalReward.__init__` banner with the term and `IG_SPEED_PREF_K`/`TARGET_SPEED`. `_speed_mps(info)` already returns m/s (absent → stopped).
- [ ] **Step 4: Tests pass; full suite green.**
- [ ] **Step 5: Commit** `feat: in-lane-gated velocity preference term (IG_SPEED_PREF_K=0.1)`

### Task 3: Observation restructure — 4-d goal block, 10-d state block

**Files:**
- Modify: `src/vae_nav/vae_goal_wrapper.py` (GOAL_DIM, `goal_block`, new `OBS_WITH_STATE`/`STATE_DIM`, observation space, `_observation`, banner)
- Test: `tests/test_vae_goal_wrapper_state_block.py` (new; plus update any test asserting the 12-d goal block)

**Interfaces:**
- Produces: `GOAL_DIM = 4`; `goal_block(fields)` uses ONLY `hero_pos`, `hero_yaw`, `goal_pos` (extra `fields` keys are ignored — the trainer's record call needs no change); `[forward, left, sin(yaw), cos(yaw)]` in the SAME order as today's first four entries. New env flag `OBS_WITH_STATE=1` → obs key `"state"`, shape (10,): `[float(has_collided), clip(speed_kmh / TARGET_SPEED, 0, 1), *action_history.flatten()]` with `TARGET_SPEED = float(os.environ.get("TARGET_SPEED", 20.0))` read at module import. `OBS_WITH_COLLISION` behaviour is UNCHANGED for the single-mode arm.

- [ ] **Step 1: Write the failing tests** (new file, stub-env style — `raw_obs` dict with "rgb"/"vector", `info` with `hero_pos/hero_yaw/goal_pos/has_collided/speed_kmh`):

```python
def test_goal_block_is_four_dim_and_goal_free_of_actions():
    gb = w.goal_block({"hero_pos": (0, 0), "hero_yaw": 0.0, "goal_pos": (10, 0)})
    assert gb.shape == (4,)
    np.testing.assert_allclose(gb, [10.0, 0.0, np.sin(0.0), np.cos(0.0)], atol=1e-6)

def test_state_block_contents_and_order():
    obs = wrapper_step(speed_kmh=10.0, has_collided=False, hist=[(0.1, 0.5), (0.2, 0.4), (0, 0), (0, 0)])
    s = obs["state"]
    assert s.shape == (10,)
    np.testing.assert_allclose(s, [0.0, 0.5, 0.1, 0.5, 0.2, 0.4, 0, 0, 0, 0], atol=1e-6)

def test_state_block_speed_clipped():
    obs = wrapper_step(speed_kmh=99.0)
    assert obs["state"][1] == pytest.approx(1.0)

def test_apply_mode_zeroes_goal_not_state():
    obs_goal, obs_explore = two_mode_step_pair()   # mode 1 then mode 0 in info
    np.testing.assert_array_equal(obs_explore["state"], obs_goal["state"])
    assert np.allclose(obs_explore["goal"], 0.0)
    assert not np.allclose(obs_goal["goal"], 0.0)

def test_observation_space_matches():
    assert obs_space["goal"].shape == (4,) and obs_space["state"].shape == (10,)
```
- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement** per the interfaces above. `goal_block` keeps the CARLA left-handed comment and the forward/left math verbatim, drops the history concat, returns 4 floats. The `state` block is built in `_observation` from `info` (has_collided) + `info["speed_kmh"]` + `self._hist` — the deque already exists. `LATENT_DIM` handling: when `PERCEPTION=vc1`+grid the wrapper's `latent_dim` comes from `source.latent_dim` already (check: it does — `self.latent_dim = self.source.latent_dim`), so 37632 flows through with NO change here; verify in the test.
- [ ] **Step 4: Fix every existing test that asserts the 12-d goal block** (grep `12` / `GOAL_DIM` / `action_history` under tests/); the HER buffer tests are Task 6's scope — here only the wrapper tests.
- [ ] **Step 5: Tests pass; full suite green.**
- [ ] **Step 6: Commit** `feat: 4-d goal block, 10-d state block (OBS_WITH_STATE=1)`

### Task 4: `TwoModeImageGoalReward` + rewritten coherence

**Files:**
- Modify: `src/vae_nav/image_goal_reward.py` (add the wrapper class; REWRITE `image_goal_coherence()`)
- Test: `tests/test_two_mode_image_goal.py` (new)

**Interfaces:**
- Produces: `TwoModeImageGoalReward(env, explore_prob=0.5, force_mode=None, max_steps=1000, seed=0)` — mode drawn per reset from `np.random.default_rng(seed)` unless forced; publishes `info["mode"]`, `info["is_success"]`, `info["has_collided"]`, `info["collision_impulse"]`, `info["ig_reward_components"]`, `info["explore_end"]`, `info["explore_dist_m"]`, and `info["arrived_this_step"]=True` on goal-mode success. Both branches call the SAME `image_goal_reward()`. Goal termination: success (via `goal_at_target` + stop window) or `simulator_invalid`. Explore termination: `EXPLORE_GOAL_DIST_M` imported from `vae_nav.gated_reward` (default 100.0) reached / invalid / cap. NO explore arrival bonus. The latch and stop window are per-episode state reset in `reset()`.
- Rewritten `image_goal_coherence()` (this replaces the file's current version): when `IMAGE_GOAL_TASK=1` AND `GATED=1` (the 2-mode pairing) it REQUIRES `OBS_WITH_STATE=1`, `OBS_WITH_MODE=1`, `PERCEPTION` in `("vc1", "vc1_vitb")`, `VC1_READOUT` in `("cls", "grid2")`, `GATED_ONE_CRITIC=1`, `EPISODE_HER=1`, `TQC_TARGET_MIN` AND `TQC_TARGET_MAX` set, and REFUSES `OBS_WITH_COLLISION=1`, any stacked reward (`LANE_FOLLOW/LANE_EXPLORE/TIME_TO_GOAL`), and `NO_TIMELIMIT=1` (training run). When `IMAGE_GOAL_TASK=1` and `GATED=0`: the EXISTING single-mode checks verbatim (OBS_WITH_COLLISION required, vc1 required, no stacking).

- [ ] **Step 1: Write the failing tests** (stub-env pattern from `tests/test_image_goal_reward.py`; `_StubEnv` returns fixed info, records actions):

```python
def test_mode_drawn_and_rewarded_identically():
    # Same info, both modes: reward identical, mode key differs.
    env = TwoModeImageGoalReward(_StubEnv(explore_prob=0.0), seed=0)
    out_g = env.step(ACT); assert out_g.info["mode"] == 1
    env2 = TwoModeImageGoalReward(_StubEnv(explore_prob=1.0), seed=0)
    out_e = env2.step(ACT); assert out_e.info["mode"] == 0
    assert out_g.reward == out_e.reward   # the UNIFIED claim

def test_goal_success_needs_sustained_stop():
    # 10 consecutive stopped-on-target steps terminate; 9 do not.
def test_latch_persists_until_reset():
def test_explore_arrival_at_path():
    # cumulative path >= EXPLORE_GOAL_DIST_M -> terminated, no bonus,
    # reward == sum of step rewards, explore_end == "reached"
def test_no_arrival_bonus_anywhere():
    # every reward in both modes < 0 even on terminal steps
def test_crossing_is_not_terminal_in_either_mode():
def test_coherence_two_mode_requires_state_and_bounds():
    # IMAGE_GOAL_TASK=1 GATED=1 without OBS_WITH_STATE=1 -> SystemExit
def test_coherence_single_mode_unchanged():
    # IMAGE_GOAL_TASK=1 GATED=0 without OBS_WITH_COLLISION=1 -> SystemExit (old behaviour)
```
- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement** both pieces. The wrapper mirrors `GatedReward`'s reset/step structure (mode draw at reset, `_t`, `_prev`, `_path_m`) but its reward is `image_goal_reward(info, disp, self._has_collided)` for BOTH modes; the stop window and success test reuse `goal_at_target`, `_speed_mps`, `STOP_*` constants; print a banner naming `explore_prob`, `force_mode`, `EXPLORE_GOAL_DIST_M`, `IG_SPEED_PREF_K`.
- [ ] **Step 4: Tests pass; full suite green.**
- [ ] **Step 5: Commit** `feat: TwoModeImageGoalReward (unified reward, no explore bonus) + 2-mode coherence`

### Task 5: `make_env` branch + `check_gated_run` acceptance

**Files:**
- Modify: `train_carla_tqc.py` (new branch immediately BEFORE the `IMAGE_GOAL_TASK` single-mode branch at ~line 466)
- Modify: `src/vae_nav/gated_config.py` (additive only: accept `PERCEPTION=vc1`; skip the `OBS_WITH_VECTOR` requirement when the 2-mode pairing is active)
- Test: `tests/test_make_env_two_mode_branch.py` (new)

**Interfaces:**
- Consumes: Task 4's wrapper + coherence, Task 1's readout, Task 3's obs.
- Branch condition (exact): `IMAGE_GOAL_TASK=1` AND `GATED=1` → run `image_goal_coherence()`, then `check_gated_run()`, then

```python
env = TwoModeImageGoalReward(
    env,
    explore_prob=float(os.environ.get("EXPLORE_PROB", 0.5)),
    force_mode=(int(os.environ.get("EVAL_FORCE_MODE", 1)) if eval_mode else None),
    max_steps=(10 ** 9 if os.environ.get("NO_TIMELIMIT", "0") == "1"
               else int(os.environ.get("MAX_EPISODE_STEPS", 1000))),
    seed=seed)
env = VAEGoalWrapper(env, source=latent_source)
return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
```
  placed so the single-mode branch's early return can never capture this pairing. The single-mode branch keeps its coherence call — but since `image_goal_coherence` now dispatches on the pairing, it must read `IMAGE_GOAL_TASK` truthiness the same way it does today.

- [ ] **Step 1: Write the failing tests** — `check_gated_run` combos with a monkeypatched env dict:

```python
def test_gated_accepts_vc1(monkeypatch): ...
def test_gated_two_mode_skips_vector_requirement(monkeypatch): ...
def test_gated_still_refuses_lane_explore(monkeypatch): ...
def test_branch_predicate():
    # extract the condition into a module-level function if it is not already
    # trivially greppable; the test pins IMAGE_GOAL_TASK=1+GATED=1 selecting the
    # two-mode wrapper class (patch ensure_carla/CarlaEnv with fakes if cheap;
    # otherwise the test asserts the branch ORDER by source inspection is not
    # acceptable -- build a minimal fake stack: coherence+check_gated_run pass
    # with the task-3/4 flags and TwoModeImageGoalReward wraps a stub env).
```
  If full `make_env` needs CARLA, test the two coherence/guard functions and the wrapper stack assembly, and verify branch order in the Task 8 smoke run's log banner (recorded as a plan checkpoint, not a test).
- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement.** In `gated_config.py` the perception check becomes:

```python
    if perception not in ("vae", "", "mobilenet", "vc1", "vc1_vitb"):
        raise ValueError(...)
```
  and the `OBS_WITH_VECTOR` requirement gains a skip for the 2-mode pairing (`IMAGE_GOAL_TASK=1`, where velocity lives in the state block). Nothing else in the file moves.
- [ ] **Step 4: Tests pass; full suite green.**
- [ ] **Step 5: Commit** `feat: make_env 2-mode branch; gated run accepts vc1`

### Task 6: HER skips explore episodes

**Files:**
- Modify: `src/vae_nav/image_goal_her.py`
- Test: `tests/test_image_goal_her.py` (extend)

**Interfaces:**
- Consumes: `EpisodeRelabelBuffer.record(..., mode=...)` (already stores nothing per-step about mode when `gated=False` — this task adds tracking).
- Produces: `ImageGoalRelabelBuffer` tracks the episode's mode per `record()` call; `flush_episode(truncated=)` returns 0 WITHOUT relabelling when any row's mode was 0 (on-policy explore rows are already in the buffer at record time — nothing to insert). New stat `n_explore_eps` exposed via `stats()`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_explore_episode_is_never_relabelled():
    buf = make_buffer(); run_episode(buf, mode=0, n=50)
    before = len(buf.buffer)
    assert buf.flush_episode() == 0
    assert len(buf.buffer) == before          # rows were inserted at record()
    assert buf.stats()["n_explore_eps"] == 1

def test_goal_episode_still_relabels():
    buf = make_buffer(); run_episode(buf, mode=1, n=50)  # with an arrival
    assert buf.flush_episode() > 0

def test_mixed_episode_defaults_to_goal_only_if_all_goal():
    # a single episode is single-mode; a mode flip mid-episode (must not happen)
    # is treated as explore -- the safe direction.
    buf = make_buffer(); run_episode(buf, mode=1, n=10); run_row(buf, mode=0, n=5)
    assert buf.flush_episode() == 0
```
- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement** — override `record()` to append the mode to `self._ep_modes` (cleared wherever the base clears its lists; mirror `_reset_episode`), and guard `flush_episode`:

```python
if getattr(self, "_ep_modes", None) and 0 in self._ep_modes:
    self.n_explore_eps += 1
    self._ep_modes = []
    self._reset_episode()
    return 0
```
  `stats()` gains `"n_explore_eps"`.
- [ ] **Step 4: Full HER suite green.**
- [ ] **Step 5: Commit** `feat: ImageGoalRelabelBuffer skips explore-mode episodes`

### Task 7: Trainer wiring — one bounded critic, plain buffer, mode in record

**Files:**
- Modify: `train_carla_tqc.py` (build_agent route at ~line 203-253; buffer selection at ~line 836-847)
- Test: `tests/test_gated_one_critic_route.py` (new, light — env-var predicates, no jax)

**Interfaces:**
- New env flag `GATED_ONE_CRITIC=1` (required by Task 4's coherence): under `GATED=1`, build_agent returns the plain `TQCLearner` WITHOUT `value_floor/ceiling` kwargs (bounds arrive via `TQC_TARGET_MIN/MAX` read at import in `critic_updater.py:38-40`), and the buffer is a plain `ReplayBuffer` (rows carry `mode` inside the observation dict). The HER selection block (line ~876) already picks `ImageGoalRelabelBuffer` on `IMAGE_GOAL_TASK=1` and the record call site (line ~1287) already passes `mode=int(inf.get("mode", 1))` — verify, do not change.

- [ ] **Step 1: Write the failing test** — extract the routing predicate into a module-level function `wants_one_critic()` (`GATED=1 and GATED_ONE_CRITIC=1`) and test it; also assert the kwargs builder excludes floor/ceiling on that route (a pure function `agent_kwargs()` is fine to extract).
- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement.** In `build_agent`, first branch:

```python
    if os.environ.get("GATED", "0") == "1" and \
       os.environ.get("GATED_ONE_CRITIC", "0") == "1":
        # ONE critic, mode-conditioned via the obs dict. Bounds come from
        # TQC_TARGET_MIN/MAX (read at import in critic_updater) -- coherence
        # refuses to start the run without them.
        plain = dict(kwargs); plain.pop("value_floor", None); plain.pop("value_ceiling", None)
        return TQCLearner(seed, observation_space.sample(),
                          action_space.sample(), **plain)
```
  (keep the kwargs dict built once; strip the two keys). In the buffer selection:

```python
    if os.environ.get("GATED", "0") == "1" and \
       os.environ.get("GATED_ONE_CRITIC", "0") != "1":
        from vae_nav.mode_split_buffer import ModeSplitBuffer
        ...
    else:
        buffer = ReplayBuffer(...)
```
  Grep the trainer for any `GatedTQCLearner`-only attribute access outside `build_agent` (known: only `INIT_FROM`'s `critic_e` list at line ~1230, which is under `INIT_FROM` and inert here) — confirm none, or guard them.
- [ ] **Step 4: Tests pass; full suite green.**
- [ ] **Step 5: Commit** `feat: GATED_ONE_CRITIC route (plain TQC + plain buffer under GATED)`

### Task 8: `run_gated_ig.sh` — the launch script

**Files:**
- Create: `run_gated_ig.sh` (sibling of `run_image_goal.sh`, structure copied — lock, GPU poll, ig-venv-or-refuse, kill/cleanup, retry, artefact checks, prove-alive doc)

**Interfaces:**
- Consumes: every flag the tasks above introduced. The script sources NOTHING and is the single authority for every value (run_image_goal.sh pattern).

- [ ] **Step 1: Write the script.** Exports (beyond the structural boilerplate copied from `run_image_goal.sh` — lock `/tmp/carla_rl_port2000.lock`, `NEED_MB=6500`, ig-venv-or-refuse):

```bash
export IMAGE_GOAL_TASK=1 GATED=1 GATED_ONE_CRITIC=1
export PERCEPTION=vc1 VC1_READOUT=grid2
export OBS_WITH_STATE=1 OBS_WITH_MODE=1
export EPISODE_HER=1 HER_MIN_DISP=5.0 IG_GOAL_DISTANCE_THRESHOLD_M=2.0
export IG_SPEED_PREF_K=0.1
export DISCOUNT=0.99 TOP_QUANTILES_DROP=2 TQC_KAPPA=1.0
export TEMP_LR=3e-4                      # the v2 divergence fix (entropy brake)
export TQC_TARGET_MIN=-260 TQC_TARGET_MAX=100   # ReViND window at gamma=0.99
export EXPLORE_PROB=0.5 EXPLORE_GOAL_DIST_M=100.0
export TARGET_SPEED=20.0 ENV_SKIP_IMAGE_FEATURES=1
export MAX_EPISODE_STEPS=1000
export GOAL_DIST_MIN="${GOAL_DIST_MIN:-5.0}" \
       GOAL_DIST_START="${GOAL_DIST_START:-25.0}" \
       GOAL_DIST_TARGET="${GOAL_DIST_TARGET:-25.0}"
export EXPERT_PROB=0.0
export TOWN_LIST=configs/town_split.json ENV_TOWNS=Town01 TOWN_ROTATE_EPISODES=0
export N_EVAL_EPISODES=0 N_EVAL_EXPLORE=0
export RUN_NAME="${RUN_NAME:-carla_image_goal_2mode}" POLICY_FOLDER="${POLICY_FOLDER:-$RUN_NAME}"
export MAX_STEPS="${MAX_STEPS:-1000000}" SEED=42 RESUME=1
```
  CARLA block identical to `run_image_goal.sh` (CARLA_START_SERVER=1, Low, offscreen, port 2000). **Scrub block difference:** `TQC_TARGET_MIN/MAX` are REQUIRED here, not scrubbed — the inverse checks assert their PRESENCE. The snapshot (`results/run_env/${RUN_NAME}.env`) gets the same anchored-prefix grep plus required-key checks for: `IMAGE_GOAL_TASK GATED GATED_ONE_CRITIC PERCEPTION VC1_READOUT OBS_WITH_STATE OBS_WITH_MODE EPISODE_HER IG_SPEED_PREF_K TEMP_LR TQC_TARGET_MIN TQC_TARGET_MAX DISCOUNT TOP_QUANTILES_DROP TQC_KAPPA EXPLORE_PROB EXPLORE_GOAL_DIST_M GOAL_DIST_MIN GOAL_DIST_START GOAL_DIST_TARGET TARGET_SPEED N_EVAL_EPISODES RUN_NAME POLICY_FOLDER MAX_STEPS SEED` — and inverse checks that `OBS_WITH_COLLISION` and `TQC_VALUE_FLOOR`/`TQC_VALUE_CEILING` are ABSENT. Banner prints the spec table.
- [ ] **Step 2: Shellcheck + dry review** (`bash -n run_gated_ig.sh`), and confirm NO other CARLA job holds the lock or GPU before any real start (`pgrep -fc 'Linux-Shippin[g]'`, `nvidia-smi`).
- [ ] **Step 3: Commit** `feat: run_gated_ig.sh — 2-mode image-goal launch`

### Task 9: 50k smoke run (the gate before the 1M)

**Files:**
- Run: `MAX_STEPS=50000 RUN_NAME=carla_image_goal_2mode_smoke bash run_gated_ig.sh`

**Interfaces:**
- Consumes: the full stack. The smoke run is an ARTEFACT check, not a courtesy.

- [ ] **Step 1: Preflight (memory rules):** kill any stray CARLA (verified kill loop over `pgrep -f '[C]arlaUE4-Linux-Shipping'` — never a pattern that matches the invoking shell); confirm port 2000 free and free VRAM ≥ 6500 MiB; confirm the snapshot file matches this task's exports.
- [ ] **Step 2: Launch and PROVE ALIVE** from another shell within ~3 minutes of `[CARLA] ready`: `pgrep -af train_carla_tqc` shows the ig-venv interpreter; the log shows, in order: `[Banner]` lines → `[Towns]` → `[Latent] backend 'vc1_vitb_grid2' (dim=37632...)` → `[CARLA] ready` → the TwoMode banner → `[VAEGoal] obs={...'state': (10,), 'mode': (1,), 'latent': (37632,)...}` → `[EpisodeHER] on` → `[Ep]` lines; `nvidia-smi` shows work. GPU utilisation is real.
- [ ] **Step 3: Watch for the five artefact signals** (check at ~10k, ~25k, ~50k):
  1. `entry_done` alive (≥ ~0.5/ep by ep 20 — the v2 fix signature);
  2. `q` within [−260, 100] (the bound banner appeared at startup);
  3. entropy NOT collapsing monotonically (temp_lr=3e-4 watch item);
  4. BOTH `succ_goal` AND `dist_explore`/`len_explore` lines advancing (the 2-mode split is live);
  5. per-step reward components sane — no positive rewards anywhere (`ig_reward_components` in the log's episode stats, or a quick buffer scan).
- [ ] **Step 4: Record the smoke verdict** in the run log; on ANY failed signal, read the log and fix before the 1M — do not just re-run.
- [ ] **Step 5: Commit** any fix the smoke run surfaces.

### Task 10: Offline 2-mode eval script

**Files:**
- Create: `scripts/image_goal_2mode_eval.py` (pattern: `image_goal_demo_350k.py` / `eval_policy.py`)

**Interfaces:**
- Consumes: `run_episodes`, `TwoModeImageGoalReward` via `make_env(eval_mode=True)`, `EVAL_FORCE_MODE`.

- [ ] **Step 1: Write the script.** Pin the full flag set at launch (like `image_goal_demo_350k.py` does with its `os.environ.setdefault` block, PLUS the new `OBS_WITH_STATE=1 OBS_WITH_MODE=1 VC1_READOUT=grid2 GATED=1 IMAGE_GOAL_TASK=1 GATED_ONE_CRITIC=1` — remember the demo lesson: flags read at import time must be exported in the SHELL, and the script's setdefaults are a second layer, not the first). Two passes: `EVAL_FORCE_MODE=1` (goal: success fraction + returns over N episodes) and `EVAL_FORCE_MODE=0` (explore: mean path metres + episode length; success is meaningless and must not be reported). Own port (`EVAL_PORT`, default 2010, NOT the training sim's streaming ports). Writes `results/two_mode_eval/<ckpt>/eval_mode{0,1}.json`.
- [ ] **Step 2: Commit** `feat: mode-forced offline eval for the 2-mode arm`

---

## Verification (whole plan)

1. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q` — full suite green after every task; zero skips introduced.
2. Task 1's probe `allclose` proof (tokens ↔ CLS).
3. Task 9's smoke run with all five artefact signals green.
4. Only then: the 1M run (`bash run_gated_ig.sh`, `carla_image_goal_2mode`), proved alive per the memory rules, watched at the smoke cadence.
5. Task 10's eval on the first saved checkpoint (50k) exercises the eval path BEFORE the 1M finishes.

## Deliberate deviations and watch items (recorded, not hidden)

- `TEMP_LR=3e-4` is the divergence DIAGNOSIS, not a proven fix; the target bounds are the hard backstop.
- `drop=2` at γ=0.99 has one negative memory (gated arms) though v2 trained fine with it — smoke-run watch item 2/3.
- The center-crop confound (Resize256/CenterCrop224) is deliberately untouched: one variable at a time.
- EXPLORE_PROB=0.5 halves goal-mode data vs v2; the framing of this run is the 2-mode mechanism, not a success SOTA (user accepted).
- The v2 single-mode arm is superseded; its checkpoints become unloadable by the obs change.
# SORB-style Agent in CARLA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a short-horizon goal-conditioned CARLA policy whose value function is a validated distance metric, usable as edge weights for graph search.

**Architecture:** TQC (distributional, 5 critics × 25 quantiles) on a pure minimum-time reward with γ=1 and a 300-step cap, so `V = −0.1 × steps` bounded in `[−30, 0]`. Edge costs come from a pessimistic quantile of the critic, converted to steps and clipped at the training horizon H=30 m. A validation harness measures V against two independent ground truths before anything downstream trusts it.

**Tech Stack:** Python 3.10, JAX/Flax via vendored `dependencies/jaxrl2`, CARLA 0.9.15, numpy, scipy 1.15.3 (`spearmanr`), pytest.

## Global Constraints

- `STEP_COST = 0.1`, `TTG_MAX_STEPS = 300`, `MAX_EPISODE_STEPS = 300` — these three must agree; the outer `TimeLimit` is the only other truncation source.
- `DISCOUNT = 1.0`. `TQC_KAPPA = 1.0`.
- `V` range is `[−30, 0]` = `[−STEP_COST × TTG_MAX_STEPS, 0]`. `TQC_TARGET_MIN = -30`, `TQC_TARGET_MAX = 0`.
- `H = 30 m` = **106 steps** at the measured **0.283 m/step**. This is both the goal ceiling and the maximum graph edge.
- `LANE_WEDGE = 0`. Arrival is the only terminal; the step cap is the only truncation.
- Goal band `2–30 m` uniform. `GOAL_DIST_MIN=2.0`, `GOAL_DIST_START=GOAL_DIST_TARGET=30.0` (this makes the curriculum inert by construction).
- `EPISODE_HER = 0`, `EXPERT_PROB = 0.2`, `INCREMENTAL_ACTIONS = 0`, `OBS_WITH_VECTOR = 0`, `MAX_STEER = 0.5`, `MAX_THROTTLE = 0.6`.
- Train on Town02; `Town03`, `Town10HD` are held out and must never be trained on.
- Run `pytest` with `PYTHONPATH=src`.
- Only one CARLA job may hold `/tmp/carla_rl_port2000.lock` at a time. Every run script takes it with `flock` and drops fd 200 before launching the server subshell.

## File Structure

| file | responsibility |
|---|---|
| `run_sorb_agent.sh` (new) | the training run: MDP env vars + CARLA supervisor |
| `tests/test_sorb_config.py` (new) | asserts the run script's constants are mutually coherent |
| `src/vae_nav/value_distance.py` (new) | critic quantiles → pessimistic distance in steps, edge rejection |
| `tests/test_value_distance.py` (new) | unit tests for the above, no JAX/CARLA needed |
| `src/vae_nav/value_validation.py` (new) | rank correlation, per-bin breakdown, acceptance gates |
| `tests/test_value_validation.py` (new) | unit tests for the metrics, synthetic data |
| `validate_value_distance.py` (new) | the harness: collects pairs in CARLA, applies the gates |

Task 1 comes first because training takes hours; the analysis code in Tasks 2–6 is written while it runs.

---

### Task 1: Training run script and its coherence test

**Files:**
- Create: `run_sorb_agent.sh`
- Create: `tests/test_sorb_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: checkpoints in `checkpoints/carla_sorb_h30/checkpoint_<step>.pkl`, each a pickle with keys `agent`, `step`, `curriculum_max`, `eval_success`, `identity`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sorb_config.py`:

```python
"""The run script's constants must agree with each other.

Every bug this catches has actually happened in this project: a step cap that
disagreed with the reward wrapper's cap, a kappa mis-scaled against the reward
so the critic silently stopped learning, and a value bound that did not match
the achievable return range.
"""
import re
import pathlib

import pytest

SCRIPT = pathlib.Path("run_sorb_agent.sh")


def exports():
    """Parse `export NAME=value` pairs from the run script.

    Handles SEVERAL pairs on one `export` line -- the script groups related
    settings that way (`export GOAL_DIST_MIN=2.0 GOAL_DIST_START=30.0 ...`),
    and a parser that took only the first would silently drop the rest and
    make the tests below pass vacuously or KeyError.
    """
    out = {}
    for line in SCRIPT.read_text().splitlines():
        m = re.match(r"^export\s+(.*)", line)
        if not m:
            continue
        body = m.group(1).split("#")[0]          # drop trailing comments
        for pair in re.finditer(r"([A-Z_]+)=(\S+)", body):
            out[pair.group(1)] = pair.group(2).strip('"')
    return out


def test_parser_reads_every_pair_on_a_shared_export_line():
    """Guards the parser itself: a silent miss makes every test below vacuous."""
    e = exports()
    for name in ("GOAL_DIST_MIN", "GOAL_DIST_START", "GOAL_DIST_TARGET",
                 "MAX_STEER", "MAX_THROTTLE", "ENV_TOWNS"):
        assert name in e, f"{name} was not parsed from the run script"


def test_script_exists():
    assert SCRIPT.exists(), f"{SCRIPT} not found"


def test_episode_caps_agree():
    e = exports()
    assert e["TTG_MAX_STEPS"] == e["MAX_EPISODE_STEPS"], (
        "TTG_MAX_STEPS and MAX_EPISODE_STEPS must match; a shorter outer "
        "TimeLimit would cut episodes before the reward wrapper's own cap and "
        "make the reported return shorter than the task allows")


def test_value_bound_matches_achievable_return():
    e = exports()
    floor = -float(e["STEP_COST"]) * int(e["TTG_MAX_STEPS"])
    assert float(e["TQC_TARGET_MIN"]) == pytest.approx(floor), (
        f"TQC_TARGET_MIN={e['TQC_TARGET_MIN']} but the achievable return floor "
        f"is {floor}. A clip inside the achievable range biases the critic; a "
        "clip outside it does nothing.")
    assert float(e["TQC_TARGET_MAX"]) == pytest.approx(0.0)


def test_horizon_is_reachable_within_the_step_cap():
    """H must be traversable with headroom, or episodes time out for reasons
    unrelated to the policy."""
    e = exports()
    metres_per_step = 0.283          # MEASURED over 75 recorded eval episodes
    steps_for_h = float(e["GOAL_DIST_TARGET"]) / metres_per_step
    assert int(e["TTG_MAX_STEPS"]) >= 2.0 * steps_for_h, (
        f"cap {e['TTG_MAX_STEPS']} gives only "
        f"{int(e['TTG_MAX_STEPS']) / steps_for_h:.1f}x the {steps_for_h:.0f} "
        "steps a goal at H needs under perfect driving")


def test_curriculum_is_inert():
    """A fixed band means nothing may advance, demote, or stall-step-back."""
    e = exports()
    assert e["GOAL_DIST_START"] == e["GOAL_DIST_TARGET"], (
        "start must equal target so the curriculum cannot advance")
    assert float(e["GOAL_DIST_MIN"]) < float(e["GOAL_DIST_TARGET"])


def test_wedge_is_off_and_discount_is_one():
    e = exports()
    assert e["LANE_WEDGE"] == "0", (
        "an absorbing failure makes V mean 'distance OR death'; the graph "
        "reads V as travel time")
    assert float(e["DISCOUNT"]) == 1.0


def test_her_off_and_expert_on():
    e = exports()
    assert e["EPISODE_HER"] == "0"
    assert float(e["EXPERT_PROB"]) == pytest.approx(0.2)


def test_heldout_towns_are_not_trained_on():
    e = exports()
    trained = set(e["ENV_TOWNS"].split(","))
    assert not (trained & {"Town03", "Town10HD"}), (
        f"held-out towns appear in ENV_TOWNS={e['ENV_TOWNS']}")


def test_goal_ceiling_within_distance_cap():
    e = exports()
    assert float(e["GOAL_DIST_TARGET"]) < float(e["MAX_DISTANCE_M"]), (
        "the env's own distance terminator would end episodes before the goal")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sorb_config.py -v`
Expected: FAIL — `test_script_exists` fails with "run_sorb_agent.sh not found", and the rest error on the missing file.

- [ ] **Step 3: Write the run script**

Create `run_sorb_agent.sh` (make it executable with `chmod +x`):

```bash
#!/usr/bin/env bash
# SORB-style agent: SHORT-HORIZON policy whose VALUE FUNCTION is the product.
#
#   r = -0.1              every step
#   r =  0   + terminate  on arrival
#            + truncate   at 300 steps
#
# V = -0.1 * steps-to-goal, bounded in [-30, 0] by construction. The graph
# supplies long-range navigation; the policy only has to be right within H=30 m.
#
# Design: docs/superpowers/specs/2026-09-01-sorb-agent-carla-design.md
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla|[r]un_|[w]atch_policy" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000

# ---- the MDP -----------------------------------------------------------------
export TIME_TO_GOAL=1
export STEP_COST=0.1
# 300 steps is 2.8x the ~106 a 30 m goal needs at the MEASURED 0.283 m/step.
# It is also what bounds the undiscounted return: 6.7x tighter than the 2000
# under which q reached -280 against a -200 floor.
export TTG_MAX_STEPS=300
export MAX_EPISODE_STEPS=300
export DISCOUNT=1.0
export TQC_KAPPA=1.0
# The achievable return range, enforced. Every quantile of a distribution
# supported on [-30, 0] lies inside it, so a target outside is unrealisable and
# clipping adds information rather than bias.
export TQC_TARGET_MIN=-30
export TQC_TARGET_MAX=0

# Sparse-reward terms are dead under TIME_TO_GOAL; pinned so nothing carries.
export K_DIST=0.0 K_SPEED=0.0 K_LIVING=0.0

# ---- NO WEDGE ------------------------------------------------------------------
# An absorbing failure makes V mean "distance to goal OR death", and the graph
# reads V as travel time. Collisions and lane crossings cost only the time they
# lose, which is self-calibrating and needs no coefficient.
export LANE_WEDGE=0

# ---- goals: 2-30 m, FIXED --------------------------------------------------------
# V must be accurate everywhere the graph queries it, and graph edges between
# adjacent trace nodes are mostly SHORT. Training only on 20-30 m would leave
# most edges extrapolated.
#
# START == TARGET makes the curriculum inert: nothing can advance, the demote
# branch needs curr_max > START, and the stall branch needs curr_max < TARGET.
export GOAL_DIST_MIN=2.0 GOAL_DIST_START=30.0 GOAL_DIST_TARGET=30.0
export GOAL_DIST_STEP=1.0
export JUNCTION_GOAL_FRAC=0.6
export SHORT_HORIZON=1
export MAX_DISTANCE_M=200

# ---- actions: ABSOLUTE, for parity with the Habitat peer -------------------------
export INCREMENTAL_ACTIONS=0
export MAX_STEER=0.5 MAX_THROTTLE=0.6
export OBS_WITH_VECTOR=0

# ---- environment -----------------------------------------------------------------
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export LANE_STRICT=0
export TARGET_SPEED=20.0
export EPISODE_HER=0
export CHAIN_MAX_GOALS=1
export EXPERT_PROB=0.2

# ---- run -------------------------------------------------------------------------
export POLICY_FOLDER="${POLICY_FOLDER:-carla_sorb_h30}"
export MAX_STEPS="${MAX_STEPS:-500000}"
export RESUME="${RESUME:-1}"
export TEMP_LR=3e-4
export EVAL_VIDEO=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t p < <(pgrep -x CarlaUE4-Linux-Shipping); [ ${#p[@]} -eq 0 ] && return 0
    for q in "${p[@]}"; do kill -9 "$q" 2>/dev/null; done; sleep 3
  done
}
cleanup() { kill_servers; echo "=== sorb agent run done ==="; }
trap cleanup EXIT

for ATTEMPT in 1 2 3 4 5; do
  echo "=== sorb agent attempt $ATTEMPT ==="
  kill_servers
  # fd 200 dropped FIRST: an orphaned grep inheriting this lock once held it
  # for 8.5 hours after the job it belonged to had exited.
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
      -windowed -ResX=640 -ResY=360 2>&1 \
    | grep -v --line-buffered "Invalid session: no stream available" \
    > logs/carla_sorb_h30.log ) &
  sleep 40
  if python -u train_carla_tqc.py; then
    echo "=== sorb agent finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chmod +x run_sorb_agent.sh && bash -n run_sorb_agent.sh && PYTHONPATH=src python -m pytest tests/test_sorb_config.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add run_sorb_agent.sh tests/test_sorb_config.py
git commit -m "feat: SORB agent run script with config coherence tests"
```

- [ ] **Step 6: Start training in the background**

Run: `nohup bash ./run_sorb_agent.sh > logs/sorb_h30.log 2>&1 &`

Then confirm the MDP printed matches the design before leaving it:

Run: `sleep 80 && grep -E "Lock\]|TimeToGoal\]|TQC\] clipping|Curriculum\]" logs/sorb_h30.log`
Expected: `[TimeToGoal] step_cost=0.1 max_steps=300 -> return in [-30.0, 0.0]`, a `[TQC] clipping critic targets to [-30.0, 0.0]` line, and a `[Curriculum] goals 2.0-30.0 m` line.

If the `[TQC] clipping` line is absent, `TQC_TARGET_MIN` did not reach `critic_updater.py` — stop and fix before spending hours of compute.

---

### Task 2: Pessimistic value aggregation

**Files:**
- Create: `src/vae_nav/value_distance.py`
- Create: `tests/test_value_distance.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pooled_quantiles(q: np.ndarray) -> np.ndarray` — `(num_qs, B, n_q)` → `(B, num_qs*n_q)`
  - `pessimistic_value(q: np.ndarray, percentile: float = 10.0) -> np.ndarray` — `(num_qs, B, n_q)` → `(B,)` in V-units
  - `ensemble_spread(q: np.ndarray) -> np.ndarray` — `(num_qs, B, n_q)` → `(B,)`
  - `value_to_steps(v: np.ndarray, step_cost: float = 0.1) -> np.ndarray` — `(B,)` V-units → `(B,)` non-negative steps

- [ ] **Step 1: Write the failing test**

Create `tests/test_value_distance.py`:

```python
"""Turning critic quantiles into a distance.

THE DIRECTION MATTERS. Distance is -V, so UNDERestimating distance means V is
too HIGH. That is the error mode that wrecks a graph: Dijkstra preferentially
selects edges it believes are cheap, so optimistic errors get chosen while
pessimistic ones are ignored. Search concentrates the error rather than
averaging it away.
"""
import numpy as np
import pytest

from vae_nav.value_distance import (ensemble_spread, pessimistic_value,
                                    pooled_quantiles, value_to_steps)

NUM_QS, N_Q = 5, 25


def make_q(values):
    """(B,) per-sample constant -> (num_qs, B, n_q)."""
    v = np.asarray(values, np.float32)
    return np.broadcast_to(v[None, :, None], (NUM_QS, v.size, N_Q)).copy()


def test_pooled_quantiles_shape():
    q = make_q([-1.0, -2.0])
    assert pooled_quantiles(q).shape == (2, NUM_QS * N_Q)


def test_pessimistic_value_on_constant_quantiles_is_that_constant():
    q = make_q([-5.0, -12.0])
    np.testing.assert_allclose(pessimistic_value(q), [-5.0, -12.0], atol=1e-5)


def test_pessimistic_value_picks_the_low_tail_not_the_mean():
    """A low percentile of V = a LARGE distance = the conservative choice."""
    q = np.zeros((NUM_QS, 1, N_Q), np.float32)
    q[..., :] = np.linspace(-20.0, 0.0, N_Q)      # spread from -20 to 0
    v = pessimistic_value(q, percentile=10.0)[0]
    assert v < np.mean(q), "must be below the mean, i.e. a longer distance"
    assert -20.0 <= v <= -14.0, f"10th percentile of [-20,0] should be ~-18, got {v}"


def test_higher_percentile_is_less_conservative():
    q = np.zeros((NUM_QS, 1, N_Q), np.float32)
    q[..., :] = np.linspace(-20.0, 0.0, N_Q)
    assert pessimistic_value(q, 10.0)[0] < pessimistic_value(q, 50.0)[0]


def test_value_to_steps_inverts_the_reward_scale():
    v = np.array([-10.6, -30.0, 0.0])
    np.testing.assert_allclose(value_to_steps(v, step_cost=0.1),
                               [106.0, 300.0, 0.0], atol=1e-4)


def test_value_to_steps_is_never_negative():
    """A critic may predict a small positive V; distance cannot be negative."""
    assert np.all(value_to_steps(np.array([0.5, -1.0]), 0.1) >= 0.0)


def test_ensemble_spread_is_zero_when_critics_agree():
    q = make_q([-7.0])
    assert ensemble_spread(q)[0] == pytest.approx(0.0, abs=1e-6)


def test_ensemble_spread_grows_when_critics_disagree():
    q = make_q([-7.0])
    q[0] -= 5.0                                   # one critic dissents
    assert ensemble_spread(q)[0] > 1.0


def test_batch_entries_are_independent():
    q = make_q([-1.0, -20.0])
    v = pessimistic_value(q)
    assert v[0] == pytest.approx(-1.0, abs=1e-5)
    assert v[1] == pytest.approx(-20.0, abs=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_value_distance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vae_nav.value_distance'`

- [ ] **Step 3: Write the implementation**

Create `src/vae_nav/value_distance.py`:

```python
"""The critic's value, read as a distance the graph can use.

UNITS. Two are in play and they differ by 1/STEP_COST, so they are named
apart everywhere:

    V-units   what the critic predicts, range [-30, 0]
    steps     -V / STEP_COST, non-negative, what Dijkstra receives

Mixing them silently is easy and would put every edge weight out by 10x.
"""
import numpy as np

STEP_COST_DEFAULT = 0.1
# The 10th percentile of the pooled quantiles. Conservative on DISTANCE.
PESSIMISM_PERCENTILE = 10.0


def pooled_quantiles(q):
    """(num_qs, B, n_q) -> (B, num_qs * n_q). All critics' quantiles together."""
    q = np.asarray(q, np.float64)
    if q.ndim != 3:
        raise ValueError(f"expected (num_qs, B, n_q), got shape {q.shape}")
    num_qs, batch, n_q = q.shape
    return q.transpose(1, 0, 2).reshape(batch, num_qs * n_q)


def pessimistic_value(q, percentile=PESSIMISM_PERCENTILE):
    """A LOW percentile of V, i.e. a LONG distance. (num_qs, B, n_q) -> (B,).

    Low rather than mean because distance is -V: an optimistically high V
    understates distance, and graph search actively selects for exactly that
    error. TQC already drops the top (most optimistic) quantiles from its
    target, so this extends the existing bias rather than fighting it.
    """
    return np.percentile(pooled_quantiles(q), percentile, axis=1)


def ensemble_spread(q):
    """Std across critics of their per-critic mean. (num_qs, B, n_q) -> (B,).

    Disagreement between independently-initialised critics is the cheapest
    available signal that a (state, goal) pair is off-distribution.
    """
    q = np.asarray(q, np.float64)
    return q.mean(axis=2).std(axis=0)


def value_to_steps(v, step_cost=STEP_COST_DEFAULT):
    """V-units -> non-negative steps.

    Clipped at zero: the critic may predict a small positive V for a state
    already at the goal, and a negative distance is meaningless downstream.
    """
    if step_cost <= 0:
        raise ValueError(f"step_cost must be positive, got {step_cost}")
    return np.maximum(0.0, -np.asarray(v, np.float64) / step_cost)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_value_distance.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vae_nav/value_distance.py tests/test_value_distance.py
git commit -m "feat: pessimistic value aggregation for graph edge weights"
```

---

### Task 3: Goal-relabelled observations and edge rejection

**Files:**
- Modify: `src/vae_nav/value_distance.py`
- Modify: `tests/test_value_distance.py`

**Interfaces:**
- Consumes: `value_to_steps`, `ensemble_spread` from Task 2; `VAEGoalWrapper.goal_block(fields) -> np.ndarray` (12-d), where `fields` has keys `hero_pos` (2,), `hero_yaw` float radians, `goal_pos` (2,), `action_history` `(4, 2)`.
- Produces:
  - `goal_relabelled_obs(base_obs: dict, fields: dict, goal_pos, goal_block_fn) -> dict`
  - `edge_reject_mask(cost_steps, spread, max_steps: float, tau: float) -> np.ndarray[bool]`
  - `H_STEPS = 106.0`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_value_distance.py`:

```python
from vae_nav.value_distance import (H_STEPS, edge_reject_mask,
                                    goal_relabelled_obs)


def fake_goal_block(fields):
    """Stand-in for VAEGoalWrapper.goal_block: only the goal offset matters."""
    d = np.asarray(fields["goal_pos"], np.float64) - np.asarray(
        fields["hero_pos"], np.float64)
    return np.concatenate([d, np.zeros(10)]).astype(np.float32)


def test_goal_relabelled_obs_rewrites_only_the_goal_key():
    base = {"latent": np.arange(4, dtype=np.float32),
            "goal": np.zeros(12, np.float32)}
    fields = {"hero_pos": (0.0, 0.0), "hero_yaw": 0.0,
              "action_history": np.zeros((4, 2))}
    out = goal_relabelled_obs(base, fields, (3.0, 4.0), fake_goal_block)
    np.testing.assert_allclose(out["goal"][:2], [3.0, 4.0])
    # The latent is what the camera SAW and must not change with the goal.
    np.testing.assert_array_equal(out["latent"], base["latent"])


def test_goal_relabelled_obs_does_not_mutate_the_input():
    base = {"latent": np.zeros(4, np.float32), "goal": np.zeros(12, np.float32)}
    fields = {"hero_pos": (0.0, 0.0), "hero_yaw": 0.0,
              "action_history": np.zeros((4, 2))}
    goal_relabelled_obs(base, fields, (1.0, 1.0), fake_goal_block)
    assert base["goal"].sum() == 0.0, "caller's dict was mutated"


def test_h_steps_matches_the_measured_traversal_rate():
    """30 m at the measured 0.283 m/step."""
    assert H_STEPS == pytest.approx(30.0 / 0.283, rel=0.01)


def test_edges_beyond_the_horizon_are_rejected():
    cost = np.array([50.0, 105.0, 107.0, 300.0])
    spread = np.zeros(4)
    rej = edge_reject_mask(cost, spread, max_steps=H_STEPS, tau=1.0)
    assert list(rej) == [False, False, True, True], (
        "V is only trained on goals within H; beyond it every query is "
        "extrapolation, which is how q reached -280 against a -200 floor")


def test_edges_with_disagreeing_critics_are_rejected():
    cost = np.array([10.0, 10.0])
    spread = np.array([0.1, 9.0])
    assert list(edge_reject_mask(cost, spread, H_STEPS, tau=1.0)) == [False, True]


def test_rejection_reasons_compose():
    cost = np.array([500.0])
    spread = np.array([99.0])
    assert edge_reject_mask(cost, spread, H_STEPS, 1.0)[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_value_distance.py -v`
Expected: FAIL — `ImportError: cannot import name 'H_STEPS'`

- [ ] **Step 3: Write the implementation**

Append to `src/vae_nav/value_distance.py`:

```python
# MEASURED over 75 recorded eval episodes: 0.283 m per step (median, max 0.327)
# at ~11 km/h on a 0.1 s timestep.
METRES_PER_STEP = 0.283
H_METRES = 30.0
H_STEPS = H_METRES / METRES_PER_STEP        # 106.0


def goal_relabelled_obs(base_obs, fields, goal_pos, goal_block_fn):
    """Copy `base_obs` with its goal block recomputed for `goal_pos`.

    NO SIMULATOR NEEDED. The VAE latent is what the camera saw at this state
    and is independent of which goal the step is credited to, so only the 12-d
    goal block is rebuilt -- through the wrapper's OWN transform, so relabelled
    observations cannot drift from on-policy ones. Same contract as
    EpisodeRelabelBuffer._relabelled.
    """
    out = dict(base_obs)
    out["goal"] = goal_block_fn({**fields, "goal_pos": goal_pos})
    return out


def edge_reject_mask(cost_steps, spread, max_steps=H_STEPS, tau=None):
    """True where an edge must NOT enter the graph.

    Two independent reasons:
      * beyond H -- V was trained only on goals within the horizon, so a longer
        query is extrapolation and unclipped extrapolation is how the critic
        left its provable range entirely;
      * critics disagree -- the cheapest available off-distribution signal.

    `tau` of None disables the disagreement test, for when no spread
    distribution has been observed yet to set it from.
    """
    cost_steps = np.asarray(cost_steps, np.float64)
    reject = cost_steps > max_steps
    if tau is not None:
        reject = reject | (np.asarray(spread, np.float64) > tau)
    return reject
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_value_distance.py -v`
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vae_nav/value_distance.py tests/test_value_distance.py
git commit -m "feat: goal relabelling and horizon/disagreement edge rejection"
```

---

### Task 4: Query the critic for arbitrary state-goal pairs

**Files:**
- Modify: `src/vae_nav/value_distance.py`
- Modify: `tests/test_value_distance.py`

**Interfaces:**
- Consumes: `pessimistic_value`, `ensemble_spread`, `value_to_steps` from Task 2. A TQC agent exposing `eval_actions(obs) -> np.ndarray` and `_critic` (a Flax `TrainState` with `.apply_fn` and `.params`).
- Produces: `critic_quantiles(agent, observations, actions) -> np.ndarray` `(num_qs, B, n_q)`; `edge_cost_steps(agent, observations, step_cost=0.1, percentile=10.0) -> (cost_steps, spread)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_value_distance.py`:

```python
from vae_nav.value_distance import critic_quantiles, edge_cost_steps


class StubCritic:
    """Mimics a Flax TrainState carrying the quantile ensemble."""

    def __init__(self, q):
        self._q = q
        self.params = {"dummy": 0}

    def apply_fn(self, variables, observations, actions):
        return self._q


class StubAgent:
    """Enough of a TQC agent for the query path, with no JAX involved."""

    def __init__(self, q, action_dim=2):
        self._critic = StubCritic(q)
        self._action_dim = action_dim
        self.seen_actions = None

    def eval_actions(self, observations):
        n = len(next(iter(observations.values())))
        return np.zeros((n, self._action_dim), np.float32)


def test_critic_quantiles_returns_the_ensemble_shape():
    q = make_q([-4.0, -8.0])
    agent = StubAgent(q)
    obs = {"latent": np.zeros((2, 4), np.float32),
           "goal": np.zeros((2, 12), np.float32)}
    out = critic_quantiles(agent, obs, agent.eval_actions(obs))
    assert out.shape == (NUM_QS, 2, N_Q)


def test_edge_cost_steps_converts_to_steps():
    """V = -10.6 must come back as ~106 steps, i.e. exactly H."""
    q = make_q([-10.6])
    agent = StubAgent(q)
    obs = {"latent": np.zeros((1, 4), np.float32),
           "goal": np.zeros((1, 12), np.float32)}
    cost, spread = edge_cost_steps(agent, obs, step_cost=0.1)
    assert cost[0] == pytest.approx(106.0, abs=0.5)
    assert spread[0] == pytest.approx(0.0, abs=1e-6)


def test_edge_cost_steps_is_pessimistic_not_mean():
    q = np.zeros((NUM_QS, 1, N_Q), np.float32)
    q[..., :] = np.linspace(-20.0, 0.0, N_Q)
    agent = StubAgent(q)
    obs = {"latent": np.zeros((1, 4), np.float32),
           "goal": np.zeros((1, 12), np.float32)}
    cost, _ = edge_cost_steps(agent, obs, step_cost=0.1)
    mean_cost = -np.mean(q) / 0.1
    assert cost[0] > mean_cost, "must over-estimate distance, never under"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_value_distance.py -v`
Expected: FAIL — `ImportError: cannot import name 'critic_quantiles'`

- [ ] **Step 3: Write the implementation**

Append to `src/vae_nav/value_distance.py`:

```python
def critic_quantiles(agent, observations, actions):
    """Raw ensemble quantiles for a batch. -> (num_qs, B, n_q).

    Reaches into `agent._critic` because jaxrl2's Agent base exposes only
    eval_actions; the critic has no public accessor. Kept in one place so the
    coupling is visible rather than scattered through the harness.
    """
    critic = agent._critic
    qs = critic.apply_fn({"params": critic.params}, observations, actions)
    return np.asarray(qs, np.float64)


def edge_cost_steps(agent, observations, step_cost=STEP_COST_DEFAULT,
                    percentile=PESSIMISM_PERCENTILE):
    """Edge weight in STEPS, plus ensemble disagreement, for a batch.

    V(s) = Q(s, pi(s)): the value of the state under the policy that will
    actually be asked to traverse the edge, not under an arbitrary action.
    """
    actions = agent.eval_actions(observations)
    q = critic_quantiles(agent, observations, actions)
    v = pessimistic_value(q, percentile)
    return value_to_steps(v, step_cost), ensemble_spread(q)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_value_distance.py -v`
Expected: PASS, 18 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vae_nav/value_distance.py tests/test_value_distance.py
git commit -m "feat: critic query path producing edge costs in steps"
```

---

### Task 5: Validation metrics and acceptance gates

**Files:**
- Create: `src/vae_nav/value_validation.py`
- Create: `tests/test_value_validation.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure metric functions).
- Produces:
  - `DISTANCE_BINS_M = [(2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0)]`
  - `spearman(pred, true) -> float`
  - `median_abs_rel_error(pred_steps, true_steps) -> float`
  - `per_bin_spearman(pred_steps, true_steps, dist_m, bins=DISTANCE_BINS_M) -> dict[str, dict]`
  - `gate_report(pred_steps, true_steps, dist_m, v_values, v_min=-30.0, v_max=0.0) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_value_validation.py`:

```python
"""Acceptance gates for V as a distance metric.

THE PRIMARY METRIC IS RANK CORRELATION, NOT ABSOLUTE ERROR. Dijkstra's path
selection depends only on the RELATIVE ordering of edge costs, so a V that is
uniformly 30% too long still routes correctly. Absolute calibration is
secondary and affects only whether reported costs are meaningful.
"""
import numpy as np
import pytest

from vae_nav.value_validation import (DISTANCE_BINS_M, gate_report,
                                      median_abs_rel_error, per_bin_spearman,
                                      spearman)


def test_spearman_is_one_for_a_monotone_relationship():
    true = np.arange(1, 21, dtype=float)
    assert spearman(true * 3.0 + 7.0, true) == pytest.approx(1.0)


def test_spearman_ignores_a_constant_scale_error():
    """A V uniformly 30% too long still routes correctly."""
    true = np.arange(1, 21, dtype=float)
    assert spearman(true * 1.3, true) == pytest.approx(1.0)


def test_spearman_is_near_zero_for_noise():
    rng = np.random.default_rng(0)
    true = np.arange(1, 201, dtype=float)
    assert abs(spearman(rng.permutation(true), true)) < 0.3


def test_median_abs_rel_error_of_a_perfect_predictor_is_zero():
    true = np.array([10.0, 50.0, 100.0])
    assert median_abs_rel_error(true, true) == pytest.approx(0.0)


def test_median_abs_rel_error_of_a_uniform_overestimate():
    true = np.array([10.0, 50.0, 100.0])
    assert median_abs_rel_error(true * 1.25, true) == pytest.approx(0.25)


def test_per_bin_reveals_a_predictor_that_fails_only_at_short_range():
    """THE CASE THIS EXISTS FOR.

    A V accurate at 25 m and meaningless at 3 m still scores well pooled,
    while every short edge in the graph is garbage. Only the bins show it.
    """
    rng = np.random.default_rng(1)
    short_d = rng.uniform(2.0, 5.0, 200)
    long_d = rng.uniform(20.0, 30.0, 200)
    dist = np.concatenate([short_d, long_d])
    true = dist / 0.283
    pred = np.concatenate([rng.permutation(short_d / 0.283),   # garbage
                           long_d / 0.283])                     # perfect
    assert spearman(pred, true) > 0.8, "pooled score hides the failure"
    bins = per_bin_spearman(pred, true, dist)
    assert bins["2.0-5.0m"]["rho"] < 0.5
    assert bins["20.0-30.0m"]["rho"] > 0.9


def test_per_bin_reports_counts_and_skips_empty_bins():
    dist = np.array([3.0, 3.5, 4.0, 25.0, 26.0, 27.0])
    true = dist / 0.283
    out = per_bin_spearman(true, true, dist)
    assert out["2.0-5.0m"]["n"] == 3
    assert "5.0-10.0m" not in out, "empty bins must not report a rho"


def test_gate_report_passes_a_good_predictor():
    rng = np.random.default_rng(2)
    dist = rng.uniform(2.0, 30.0, 400)
    true = dist / 0.283
    pred = true * rng.uniform(0.95, 1.05, dist.size)
    v = -0.1 * pred
    rep = gate_report(pred, true, dist, v)
    assert rep["passed"] is True
    assert rep["gates"]["v_in_bounds"]["passed"] is True


def test_gate_report_fails_when_v_leaves_the_achievable_range():
    """This is the failure that actually happened: q = -280 against a -200 floor."""
    dist = np.array([10.0, 20.0])
    true = dist / 0.283
    v = np.array([-5.0, -45.0])           # -45 is outside [-30, 0]
    rep = gate_report(true, true, dist, v)
    assert rep["gates"]["v_in_bounds"]["passed"] is False
    assert rep["passed"] is False


def test_gate_report_fails_on_poor_rank_correlation():
    rng = np.random.default_rng(3)
    dist = rng.uniform(2.0, 30.0, 300)
    true = dist / 0.283
    pred = rng.permutation(true)
    rep = gate_report(pred, true, dist, -0.1 * pred)
    assert rep["gates"]["spearman"]["passed"] is False
    assert rep["passed"] is False


def test_gate_report_is_json_serialisable():
    import json
    dist = np.array([3.0, 10.0, 25.0])
    true = dist / 0.283
    json.dumps(gate_report(true, true, dist, -0.1 * true))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_value_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vae_nav.value_validation'`

- [ ] **Step 3: Write the implementation**

Create `src/vae_nav/value_validation.py`:

```python
"""Does V actually predict cost-to-go?

This has been asserted repeatedly in this project and never measured. The
gates below are the measurement.

RANK CORRELATION IS PRIMARY. Dijkstra depends only on the relative ordering of
edge costs, so a uniformly-scaled V still routes correctly; absolute error
affects only whether the reported cost means anything.
"""
import numpy as np
from scipy.stats import spearmanr

# Bins chosen to match the training band. Graph edges after thinning are mostly
# SHORT, so the 2-5 m bin is the one most likely to be quietly broken.
DISTANCE_BINS_M = [(2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0)]

# PROVISIONAL. These are estimates, not derived values -- there is no prior
# measurement of V calibration in this project to set them from. Revisit once
# the first report exists.
GATE_SPEARMAN = 0.8
GATE_MEDIAN_REL_ERROR = 0.25
GATE_PER_BIN_SPEARMAN = 0.5


def spearman(pred, true):
    """Rank correlation. Returns 0.0 rather than nan for degenerate input."""
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    if pred.size < 3 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return 0.0
    rho = spearmanr(pred, true).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def median_abs_rel_error(pred_steps, true_steps):
    """Median |pred - true| / true. Pairs with true == 0 are dropped."""
    pred = np.asarray(pred_steps, np.float64)
    true = np.asarray(true_steps, np.float64)
    ok = true > 0
    if not ok.any():
        return float("nan")
    return float(np.median(np.abs(pred[ok] - true[ok]) / true[ok]))


def per_bin_spearman(pred_steps, true_steps, dist_m, bins=DISTANCE_BINS_M):
    """Rank correlation within each distance band.

    Pooled rho can look healthy while an entire band is garbage -- and the
    short band is exactly where most graph edges live.
    """
    pred = np.asarray(pred_steps, np.float64)
    true = np.asarray(true_steps, np.float64)
    dist = np.asarray(dist_m, np.float64)
    out = {}
    for lo, hi in bins:
        sel = (dist >= lo) & (dist < hi)
        if sel.sum() < 3:
            continue
        out[f"{lo}-{hi}m"] = {"n": int(sel.sum()),
                              "rho": spearman(pred[sel], true[sel])}
    return out


def gate_report(pred_steps, true_steps, dist_m, v_values,
                v_min=-30.0, v_max=0.0):
    """Every acceptance gate, with the numbers behind each verdict."""
    v = np.asarray(v_values, np.float64)
    rho = spearman(pred_steps, true_steps)
    rel = median_abs_rel_error(pred_steps, true_steps)
    bins = per_bin_spearman(pred_steps, true_steps, dist_m)
    n_out = int(((v < v_min) | (v > v_max)).sum())
    worst_bin = min((b["rho"] for b in bins.values()), default=float("nan"))

    gates = {
        "spearman": {"value": rho, "threshold": GATE_SPEARMAN,
                     "passed": rho > GATE_SPEARMAN},
        "median_rel_error": {"value": rel, "threshold": GATE_MEDIAN_REL_ERROR,
                             "passed": bool(rel < GATE_MEDIAN_REL_ERROR)},
        "v_in_bounds": {"value": n_out, "threshold": 0,
                        "passed": n_out == 0,
                        "range": [v_min, v_max]},
        "per_bin_spearman": {"value": worst_bin,
                             "threshold": GATE_PER_BIN_SPEARMAN,
                             "passed": bool(bins) and worst_bin > GATE_PER_BIN_SPEARMAN},
    }
    return {"n_pairs": int(np.asarray(true_steps).size),
            "bins": bins,
            "gates": gates,
            "passed": all(g["passed"] for g in gates.values())}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_value_validation.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vae_nav/value_validation.py tests/test_value_validation.py
git commit -m "feat: acceptance gates for V as a distance metric"
```

---

### Task 6: Validation harness

**Files:**
- Create: `validate_value_distance.py`
- Modify: `tests/test_value_validation.py`

**Interfaces:**
- Consumes: `edge_cost_steps`, `goal_relabelled_obs`, `H_STEPS` (Tasks 3–4); `gate_report`, `per_bin_spearman` (Task 5); `train_carla_tqc.make_env` / `build_agent`; `vae_nav.policy_io.load_agent_state`.
- Produces: `results/value_validation_<town>.json` and a `[Gate]` summary on stdout.

- [ ] **Step 1: Write the failing test for the pair collector**

Append to `tests/test_value_validation.py`:

```python
def test_collect_pairs_keeps_only_successful_episodes():
    """Realised steps exist only where the goal was reached.

    An episode that timed out has no ground-truth distance -- counting its
    length as the true distance would teach the gate that failures are simply
    long journeys.
    """
    from validate_value_distance import pairs_from_episodes

    episodes = [
        {"arrived": True, "steps": 80, "goal_dist_m": 22.0, "v0": -8.0},
        {"arrived": False, "steps": 300, "goal_dist_m": 28.0, "v0": -30.0},
        {"arrived": True, "steps": 30, "goal_dist_m": 8.0, "v0": -3.0},
    ]
    pred, true, dist, v = pairs_from_episodes(episodes, step_cost=0.1)
    assert list(true) == [80.0, 30.0]
    assert list(dist) == [22.0, 8.0]
    np.testing.assert_allclose(pred, [80.0, 30.0], atol=1e-6)


def test_collect_pairs_returns_empty_arrays_when_nothing_arrived():
    from validate_value_distance import pairs_from_episodes

    pred, true, dist, v = pairs_from_episodes(
        [{"arrived": False, "steps": 300, "goal_dist_m": 20.0, "v0": -30.0}],
        step_cost=0.1)
    assert len(pred) == len(true) == len(dist) == len(v) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_value_validation.py -k pairs -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'validate_value_distance'`

- [ ] **Step 3: Write the harness**

Create `validate_value_distance.py`:

```python
"""Measure whether V predicts cost-to-go, against two independent truths.

    realised   successful episodes: V(s_0, g) vs steps actually taken
    geodesic   GlobalRoutePlanner along-road distance / 0.283 m per step

The second covers pairs the policy fails, where realised steps do not exist.

A NEGATIVE CONTROL is run alongside: the identical evaluation against an
UNTRAINED critic. If a random V scores rho = 0.6 through goal-block geometry
alone, then 0.8 from the trained one is a far weaker result than it looks.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from vae_nav.value_distance import (METRES_PER_STEP, edge_cost_steps,  # noqa: E402
                                    value_to_steps)
from vae_nav.value_validation import gate_report  # noqa: E402


def pairs_from_episodes(episodes, step_cost=0.1):
    """Successful episodes -> (pred_steps, true_steps, dist_m, v0).

    ONLY ARRIVALS. A timed-out episode has no ground-truth distance; treating
    its length as the truth would teach the gate that failure is just a long
    journey.
    """
    pred, true, dist, v0 = [], [], [], []
    for ep in episodes:
        if not ep.get("arrived"):
            continue
        pred.append(float(value_to_steps(np.array([ep["v0"]]), step_cost)[0]))
        true.append(float(ep["steps"]))
        dist.append(float(ep["goal_dist_m"]))
        v0.append(float(ep["v0"]))
    return (np.asarray(pred), np.asarray(true),
            np.asarray(dist), np.asarray(v0))


def run_episodes(agent, env, n_episodes, seed0, step_cost, max_steps):
    """Roll out, recording V at the START state against the realised length."""
    out = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed0 + i)
        start = np.asarray(info["hero_pos"], np.float64)[:2]
        goal = np.asarray(info["goal_pos"], np.float64)[:2]
        batched = {k: np.asarray(v)[None] for k, v in obs.items()}
        cost0, spread0 = edge_cost_steps(agent, batched, step_cost)
        v0 = -float(cost0[0]) * step_cost

        arrived, steps, done = False, 0, False
        while not done and steps < max_steps:
            obs, _r, term, trunc, info = env.step(agent.eval_actions(obs))
            steps += 1
            arrived = arrived or bool(info.get("arrived") or info.get("is_success"))
            done = bool(term or trunc)
        out.append({"arrived": arrived, "steps": steps,
                    "goal_dist_m": float(np.linalg.norm(goal - start)),
                    "v0": v0, "spread0": float(spread0[0])})
        print(f"  ep {i:3d}  arrived={arrived}  steps={steps:4d}  "
              f"V0={v0:7.2f}  pred={cost0[0]:6.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--towns", default="Town02,Town03,Town10HD")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--step-cost", type=float,
                    default=float(os.environ.get("STEP_COST", 0.1)))
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("TTG_MAX_STEPS", 300)))
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    import train_carla_tqc as T
    from vae_nav.latent_source import LatentSource
    from vae_nav.policy_io import load_agent_state, param_fingerprint

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    if "agent" not in ck:
        raise SystemExit(f"{args.ckpt} holds no weights (keys: {sorted(ck)})")
    print(f"[Validate] checkpoint step={ck.get('step')}")
    print(f"[Validate] {param_fingerprint(ck['agent'])}")

    latent_source = LatentSource(os.environ.get("VAE_CKPT", ""))
    towns = [t for t in args.towns.split(",") if t]

    probe = T.make_env(towns[0], args.seed, latent_source, eval_mode=True,
                       port=args.port)
    trained = T.build_agent(probe.observation_space, probe.action_space, args.seed)
    # NEGATIVE CONTROL: same architecture, same seed, NO loaded weights.
    control = T.build_agent(probe.observation_space, probe.action_space,
                            args.seed + 1)
    probe.close()
    load_agent_state(trained, ck["agent"])

    os.makedirs(args.out_dir, exist_ok=True)
    for town in towns:
        env = T.make_env(town, args.seed, latent_source, eval_mode=True,
                         port=args.port)
        try:
            print(f"\n=== {town} (trained) ===", flush=True)
            eps = run_episodes(trained, env, args.episodes, args.seed,
                               args.step_cost, args.max_steps)
            print(f"\n=== {town} (UNTRAINED negative control) ===", flush=True)
            ctl = run_episodes(control, env, args.episodes, args.seed,
                               args.step_cost, args.max_steps)
        finally:
            env.close()

        rep = gate_report(*pairs_from_episodes(eps, args.step_cost))
        ctl_rep = gate_report(*pairs_from_episodes(ctl, args.step_cost))
        rep["negative_control_spearman"] = ctl_rep["gates"]["spearman"]["value"]
        rep["town"] = town
        rep["arrival_rate"] = float(np.mean([e["arrived"] for e in eps]))
        rep["heldout"] = town in ("Town03", "Town10HD")

        path = os.path.join(args.out_dir, f"value_validation_{town}.json")
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)

        print(f"\n[Gate] {town}  {'PASS' if rep['passed'] else 'FAIL'}  "
              f"(n={rep['n_pairs']}, arrival={rep['arrival_rate']:.2f}"
              f"{', HELD OUT' if rep['heldout'] else ''})")
        for name, g in rep["gates"].items():
            mark = "ok  " if g["passed"] else "FAIL"
            print(f"   {mark} {name:18s} {g['value']} vs {g['threshold']}")
        print(f"   negative control rho = {rep['negative_control_spearman']:.3f} "
              "(trained rho must beat this by a clear margin)")
        for b, s in rep["bins"].items():
            print(f"     bin {b:12s} n={s['n']:4d}  rho={s['rho']:.3f}")
        print(f"   -> {path}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_value_validation.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Commit**

```bash
git add validate_value_distance.py tests/test_value_validation.py
git commit -m "feat: V-vs-realised-time validation harness with negative control"
```

- [ ] **Step 6: Run the whole suite**

Run: `PYTHONPATH=src python -m pytest tests/ -q`
Expected: PASS. The pre-existing suites (`test_her_time_to_goal.py`, `test_wedge_truncate.py`, `test_wedge_lane_filter.py`, `test_vae_nav_stack.py`, `test_vec_training_loop.py`) must still pass — nothing in this plan changes their behaviour.

- [ ] **Step 7: Validate the trained checkpoint**

Only once training from Task 1 has produced a checkpoint, and with training **stopped** so the CARLA lock is free:

```bash
POLICY_FOLDER=carla_sorb_h30 \
STEP_COST=0.1 TTG_MAX_STEPS=300 MAX_EPISODE_STEPS=300 \
LANE_WEDGE=0 OBS_WITH_VECTOR=0 INCREMENTAL_ACTIONS=0 \
MAX_STEER=0.5 MAX_THROTTLE=0.6 ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120 \
VAE_CKPT=checkpoints/lane_vae_120/best.pkl \
GOAL_DIST_MIN=2.0 GOAL_DIST_MAX=30.0 MAX_DISTANCE_M=200 \
python -u validate_value_distance.py \
  --ckpt checkpoints/carla_sorb_h30/checkpoint_500000.pkl \
  --towns Town02,Town03,Town10HD --episodes 60
```

Expected: three `[Gate]` blocks. Town02 is in-distribution; Town03 and Town10HD are held out and are the ones that matter for any claim about generalisation.

**If the gates fail**, the order of response is fixed by the design: re-enable HER (`EPISODE_HER=1`) for denser short-range distance supervision, then more steps, then reconsider a supervised distance head. Do not tune the gate thresholds to make a failing V pass.

---

## Self-Review

**Spec coverage.** Every section of the design maps to a task: the MDP and all its constants → Task 1 (asserted, not just written); pessimistic aggregation and the units split → Tasks 2 and 4; goal relabelling without a simulator, horizon clipping, and disagreement rejection → Task 3; rank correlation, per-bin breakdown, the four gates, the negative control, held-out towns, and the failure branch → Tasks 5 and 6. The directed-graph requirement is honoured by construction — `edge_cost_steps` takes an explicit `(state, goal)` pair and is called once per direction; the graph spec consumes it. Out-of-scope items (node selection, thinning, Dijkstra, the long-horizon baseline) have no tasks, correctly.

**Placeholder scan.** No TBDs, no "add error handling", no "similar to Task N". Every code step carries runnable code. The one deliberately unset value, `tau`, is a documented `None` default with the reason attached, not a placeholder — the spec lists it as an open question because no spread distribution exists to set it from yet.

**Type consistency.** `pessimistic_value`, `ensemble_spread`, `value_to_steps`, `pooled_quantiles`, `goal_relabelled_obs`, `edge_reject_mask`, `critic_quantiles`, `edge_cost_steps`, `H_STEPS`, `METRES_PER_STEP`, `STEP_COST_DEFAULT`, `PESSIMISM_PERCENTILE` are each defined once and used under the same names throughout. Quantile arrays are `(num_qs, B, n_q)` everywhere. `gate_report(pred_steps, true_steps, dist_m, v_values)` matches `pairs_from_episodes`' four-tuple return order exactly, which is what lets Task 6 splat it.

**One gap found and closed during review:** Task 6's harness imports `METRES_PER_STEP`, which Task 3 defines — the import is present and the constant is exported from the same module, so no circular or missing dependency.

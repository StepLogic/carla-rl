"""Recording expert (and DAgger) episodes for imitation (2026-11-04).

Imitation-first plan, Tasks 4-5. This module holds the pure logic -- the
held-out guard, the per-episode schedule, recovery noise, which action is
executed and which is the label, and the M0 expert gate -- plus the recording
loop itself (run_record, Task 5), which drives the unchanged env stack.

RECOVERY NOISE (DART). With the expert driving, the EXECUTED steering gets
N(0, sigma) noise while the stored LABEL stays the expert's clean action, so
the data shows how to come back from drifting: the expert's next action is
computed from the perturbed state it was pushed into.
"""
import numpy as np

HELD_OUT_TOWNS = ("Town02", "Town03")
FAULT_ENDS = ("crossing", "off_road", "collision")
SIGMAS = (0.0, 0.1, 0.2)


def assert_recordable_town(town):
    """Refuse to record in a held-out town: those towns are for evaluation only."""
    if town in HELD_OUT_TOWNS:
        raise ValueError(f"{town} is held out for evaluation: no expert data may be "
                         f"recorded there (held out: {', '.join(HELD_OUT_TOWNS)})")


def episode_plan(seed):
    """Deterministic per-episode schedule: alternate goal (1) / explore (0)
    episodes, and cycle the recovery-noise level through SIGMAS."""
    seed = int(seed)
    return {"mode": 1 if seed % 2 == 0 else 0, "sigma": SIGMAS[(seed // 2) % len(SIGMAS)]}


def effective_sigma(sigma, dry):
    """The recovery-noise level actually applied. A dry run is the M0 gate,
    which measures the expert itself, so it runs without noise: in the first
    diagnostic (job 463139) every fault came from a sigma=0.2 episode."""
    return 0.0 if dry else float(sigma)


def dart_action(a_env, sigma, rng, low, high):
    """The executed action: N(0, sigma) added to the steering (index 0) only,
    clipped to the env box. A copy, and exactly `a_env` when sigma == 0."""
    out = np.array(a_env, dtype=np.float32, copy=True)
    if sigma > 0.0:
        out[0] += np.float32(rng.normal(0.0, sigma))
    return np.clip(out, low, high).astype(np.float32)


def record_step(driver, a_policy, a_expert, sigma, rng, low, high):
    """(a_exec, a_label) for one step. The label is always the expert's action;
    what is executed is the noisy expert ("expert") or the policy ("policy")."""
    a_label = np.asarray(a_expert, np.float32)
    if driver == "expert":
        return dart_action(a_label, sigma, rng, low, high), a_label
    if driver == "policy":
        return np.asarray(a_policy, np.float32), a_label
    raise ValueError(f"driver must be 'expert' or 'policy', got {driver!r}")


def keep_episode(n_steps):
    """An episode with no recorded step (no plan at step 0) is discarded."""
    return int(n_steps) > 0


def gate_report(goal_eps, explore_eps, min_arrival=0.95, min_m_between=1000.0,
                min_explore_m=2000.0):
    """The M0 expert gate. Goal episodes give the arrival rate; explore episodes
    give metres between failures (a failure = a crossing, off-road or collision
    end -- a route that simply ends, or the clock, is not one)."""
    n_goal = len(goal_eps)
    arrival = (sum(1 for e in goal_eps if e["end"] == "reached") / n_goal
               if n_goal else 0.0)
    metres = float(sum(float(e["path_m"]) for e in explore_eps))
    faults = sum(1 for e in explore_eps if e["end"] in FAULT_ENDS)
    m_between = metres / faults if faults else float("inf")
    reasons = []
    if arrival < min_arrival:
        reasons.append(f"arrival rate {arrival:.3f} < {min_arrival:g}")
    if metres < min_explore_m:
        reasons.append(f"insufficient explore driving: {metres:.0f} m < {min_explore_m:g} m")
    elif m_between < min_m_between:
        reasons.append(f"{m_between:.0f} m between failures < {min_m_between:g} m")
    return {"arrival_rate": arrival, "goal_episodes": n_goal, "explore_m": metres,
            "faults": faults, "m_between": m_between, "pass": not reasons,
            "reason": "; ".join(reasons) or "ok"}

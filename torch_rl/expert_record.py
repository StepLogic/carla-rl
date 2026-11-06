"""Recording expert (and DAgger) episodes for imitation (2026-11-04).

Imitation-first plan, Tasks 4-5. This module holds the pure logic -- the
held-out guard, the per-episode schedule, recovery noise, which action is
executed and which is the label, and the M0 expert gate -- plus the recording
loop itself (run_record, Task 5), which drives the unchanged env stack.

RECOVERY NOISE (DART). With the expert driving, the EXECUTED steering gets
N(0, sigma) noise while the stored LABEL stays the expert's clean action, so
the data shows how to come back from drifting: the expert's next action is
computed from the perturbed state it was pushed into.

COVERAGE (user 2026-11-06, "PERTURB EXPERT MORE SO WE HAVE COVERAGE DATA"). The
critic had only ever seen small i.i.d. steering noise, so it had no evidence
about the sustained hard steering the collapsed fine-tunes drifted into. All off
by default: EXPERT_SIGMAS sets the steering levels, throttle_sigma adds throttle
noise, and BurstPerturber holds one (steer, throttle) offset for several steps.
The label stays the expert's clean action throughout.
"""
import os

import numpy as np

HELD_OUT_TOWNS = ("Town02", "Town03")
FAULT_ENDS = ("crossing", "off_road", "collision")
SIGMAS = tuple(float(x) for x in os.environ.get("EXPERT_SIGMAS", "0,0.1,0.2").split(","))


def assert_recordable_town(town, trace_only=False):
    """Refuse to record in a held-out town: those towns are for evaluation only.

    trace_only=True is the ONE documented bypass (user 2026-11-06, single-trace
    evaluation maps): EXPERT_TRACE drives the held-out towns to build their
    EVALUATION maps. Its trace is written in the map-trace format, which no
    training loader reads, and expert_trace.trace_dir refuses any root where
    training data lives, so nothing it records can become training data."""
    if town in HELD_OUT_TOWNS and not trace_only:
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


def dart_action(a_env, sigma, rng, low, high, throttle_sigma=0.0):
    """The executed action: N(0, sigma) added to the steering (index 0) and,
    when throttle_sigma > 0, N(0, throttle_sigma) to the throttle (index 1),
    clipped to the env box. A copy, and exactly `a_env` when both are 0."""
    out = np.array(a_env, dtype=np.float32, copy=True)
    if sigma > 0.0:
        out[0] += np.float32(rng.normal(0.0, sigma))
    if throttle_sigma > 0.0:
        out[1] += np.float32(rng.normal(0.0, throttle_sigma))
    return np.clip(out, low, high).astype(np.float32)


class BurstPerturber:
    """Temporally correlated perturbation for coverage: with probability `prob`
    per step (when no burst is running) start a burst of k steps, k uniform in
    `steps`, during which one constant offset is added to the executed action --
    steering magnitude uniform in `steer`, throttle magnitude in `throttle`,
    each with a random sign -- clipped to the box. prob 0 = off."""

    def __init__(self, prob=0.0, steps=(5, 15), steer=(0.3, 1.0), throttle=(0.0, 0.5)):
        self.prob, self.steps = float(prob), (int(steps[0]), int(steps[1]))
        self.steer, self.throttle = tuple(map(float, steer)), tuple(map(float, throttle))
        self.left = 0
        self.offset = np.zeros(2, np.float32)

    def reset(self):
        self.left = 0

    def apply(self, a, rng, low, high):
        if self.prob <= 0.0:
            return a
        if self.left == 0 and rng.random() < self.prob:
            self.left = int(rng.integers(self.steps[0], self.steps[1] + 1))
            self.offset = np.array(
                [rng.uniform(*self.steer) * rng.choice((-1.0, 1.0)),
                 rng.uniform(*self.throttle) * rng.choice((-1.0, 1.0))], np.float32)
        if self.left == 0:
            return a
        self.left -= 1
        return np.clip(np.asarray(a, np.float32) + self.offset, low, high).astype(np.float32)


def record_step(driver, a_policy, a_expert, sigma, rng, low, high, throttle_sigma=0.0,
                burst=None):
    """(a_exec, a_label) for one step. The label is always the expert's action;
    what is executed is the noisy expert ("expert") -- DART noise, then any
    coverage burst -- or the policy ("policy")."""
    a_label = np.asarray(a_expert, np.float32)
    if driver == "expert":
        a_exec = dart_action(a_label, sigma, rng, low, high, throttle_sigma=throttle_sigma)
        if burst is not None:
            a_exec = burst.apply(a_exec, rng, low, high)
        return a_exec, a_label
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

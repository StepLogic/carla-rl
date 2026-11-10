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


def expert_bias():
    """EXPERT_BIAS="steer,throttle" (env units; user 2026-11-06, "see if the expert
    behavior scores the best reward"): a constant offset added to every EXECUTED
    expert action, or None when unset. The label stays the expert's own action."""
    raw = os.environ.get("EXPERT_BIAS", "").strip()
    if not raw:
        return None
    vals = np.array([float(x) for x in raw.split(",")], np.float32)
    if vals.shape != (2,):
        raise ValueError(f"EXPERT_BIAS={raw!r}: give two values, 'steer,throttle'")
    return vals


def episode_returns(rewards, gamma):
    """(sum, discounted sum) of one episode's rewards."""
    r = np.asarray(rewards, np.float64)
    if r.size == 0:
        return 0.0, 0.0
    return float(r.sum()), float(np.sum(r * float(gamma) ** np.arange(r.size)))


def record_step(driver, a_policy, a_expert, sigma, rng, low, high, throttle_sigma=0.0,
                burst=None, bias=None):
    """(a_exec, a_label) for one step. The label is always the expert's action;
    what is executed is the noisy expert ("expert") -- DART noise, then any
    coverage burst, then any constant EXPERT_BIAS -- or the policy ("policy")."""
    a_label = np.asarray(a_expert, np.float32)
    if driver == "expert":
        a_exec = dart_action(a_label, sigma, rng, low, high, throttle_sigma=throttle_sigma)
        if burst is not None:
            a_exec = burst.apply(a_exec, rng, low, high)
        if bias is not None:
            a_exec = np.clip(np.asarray(a_exec, np.float32) + bias, low, high).astype(np.float32)
        return a_exec, a_label
    if driver == "policy":
        return np.asarray(a_policy, np.float32), a_label
    raise ValueError(f"driver must be 'expert' or 'policy', got {driver!r}")


CHAIN_SEGMENTS_MAX = 100


def chain_segment_id(seed, seg):
    """Shard id of goal segment `seg` of the chained recording started at `seed`
    (EXPERT_RECORD_CHAIN, plan 2026-11-09 B1): seed * 100 + seg. Every segment is
    its own episode shard with one goal image."""
    seg = int(seg)
    if not 0 <= seg < CHAIN_SEGMENTS_MAX:
        raise ValueError(f"segment {seg} outside 0..{CHAIN_SEGMENTS_MAX - 1}")
    return int(seed) * CHAIN_SEGMENTS_MAX + seg


def chain_continues(chain_on, mode, end, info):
    """After a goal arrival in a chained recording, the same car drives on to the
    next goal (the env's soft reset). Never for explore episodes, faults, or once
    the env stops offering a next goal (the chain's 200 m are done)."""
    return (bool(chain_on) and int(mode) == 1 and end == "reached"
            and bool(info.get("chain_continue")))


def map_route_metres(node_xy, route, start_xy, goal_xy):
    """Metres of a MAP route as the car would drive it (EXPERT_RECORD_MAP, plan
    2026-11-09 B2): car -> first node -> ... along node positions -> last node ->
    goal. The image maps' edge costs are unit hops, not metres, so they cannot
    give this."""
    xy = np.asarray(node_xy, np.float64)[list(route)]
    s = np.asarray(start_xy, np.float64)[:2]
    g = np.asarray(goal_xy, np.float64)[:2]
    along = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
    return float(np.hypot(*(xy[0] - s)) + along + np.hypot(*(xy[-1] - g)))


def map_route_skip(route, map_m, road_m, turns, min_turns=1, max_ratio=1.2,
                   max_legs=CHAIN_SEGMENTS_MAX):
    """-> None to record a map-subgoal episode, else why it is skipped.

    The expert drives the planner's ROAD route to the real goal while the goal
    images follow the MAP route, so the two must agree: a map route with at least
    one subgoal node ("no_route"), no more legs than shard ids ("legs": one leg per
    node after the start plus the final one), a road route ("no_road"), at least
    `min_turns` junction turns on it ("turns"), and a map route no longer than
    `max_ratio` x the road route ("detour")."""
    if route is None or len(route) < 2:
        return "no_route"
    if len(route) > int(max_legs):
        return "legs"
    if road_m is None or road_m <= 0:
        return "no_road"
    if int(turns) < int(min_turns):
        return "turns"
    if map_m is None or float(map_m) > float(max_ratio) * float(road_m):
        return "detour"
    return None


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

"""Hindsight rows vs the FULL on-policy goal reward, including along-road progress.

WHY THIS FILE EXISTS (2026-09-26). test_her_relabel_matches_reward checked the
relabeller against goal_reward_sorb -- but the goal wrapper adds POTENTIAL_K *
(metres of along-road progress) ON TOP of goal_reward_sorb, and v13-v15 ran with
POTENTIAL_K=1.0. The relabeller never had that term, so every hindsight row was
costed by a different reward function than the on-policy rows beside it. The
oracle here is the wrapper's WHOLE reward:

    sorb:      goal_reward_sorb(...) + POTENTIAL_K * progress   (non-terminal steps)
    progress:  goal_reward_progress(...)   Bono et al. 2023, arXiv 2309.16634:
               r = K*1[arrived] - Delta(route) - lambda, K=10, lambda=0.01

Progress toward a HINDSIGHT goal needs a road distance to that goal. The unit
tests use a straight road, where the true along-road distance is |gx - x|, so the
oracle can compute it independently of the planner. A separate test drives the
REAL Town05 planner offline (no server) and checks the relabeller's progress
against an independent fresh plan.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vae_nav import gated_reward as gr  # noqa: E402
from vae_nav.episode_her import EpisodeRelabelBuffer  # noqa: E402

RADIUS, CLOCK, TRIP, FINE = 2.0, 1000, 20, 10.0
BROKEN = ["carla.LaneMarkingType.Broken"]
SOLID = ["carla.LaneMarkingType.Solid"]


class StraightRoad:
    """A route planner for a straight road along x: the along-road distance is
    exactly |goal_x - x|. Same interface as route_potential.XYRoute."""

    def __init__(self):
        self.goal = None
        self.plans = 0

    def reset(self, origin_xy, goal_xy, origin_z=0.0, goal_z=0.0):
        self.goal = np.asarray(goal_xy, np.float64)
        self.plans += 1
        return abs(self.goal[0] - float(origin_xy[0]))

    def remaining(self, xy):
        return abs(self.goal[0] - float(np.asarray(xy)[0]))


def road_d(p, g):
    return abs(float(g[0]) - float(p[0]))


@pytest.fixture
def base(monkeypatch):
    from vae_nav import lane_explore_reward as ler
    monkeypatch.setattr(ler, "LANE_ANY_FATAL", False)
    for k, v in dict(GOAL_TRUNCATE_COUNTS=True, LANE_TERMINAL=True,
                     STEP_COST=1.0, GOAL_ARRIVAL_BONUS=10.0,
                     GOAL_OFF_ROAD_PENALTY=FINE, GOAL_LANE_PENALTY=FINE,
                     GOAL_COLLISION_PENALTY=FINE,
                     GOAL_OFFROAD_TRUNCATE_STEPS=TRIP,
                     GOAL_COLLISION_TRUNCATE_STEPS=TRIP,
                     GOAL_CROSSING_TRUNCATE_STEPS=TRIP,
                     GOAL_MAX_STEPS=CLOCK, GOAL_SPEED_BONUS=0.75,
                     PROGRESS_K=10.0, PROGRESS_SLACK=0.01,
                     PROGRESS_FINES=False, PROGRESS_MAX_STEP_M=5.0).items():
        monkeypatch.setattr(gr, k, v)
    return monkeypatch


@pytest.fixture
def v15(base):
    """What v15 actually runs: sorb counting + POTENTIAL_K=1.0 shaping."""
    base.setattr(gr, "GOAL_REWARD_KIND", "sorb")
    base.setattr(gr, "POTENTIAL_K", 1.0)
    return base


@pytest.fixture
def paper(base):
    """Bono et al. 2023's reward (POTENTIAL_K left at 1.0 on purpose: it must be
    ignored, not added twice)."""
    base.setattr(gr, "GOAL_REWARD_KIND", "progress")
    base.setattr(gr, "POTENTIAL_K", 1.0)
    return base


class FakeBuffer:
    def __init__(self):
        self.rows = []

    def insert(self, row, mode=None, segment=None):
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)


def goal_block(fields):
    return np.asarray(fields["goal_pos"], np.float32) - np.asarray(
        fields["pos"], np.float32)


def gen_episode(rng, mid_fatal=False):
    n = int(rng.integers(25, 90))
    infos, legs = [], []
    for t in range(n):
        info = {"speed_kmh": float(rng.uniform(5, 30)), "target_speed_kmh": 20.0}
        if rng.random() < 0.10:
            info.update(lane_any=True, lane_marking_types=BROKEN,
                        in_junction=True)                      # forgiven
        if rng.random() < 0.06:
            info["off_road"] = True
        if rng.random() < 0.03:
            info["collision"] = True
        infos.append(info)
        if rng.random() < 0.05 and t < n - 1:
            legs.append(t)
    infos[-1].update(lane_any=True, lane_marking_types=SOLID, in_junction=False)
    if mid_fatal:
        k = int(rng.integers(3, n - 3))
        infos[k].update(lane_any=True, lane_marking_types=SOLID, in_junction=False)
    # straight road, uneven speed so progress varies step to step
    x = np.cumsum(rng.uniform(0.2, 1.4, n)) - 0.5
    pos = [(float(v), 0.0) for v in x]
    return infos, pos, legs


def make_her(seed, route):
    buf = FakeBuffer()
    her = EpisodeRelabelBuffer(
        buf, goal_block, goal_radius=RADIUS, step_cost=1.0,
        time_to_goal=True, gated=True, recost_counting=True,
        k_goal=600.0, k_collision=5.0, k_lane=5.0, k_living=0.5, k_dist=1.0,
        seed=seed, route_provider=(lambda: route) if route is not None else None)
    return her, buf


def drive(her, infos, pos, legs):
    for t, (info, p) in enumerate(zip(infos, pos)):
        txn = {"observations": {"goal": np.zeros(2, np.float32),
                                "latent": np.full(4, float(t), np.float32)},
               "next_observations": {"goal": np.zeros(2, np.float32),
                                     "latent": np.full(4, t + 1.0, np.float32)},
               "actions": np.zeros(2, np.float32), "rewards": np.float32(0.0),
               "masks": np.float32(1.0), "dones": np.float32(0.0)}
        her.record(txn, {"pos": p}, p, hit=bool(info.get("collision", False)),
                   lane=bool(info.get("lane_any", False)),
                   off_road=bool(info.get("off_road", False)),
                   speed_kmh=info["speed_kmh"],
                   target_speed_kmh=info["target_speed_kmh"],
                   lane_fatal=gr.goal_lane_departed(info))
        if t in legs:
            her.end_leg()
    her.end_leg()


def leg_start(i, legs):
    s = 0
    for e in sorted(legs):
        if e < i:
            s = e + 1
    return s


def full_reward(infos, pos, ls, i, tp):
    """The goal WRAPPER's reward for step i with goal pos[tp] -- the oracle."""
    g = pos[tp]
    j = min(i + 1, len(pos) - 1)
    info = dict(infos[i])
    info["is_success"] = bool(np.linalg.norm(np.asarray(pos[j]) - np.asarray(g))
                              < RADIUS)
    seg = infos[ls:i + 1]
    counts = dict(offroad_steps=sum(bool(s.get("off_road")) for s in seg),
                  collision_steps=sum(bool(s.get("collision")) for s in seg),
                  crossing_steps=sum(bool(s.get("lane_any")) for s in seg))
    d0, d1 = road_d(pos[i], g), road_d(pos[j], g)
    if gr.GOAL_REWARD_KIND == "progress":
        return gr.goal_reward_progress(info, i - ls, d0, d1, max_steps=CLOCK,
                                       **counts)
    r, term, trunc, arrived = gr.goal_reward_sorb(
        info, i - ls, max_steps=CLOCK, step_cost=1.0, **counts)
    if gr.POTENTIAL_K and not (term or trunc):
        r += gr.POTENTIAL_K * (d0 - d1)      # GatedReward.step's shaping
    return r, term, trunc, arrived


def audit(seeds=250, mid_fatal=False):
    n = bad = 0
    stats = dict(arrive=0, progress_pos=0, progress_neg=0)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        infos, pos, legs = gen_episode(rng, mid_fatal=mid_fatal)
        route = StraightRoad()
        her, buf = make_her(seed, route)
        drive(her, infos, pos, legs)
        n_on = len(infos)
        her.flush_episode()
        rows = buf.rows[n_on:]
        assert len(rows) == len(her.last_pairs)
        for (i, tp), row in zip(her.last_pairs, rows):
            r, term, trunc, arrived = full_reward(infos, pos, leg_start(i, legs),
                                                  i, tp)
            n += 1
            if (abs(float(row["rewards"]) - r) > 1e-4
                    or float(row["masks"]) != (0.0 if term else 1.0)
                    or trunc or (term and not arrived)):
                bad += 1
            stats["arrive"] += int(arrived)
            d = road_d(pos[i], pos[tp]) - road_d(pos[min(i + 1, len(pos) - 1)],
                                                   pos[tp])
            stats["progress_pos"] += int(d > 1e-6)
            stats["progress_neg"] += int(d < -1e-6)
    return n, bad, stats


@pytest.mark.parametrize("mid_fatal", [False, True])
def test_v15_hindsight_rows_now_carry_the_shaping(v15, mid_fatal):
    """sorb + POTENTIAL_K=1.0, exactly v15: every hindsight row equals the
    wrapper's full reward, progress term included."""
    n, bad, st = audit(mid_fatal=mid_fatal)
    assert bad == 0, f"{bad}/{n} hindsight rows disagree with the wrapper"
    assert n > 2000 and st["arrive"] > 50 and st["progress_pos"] > 1000


@pytest.mark.parametrize("mid_fatal", [False, True])
def test_paper_reward_hindsight_rows_match(paper, mid_fatal):
    n, bad, st = audit(mid_fatal=mid_fatal)
    assert bad == 0, f"{bad}/{n} hindsight rows disagree with the paper reward"
    assert n > 2000 and st["arrive"] > 50 and st["progress_pos"] > 1000


def test_paper_reward_terms(paper):
    """The three terms, at the paper's constants, and nothing else."""
    info = {"speed_kmh": 20.0, "target_speed_kmh": 20.0}
    r, term, trunc, arr = gr.goal_reward_progress(info, 5, 50.0, 49.7)
    assert r == pytest.approx(0.3 - 0.01) and not term and not trunc
    r, *_ = gr.goal_reward_progress(info, 5, 50.0, 50.4)          # drove away
    assert r == pytest.approx(-0.4 - 0.01)
    r, term, _, arr = gr.goal_reward_progress(dict(info, is_success=True), 5,
                                               1.0, 0.5)
    assert arr and term and r == pytest.approx(10.0 - 0.01)
    # no fines, no speed bonus, no bill: a fault step is still just progress
    r, *_ = gr.goal_reward_progress(dict(info, off_road=True, collision=True,
                                         lane_any=True, in_junction=True,
                                         lane_marking_types=BROKEN),
                                    5, 50.0, 49.7)
    assert r == pytest.approx(0.3 - 0.01)


def test_paper_reward_lane_terminal_has_no_bill(paper):
    info = {"lane_any": True, "lane_marking_types": SOLID, "in_junction": False}
    r, term, trunc, arr = gr.goal_reward_progress(info, 100, 50.0, 49.7)
    assert term and not arr and not trunc
    assert r == pytest.approx(0.3 - 0.01)          # this step only


def test_non_physical_jump_pays_nothing(paper):
    """A replan or goal switch moves the distance by tens of metres in a step;
    paying it would be a fake reward."""
    r, *_ = gr.goal_reward_progress({}, 5, 50.0, 10.0)
    assert r == pytest.approx(-0.01)
    r, *_ = gr.goal_reward_progress({}, 5, None, 10.0)
    assert r == pytest.approx(-0.01)


def test_missing_route_planner_raises_instead_of_costing_zero(v15):
    rng = np.random.default_rng(0)
    infos, pos, legs = gen_episode(rng)
    her, _ = make_her(0, route=None)
    drive(her, infos, pos, legs)
    with pytest.raises(RuntimeError, match="no route planner"):
        her.flush_episode()


def test_no_progress_term_needs_no_planner(base):
    """sorb with POTENTIAL_K=0 (pre-v13 recipes) must not demand a planner."""
    base.setattr(gr, "GOAL_REWARD_KIND", "sorb")
    base.setattr(gr, "POTENTIAL_K", 0.0)
    rng = np.random.default_rng(0)
    infos, pos, legs = gen_episode(rng)
    her, _ = make_her(0, route=None)
    drive(her, infos, pos, legs)
    her.flush_episode()


def test_consecutive_rows_reuse_one_plan(v15):
    """Rows sharing a hindsight goal advance one plan's pointer, as on-policy
    does across a goal's steps -- far fewer plans than rows."""
    rng = np.random.default_rng(3)
    infos, pos, legs = gen_episode(rng)
    route = StraightRoad()
    her, _ = make_her(3, route)
    drive(her, infos, pos, legs)
    her.flush_episode()
    assert her.last_pairs
    assert route.plans <= len(her.last_pairs)


# ---- the real planner, offline ----------------------------------------------

TOWN05 = ("/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/"
          "OpenDrive/Town05.xodr")


@pytest.mark.skipif(not os.path.exists(TOWN05), reason="no Town05.xodr")
def test_real_town05_progress_matches_an_independent_plan():
    """Drive-like points along a real Town05 route: the relabeller's per-step
    progress (plan once, pointer) must match an INDEPENDENT fresh plan from each
    point, and sum exactly to the start-to-end route distance."""
    carla = pytest.importorskip("carla")
    import vae_nav.route_potential as RP
    from vae_nav.route_potential import XYRoute
    # the measure the paper reward runs on (gated_config refuses anything else)
    _mode = RP.REMAINING_MODE
    RP.REMAINING_MODE = "project"
    m = carla.Map("Town05", open(TOWN05).read())
    route = XYRoute(m)
    wps = m.generate_waypoints(2.0)
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(20):
        a = wps[int(rng.integers(len(wps)))]
        path = [a]
        for _ in range(60):
            nxt = path[-1].next(0.5)
            if not nxt:
                break
            path.append(nxt[0])
        if len(path) < 40:
            continue
        pts = [(w.transform.location.x, w.transform.location.y) for w in path]
        zs = [w.transform.location.z for w in path]
        goal = pts[-1]
        route.reset(pts[0], goal, zs[0], zs[-1])
        d = [route.remaining(p) for p in pts]
        indep = XYRoute(m)
        for k in range(0, len(pts) - 1, 7):
            indep.reset(pts[k], goal, zs[k], zs[-1])
            assert abs(d[k] - indep.remaining(pts[k])) < 1.5, (k, d[k])
        steps = np.diff(d)
        assert np.all(steps <= 1e-6), "progress along the route must not reverse"
        assert abs((d[0] - d[-1]) - sum(-steps)) < 1e-6
        checked += 1
    RP.REMAINING_MODE = _mode
    assert checked >= 10


@pytest.mark.skipif(not os.path.exists(TOWN05), reason="no Town05.xodr")
def test_hop_measure_is_a_sawtooth_and_project_is_not():
    """Pins the defect the paper reward must not run on, and the fix."""
    carla = pytest.importorskip("carla")
    import vae_nav.route_potential as RP
    from vae_nav.route_potential import XYRoute
    m = carla.Map("Town05", open(TOWN05).read())
    wps = m.generate_waypoints(2.0)
    rng = np.random.default_rng(1)
    paths = []
    while len(paths) < 15:
        a = wps[int(rng.integers(len(wps)))]
        path = [a]
        for _ in range(150):
            nx = path[-1].next(0.28)
            if not nx:
                break
            path.append(nx[0])
        if len(path) >= 120:
            paths.append([(w.transform.location.x, w.transform.location.y)
                          for w in path])
    neg = {}
    _mode = RP.REMAINING_MODE
    try:
        for mode in ("hop", "project"):
            RP.REMAINING_MODE = mode
            r = XYRoute(m)
            steps = []
            for pts in paths:
                if not r.reset(pts[0], pts[-1]):
                    continue
                d = [r.remaining(p) for p in pts]
                if any(x is None for x in d):
                    continue
                steps.append(-np.diff(np.array(d)))
            s = np.concatenate(steps)
            neg[mode] = float(np.mean(s < 0))
    finally:
        RP.REMAINING_MODE = _mode
    assert neg["hop"] > 0.3, neg          # the sawtooth, as measured (43%)
    assert neg["project"] == 0.0, neg     # forward driving never reads backwards

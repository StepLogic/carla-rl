"""The CARLA replication of the Gibson image-goal benchmark (user 2026-11-01).

Mirrors the Mezghani ImageNav protocol a peer session runs on Gibson: a FIXED
episode set, difficulty bins by GEODESIC (here: route) distance, a fixed count
per difficulty per scene, SR and SPL per bin, arms paired per episode with exact
McNemar and Holm correction. NRNS's straight/curved split (route/euclidean
<= 1.20 is straight) is tagged on the same episodes.

Gibson: 14 houses x 3 bins x 100 = 4,200; bins 1.5-3 / 3-5 / 5-10 m.
CARLA:   6 held-out towns x 3 bins x 100 = 1,800; bins 5-10 / 10-15 / 15-25 m,
         our trained 5-25 m band split the same way.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav import carla_benchmark as cb  # noqa: E402


# ---- bins and path type ----------------------------------------------------

@pytest.mark.parametrize("route,want", [
    (4.99, None), (5.0, "easy"), (9.99, "easy"), (10.0, "medium"),
    (14.99, "medium"), (15.0, "hard"), (25.0, "hard"), (25.01, None)])
def test_bins_are_half_open_by_route_distance(route, want):
    assert cb.assign_bin(route) == want


def test_no_route_means_no_bin():
    """An episode the planner could not route has no geodesic distance, and
    Gibson's bins are geodesic; it is dropped, never binned by straight line."""
    assert cb.assign_bin(None) is None


@pytest.mark.parametrize("route,euclid,want", [
    (12.0, 10.0, "straight"), (12.0, 10.01, "straight"),
    (12.1, 10.0, "curved"), (20.0, 5.0, "curved")])
def test_nrns_straight_rule(route, euclid, want):
    assert cb.path_kind(route, euclid) == want


# ---- selecting the fixed episode set ---------------------------------------

def _probe(seed, route, euclid=None, town="Town02"):
    return {"town": town, "seed": seed, "route_m": route,
            "euclid_m": route if euclid is None else euclid}


def test_selection_takes_the_first_n_per_bin_in_seed_order():
    probes = ([_probe(s, 7.0) for s in range(100, 110)]
              + [_probe(s, 12.0) for s in range(200, 210)]
              + [_probe(s, 20.0) for s in range(300, 310)])
    eps = cb.select_episodes(probes, per_bin=3)
    by = {}
    for e in eps:
        by.setdefault(e["bin"], []).append(e["seed"])
    assert by == {"easy": [100, 101, 102], "medium": [200, 201, 202],
                  "hard": [300, 301, 302]}


def test_selection_is_per_town():
    probes = ([_probe(s, 7.0, town="Town02") for s in range(5)]
              + [_probe(s, 7.0, town="Town03") for s in range(10, 15)])
    eps = cb.select_episodes(probes, per_bin=2, bins=("easy",))
    assert sorted((e["town"], e["seed"]) for e in eps) == [
        ("Town02", 0), ("Town02", 1), ("Town03", 10), ("Town03", 11)]


def test_a_short_bin_is_reported_not_silently_padded():
    probes = [_probe(s, 7.0) for s in range(2)]
    with pytest.raises(ValueError, match="Town02.*easy.*2.*3"):
        cb.select_episodes(probes, per_bin=3, bins=("easy",))


def test_duplicate_seeds_within_a_town_are_refused():
    """Every episode must be a distinct seed, or the pairing double-counts."""
    with pytest.raises(ValueError, match="duplicate"):
        cb.select_episodes([_probe(1, 7.0), _probe(1, 7.5)], per_bin=1,
                           bins=("easy",))


def test_episodes_carry_their_metadata():
    e = cb.select_episodes([_probe(4, 12.6, euclid=10.0)], per_bin=1,
                           bins=("medium",))[0]
    assert e == {"town": "Town02", "seed": 4, "bin": "medium",
                 "route_m": 12.6, "euclid_m": 10.0, "kind": "curved"}


# ---- paired statistics -----------------------------------------------------

def test_mcnemar_exact():
    assert cb.mcnemar_p(0, 0) == 1.0
    assert cb.mcnemar_p(5, 5) == 1.0
    assert cb.mcnemar_p(10, 0) == pytest.approx(2 / 2 ** 10)
    assert cb.mcnemar_p(0, 10) == pytest.approx(2 / 2 ** 10)


def test_holm_step_down():
    # sorted p = .01 .02 .04 with m = 3 -> .03, .04, .04 (monotone), mapped back
    adj = cb.holm([0.04, 0.01, 0.02])
    assert adj == pytest.approx([0.04, 0.03, 0.04])


def test_holm_caps_at_one():
    assert cb.holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


def _row(seed, reached, spl=None, bin_="easy", kind="straight", town="Town02",
         reached_2m=None):
    return {"seed": seed, "town": town, "bin": bin_, "kind": kind,
            "reached": reached, "spl": (1.0 if reached else 0.0) if spl is None else spl,
            "reached_2m": reached if reached_2m is None else reached_2m}


def test_paired_table_per_bin_and_overall():
    true = [_row(1, True), _row(2, True), _row(3, False), _row(4, True, bin_="hard")]
    wrong = [_row(1, False), _row(2, True), _row(3, False), _row(4, False, bin_="hard")]
    t = cb.paired_table(true, wrong, by="bin")
    assert t["easy"]["n"] == 3 and t["easy"]["only_true"] == 1 and t["easy"]["only_wrong"] == 0
    assert t["easy"]["sr_true"] == pytest.approx(2 / 3)
    assert t["hard"]["only_true"] == 1
    assert t["all"]["n"] == 4 and t["all"]["only_true"] == 2
    # Holm across the BINS (the family), not including the pooled row
    assert "p_holm" in t["easy"] and "p_holm" in t["hard"]


def test_pairing_refuses_mismatched_episode_sets():
    with pytest.raises(ValueError, match="same episodes"):
        cb.paired_table([_row(1, True)], [_row(2, True)], by="bin")


def test_the_2m_criterion_is_reported_alongside():
    true = [_row(1, False, reached_2m=True), _row(2, True)]
    wrong = [_row(1, False), _row(2, False)]
    t = cb.paired_table(true, wrong, by="bin")
    assert t["easy"]["sr_true"] == pytest.approx(0.5)
    assert t["easy"]["sr_true_2m"] == pytest.approx(1.0)


# ---- the env's arrival radius ------------------------------------------------

def test_the_arrival_radius_is_the_shared_setting():
    """The env, SPL and HER must read ONE radius, or a 1 m eval would score
    against a 2 m arrival. The env imports CARLA, so this checks the source."""
    src = open(os.path.join(ROOT, "src", "configs",
                            "train_env_config_v4_image_goal.py")).read()
    assert 'ARRIVAL_RADIUS_M = float(os.environ.get("GOAL_RADIUS_M", 2.0))' in src
    assert "distance_to_goal < ARRIVAL_RADIUS_M" in src
    assert "distance_to_goal < 2 and" not in src
    trainer = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'GOAL_RADIUS_M = env_float("GOAL_RADIUS_M", 2.0)' in trainer


# ---- scoring the 2 m criterion inside a 1 m run ------------------------------
# The eval ends at the Gibson radius (1 m). Our training radius (2 m) is scored
# from the same run: the first step within 2 m AND slow enough to arrive. If a
# fault ends the episode after that step, a 2 m eval would already have ended
# it as a success; a fault ON that step is a failure, as the env rules it.

def test_reaching_1m_counts_at_2m_too():
    tr = cb.EpisodeTracker(goal_pos=(10.0, 0.0), start_pos=(0.0, 0.0), max_speed=10.0)
    for x, v in [(5.0, 20.0), (8.5, 8.0), (9.5, 5.0)]:
        tr.step((x, 0.0), v)
    r = tr.finish(reached=True, t_end=3, route_m=10.0)
    assert r["reached"] and r["reached_2m"]


def test_slow_inside_2m_then_crash_counts_at_2m():
    tr = cb.EpisodeTracker(goal_pos=(10.0, 0.0), start_pos=(0.0, 0.0), max_speed=10.0)
    tr.step((8.5, 0.0), 6.0)       # step 1: inside 2 m (1.5) and slow
    tr.step((9.0, 0.0), 6.0)       # step 2: a fault ends it here
    r = tr.finish(reached=False, t_end=2, route_m=10.0)
    assert not r["reached"] and r["reached_2m"]


def test_fast_through_2m_does_not_count():
    tr = cb.EpisodeTracker(goal_pos=(10.0, 0.0), start_pos=(0.0, 0.0), max_speed=10.0)
    tr.step((8.5, 0.0), 30.0)      # inside 2 m but too fast to arrive
    tr.step((12.5, 0.0), 30.0)
    r = tr.finish(reached=False, t_end=2, route_m=10.0)
    assert not r["reached_2m"]


def test_a_fault_on_the_reaching_step_is_a_failure():
    tr = cb.EpisodeTracker(goal_pos=(10.0, 0.0), start_pos=(0.0, 0.0), max_speed=10.0)
    tr.step((8.5, 0.0), 5.0)       # step 1 is both inside-2m-slow AND the fatal step
    r = tr.finish(reached=False, t_end=1, route_m=10.0)
    assert not r["reached_2m"]


def test_spl_at_each_radius_uses_its_own_success_region():
    """SPL's shortest path is to the success REGION, route minus radius, and the
    path is what was driven when that region was entered."""
    tr = cb.EpisodeTracker(goal_pos=(10.0, 0.0), start_pos=(0.0, 0.0), max_speed=10.0,
                           radius=1.0, radius_2m=2.0)
    for x in (2.0, 4.0, 6.0, 8.5, 9.2):            # 2 m in at x=8.5, 1 m in at 9.2
        tr.step((x, 0.0), 5.0)
    r = tr.finish(reached=True, t_end=5, route_m=10.0)
    assert r["path_m"] == pytest.approx(9.2)
    assert r["spl"] == pytest.approx((10.0 - 1.0) / 9.2)
    assert r["spl_2m"] == pytest.approx((10.0 - 2.0) / 8.5)


def test_failures_score_zero_spl():
    tr = cb.EpisodeTracker(goal_pos=(10.0, 0.0), start_pos=(0.0, 0.0), max_speed=10.0)
    tr.step((3.0, 0.0), 5.0)
    r = tr.finish(reached=False, t_end=1, route_m=10.0)
    assert r["spl"] == 0.0 and r["spl_2m"] == 0.0

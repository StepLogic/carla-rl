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
    (9.99, None), (10.0, "easy"), (24.99, "easy"), (25.0, "medium"),
    (49.99, "medium"), (50.0, "hard"), (100.0, "hard"), (100.01, None)])
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
    probes = ([_probe(s, 15.0) for s in range(100, 110)]
              + [_probe(s, 30.0) for s in range(200, 210)]
              + [_probe(s, 70.0) for s in range(300, 310)])
    eps = cb.select_episodes(probes, per_bin=3)
    by = {}
    for e in eps:
        by.setdefault(e["bin"], []).append(e["seed"])
    assert by == {"easy": [100, 101, 102], "medium": [200, 201, 202],
                  "hard": [300, 301, 302]}


def test_selection_is_per_town():
    probes = ([_probe(s, 15.0, town="Town02") for s in range(5)]
              + [_probe(s, 15.0, town="Town03") for s in range(10, 15)])
    eps = cb.select_episodes(probes, per_bin=2, bins=("easy",))
    assert sorted((e["town"], e["seed"]) for e in eps) == [
        ("Town02", 0), ("Town02", 1), ("Town03", 10), ("Town03", 11)]


def test_a_short_bin_is_reported_not_silently_padded():
    probes = [_probe(s, 15.0) for s in range(2)]
    with pytest.raises(ValueError, match="Town02.*easy.*2.*3"):
        cb.select_episodes(probes, per_bin=3, bins=("easy",))


def test_duplicate_seeds_within_a_town_are_refused():
    """Every episode must be a distinct seed, or the pairing double-counts."""
    with pytest.raises(ValueError, match="duplicate"):
        cb.select_episodes([_probe(1, 15.0), _probe(1, 15.5)], per_bin=1,
                           bins=("easy",))


def test_episodes_carry_their_metadata():
    e = cb.select_episodes([_probe(4, 37.8, euclid=30.0)], per_bin=1,
                           bins=("medium",))[0]
    assert e == {"town": "Town02", "seed": 4, "bin": "medium",
                 "route_m": 37.8, "euclid_m": 30.0, "kind": "curved"}


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


def test_max_speed_zero_means_no_speed_condition():
    """User 2026-11-07, "drop the 5km/h rule": GOAL_ARRIVAL_MAX_SPEED=0, which the
    reward already reads as OFF (gated_reward._stopped_enough). The eval's 2 m
    scoring must agree: a fast pass through 2 m counts, not only a standstill."""
    tr = cb.EpisodeTracker(goal_pos=(10.0, 0.0), start_pos=(0.0, 0.0), max_speed=0.0)
    tr.step((8.5, 0.0), 30.0)      # inside 2 m at 30 km/h
    tr.step((12.5, 0.0), 30.0)
    r = tr.finish(reached=False, t_end=2, route_m=10.0)
    assert r["reached_2m"] and r["reach2_step"] == 1


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


# ---- per-band generation (2026-11-01) ----------------------------------------
# The probe found our 5-25 m goals are almost all ~20-25 m: 85% of origins sit
# before a junction and GOAL_PAST_JUNCTION keeps only distances >= 15 m past
# it. So each difficulty is GENERATED from its own band (as Mezghani's set was
# generated per difficulty), and an episode belongs to a bin only if its ROUTE
# distance also lands in that bin. The band is part of the episode: replaying
# a seed under a different band would be a different episode.

def test_a_banded_probe_counts_only_in_its_own_bin():
    probes = [dict(_probe(1, 15.0), band="easy"),       # in band -> kept
              dict(_probe(2, 30.0), band="easy"),       # routed long -> dropped
              dict(_probe(3, 30.0), band="medium")]
    eps = cb.select_episodes(probes, per_bin=1, bins=("easy", "medium"))
    assert [(e["seed"], e["bin"], e["band"]) for e in eps] == [
        (1, "easy", "easy"), (3, "medium", "medium")]


def test_unbanded_probes_still_bin_by_route():
    eps = cb.select_episodes([_probe(1, 30.0)], per_bin=1, bins=("medium",))
    assert eps[0]["bin"] == "medium" and "band" not in eps[0]


def test_bands_must_be_known_bins():
    with pytest.raises(ValueError, match="band"):
        cb.select_episodes([dict(_probe(1, 15.0), band="extra")], per_bin=1,
                           bins=("easy",))


# ---- approach statistics ("why does it blow past it?", 2026-11-04) ----------

def test_a_fast_pass_records_its_entry_speed_and_closest_approach():
    """Straight through the goal at 18 km/h: it enters the 2 m radius, is
    never slow enough to count, and the closest approach is the pass."""
    tr = cb.EpisodeTracker((10.0, 0.0), (0.0, 0.0), max_speed=10.0, radius=1.0,
                           radius_2m=2.0)
    for x in range(1, 21):
        tr.step((float(x), 0.3), 18.0)
    out = tr.finish(reached=False, t_end=20, route_m=10.0)
    assert out["closest_m"] == pytest.approx(0.3)
    assert out["entry_speed_kmh"] == pytest.approx(18.0)
    assert out["min_speed_inside_kmh"] == pytest.approx(18.0)
    assert not out["reached_2m"]


def test_a_slow_arrival_records_the_slowdown():
    tr = cb.EpisodeTracker((10.0, 0.0), (0.0, 0.0), max_speed=10.0, radius=1.0,
                           radius_2m=2.0)
    for x, v in zip((2, 4, 6, 8, 9, 9.5), (20, 20, 18, 14, 9, 4)):
        tr.step((float(x), 0.0), float(v))
    out = tr.finish(reached=True, t_end=6, route_m=10.0)
    # At x=8 the car is exactly 2.0 m out; the radius test is strict, so it
    # first enters at x=9, at 9 km/h.
    assert out["entry_speed_kmh"] == pytest.approx(9.0)
    assert out["min_speed_inside_kmh"] == pytest.approx(4.0)
    assert out["reached_2m"]


def test_an_episode_that_never_gets_near_has_no_entry():
    tr = cb.EpisodeTracker((50.0, 0.0), (0.0, 0.0), max_speed=10.0)
    for x in range(1, 6):
        tr.step((float(x), 0.0), 15.0)
    out = tr.finish(reached=False, t_end=5, route_m=50.0)
    assert out["entry_speed_kmh"] is None and out["min_speed_inside_kmh"] is None
    assert out["closest_m"] == pytest.approx(45.0)

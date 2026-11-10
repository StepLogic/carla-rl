"""Map-subgoal recording (plan 2026-11-09, B2; EXPERT_RECORD_MAP=1 with EVAL_MAP_DIR).

On the training-town multi-turn validation set c0 mostly fails by taking the wrong
branch at junctions while following MAP subgoals (Gate A: wrong branches 62 vs lane
faults 12 in Town01's image-map arm, 82 vs 60 in Town05). B2 records the expert
driving multi-turn routes while the goal input is the map's NEXT NODE IMAGE -- the
deployment interface. Start and goal are localised by true pose (recording may use
privileged information; the policy never sees it), the subgoal switches when the car
passes the node (image_map.Navigator, pose), and each leg is its own shard with that
node's image as its goal (id seed * 100 + leg, as the chained recording). The last
leg's goal is the episode's real goal image. Off by default.

The expert drives the ROAD route to the real goal while the images follow the MAP
route, so an episode is recorded only when the two agree: a map route exists, its
length along the nodes is at most EXPERT_RECORD_MAP_MAX_RATIO x the road route, and
the road route has at least EXPERT_RECORD_MAP_MIN_TURNS junction turns.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.expert_record import (CHAIN_SEGMENTS_MAX, map_route_metres,  # noqa: E402
                                    map_route_skip)

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def _record_src():
    rec = SRC[SRC.index("def run_record("):]
    return rec[:rec.index("\ndef ", 10)]


# ---- the pure rules (torch_rl.expert_record) --------------------------------------

def test_map_route_metres_runs_car_to_nodes_to_goal():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    assert map_route_metres(xy, [0, 1, 2], (-2.0, 0.0), (25.0, 0.0)) == 27.0
    assert map_route_metres(xy, [1], (8.0, 0.0), (12.0, 0.0)) == 4.0


def test_an_agreeing_route_with_enough_turns_is_recorded():
    route = list(range(12))
    assert map_route_skip(route, 115.0, 100.0, turns=2, min_turns=1, max_ratio=1.2) is None
    assert map_route_skip(route, 95.0, 100.0, turns=1, min_turns=1, max_ratio=1.2) is None


def test_routes_the_expert_would_not_follow_are_skipped():
    assert map_route_skip(None, None, 100.0, turns=2) == "no_route"
    assert map_route_skip([7], 3.0, 100.0, turns=2) == "no_route"      # start == goal node
    assert map_route_skip([0, 1, 2], 130.0, 100.0, turns=2, max_ratio=1.2) == "detour"
    assert map_route_skip([0, 1, 2], 100.0, None, turns=2) == "no_road"
    assert map_route_skip([0, 1, 2], 100.0, 100.0, turns=0, min_turns=1) == "turns"


def test_a_route_with_more_legs_than_shard_ids_is_skipped():
    long = list(range(CHAIN_SEGMENTS_MAX + 1))       # legs = nodes after the start + 1
    assert map_route_skip(long, 100.0, 100.0, turns=2) == "legs"
    assert map_route_skip(long[:-1], 100.0, 100.0, turns=2) is None


# ---- the recording loop (train_carla_sac.run_record; needs CARLA, so source checks) --

def test_the_flag_needs_a_map_goal_episodes_and_the_expert():
    assert 'EXPERT_RECORD_MAP = os.environ.get("EXPERT_RECORD_MAP", "0") == "1"' in SRC
    assert "EXPERT_RECORD_MAP and not EVAL_MAP_DIR" in SRC
    assert "EXPERT_RECORD_MAP and not EXPERT_GOAL_ONLY" in SRC
    assert "EXPERT_RECORD_MAP and EXPERT_RECORD_CHAIN" in SRC
    assert 'EXPERT_RECORD_MAP and EXPERT_DRIVER != "expert"' in SRC


def test_legs_switch_by_pose_and_retarget_the_goal_image():
    rec = _record_src()
    assert "image_map.Navigator(gmap, route, switch_m=EVAL_MAP_SWITCH_M" in rec
    assert "nav.update(pos, " in rec
    assert 'end, truncated = "subgoal", True' in rec           # a leg is a cut, not a terminal
    assert "goal_wrapper.retarget(map_frames[nav.current()], info)" in rec
    assert "goal_wrapper.retarget(final_frame, info)" in rec    # the last leg: the real goal
    assert "if EXPERT_RECORD_CHAIN or EXPERT_RECORD_MAP else seed" in rec


def _branch_body(src, head):
    """The lines of the block opened by the line containing `head` (by indentation)."""
    lines = src[src.index(head):].splitlines()
    indent = len(src[:src.index(head)].splitlines()[-1])     # the `if` line's indentation
    body = []
    for ln in lines[1:]:
        if ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
            break
        body.append(ln)
    return "\n".join(body)


def test_a_subgoal_switch_continues_without_resetting_the_env():
    body = _branch_body(_record_src(), 'if end == "subgoal":')
    assert body.strip().endswith("continue")
    assert "env.reset" not in body
    assert "_leg_goal(nav, final_frame, info)" in body        # the next leg's goal image


def test_skipped_episodes_and_lost_legs_are_not_recorded():
    rec = _record_src()
    assert "map_route_skip(" in rec and "map_skipped" in rec
    assert 'end != "map_lost"' in rec              # a leg whose subgoal fell behind is dropped


# ---- the overlay (slurm/il_record_map.env) ------------------------------------------

def test_the_overlay_records_the_validation_band_with_the_map_flag():
    import subprocess
    out = subprocess.run(["bash", "-c", '. "$REPO/slurm/il_record_map.env" && env'],
                         env={"REPO": ROOT, "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                              "HOME": os.environ.get("HOME", "/tmp")},
                         capture_output=True, text=True, check=True).stdout
    env = dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)
    assert env["EXPERT_RECORD_MAP"] == "1" and env["EXPERT_GOAL_ONLY"] == "1"
    # bench_turns.env's band: 50-149 m along the road, 1 m arrival
    assert (env["GOAL_DIST_MIN"], env["GOAL_DIST_MAX"], env["GOAL_DIST_STEP"],
            env["GOAL_RADIUS_M"]) == ("50", "149", "1", "1.0")
    assert env["GOAL_ARRIVAL_MAX_SPEED"] == "0" and env["OFF_ROAD_JUNCTION_M"] == "3.5"
    assert env["GOAL_KEEP_ON_PASS"] == "1" and env["GOAL_CAM_MATCH"] == "1"
    assert env["SPEED_OBS"] == "1" and env["MONO_CHW"] == "1"

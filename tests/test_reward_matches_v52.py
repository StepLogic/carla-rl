"""The launcher's reward must be v52's, term for term.

This arm reuses the reward that actually trains in CARLA. "Reuses" is a claim
about the ENVIRONMENT the job runs with, not about intent -- and the first
version of slurm/train_sac.sbatch inherited its exports from the smoke job, so
21 terms silently fell back to CODE DEFAULTS that are not v52's: off-road cost
NOTHING (EXPLORE_OFF_ROAD_PENALTY 10 -> 0), collisions were not terminal
(EXPLORE_COLLISION_FATAL 1 -> 0), and both lane penalties were half. The run was
already going before the user's "collision should be terminal" led to the diff.

A reward term that is not exported is not neutral: it takes whatever the code
defaults to, which is a different experiment wearing the same name.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "clean_fix1", "results", "run_env",
                    "carla_gated_v52_segvae_samesetup_t05.env")
SBATCH = os.path.join(ROOT, "slurm", "train_sac.sbatch")
# The reward lives in ONE file that launchers source, so a new launcher
# inherits it instead of copying it (user 2026-10-14: make v52 the default
# going forward). Copying is how it drifted the first time.
REWARD_ENV = os.path.join(ROOT, "slurm", "v52_reward.env")
REWARD_SRC = ("src/vae_nav/gated_reward.py", "src/vae_nav/lane_explore_reward.py",
              "src/vae_nav/lane_follow_reward.py")


def _v52():
    out = {}
    for line in open(SNAP):
        k, _, v = line.rstrip("\n").partition("=")
        if k:
            out[k] = v
    return out


def _exported():
    """Every assignment on every export line -- `export A=1 B=2` sets both."""
    out = {}
    lines = list(open(REWARD_ENV)) + list(open(SBATCH))
    for line in lines:
        line = line.strip()
        if not line.startswith("export "):
            continue
        for k, v in re.findall(r"([A-Z0-9_]+)=([^\s]+)", line[len("export "):]):
            # LAST assignment wins, as the shell does. setdefault took the
            # FIRST, so a later deliberate override read as "no drift" while
            # the run used the other value -- the test would have lied.
            out[k] = v
    return out


def _reward_env_names():
    src = ""
    for f in REWARD_SRC:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            src += open(p).read()
    return dict(re.findall(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"\s*,\s*([^)]+)\)', src))


# DELIBERATE DEVIATIONS FROM v52, each with the reason it was made. Anything
# that differs and is NOT listed here is accidental drift and fails the test.
# This is the point of the guard: a reward may change, but never by accident.
DELIBERATE = {
    "EXPLORE_LANE_MODE":
        "2026-10-19: crossings fined, not terminal. v9 (terminal, the user's "
        "rule) never learned to stay in lane: goal arm 0.03-0.16 at 212k vs "
        "v8's 0.55 fined; the user approved going back",
    "LANE_DEPARTURE_PENALTY":
        "2026-10-19 after v8 parked: at 10 a single crossing outweighed parking "
        "for the rest of the episode (parked ~ -10 at gamma 0.99)",
    "POTENTIAL_K":
        "2026-10-14 after v6: a dense potential-based goal term. Both modes are "
        "paid the same reward, so goal mode had nothing to make it behave "
        "differently and its arm decayed 0.100 -> 0.010 while explore improved "
        "to 0.20. Potential-based, so the optimal policy is unchanged",
    "EXPLORE_OFF_ROAD_FATAL":
        "2026-10-14: same reason -- off-road is the other door the agent drifts "
        "through; it is now a per-step fine of 10",
}


def _same(a, b):
    try:
        return float(a) == float(b)        # "1" and "1.0" are one setting
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


@pytest.mark.skipif(not os.path.exists(SNAP), reason="v52 snapshot not present")
def test_every_reward_term_matches_v52():
    v52, exported, names = _v52(), _exported(), _reward_env_names()
    drift = []
    for name, default in names.items():
        if name not in v52:
            continue                       # v52 never set it; nothing to match
        mine = exported.get(name, default.strip().strip('"').strip("'"))
        if not _same(v52[name], mine) and name not in DELIBERATE:
            src = "exported" if name in exported else "CODE DEFAULT"
            drift.append(f"{name}: v52={v52[name]} launcher={mine} ({src})")
    assert not drift, ("the launcher's reward differs from v52 in ways nobody "
                       "declared:\n  " + "\n  ".join(drift))


def test_every_declared_deviation_really_deviates():
    """A stale entry in DELIBERATE would silently excuse a future accident.

    The baseline is v52's recorded value where it has one, and the CODE DEFAULT
    otherwise -- v52 never exported EXPLORE_LANE_MODE, so what it actually ran
    with was the default "terminal", and that is what this deviates from."""
    v52, exported, names = _v52(), _exported(), _reward_env_names()
    for name in DELIBERATE:
        base = v52.get(name)
        if base is None:
            assert name in names, f"{name} is not a reward setting at all"
            base = names[name].strip().strip('"').strip("'")
        assert not _same(base, exported.get(name)), (
            f"{name} no longer differs from its v52 baseline ({base}); "
            "remove it from DELIBERATE")


def test_which_events_end_an_episode():
    """Collisions end the episode (user). Crossings are FINED again after v9
    showed terminal crossings stop learning (user-approved 2026-10-19).
    Off-road is a per-step fine."""
    e = _exported()
    assert e.get("EXPLORE_LANE_MODE") == "penalise", "crossings are fined"
    assert e.get("EXPLORE_COLLISION_FATAL") == "1", "collisions are terminal"
    assert e.get("EXPLORE_OFF_ROAD_FATAL") == "0", "off-road is a fine"


def test_the_reward_is_a_sourced_file_not_an_inlined_copy():
    """The default must be inherited, not pasted: 21 terms drifted when one
    launcher carried its own copy."""
    assert os.path.exists(REWARD_ENV)
    assert 'v52_reward.env' in open(SBATCH).read(), \
        "the launcher must source the reward file rather than inline it"


@pytest.mark.skipif(not os.path.exists(SNAP), reason="v52 snapshot not present")
def test_collisions_are_terminal():
    """User 2026-10-14: "collision should be terminal". v52 had it; the first
    version of this launcher did not export it and the default is off."""
    assert _exported().get("EXPLORE_COLLISION_FATAL") == "1"


@pytest.mark.skipif(not os.path.exists(SNAP), reason="v52 snapshot not present")
def test_leaving_the_road_actually_costs_something():
    """It fell back to 0.0, so driving off the road was free."""
    assert float(_exported().get("EXPLORE_OFF_ROAD_PENALTY", 0)) > 0


def test_progress_is_measured_along_the_lane_as_v52_did():
    """v52 left LANE_PROGRESS_MODE at its code default, `projected`, so the
    term-by-term comparison above never looked at it -- and the launcher's
    `heading` (which pays circling) trained seven runs unnoticed."""
    exported = _exported()
    assert exported.get("LANE_PROGRESS_MODE") == "projected", exported.get(
        "LANE_PROGRESS_MODE")
    import re
    assert not re.search(r"LANE_PROGRESS_MODE=", open(SBATCH).read()), \
        "the launcher must not override the progress mode"


def test_parking_is_not_free():
    """v8 parked because standing still cost almost nothing (2026-10-19)."""
    e = _exported()
    assert float(e.get("EXPLORE_IDLE_PENALTY", 0)) > 0
    assert float(e.get("EXPLORE_IDLE_SPEED_KMH", 0)) < float(e["GOAL_ARRIVAL_MAX_SPEED"])
    assert e.get("EXPLORE_DIST_MODE") == "net"

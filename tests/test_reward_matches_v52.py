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
            out.setdefault(k, v)
    return out


def _reward_env_names():
    src = ""
    for f in REWARD_SRC:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            src += open(p).read()
    return dict(re.findall(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"\s*,\s*([^)]+)\)', src))


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
        if not _same(v52[name], mine):
            src = "exported" if name in exported else "CODE DEFAULT"
            drift.append(f"{name}: v52={v52[name]} launcher={mine} ({src})")
    assert not drift, "the launcher's reward is not v52's:\n  " + "\n  ".join(drift)


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

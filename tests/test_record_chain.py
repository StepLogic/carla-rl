"""Chained DAgger recording (plan 2026-11-09, B1; EXPERT_RECORD_CHAIN=1).

c0 drives chained goals past junctions (GOAL_CHAIN_STYLE=past_junction, to 200 m)
while the expert labels every frame. After an arrival the SAME car drives on: the
env's soft reset (GatedReward, as in the chain eval) places the next goal and its
image. Each goal's segment is saved as its own episode shard -- a shard has ONE
goal image, which the cloning loss reads -- with id seed * 100 + segment, and the
policy's GRU and the expert's plan restart for the new goal. Off by default: every
earlier recording is unchanged.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.expert_record import chain_continues, chain_segment_id  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_segment_ids_are_seed_times_100_plus_segment():
    assert chain_segment_id(900000, 0) == 90000000
    assert chain_segment_id(900002, 7) == 90000207
    with pytest.raises(ValueError):
        chain_segment_id(900000, 100)
    with pytest.raises(ValueError):
        chain_segment_id(900000, -1)


def test_only_a_chained_goal_arrival_continues():
    on = {"chain_continue": True}
    assert chain_continues(True, 1, "reached", on)
    assert not chain_continues(False, 1, "reached", on)          # flag off
    assert not chain_continues(True, 0, "reached", on)           # explore episode
    assert not chain_continues(True, 1, "crossing", on)          # a fault ends it
    assert not chain_continues(True, 1, "reached", {})           # chain complete (200 m)


def test_run_record_uses_them_only_when_asked():
    assert 'EXPERT_RECORD_CHAIN = os.environ.get("EXPERT_RECORD_CHAIN", "0") == "1"' in SRC
    rec = SRC[SRC.index("def run_record("):]
    rec = rec[:rec.index("\ndef ", 10)]
    assert "chain_continues(EXPERT_RECORD_CHAIN, mode, end, info)" in rec
    assert "chain_segment_id(seed, seg) if EXPERT_RECORD_CHAIN else seed" in rec
    assert "env.reset()" in rec and 'info.get("chain_soft_reset")' in rec
    # a new goal: the expert re-plans to it and the policy's memory restarts
    i = rec.index('info.get("chain_soft_reset")')
    tail = rec[i:i + 900]
    assert 'expert.reset(info["goal_location"])' in tail
    assert "agent.reset_hidden(1)" in tail


def test_chained_recording_needs_the_chain_turned_on_in_the_env():
    assert "EXPERT_RECORD_CHAIN and not EVAL_CHAIN" in SRC


def test_goal_only_recording_takes_the_even_seeds_as_on_vint_carla():
    """EXPERT_GOAL_ONLY=1 (ported from vint-carla 598a593, same name and meaning):
    seeds EXPERT_SEED0 + 2k -- episode_plan makes even seeds the goal episodes, and
    the noise level still cycles through every sigma."""
    from torch_rl.expert_record import SIGMAS, episode_plan
    rec = SRC[SRC.index("def run_record("):]
    assert "EXPERT_SEED0 + (2 * k if EXPERT_GOAL_ONLY else k)" in rec
    assert "EXPERT_GOAL_ONLY=1 needs an even EXPERT_SEED0" in rec
    plans = [episode_plan(900000 + 2 * k) for k in range(6)]
    assert all(p["mode"] == 1 for p in plans)
    assert {p["sigma"] for p in plans} == set(SIGMAS)

"""FIXED_ALPHA (user 2026-11-08: "lets start with a fixed alpha", value 0.02): the SAC
temperature is held at a given value -- it starts there, no update moves it, and a
checkpoint load does not bring a learned one back. 0 (the default) keeps it learned.

Why: in runs P and P2 (SAC from scratch) the auto-tuned temperature fell from 0.1 to
1e-4 because the actor's entropy (~1.3) never reached the -2 target; the evals were best
while it passed 0.036 (25k) to 0.012 (50k) and fell apart once it was gone.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")


def _agent(**kw):
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    agent = GatedRecurrentSAC(_cfg(**kw))
    return agent, _chunk(agent.cfg, [1, 0])


def test_a_fixed_alpha_starts_at_its_value():
    agent, _ = _agent(fixed_alpha=0.02)
    assert agent.alpha_g == pytest.approx(0.02) and agent.alpha_e == pytest.approx(0.02)


def test_a_fixed_alpha_never_moves():
    agent, ch = _agent(fixed_alpha=0.02)
    for _ in range(3):
        agent.update(ch)
    assert agent.alpha_g == pytest.approx(0.02, rel=1e-6)
    assert agent.alpha_e == pytest.approx(0.02, rel=1e-6)


def test_zero_keeps_the_temperature_learned():
    agent, ch = _agent(fixed_alpha=0.0)
    a0 = agent.alpha_g
    agent.update(ch)
    assert agent.alpha_g != a0


def test_a_checkpoint_load_keeps_the_fixed_value(tmp_path):
    learned, ch = _agent()
    learned.update(ch)                          # a learned temperature, not 0.02
    learned.save(str(tmp_path))
    fixed, _ = _agent(fixed_alpha=0.02)
    fixed.load(str(tmp_path))
    assert fixed.alpha_g == pytest.approx(0.02) and fixed.alpha_e == pytest.approx(0.02)


def test_the_learner_description_says_so():
    agent, _ = _agent(fixed_alpha=0.02)
    assert "temperature FIXED at 0.02" in agent.describe()
    assert "FIXED" not in _agent()[0].describe()

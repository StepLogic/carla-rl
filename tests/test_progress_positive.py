"""GOAL_PROGRESS_POSITIVE: pay max(d_prev - d, 0) instead of the signed delta.

User 2026-10-04, on top of the DEBiT goal reward. The signed form is a potential
function and telescopes to d_start - d_end, so shaping cannot change the optimal
policy. Clipping at zero DELIBERATELY breaks that: driving away becomes free.
These tests pin both the intended behaviour and the hazard it creates, because
an approach-retreat cycle now pays every time round and nothing in the reward
says otherwise.

The clip lives in gated_reward.progress_term, which episode_her binds directly
(`self._progress_term = _gr.progress_term`), so hindsight rows get the same
definition. A relabeller using the signed form against an env that clips would
fit two inconsistent MDPs over one set of states -- the bug class that mis-costed
83% of v13-v15's hindsight rows.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))


@pytest.fixture
def gr(monkeypatch):
    from vae_nav import gated_reward as _gr
    return _gr


def test_signed_by_default_so_every_existing_run_is_unchanged(gr, monkeypatch):
    monkeypatch.setattr(gr, "PROGRESS_POSITIVE", False)
    assert gr.progress_term(10.0, 12.0) == pytest.approx(-2.0)
    assert gr.progress_term(12.0, 10.0) == pytest.approx(+2.0)


def test_moving_away_is_free_when_clipped(gr, monkeypatch):
    monkeypatch.setattr(gr, "PROGRESS_POSITIVE", True)
    assert gr.progress_term(10.0, 12.0) == 0.0          # regress costs nothing
    assert gr.progress_term(12.0, 10.0) == pytest.approx(+2.0)


def test_the_oscillation_it_permits_is_real_and_measured_here(gr, monkeypatch):
    """THE HAZARD, pinned rather than hidden. Approach 2 m, retreat 2 m, repeat:
    the signed form pays 0 over the cycle, the clipped form pays 2 every cycle
    while the car ends where it started."""
    monkeypatch.setattr(gr, "PROGRESS_POSITIVE", False)
    signed = gr.progress_term(12.0, 10.0) + gr.progress_term(10.0, 12.0)
    monkeypatch.setattr(gr, "PROGRESS_POSITIVE", True)
    clipped = gr.progress_term(12.0, 10.0) + gr.progress_term(10.0, 12.0)
    assert signed == pytest.approx(0.0)
    assert clipped == pytest.approx(2.0)


def test_the_jump_guard_still_fires_before_the_clip(gr, monkeypatch):
    """A route replan looks like a huge step. It must read 0 under BOTH modes --
    clipping must not turn a 50 m planner artefact into a 50 m payout."""
    monkeypatch.setattr(gr, "PROGRESS_POSITIVE", True)
    monkeypatch.setattr(gr, "PROGRESS_MAX_STEP_M", 5.0)
    assert gr.progress_term(60.0, 10.0) == 0.0
    assert gr.progress_term(10.0, 60.0) == 0.0


def test_missing_route_is_still_zero(gr, monkeypatch):
    monkeypatch.setattr(gr, "PROGRESS_POSITIVE", True)
    assert gr.progress_term(None, 10.0) == 0.0
    assert gr.progress_term(10.0, None) == 0.0


def test_the_relabeller_binds_the_same_function(gr):
    """If HER ever stops sharing progress_term, the clip would apply to live rows
    and not to hindsight ones."""
    import inspect

    from vae_nav import episode_her
    src = inspect.getsource(episode_her)
    assert "_gr.progress_term" in src, (
        "episode_her no longer binds gated_reward.progress_term; the clip would "
        "then apply to env rows but not relabelled ones")

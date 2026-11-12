"""Discounted lane-terminal bill (user 2026-09-29, option 2)."""
import pytest

import vae_nav.gated_reward as gr


def test_bill_values(monkeypatch):
    monkeypatch.setattr(gr, "LANE_TERMINAL_BILL", "discounted")
    monkeypatch.setattr(gr, "_BILL_GAMMA", 0.99)
    assert gr.lane_terminal_bill(0) == pytest.approx(1.0)
    assert gr.lane_terminal_bill(99) == pytest.approx((1 - 0.99 ** 100) / 0.01)
    assert gr.lane_terminal_bill(10_000) == pytest.approx(100.0, abs=1e-3)
    monkeypatch.setattr(gr, "LANE_TERMINAL_BILL", "clock")
    assert gr.lane_terminal_bill(99) == 100.0


def test_broken_crossing_ends_goal_step_with_discounted_bill(monkeypatch):
    monkeypatch.setattr(gr, "LANE_TERMINAL", True)
    monkeypatch.setattr(gr, "GOAL_TRUNCATE_COUNTS", True)
    monkeypatch.setattr(gr, "LANE_TERMINAL_BILL", "discounted")
    monkeypatch.setattr(gr, "_BILL_GAMMA", 0.99)
    info = {"lane_any": True, "lane_invasion_any": True,
            "lane_marking_types": ["Broken"], "in_junction": False}
    r, term, trunc, arrived = gr.goal_reward_sorb(info, t=100, max_steps=1000)
    assert term and not trunc and not arrived
    assert r == pytest.approx(-gr.lane_terminal_bill(900))
    # a junction crossing is forgiven: the step goes on
    info_j = dict(info, in_junction=True)
    _, term_j, _, _ = gr.goal_reward_sorb(info_j, t=100, max_steps=1000)
    assert not term_j

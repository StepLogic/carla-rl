"""--collect-action-mode picks the collection action: mode or sample.

USER 2026-09-13: "one trace in town 2 with sampled actions". Every earlier
trace used agent.eval_actions (dist.mode(), deterministic).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeAgent:
    def __init__(self):
        self.calls = []

    def eval_actions(self, obs):
        self.calls.append("eval")
        return "mode-action"

    def sample_actions(self, obs):
        self.calls.append("sample")
        return "sampled-action"


def test_eval_uses_the_mode():
    import eval_placement as ep
    a = FakeAgent()
    assert ep.collect_action(a, {}, "eval") == "mode-action"
    assert a.calls == ["eval"]


def test_sample_uses_sample_actions():
    import eval_placement as ep
    a = FakeAgent()
    assert ep.collect_action(a, {}, "sample") == "sampled-action"
    assert a.calls == ["sample"]


def test_default_is_eval():
    import eval_placement as ep
    a = FakeAgent()
    ep.collect_action(a, {})
    assert a.calls == ["eval"]


def test_rejects_unknown_mode():
    import eval_placement as ep
    with pytest.raises(ValueError):
        ep.collect_action(FakeAgent(), {}, "greedy")

"""scripts/vint/patch_vint_start_pad.py: ViNT trains from frame 0 with the context
padded by the first frame (VINT_START_PAD=1), as the driver feeds it at a start
(torch_rl.vint_topomap.context_steps / ViNTDriver pad with the first frame).
ViNT's own loader starts at frame context*spacing, so a start never appears in
training (v1's start-up failures: it turns toward the goal from a standstill)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "vint"))

import patch_vint_start_pad as patch  # noqa: E402

SRC = '''import os
class D:
    context_size, waypoint_spacing, end_slack, len_traj_pred = 5, 3, 0, 5

    def begin(self, traj_len):
            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            return begin_time, end_time

    def ctx(self, f_curr, curr_time):
            context_times = list(range(curr_time + -self.context_size * self.waypoint_spacing,
                                       curr_time + 1, self.waypoint_spacing))
            context = [(f_curr, t) for t in context_times]
            return context
'''


def _cls(src):
    ns = {}
    exec(src, ns)
    return ns["D"]()


def test_off_by_default_nothing_changes(monkeypatch):
    monkeypatch.delenv("VINT_START_PAD", raising=False)
    d = _cls(patch.apply(SRC))
    assert d.begin(100)[0] == 15
    assert d.ctx("a", 15) == [("a", t) for t in (0, 3, 6, 9, 12, 15)]


def test_on_samples_start_at_frame_0_with_the_first_frame_as_padding(monkeypatch):
    monkeypatch.setenv("VINT_START_PAD", "1")
    d = _cls(patch.apply(SRC))
    assert d.begin(100)[0] == 0
    assert d.ctx("a", 4) == [("a", 0), ("a", 0), ("a", 0), ("a", 0), ("a", 1), ("a", 4)]


def test_idempotent_and_refuses_unknown_code():
    once = patch.apply(SRC)
    assert patch.apply(once) == once
    with pytest.raises(SystemExit):
        patch.apply("class D:\n    pass\n")

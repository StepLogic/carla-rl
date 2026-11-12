"""ViNT's _build_index stores the farthest goal in FRAMES (max_dist_cat *
waypoint_spacing) and _sample_goal multiplies the sampled offset by
waypoint_spacing again. At spacing 1 (every released dataset) the two agree; at
our stride 3 goals land up to 3x too far (past the episode: the probe's
"AssertionError: 82 an 71") and distance labels reach 60 against max_dist_cat 20.
scripts/vint/patch_vint_dataset.py makes the stored bound a count of STRIDES."""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH = os.path.join(ROOT, "scripts", "vint", "patch_vint_dataset.py")
BUGGY = ("                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, "
         "traj_len - curr_time - 1)\n")


def _fake(tmp_path, line=BUGGY):
    p = tmp_path / "vint_dataset.py"
    p.write_text("class ViNT_Dataset:\n    def _build_index(self):\n" + line + "        pass\n")
    return p


def _run(p):
    return subprocess.run([sys.executable, PATCH, str(p)], capture_output=True, text=True)


def test_the_patch_bounds_goals_in_strides(tmp_path):
    p = _fake(tmp_path)
    r = _run(p)
    assert r.returncode == 0, r.stderr
    src = p.read_text()
    assert ("max_goal_distance = min(self.max_dist_cat, (traj_len - curr_time - 1) "
            "// self.waypoint_spacing)") in src
    assert "self.max_dist_cat * self.waypoint_spacing" not in src


def test_the_patch_is_idempotent_and_refuses_unknown_code(tmp_path):
    p = _fake(tmp_path)
    assert _run(p).returncode == 0
    once = p.read_text()
    assert _run(p).returncode == 0 and p.read_text() == once
    other = tmp_path / "other"
    other.mkdir()
    r = _run(_fake(other, line="                max_goal_distance = 7\n"))
    assert r.returncode != 0


def test_the_patched_bound_keeps_every_goal_inside_the_episode():
    """The arithmetic the patch relies on: offset <= bound, goal = curr + offset * s."""
    for traj_len in (31, 40, 71, 200):
        for s in (1, 3):
            for curr in range(5 * s, traj_len - 5 * s):
                bound = min(20, (traj_len - curr - 1) // s)
                assert curr + bound * s <= traj_len - 1 and bound <= 20

"""Choosing the cloning checkpoint on training-town validation (imitation-first
plan, Task 7). The validation evals run every checkpoint with the true goal
image and with a wrong one; a clone that reaches the place in the picture
should NOT reach it when shown somewhere else, so a checkpoint whose wrong-goal
success is too high is following the road, not the goal."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from select_checkpoint import pool_eval_dirs, select_checkpoint  # noqa: E402


def test_highest_true_among_qualifying():
    assert select_checkpoint({2500: (0.6, 0.05), 5000: (0.7, 0.2), 7500: (0.65, 0.1)}) == 7500


def test_ties_go_to_the_later_step():
    assert select_checkpoint({2500: (0.7, 0.0), 5000: (0.7, 0.05), 7500: (0.6, 0.0)}) == 5000


def test_falls_back_to_the_largest_gap_when_none_qualify():
    # 5000 has the best true rate, but 7500 separates true from wrong the most
    assert select_checkpoint({2500: (0.6, 0.3), 5000: (0.8, 0.4), 7500: (0.7, 0.2)}) == 7500


def _eval(root, town, step, sr, n, wrong=False):
    d = os.path.join(root, f"il_bc_r0_val_{town}", f"eval_step_{step}" + ("_wrong" if wrong else ""))
    os.makedirs(d)
    with open(os.path.join(d, "eval.json"), "w") as fh:
        json.dump({"checkpoint": f"/x/checkpoints/il_bc_r0/step_{step}", "episodes": n,
                   "success_rate": sr, "towns": [town], "deterministic": True}, fh)
    return d


def test_pools_towns_by_episodes_and_tells_wrong_from_true(tmp_path):
    dirs = [_eval(tmp_path, "Town05", 2500, 0.75, 40), _eval(tmp_path, "Town05", 2500, 0.05, 40, True),
            _eval(tmp_path, "Town01", 2500, 0.50, 20), _eval(tmp_path, "Town01", 2500, 0.20, 20, True)]
    res = pool_eval_dirs(dirs)
    assert set(res) == {2500}
    assert res[2500] == pytest.approx((40 / 60, 6 / 60))


def test_a_step_missing_its_wrong_goal_eval_is_refused(tmp_path):
    dirs = [_eval(tmp_path, "Town05", 2500, 0.75, 40), _eval(tmp_path, "Town05", 2500, 0.05, 40, True),
            _eval(tmp_path, "Town05", 5000, 0.80, 40)]
    with pytest.raises(ValueError, match="5000"):
        pool_eval_dirs(dirs)

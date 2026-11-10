"""scripts/paired_wrong.py: the paired true- vs wrong-goal test every eval pair prints
(slurm/il_eval_pair.sh). It lived only in ~/paired_wrong.py on the cluster; moved
into the repo unchanged (md5 08825fc4e0d5cb7456bf761c3f30a536), plan 2026-11-09 F2.
"""
import json
import os
import subprocess
import sys
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "paired_wrong.py")


def _eval(d, reached):
    os.makedirs(d, exist_ok=True)
    eps = [{"seed": s, "reached": r} for s, r in reached.items()]
    json.dump({"success_rate": sum(reached.values()) / len(reached), "episodes_detail": eps},
              open(os.path.join(d, "eval.json"), "w"))


def test_counts_gap_and_exact_mcnemar(tmp_path):
    true = {s: s < 7 for s in range(10)}              # 7 of 10
    wrong = {s: s in (0, 1, 8) for s in range(10)}    # 3 of 10; 8 is wrong-only
    _eval(tmp_path / "t", true)
    _eval(tmp_path / "w", wrong)
    out = subprocess.run([sys.executable, "-I", SCRIPT, str(tmp_path / "t"), str(tmp_path / "w")],
                         capture_output=True, text=True, check=True).stdout
    assert "paired on 10 episodes" in out
    assert "true 0.700   wrong 0.300   gap +0.400" in out
    assert "both 2  neither 2  only-true 5  only-wrong 1" in out
    p = min(1.0, 2 * sum(comb(6, k) for k in range(2)) / 2 ** 6)
    assert f"McNemar exact two-sided p = {p:.4f}" in out


def test_only_shared_seeds_are_paired(tmp_path):
    _eval(tmp_path / "t", {1: True, 2: True, 3: False})
    _eval(tmp_path / "w", {2: False, 3: False, 4: True})
    out = subprocess.run([sys.executable, "-I", SCRIPT, str(tmp_path / "t"), str(tmp_path / "w")],
                         capture_output=True, text=True, check=True).stdout
    assert "paired on 2 episodes" in out
    assert "only-true 1  only-wrong 0" in out


def test_the_eval_pair_calls_the_repo_copy():
    src = open(os.path.join(ROOT, "slurm", "il_eval_pair.sh")).read()
    assert "scripts/paired_wrong.py" in src and "~/paired_wrong.py" not in src

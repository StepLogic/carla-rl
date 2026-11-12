"""scripts/bench_map_report.py for any checkpoint's benchmark tree (2026-11-06).

The report hard-coded v30's run names (bench_long_true_*, bench_map_*) and its
step (300000). The c0 benchmark (slurm/bench_c0.sh) writes bench_c0_true_* and
bench_c0map_* at step 15000, so the globs are arguments, v30's the defaults; and
a per-bin SR / SPL table joins the paired McNemar blocks.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import bench_map_report as rep  # noqa: E402


def _row(seed, band, reached, end="reached", path=20.0, shortest=10.0):
    return {"town": "Town02", "seed": seed, "band": band, "reached": reached,
            "reached_2m": reached, "end": end, "path_m": path, "shortest_m": shortest}


def _write(root, run, sub, rows):
    d = os.path.join(root, run, sub)
    os.makedirs(d)
    json.dump({"episodes_detail": rows}, open(os.path.join(d, "eval.json"), "w"))


def _tree(tmp_path):
    root = str(tmp_path)
    _write(root, "bench_c0_true_Town02_easy", "eval_step_15000",
           [_row(1, "easy", True), _row(2, "easy", False, end="off_road")])
    _write(root, "bench_c0map_entropy_Town02_easy", "eval_step_15000_map-entropy",
           [_row(1, "easy", True, path=10.0), _row(2, "easy", False, end="unroutable")])
    return root


def test_the_default_globs_are_v30s():
    a = rep.parse_args([])
    assert a.mapless_glob == "bench_long_true_*_*/eval_step_300000/eval.json"
    assert a.map_glob == "bench_map_{pl}_*_*/eval_step_300000_map-{pl}/eval.json"


def test_a_c0_tree_is_read_with_its_own_globs(tmp_path, capsys):
    root = _tree(tmp_path)
    rep.main(["--root", root, "--placements", "entropy",
              "--mapless-glob", "bench_c0_true_*_*/eval_step_15000/eval.json",
              "--map-glob", "bench_c0map_{pl}_*_*/eval_step_15000_map-{pl}/eval.json"])
    out = capsys.readouterr().out
    assert "mapless 2 episodes; arms [('entropy', 2)]; 2 episodes present in every arm" in out


def test_the_per_bin_table_gives_sr_and_spl(tmp_path, capsys):
    """SPL per episode = reached * shortest / max(path, shortest); unroutable = 0."""
    root = _tree(tmp_path)
    rep.main(["--root", root, "--placements", "entropy",
              "--mapless-glob", "bench_c0_true_*_*/eval_step_15000/eval.json",
              "--map-glob", "bench_c0map_{pl}_*_*/eval_step_15000_map-{pl}/eval.json"])
    out = capsys.readouterr().out
    # mapless: 1 of 2 reached, SPL (10/20)/2 = 0.25; entropy: 1 of 2, SPL (10/10)/2 = 0.5
    assert "| mapless | easy | 2 | 0.500 | 0.500 | 0.250 |" in out
    assert "| entropy | easy | 2 | 0.500 | 0.500 | 0.500 |" in out
    assert "| mapless | pooled | 2 | 0.500 | 0.500 | 0.250 |" in out

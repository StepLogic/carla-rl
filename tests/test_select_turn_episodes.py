"""scripts/select_turn_episodes.py: junction-turn bands (2026-11-12, goal-directedness
bands of exactly 1, exactly 2 and 3+ junction turns, all from the 50-149 m window)."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "select_turn_episodes.py")


def _probes(tmp_path):
    rows = []
    for seed in range(100):                       # turns cycle 0..4, all lengths in-window
        rows.append({"town": "Town02", "seed": 41000 + seed, "band": "turns",
                     "route_m": 60.0 + seed, "euclid_m": 40.0 + seed / 2,
                     "turns": seed % 5, "junctions": seed % 5 + 1})
    rows.append({"town": "Town02", "seed": 49999, "band": "turns", "route_m": 160.0,
                 "euclid_m": 100.0, "turns": 1, "junctions": 2})   # too long: never kept
    p = tmp_path / "probe.json"
    p.write_text(json.dumps(rows))
    return p


def _run(tmp_path, *args):
    out = tmp_path / "eps.json"
    r = subprocess.run([sys.executable, SCRIPT, str(out), str(_probes(tmp_path)), *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return json.loads(out.read_text())


def test_default_is_unchanged_two_or_more_turns_labelled_hard(tmp_path):
    eps = _run(tmp_path, "--per-town", "10")
    assert len(eps) == 10
    assert all(e["turns"] >= 2 and e["bin"] == "hard" for e in eps)


def test_exactly_one_turn_band(tmp_path):
    eps = _run(tmp_path, "--per-town", "10", "--min-turns", "1", "--max-turns", "1",
               "--bin", "turns1")
    assert len(eps) == 10
    assert all(e["turns"] == 1 and e["bin"] == "turns1" for e in eps)
    assert all(50.0 <= e["route_m"] < 150.0 for e in eps)


def test_three_or_more_band_has_no_upper_limit(tmp_path):
    eps = _run(tmp_path, "--per-town", "10", "--min-turns", "3", "--bin", "turns3plus")
    assert {e["turns"] for e in eps} == {3, 4}
    assert all(e["bin"] == "turns3plus" for e in eps)


def test_too_few_qualifying_episodes_fails(tmp_path):
    out = tmp_path / "eps.json"
    r = subprocess.run([sys.executable, SCRIPT, str(out), str(_probes(tmp_path)),
                        "--per-town", "50", "--min-turns", "1", "--max-turns", "1"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "qualifying" in (r.stderr + r.stdout)

"""The chooser must tolerate the dead-end nodes every multi-episode trace has.

Measured 2026-09-11: best attainable directed reach on the Town02 traces was
0.995, so a 1.0 requirement chose nothing and failed every stage-3 run."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "calibrate_critic", pathlib.Path(__file__).resolve().parents[1] / "calibrate_critic.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _row(e, c, nodes, reach):
    return {"edge_cutoff": e, "consistency": c, "nodes": nodes,
            "reach_directed": reach}


def test_requires_every_trace_and_prefers_smallest_cutoff_then_fewest_nodes():
    results = {
        "a": [_row(200, 50, 900, 0.999), _row(200, 100, 800, 0.995),
              _row(400, 50, 950, 0.999), _row(400, 100, 700, 0.999)],
        "b": [_row(200, 50, 910, 0.996), _row(200, 100, 790, 0.985),
              _row(400, 50, 960, 0.999), _row(400, 100, 690, 0.999)],
    }
    chosen, recs = cc.choose_setting(results, 0.99)
    # (200,100) fails on trace b; (200,50) is the smallest cutoff that passes
    assert chosen == (200, 50)
    assert [r["reach_directed"] for r in recs] == [0.999, 0.996]
    chosen, _ = cc.choose_setting(results, 0.98)
    assert chosen == (200, 100)            # now admissible and fewer nodes
    assert cc.choose_setting(results, 1.0) == (None, [])

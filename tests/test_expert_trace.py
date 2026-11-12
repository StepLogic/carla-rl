"""The single-trace evaluation maps' trace (user 2026-11-06): the expert drives one
closed coverage tour of a HELD-OUT town while the clone rides along for its
entropy. Pinned here: the one documented bypass of the held-out guard, that the
trace can never land where training data is read from, and the chunked writer
that keeps run_trace's on-disk format (so image_map.load_trace reads it) while
recording which uninterrupted DRIVE each chunk belongs to and how far along the
tour it got, so a crash resumes at the crash point instead of starting over.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.expert_record import assert_recordable_town  # noqa: E402
from torch_rl.expert_trace import ChunkWriter, drive_ids, trace_dir  # noqa: E402
from vae_nav import image_map  # noqa: E402


def test_trace_only_is_the_one_bypass_of_the_held_out_guard():
    for t in ("Town02", "Town03", "Town07"):
        with pytest.raises(ValueError, match="held out"):
            assert_recordable_town(t)
        assert_recordable_town(t, trace_only=True)


def test_the_trace_never_lands_under_the_expert_data_root(tmp_path):
    rec = tmp_path / "checkpoints" / "il_data"
    with pytest.raises(ValueError, match="training"):
        trace_dir(str(rec), "Town02", record_root=str(rec))
    with pytest.raises(ValueError, match="training"):
        trace_dir(str(rec / "maps"), "Town02", record_root=str(rec))
    with pytest.raises(ValueError, match="training"):
        trace_dir(str(tmp_path / "checkpoints" / "il_data_cover"), "Town02", record_root=str(rec))
    d = trace_dir(str(tmp_path / "checkpoints" / "trace_expert"), "Town02", record_root=str(rec))
    assert d == str(tmp_path / "checkpoints" / "trace_expert" / "trace_expert_Town02")


def _sample(t, x):
    return {"frame": np.full((4, 4, 3), t % 255, np.uint8), "pos": np.array([x, 0.0]),
            "yaw": 0.0, "speed": 15.0, "action": np.zeros(2, np.float32), "reward": 0.0,
            "entropy": -2.0 + 0.01 * t, "entropy_h0": -2.5, "q_mean": 1.0, "q_std": 0.1,
            "t": t, "on_line": False}


def test_chunks_are_one_drive_that_load_trace_reads_as_continuous(tmp_path):
    w = ChunkWriter(str(tmp_path), chunk=3, meta={"town": "Town02"})
    for t in range(7):
        w.add(_sample(t, 4.0 * t), plan_index=10 + t)
    w.close(end="route_end", done=True)
    index = json.load(open(tmp_path / "index.json"))
    assert [s["kept"] for s in index["segments"]] == [3, 3, 1]
    assert {s["drive"] for s in index["segments"]} == {0}
    assert not any(s["jumped"] for s in index["segments"])
    assert index["segments"][-1]["plan_index"] == 16
    assert index["done"] is True
    tr = image_map.load_trace(str(tmp_path))
    assert len(tr["pos"]) == 7 and tr["entropy"][6] == pytest.approx(-1.94)
    assert drive_ids(tr, index).tolist() == [0] * 7
    # a single drive builds one chain of edges with no break at the chunk joints
    g = image_map.build_graph(tr, [0, 3, 6])
    assert set(g.edges) == {(0, 1), (1, 2)}


def test_an_env_ending_the_episode_starts_a_new_drive_in_the_same_run(tmp_path):
    w = ChunkWriter(str(tmp_path), chunk=10, meta={"town": "Town02"})
    for t in range(3):
        w.add(_sample(t, 4.0 * t), plan_index=t)
    w.new_drive(end="collision", jumped=False)
    for t in range(3, 5):
        w.add(_sample(t, 4.0 * t), plan_index=t)
    w.new_drive(end="stuck", jumped=True)
    w.add(_sample(5, 20.0), plan_index=9)
    w.close(end="route_end", done=True)
    segs = json.load(open(tmp_path / "index.json"))["segments"]
    assert [(s["drive"], s["end"], s["continues"], s["jumped"]) for s in segs] == [
        (0, "collision", False, False), (1, "stuck", False, False), (2, "route_end", False, True)]


def test_a_resume_starts_a_new_drive_where_the_last_chunk_stopped(tmp_path):
    w = ChunkWriter(str(tmp_path), chunk=2, meta={"town": "Town02"})
    for t in range(5):
        w.add(_sample(t, 0.4 * t), plan_index=100 + t)
    w.flush()                                  # the job dies here: the partial chunk is kept
    r = ChunkWriter(str(tmp_path), chunk=2, meta={"town": "Town02"})
    assert r.drive == 1 and r.resume_plan_index == 104
    assert r.steps == 5 and r.done is False
    for t in range(5, 8):
        r.add(_sample(t, 0.4 * t), plan_index=100 + t)
    r.close(end="route_end", done=True)
    index = json.load(open(tmp_path / "index.json"))
    tr = image_map.load_trace(str(tmp_path))
    assert drive_ids(tr, index).tolist() == [0] * 5 + [1] * 3
    assert ChunkWriter(str(tmp_path), chunk=2, meta={}).done is True

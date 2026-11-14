"""Expert rows for learner C behind the CONTROL STREAM (torch_rl/frl/expert.episode_transitions_control;
user 2026-11-14: "Queue online train for best fixes with SAC with expert pinned rows").

The pinned expert rows must be what the online loop would have stored had the expert driven it: the SAME
decision windows, executed action, summed and relabelled reward, arrival cut and terminal mask as the c0 arm's
rows (episode_transitions) -- and features from the trunk the run acts through. With the control stream as
that trunk ONLY THE FEATURES CHANGE. Every defect here is silent (a critic pretrained on rows whose features
or rewards are one tick off still trains), so the trunk is faked: it records which ticks, speeds and mode it
was asked for and returns features that name their tick, which makes every alignment readable.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")

from torch_rl import pixel_rows as _pxr  # noqa: E402
from torch_rl.frl import expert as fx  # noqa: E402

KMH = 20.0          # pixel_rows.SPEED_SCALE_KMH: the state's speed column is km/h / 20


@pytest.fixture(scope="module", autouse=True)
def _chw_speed_rows():
    """The row layout is a module constant of torch_rl.pixel_rows, read from the environment at its
    FIRST import; in a full-suite run another test module imports it first with neither flag set, and
    the c0 trunk this file compares against then has a one-column state ([mode], no speed). The
    setdefault lines above only help when this file runs alone, so the layout is forced here for this
    module's tests, before the c0 fixture is built, and put back afterwards."""
    mp = pytest.MonkeyPatch()
    mp.setattr(_pxr, "MONO_CHW", True)
    mp.setattr(_pxr, "SPEED_OBS", True)
    mp.setattr(_pxr, "PRIV_OBS", False)
    yield
    mp.undo()


class _FakeTrunk:
    """carla.ControlTrunk's interface for recorded frames. Feature j is filled with its TICK NUMBER and its
    state is [mode, the speed it was handed for that tick / 20], so a row shows where it came from."""

    def __init__(self, repeat, emb_dim=6, state_dim=2):
        self.repeat, self.emb_dim, self.state_dim = int(repeat), int(emb_dim), int(state_dim)
        self.calls = []

    def embed_ticks(self, frames, binoc, mode, kmh, ticks):
        ticks = [int(t) for t in ticks]
        self.calls.append({"ticks": ticks, "mode": int(mode), "kmh": np.asarray(kmh, np.float32).copy(),
                           "frames": len(frames), "binoc": len(binoc)})
        emb = np.stack([np.full(self.emb_dim, float(t), np.float32) for t in ticks])
        states = np.zeros((len(ticks), self.state_dim), np.float32)
        states[:, 0] = float(mode)
        states[:, 1] = [float(kmh[t]) / KMH for t in ticks]
        return emb, states


def _episode(T, mode=0, end="crossing", terminal=True, px=4, binoc=3, seed=0):
    g = np.random.default_rng(seed)
    return {"frames": g.integers(0, 256, (T + 1, px, px, 3), dtype=np.uint8),
            "binoc": g.standard_normal((T + 1, binoc)).astype(np.float16),
            "act_exec": g.uniform(-1, 1, (T, 2)).astype(np.float32),
            "act_label": g.uniform(-1, 1, (T, 2)).astype(np.float32),
            "reward": np.arange(T, dtype=np.float32) + 1.0,            # 1, 2, 3, ...
            "speed": np.full(T, 10.0, np.float32), "mode": np.array(mode),
            "terminal": np.array(terminal), "truncated": np.array(not terminal),
            "end": np.array(end), "driver": np.array("expert")}


def _rows(ep, k, trunk=None, idle_penalty=0.0, junction_bonus=0.0):
    trunk = trunk or _FakeTrunk(repeat=k)
    return fx.episode_transitions_control(trunk, ep, k=k, idle_penalty=idle_penalty,
                                          junction_bonus=junction_bonus), trunk


def test_every_tick_gets_one_decision_across_the_phases():
    """At action repeat k a decision starts every k-th tick, so one pass (phase 0) would use an eighth of
    the recorded ticks at k=8. Phases 0..k-1 start one tick later each; together every tick is the first
    tick of exactly one decision."""
    tr, trunk = _rows(_episode(11), k=4)
    # one trunk pass per phase, over that phase's window starts plus the end of its last window
    assert [c["ticks"] for c in trunk.calls] == [[0, 4, 8, 11], [1, 5, 9, 11], [2, 6, 10, 11], [3, 7, 11]]
    assert sorted(tr["emb"][:, 0].tolist()) == list(range(11))           # each tick starts one decision
    assert len(tr["rewards"]) == 11
    assert all(c["frames"] == 12 and c["binoc"] == 12 for c in trunk.calls)   # the whole episode is offered


def test_a_row_is_its_windows_first_tick_its_end_the_executed_action_and_the_summed_reward():
    T, k = 11, 4
    ep = _episode(T, mode=0)
    base = 0.1 * (np.arange(T, dtype=np.float32) + 1.0)
    ep["reward"] = base.copy()
    ep["reward"][5] += 10.0                          # a tick that was paid the explore junction bonus
    ep["speed"] = np.arange(T, dtype=np.float32)     # ticks 0 and 1 are below the idle speed (2 km/h)
    tr, _ = _rows(ep, k, idle_penalty=0.5, junction_bonus=10.0)
    want = base.copy()                               # the bonus removed ...
    want[[0, 1]] -= 0.5                              # ... and the idle charge added
    kmh = np.concatenate([[0.0], ep["speed"]])
    j = 0
    for phase in range(k):
        for f in range(phase, T, k):
            end = min(f + k, T)
            assert np.all(tr["emb"][j] == f), (phase, f)                 # the feature AT the first tick
            assert np.all(tr["next_emb"][j] == end), (phase, f)          # and at the window's end
            assert np.allclose(tr["states"][j], [0.0, kmh[f] / KMH])
            assert np.allclose(tr["next_states"][j], [0.0, kmh[end] / KMH])
            assert np.array_equal(tr["actions"][j], ep["act_exec"][f])   # EXECUTED, at the first tick
            assert not np.array_equal(tr["actions"][j], ep["act_label"][f])
            assert np.isclose(tr["rewards"][j], want[f:end].sum(), atol=1e-5), (phase, f)
            j += 1
    assert j == len(tr["rewards"]) == T


def test_the_goal_arm_keeps_its_rewards_above_half_a_junction_bonus():
    """The junction bonus was an EXPLORE reward; in goal mode a large tick reward is something else (the
    recorded shaping) and must not lose 10."""
    ep = _episode(8, mode=1)
    ep["reward"] = np.full(8, 6.0, np.float32)
    tr, trunk = _rows(ep, 4, junction_bonus=10.0)
    assert np.allclose(tr["rewards"][:2], [24.0, 24.0])
    assert all(c["mode"] == 1 for c in trunk.calls)
    assert np.all(tr["states"][:, 0] == 1.0)


def test_the_trunk_is_given_the_speed_each_frame_was_taken_at():
    """Shards store speed[t] from the step that produced frame t+1 (torch_rl/expert_data.py), so frame f
    was taken at speed[f-1] and frame 0 at rest. The online row carries the speed when its frame was taken;
    handing the stream speed[f] instead would be one tick in the future."""
    ep = _episode(9)
    ep["speed"] = np.arange(9, dtype=np.float32) + 1.0
    _, trunk = _rows(ep, 4)
    for c in trunk.calls:
        assert np.array_equal(c["kmh"], np.concatenate([[0.0], ep["speed"]]).astype(np.float32))


def test_a_reached_episode_loses_its_arrival_and_keeps_bootstrapping():
    cut = fx.ARRIVAL_CUT
    T, k = cut + 10, 4
    ep = _episode(T, mode=1, end="reached", terminal=True)
    tr, trunk = _rows(ep, k)
    assert [c["ticks"] for c in trunk.calls] == [[0, 4, 8, 10], [1, 5, 9, 10], [2, 6, 10], [3, 7, 10]]
    assert max(max(c["ticks"]) for c in trunk.calls) == T - cut          # nothing past the cut is embedded
    assert len(tr["rewards"]) == 10
    assert tr["masks"].tolist() == [1.0] * 10                            # cut = truncated: it bootstraps
    assert tr["rewards"].max() == 7 + 8 + 9 + 10                         # no tick past the cut is paid
    assert np.allclose(tr["rewards"][:3], [1 + 2 + 3 + 4, 5 + 6 + 7 + 8, 9 + 10])


def test_a_fault_ends_only_the_window_it_happened_in():
    tr, _ = _rows(_episode(11, end="crossing", terminal=True), k=4)
    assert tr["masks"].tolist() == [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]     # the last window of each phase
    tr, _ = _rows(_episode(11, end="timeout", terminal=False), k=4)
    assert tr["masks"].tolist() == [1] * 11                              # a time-out bootstraps


def test_a_reached_episode_no_longer_than_the_cut_gives_no_rows_with_the_right_shapes():
    trunk = _FakeTrunk(repeat=4, emb_dim=7, state_dim=2)
    tr, _ = _rows(_episode(fx.ARRIVAL_CUT, mode=1, end="reached"), 4, trunk=trunk)
    assert trunk.calls == []
    assert tr["emb"].shape == (0, 7) and tr["next_emb"].shape == (0, 7)
    assert tr["states"].shape == (0, 2) and tr["next_states"].shape == (0, 2)
    assert tr["actions"].shape == (0, 2) and tr["rewards"].shape == (0,) and tr["masks"].shape == (0,)
    assert tr["emb"].dtype == np.float16 and tr["rewards"].dtype == np.float32
    # and it concatenates with real rows, which is what the build script does with the parts
    both = fx.concat([tr, _rows(_episode(9), 4, trunk=_FakeTrunk(repeat=4, emb_dim=7))[0]])
    assert both["emb"].shape == (9, 7)


def test_a_stream_pretrained_at_another_decision_rate_is_refused():
    """The stream's frame pair and GRU step are `repeat` ticks apart; rows at another k would feed the
    critic features the online trunk never produces."""
    trunk = _FakeTrunk(repeat=8)
    with pytest.raises(ValueError, match="8"):
        fx.episode_transitions_control(trunk, _episode(20), k=4, idle_penalty=0.0, junction_bonus=0.0)
    assert trunk.calls == []


def test_features_of_the_wrong_shape_are_refused_not_stored():
    class _Bad(_FakeTrunk):
        def embed_ticks(self, *a, **kw):
            emb, states = super().embed_ticks(*a, **kw)
            return emb[:, :-1], states                                   # one column short
    with pytest.raises(ValueError, match="emb"):
        fx.episode_transitions_control(_Bad(repeat=4), _episode(9), k=4, idle_penalty=0.0,
                                       junction_bonus=0.0)


@pytest.fixture(scope="module")
def c0_trunk(tmp_path_factory):
    """A tiny c0 (tests/test_frl_expert.py's), for the row-for-row comparison with the c0 arm."""
    from torch_rl import pixel_rows as pxr
    from torch_rl.frl import carla as fc
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig
    sd = pxr.mono_state_dim(pxr.BINOC_DIM)
    cfg = GatedRecurrentSACConfig(
        mono_encoder="resnet", embed_dim=3, grid=pxr.MONO_IMG, action_dim=2, mono_head="ln_tanh",
        goal_tokens=False, prev_action=False, core_state_cols=pxr.mono_core_state_cols(pxr.BINOC_DIM),
        state_dim=sd, mode_index=0, one_critic=True, goal_cols=tuple(range(sd)), explore_cols=(),
        hidden=16, visual_dim=16, mlp=(8, 8), n_critics=2, device="cpu", compile=False, seed=0)
    d = str(tmp_path_factory.mktemp("c0x"))
    GatedRecurrentSAC(cfg).save(d)
    return fc.C0Trunk(d, "cpu")


def test_everything_but_the_features_is_the_c0_arms_row(c0_trunk):
    """The two builders on the same episodes: the same keys, dtypes and shapes, and IDENTICAL actions,
    rewards and masks -- a fault, an arrival (cut) and a time-out, with the reward relabelling on."""
    from torch_rl import pixel_rows as pxr
    k = 4
    big = dict(px=pxr.MONO_IMG, binoc=pxr.BINOC_DIM)
    eps = [_episode(11, mode=0, seed=1, **big),
           _episode(fx.ARRIVAL_CUT + 9, mode=1, end="reached", seed=2, **big),
           _episode(10, mode=1, end="timeout", terminal=False, seed=3, **big)]
    for ep in eps:
        ep["reward"] = 0.1 * ep["reward"]
        ep["reward"][2] += 10.0
        ep["speed"][:3] = 1.0
        ctrl = _FakeTrunk(repeat=k, emb_dim=c0_trunk.emb_dim, state_dim=c0_trunk.state_dim)
        a = fx.episode_transitions(c0_trunk, ep, k=k, idle_penalty=0.5, junction_bonus=10.0)
        b = fx.episode_transitions_control(ctrl, ep, k=k, idle_penalty=0.5, junction_bonus=10.0)
        assert list(a.keys()) == list(b.keys())
        for key in a:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
        for key in ("actions", "rewards", "masks"):
            assert np.array_equal(a[key], b[key]), key
        assert len(a["rewards"]) > 0


# ---- scripts/build_expert_replay.py --control ------------------------------------------------------

def _build_script():
    spec = importlib.util.spec_from_file_location(
        "build_expert_replay", os.path.join(ROOT, "scripts", "build_expert_replay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_build_script_wants_exactly_one_trunk(tmp_path, monkeypatch):
    mod = _build_script()
    common = ["build_expert_replay.py", "--data", str(tmp_path), "--k", "8", "--out",
              str(tmp_path / "x.npz")]
    for extra in ([], ["--c0", str(tmp_path / "c0"), "--control", str(tmp_path / "ctrl")]):
        monkeypatch.setattr(sys, "argv", common + extra)
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert "exactly one" in str(e.value)                             # ours, not argparse's exit 2
    assert not (tmp_path / "x.npz").exists()


def _control_dir(tmp_path, k):
    d = tmp_path / "ctrl" / "step_1"
    d.mkdir(parents=True)
    torch.save({"args": {"n_frames": 2, "repeat": k, "px": 84, "latent": 50, "hidden": 6}, "net": {}},
               str(d / "params.pt"))
    return str(d)


def _data_root(tmp_path, name, town="TownXX", n=5):
    d = tmp_path / name / town
    d.mkdir(parents=True)
    for i in range(n):
        np.savez(str(d / f"ep_{i:06d}.npz"), **_episode(9 + i, seed=i))  # T = 9 .. 13, all faults
    return str(tmp_path / name)


def _run_build(mod, monkeypatch, control, data, out, k):
    monkeypatch.setattr(sys, "argv", [
        "build_expert_replay.py", "--control", control, "--data", data, "--towns", "TownXX", "--k", str(k),
        "--out", out, "--device", "cpu", "--idle-penalty", "0", "--junction-bonus", "0"])
    mod.main()
    z = np.load(out, allow_pickle=False)
    return z, json.loads(str(z["meta"]))


def test_the_build_script_builds_control_rows_and_records_the_stream(tmp_path, monkeypatch):
    from torch_rl.frl import carla as fc
    k = 4
    control = _control_dir(tmp_path, k)
    made = []

    class _Trunk(_FakeTrunk):
        def __init__(self, ckpt_dir, device):
            super().__init__(repeat=k, emb_dim=6, state_dim=2)
            made.append((ckpt_dir, str(device)))

    monkeypatch.setattr(fc, "ControlTrunk", _Trunk, raising=False)
    root = _data_root(tmp_path, "rootA")
    z, meta = _run_build(_build_script(), monkeypatch, control, root, str(tmp_path / "one.npz"), k)
    assert made == [(control, "cpu")]
    # 5 expert shards; the hold-out rule keeps the first 3 (and every 10th) out -> T = 12 and 13 are used,
    # and a fault episode gives every tick one decision
    assert len(z["rewards"]) == 12 + 13 and z["emb"].shape == (25, 6) and z["emb"].dtype == np.float16
    assert int(z["k"]) == k
    assert str(z["c0_md5"]) == fc.params_md5(control)                    # the run checks THIS trunk's md5
    assert meta["data"] == root                                          # one root: as before
    assert meta["towns"]["TownXX"] == {"shards": 5, "expert": 5, "used": 2, "probe_held_out": "rule",
                                       "transitions": 25}
    assert meta["trunk"] == "control" and meta["trunk_dir"] == os.path.abspath(control)
    assert (meta["n_frames"], meta["repeat"], meta["px"]) == (2, k, 84)
    assert meta["k"] == k and meta["arrival_cut"] == fx.ARRIVAL_CUT
    # ... and learner C's loader reads it
    rb = fx.ExpertReplay.load(str(tmp_path / "one.npz"), seed=0)
    assert len(rb) == 25 and rb.k == k and rb.meta["trunk"] == "control"
    b = rb.sample(8)
    assert b["observations"]["embeddings"].shape == (8, 6) and b["observations"]["states"].shape == (8, 2)


def test_several_data_roots_are_selected_root_by_root(tmp_path, monkeypatch):
    """Two recordings number their episodes independently (ep_000003 exists in both), so the hold-out rule
    and the expert filter are applied inside each root, never to the pooled names."""
    from torch_rl.frl import carla as fc
    k = 4
    control = _control_dir(tmp_path, k)

    class _Trunk(_FakeTrunk):
        def __init__(self, ckpt_dir, device):
            super().__init__(repeat=k, emb_dim=6, state_dim=2)

    monkeypatch.setattr(fc, "ControlTrunk", _Trunk, raising=False)
    a, b = _data_root(tmp_path, "rootA"), _data_root(tmp_path, "rootB", n=4)
    z, meta = _run_build(_build_script(), monkeypatch, control, f"{a},{b}", str(tmp_path / "two.npz"), k)
    # rootA keeps T = 12, 13 (as alone); rootB has 4 shards and keeps its 4th, T = 12
    assert len(z["rewards"]) == 25 + 12
    assert meta["data"] == f"{a},{b}"
    t = meta["towns"]["TownXX"]
    assert (t["shards"], t["expert"], t["used"], t["transitions"]) == (9, 9, 3, 37)
    assert t["roots"] == {a: {"shards": 5, "expert": 5, "used": 2, "transitions": 25},
                          b: {"shards": 4, "expert": 4, "used": 1, "transitions": 12}}

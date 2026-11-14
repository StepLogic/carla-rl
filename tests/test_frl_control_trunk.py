"""The control stream as learner C's frozen trunk (torch_rl/frl/carla.ControlTrunk; user 2026-11-14:
"Queue online train for best fixes wirh SAC with expert pinned rows").

c0's features carry lateral offset but not heading (R^2 0.046); the pretrained control stream
(torch_rl/control_cnn.ControlGRU: two frames one decision apart -> small CNN -> GRU that also takes
the speed) carries both (0.81 / 0.85 at action repeat 8). Here it replaces c0's GRU state in learner
C's embeddings; DEBiT's binocular features stay as they are. Every defect this glue can have is
SILENT -- a frame pair at the wrong spacing, the speed in the wrong unit, expert rows computed by a
different function than the online rows, a checkpoint scored at another decision rate -- so each is
pinned here.
"""
import json
import os
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from torch_rl import pixel_rows as pxr  # noqa: E402
from torch_rl.control_cnn import ControlGRU  # noqa: E402
from torch_rl.control_data import stack_frames  # noqa: E402
from torch_rl.frl import carla as fc  # noqa: E402
from torch_rl.frl import expert as fx  # noqa: E402
from tests.test_frl_carla import _FakeEnv, _row_of, c0_dir  # noqa: E402,F401  (the tiny c0 fixture)

BINOC = pxr.BINOC_DIM
K, PX, LATENT, HIDDEN = 8, 84, 8, 16          # the real stream: repeat 8, 84 px, latent 50, hidden 128
EMB = HIDDEN + BINOC


@pytest.fixture(scope="module", autouse=True)
def _chw_speed_rows():
    """THE ROW LAYOUT IS A MODULE CONSTANT of torch_rl.pixel_rows, read from the environment at ITS
    first import -- and in a full-suite run another test module imports it first with neither flag
    set (tests/test_bc_shard_list.py, alphabetically ahead of this file). The setdefault lines above
    only help when this file runs alone: the first full-suite run failed 13 of these tests on
    "MONO_CHW=1 and SPEED_OBS=1" while the file alone passed. So the layout the control stream reads
    is forced for this module's tests, before any other fixture is built, and put back afterwards."""
    mp = pytest.MonkeyPatch()
    mp.setattr(pxr, "MONO_CHW", True)
    mp.setattr(pxr, "SPEED_OBS", True)
    mp.setattr(pxr, "PRIV_OBS", False)
    yield
    mp.undo()


def _save_stream(d, seed, repeat=K, n_frames=2):
    """A tiny random stream saved the way scripts/pretrain_control.py saves one."""
    torch.manual_seed(seed)
    net = ControlGRU(latent_dim=LATENT, hidden_dim=HIDDEN, px=PX, n_frames=n_frames)
    os.makedirs(d, exist_ok=True)
    torch.save({"net": net.state_dict(),
                "args": {"n_frames": n_frames, "repeat": repeat, "px": PX, "latent": LATENT,
                         "hidden": HIDDEN},
                "row": {"update": 0}}, os.path.join(d, "params.pt"))
    return d


@pytest.fixture(scope="module")
def control_dir(tmp_path_factory):
    return _save_stream(str(tmp_path_factory.mktemp("stream") / "step_6000"), seed=0)


def _net(d):
    ck = torch.load(os.path.join(d, "params.pt"), map_location="cpu")
    a = ck["args"]
    net = ControlGRU(latent_dim=a["latent"], hidden_dim=a["hidden"], px=a["px"],
                     n_frames=a["n_frames"])
    net.load_state_dict(ck["net"])
    return net.train(False)


def _pretraining_input(stack_u8):
    """scripts/pretrain_control.batch: the uint8 stack as float in 0-255, area-resized; the net
    itself divides by 255."""
    x = torch.from_numpy(np.asarray(stack_u8)[None]).float()
    return F.interpolate(x, size=(PX, PX), mode="area")


def _pair(frames, i, j):
    return np.concatenate([frames[i].transpose(2, 0, 1), frames[j].transpose(2, 0, 1)])


def _row(seed, mode=1.0, kmh=7.0):
    g = np.random.default_rng(seed)
    frame = g.integers(0, 256, (pxr.MONO_IMG, pxr.MONO_IMG, 3), dtype=np.uint8)
    return pxr.mono_row(frame, mode, g.standard_normal(BINOC).astype(np.float32), speed_kmh=kmh)


# ---- the trunk ------------------------------------------------------------------------------------

def test_the_trunk_feeds_the_stream_what_pretraining_fed_it(control_dir):
    trunk = fc.ControlTrunk(control_dir, "cpu")
    net = _net(control_dir)
    g = np.random.default_rng(0)
    frames = g.integers(0, 256, (2, pxr.MONO_IMG, pxr.MONO_IMG, 3), dtype=np.uint8)
    binoc = g.standard_normal((2, BINOC)).astype(np.float32)
    kmh = [7.0, 12.5]
    rows = [pxr.mono_row(frames[i], 1.0, binoc[i], speed_kmh=kmh[i]) for i in range(2)]
    o0, h = trunk.step(rows[0], trunk.zero_hidden())
    o1, h = trunk.step(rows[1], h)
    with torch.no_grad():
        # the first decision of an episode repeats its own frame; the next pairs the PREVIOUS
        # decision's frame with its own, oldest first -- and the GRU carries on
        w0, g0 = net(_pretraining_input(_pair(frames, 0, 0)), torch.tensor([[kmh[0]]]), None)
        w1, _ = net(_pretraining_input(_pair(frames, 0, 1)), torch.tensor([[kmh[1]]]), g0)
    assert o0["embeddings"].shape == (1, EMB) and o0["states"].shape == (1, 2)
    assert torch.allclose(o0["embeddings"][0, :HIDDEN], w0[0], atol=1e-5)
    assert torch.allclose(o1["embeddings"][0, :HIDDEN], w1[0], atol=1e-5)
    assert torch.equal(o1["embeddings"][0, HIDDEN:], torch.from_numpy(binoc[1]))      # DEBiT, untouched
    assert torch.allclose(o1["states"][0], torch.tensor([1.0, 12.5 / pxr.SPEED_SCALE_KMH]))
    # THE GRU IS FED km/h, as in pretraining -- the row holds km/h / 20, and feeding that instead
    # would be a silent 20x error in the one scalar the stream reads
    with torch.no_grad():
        scaled, _ = net(_pretraining_input(_pair(frames, 0, 0)),
                        torch.tensor([[kmh[0] / pxr.SPEED_SCALE_KMH]]), None)
    assert not torch.allclose(o0["embeddings"][0, :HIDDEN], scaled[0], atol=1e-4)
    # a fresh hidden forgets the previous FRAME as well as the GRU state (a new episode must not
    # pair its first frame with the last frame of the one before)
    fresh, _ = trunk.step(rows[1], trunk.zero_hidden())
    with torch.no_grad():
        w_fresh, _ = net(_pretraining_input(_pair(frames, 1, 1)), torch.tensor([[kmh[1]]]), None)
    assert torch.allclose(fresh["embeddings"][0, :HIDDEN], w_fresh[0], atol=1e-5)
    assert not torch.allclose(fresh["embeddings"][0, :HIDDEN], o1["embeddings"][0, :HIDDEN], atol=1e-4)
    assert (trunk.emb_dim, trunk.state_dim, trunk.hidden_dim, trunk.binoc_dim) == (EMB, 2, HIDDEN, BINOC)
    assert (trunk.repeat, trunk.n_frames, trunk.px) == (K, 2, PX)
    assert trunk.c0_dir == control_dir                           # its identity for the md5 bookkeeping
    assert not any(p.requires_grad for p in trunk.net.parameters()) and not trunk.net.training


def test_recorded_frames_give_the_features_the_online_trunk_computes(control_dir):
    """THE ONE THAT MATTERS. The pinned expert rows are built from recorded uint8 frames
    (embed_ticks); the RL's own rows come from the env one decision at a time (step). The critic is
    trained on both in one batch, so they must be the same function of the same frames."""
    trunk = fc.ControlTrunk(control_dir, "cpu")
    g = np.random.default_rng(1)
    T = 2 * K + 3
    frames = g.integers(0, 256, (T + 1, pxr.MONO_IMG, pxr.MONO_IMG, 3), dtype=np.uint8)
    binoc = g.standard_normal((T + 1, BINOC)).astype(np.float16)     # as the shards store it
    kmh = g.uniform(0.0, 30.0, T + 1).astype(np.float32)             # the speed frame f was taken at
    ticks = [0, K, 2 * K]
    emb, states = trunk.embed_ticks(frames, binoc, 1, kmh, ticks)
    assert emb.shape == (3, EMB) and emb.dtype == np.float32
    assert states.shape == (3, 2) and states.dtype == np.float32
    h = trunk.zero_hidden()
    for j, f in enumerate(ticks):
        obs, h = trunk.step(pxr.mono_row(frames[f], 1.0, binoc[f], speed_kmh=float(kmh[f])), h)
        assert np.allclose(emb[j], obs["embeddings"][0].numpy(), atol=1e-5), f
        assert np.allclose(states[j], obs["states"][0].numpy(), atol=1e-7), f
    assert not np.allclose(emb[1, :HIDDEN], emb[2, :HIDDEN], atol=1e-3)   # the test can tell rows apart


def test_a_later_phase_starts_from_a_zero_state_on_pretrainings_clamped_pair(control_dir):
    """Phases 1..k-1 of the expert rows start mid-episode. Their first pair is what pretraining fed
    for an early chunk start -- control_data.stack_frames, the frame `repeat` ticks back clamped to
    frame 0 -- from a zero GRU state."""
    trunk = fc.ControlTrunk(control_dir, "cpu")
    net = _net(control_dir)
    g = np.random.default_rng(2)
    T = K + 6
    frames = g.integers(0, 256, (T + 1, pxr.MONO_IMG, pxr.MONO_IMG, 3), dtype=np.uint8)
    binoc = g.standard_normal((T + 1, BINOC)).astype(np.float16)
    kmh = g.uniform(0.0, 30.0, T + 1).astype(np.float32)
    ticks = [3, 3 + K]
    emb, states = trunk.embed_ticks(frames, binoc, 0, kmh, ticks)
    h = None
    with torch.no_grad():
        for j, t in enumerate(ticks):
            out, h = net(_pretraining_input(stack_frames(frames, t, 2, K)),
                         torch.tensor([[float(kmh[t])]]), h)
            assert np.allclose(emb[j, :HIDDEN], out[0].numpy(), atol=1e-5), t
            assert np.allclose(emb[j, HIDDEN:], binoc[t].astype(np.float32))
    assert np.allclose(states[:, 0], 0.0)                                # the mode bit (explore)
    assert np.allclose(states[:, 1], kmh[ticks] / pxr.SPEED_SCALE_KMH)
    e0, s0 = trunk.embed_ticks(frames, binoc, 0, kmh, [])
    assert e0.shape == (0, EMB) and s0.shape == (0, 2)


def test_the_trunk_refuses_rows_and_streams_it_cannot_read(control_dir, tmp_path, monkeypatch):
    trunk = fc.ControlTrunk(control_dir, "cpu")
    with pytest.raises(ValueError, match="row"):
        trunk.step(_row(0)[:-1], trunk.zero_hidden())            # not [frame | mode | speed | binoc]
    three = _save_stream(str(tmp_path / "three" / "step_1"), seed=0, n_frames=3)
    with pytest.raises(ValueError, match="n_frames"):
        fc.ControlTrunk(three, "cpu")                            # the hidden carries ONE previous frame
    monkeypatch.setattr(pxr, "SPEED_OBS", False)
    with pytest.raises(ValueError, match="SPEED_OBS"):
        fc.ControlTrunk(control_dir, "cpu")


# ---- settings -------------------------------------------------------------------------------------

def test_settings_read_the_control_stream_and_the_warm_up():
    s = fc.LoopSettings.from_env({"FRL_CONTROL": "/x/step_6000", "FRL_IQL_PRETRAIN": "30000"})
    assert (s.control, s.iql_pretrain) == ("/x/step_6000", 30000)
    d = fc.LoopSettings.from_env({})
    assert (d.control, d.iql_pretrain) == ("", 0)                # every earlier run: c0, no warm-up
    # the stream's PATH may differ between machines (its identity is the md5 in the checkpoint) and a
    # resumed learner is past its warm-up
    assert {"control", "iql_pretrain"} <= set(fc.RESUME_MAY_DIFFER)


def test_settings_that_cannot_mean_anything_are_refused_before_a_server_starts():
    ok = {"MONO_CHW": "1", "SPEED_OBS": "1"}
    assert fc.frl_refusals(["Town05"], False, {**ok, "FRL_CONTROL": "/x", "FRL_EXPERT": "/e.npz",
                                               "FRL_IQL_PRETRAIN": "30000"}) == []
    assert any("FRL_ACTOR" in m for m in fc.frl_refusals(
        ["Town05"], False, {**ok, "FRL_CONTROL": "/x", "FRL_ACTOR": "c0"}))
    assert any("FRL_IQL_PRETRAIN" in m for m in fc.frl_refusals(
        ["Town05"], False, {**ok, "FRL_IQL_PRETRAIN": "5"}))
    assert fc.frl_refusals(["Town05"], False, {**ok, "FRL_ACTOR": "c0"}) == []   # the c0-head arm stays
    assert fc.frl_refusals(["Town05"], False, {**ok, "FRL_IQL_PRETRAIN": "0"}) == []


# ---- checkpoints ----------------------------------------------------------------------------------

def _control_checkpoint(control_dir, d):
    trunk = fc.ControlTrunk(control_dir, "cpu")
    learner = fc.make_learner(trunk.emb_dim, trunk.state_dim, 2, "cpu", seed=0)
    fc.save_checkpoint(d, learner, step=7, c0_dir=control_dir,
                       settings=fc.LoopSettings(control=control_dir))
    return trunk, learner


def test_a_checkpoint_made_on_the_control_stream_drives_the_harness_with_it(
        control_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("ACTION_REPEAT", str(K))
    monkeypatch.delenv("FRL_CONTROL", raising=False)
    monkeypatch.setenv("FRL_C0", "/no/such/c0")                  # set for the c0 arms; NOT read here
    d = str(tmp_path / "step_7")
    trunk, learner = _control_checkpoint(control_dir, d)
    pol = fc.load_policy(d, "cpu")
    assert isinstance(pol.trunk, fc.ControlTrunk) and pol.trunk.c0_dir == os.path.abspath(control_dir)
    r0, r1 = _row(3), _row(4, kmh=11.0)
    a0, h = pol.act(r0, pol.reset_hidden(1), goal=None, deterministic=True, noise_scale=0.0)
    a1, h = pol.act(r1, h, goal=None, deterministic=True)
    o0, g = trunk.step(r0, trunk.zero_hidden())
    o1, _ = trunk.step(r1, g)
    with torch.no_grad():
        want = [torch.tanh(learner.actor.pi.dist_params(learner.actor.trunk(o))[0])[0].numpy()
                for o in (o0, o1)]
    assert a0.shape == (2,) and np.allclose(a0, want[0], atol=1e-6)
    assert np.allclose(a1, want[1], atol=1e-6)                   # the second decision used the pair


def test_a_control_checkpoint_refuses_another_stream_and_another_decision_rate(
        control_dir, tmp_path, monkeypatch):
    d = str(tmp_path / "step_7")
    _control_checkpoint(control_dir, d)
    monkeypatch.setenv("ACTION_REPEAT", str(K))
    other = _save_stream(str(tmp_path / "other" / "step_1"), seed=5)     # same shape, other weights
    monkeypatch.setenv("FRL_CONTROL", other)
    with pytest.raises(ValueError, match="md5"):
        fc.load_policy(d, "cpu")
    same = str(tmp_path / "copy")
    shutil.copytree(control_dir, same)                           # the same stream elsewhere is fine
    monkeypatch.setenv("FRL_CONTROL", same)
    assert fc.load_policy(d, "cpu").trunk.c0_dir == same
    # A POLICY TRAINED AT ONE DECISION PER 8 TICKS, SCORED AT ANOTHER RATE, would be handed frame
    # pairs at the wrong spacing and nothing would look wrong -- so it is refused
    monkeypatch.setenv("ACTION_REPEAT", "1")
    with pytest.raises(ValueError, match="ACTION_REPEAT"):
        fc.load_policy(d, "cpu")


def test_checkpoints_from_before_these_settings_still_load_and_resume(c0_dir, tmp_path):  # noqa: F811
    trunk = fc.C0Trunk(c0_dir, "cpu")
    learner = fc.make_learner(trunk.emb_dim, trunk.state_dim, 2, "cpu", seed=0)
    d = str(tmp_path / "step_3")
    fc.save_checkpoint(d, learner, step=3, c0_dir=c0_dir, settings=fc.LoopSettings())
    p = os.path.join(d, "config.json")
    meta = json.load(open(p))
    for key in ("control", "iql_pretrain"):
        del meta["settings"][key]                                # as every checkpoint made so far
    json.dump(meta, open(p, "w"))
    assert isinstance(fc.load_policy(d, "cpu").trunk, fc.C0Trunk)
    assert fc.restore_learner(learner, d, fc.LoopSettings(), c0_dir) == 3


# ---- the run --------------------------------------------------------------------------------------

def _expert_npz(path, trunk_md5, k=K, n=12):
    g = np.random.default_rng(0)
    tr = {"emb": g.standard_normal((n, EMB)).astype(np.float16),
          "next_emb": g.standard_normal((n, EMB)).astype(np.float16),
          "states": g.uniform(0, 1, (n, 2)).astype(np.float32),
          "next_states": g.uniform(0, 1, (n, 2)).astype(np.float32),
          "actions": g.uniform(-1, 1, (n, 2)).astype(np.float32),
          "rewards": g.standard_normal(n).astype(np.float32), "masks": np.ones(n, np.float32)}
    return fx.save_replay(str(path), tr, k=k, c0_md5=trunk_md5, meta={"trunk": "control"})


_UNSET = ("FRL_C0", "FRL_EXPERT", "FRL_EXPERT_MODE", "FRL_ACTOR", "FRL_IQL_PRETRAIN", "FRL_PARALLEL",
          "FRL_RESUME", "FRL_RESUME_NEW_ACTOR", "JUNCTION_RAMP_START")


def _control_env(monkeypatch, control_dir, **extra):
    base = {"FRL_CONTROL": control_dir, "FRL_DEVICE": "cpu", "FRL_MAX_STEPS": "4",
            "FRL_START_TRAINING": "100", "FRL_CKPT_EVERY": "4", "FRL_LOG_EVERY": "100",
            "FRL_REPLAY": "64", "FRL_PROBE_CACHE": "", "MONO_CHW": "1", "SPEED_OBS": "1",
            "ACTION_REPEAT": str(K)}
    base.update(extra)
    for key in _UNSET:
        monkeypatch.delenv(key, raising=False)
    for key, v in base.items():
        monkeypatch.setenv(key, v)


def _fake_warmup(monkeypatch, env, calls):
    import torch_rl.frl.iql as iql

    def fake(learner, sample, updates, **kw):
        calls.append({"learner": learner, "updates": updates, "rows": int(sample()["rewards"].shape[0]),
                      "env_steps": len(env.actions), "kw": sorted(kw)})
        return {"critic_loss": 0.5}

    monkeypatch.setattr(iql, "iql_warmup", fake, raising=False)


def test_run_carla_trains_on_the_control_stream_without_c0(control_dir, tmp_path, monkeypatch):
    md5 = fc.params_md5(control_dir)
    npz = _expert_npz(tmp_path / "x.npz", md5)
    # a steering-probe cache is c0's features: with the control stream the probe must stay off, so
    # a path that does not exist must not be opened
    _control_env(monkeypatch, control_dir, FRL_EXPERT=npz, FRL_EXPERT_MODE="pinned",
                 FRL_IQL_PRETRAIN="3", FRL_PROBE_CACHE="/no/such/cache.npz")
    env = _FakeEnv()
    obs, info = env.reset()
    calls, msgs = [], []
    _fake_warmup(monkeypatch, env, calls)
    out = str(tmp_path / "run")
    fc.run_carla(env, obs, info, None, out, row_of=_row_of, to_env=fc.box_map, log=msgs.append)
    # the warm-up: once, on expert batches of the loop's batch size, before any env step
    assert [(c["updates"], c["rows"], c["env_steps"]) for c in calls] == [(3, 256, 0)]
    assert "log" in calls[0]["kw"]
    cfg = json.load(open(os.path.join(out, "step_4", "config.json")))
    assert cfg["c0_md5"] == md5 and cfg["c0"] == os.path.abspath(control_dir)
    assert cfg["settings"]["control"] == control_dir and cfg["settings"]["iql_pretrain"] == 3
    assert cfg["emb_dim"] == EMB and cfg["state_dim"] == 2
    assert any("CONTROL STREAM" in m for m in msgs)
    assert any("IQL" in m and "3 updates" in m for m in msgs)
    assert isinstance(fc.load_policy(os.path.join(out, "step_4"), "cpu").trunk, fc.ControlTrunk)


def test_run_carla_refuses_what_a_control_run_cannot_mean(control_dir, tmp_path, monkeypatch):
    md5 = fc.params_md5(control_dir)
    cases = [
        ("FRL_CONTROL.*ACTION_REPEAT", dict(ACTION_REPEAT="2")),   # the stream decides every 8 ticks
        ("FRL_ACTOR", dict(FRL_ACTOR="c0")),                       # c0's head reads c0's trunk
        ("FRL_IQL_PRETRAIN", dict(FRL_IQL_PRETRAIN="5")),          # no expert rows to learn from
        ("md5", dict(FRL_EXPERT=_expert_npz(tmp_path / "c0built.npz", "0" * 32))),   # another trunk's rows
        ("built at action repeat", dict(FRL_EXPERT=_expert_npz(tmp_path / "k2.npz", md5, k=2))),
    ]
    for match, extra in cases:
        _control_env(monkeypatch, control_dir, **extra)
        env = _FakeEnv()
        obs, info = env.reset()
        with pytest.raises(SystemExit, match=match):
            fc.run_carla(env, obs, info, None, str(tmp_path / "run"), row_of=_row_of,
                         to_env=fc.box_map, log=lambda m: None)
        assert env.actions == [], match                            # refused before any env step


def _expert(control_dir, tmp_path):
    return fx.ExpertReplay.load(_expert_npz(tmp_path / "e.npz", fc.params_md5(control_dir)), seed=0)


def _loop_settings(**kw):
    base = dict(max_steps=4, start_training=100, utd=1, batch=4, replay=64, ckpt_every=4,
                log_every=100, device="cpu", seed=0, expert_mode="pinned")
    base.update(kw)
    return fc.LoopSettings(**base)


@pytest.mark.parametrize("parallel", [0, 1])
def test_the_warm_up_runs_once_on_expert_rows_before_the_first_env_step(
        control_dir, tmp_path, monkeypatch, parallel):
    loop = fc.train_parallel if parallel else fc.train
    expert = _expert(control_dir, tmp_path)
    env = _FakeEnv()
    obs, info = env.reset()
    calls, msgs = [], []
    _fake_warmup(monkeypatch, env, calls)
    learner, rb, _ = loop(env, obs, info, str(tmp_path / "run"),
                          trunk=fc.ControlTrunk(control_dir, "cpu"), row_of=_row_of,
                          settings=_loop_settings(iql_pretrain=7, parallel=parallel),
                          c0_dir=control_dir, log=msgs.append, expert=expert)
    assert len(calls) == 1 and calls[0]["learner"] is learner
    assert (calls[0]["updates"], calls[0]["rows"], calls[0]["env_steps"]) == (7, 4, 0)
    assert rb.n_pinned == len(expert) and len(env.actions) == 4
    assert any("[IQL]" in m for m in msgs)
    # off by default: no call
    env2 = _FakeEnv()
    obs, info = env2.reset()
    calls2 = []
    _fake_warmup(monkeypatch, env2, calls2)
    loop(env2, obs, info, str(tmp_path / "run_off"), trunk=fc.ControlTrunk(control_dir, "cpu"),
         row_of=_row_of, settings=_loop_settings(parallel=parallel), c0_dir=control_dir,
         log=lambda m: None, expert=expert)
    assert calls2 == []


def test_a_resumed_run_does_not_repeat_the_warm_up(control_dir, tmp_path, monkeypatch):
    expert = _expert(control_dir, tmp_path)
    calls, msgs = [], []

    def run(out, **kw):
        env = _FakeEnv()
        obs, info = env.reset()
        _fake_warmup(monkeypatch, env, calls)
        return fc.train_parallel(env, obs, info, str(out), trunk=fc.ControlTrunk(control_dir, "cpu"),
                                 row_of=_row_of, settings=_loop_settings(iql_pretrain=7, parallel=1, **kw),
                                 c0_dir=control_dir, log=msgs.append, expert=expert)

    run(tmp_path / "a")
    assert len(calls) == 1
    run(tmp_path / "b", resume=str(tmp_path / "a" / "step_4"), max_steps=8, ckpt_every=8)
    assert len(calls) == 1                                       # the restored learner is past it
    assert any("[IQL]" in m and "skipped" in m for m in msgs)
    assert (tmp_path / "b" / "step_8" / "config.json").exists()


@pytest.mark.parametrize("parallel", [0, 1])
def test_a_warm_up_with_no_expert_rows_is_an_error_not_a_silent_skip(
        control_dir, tmp_path, monkeypatch, parallel):
    loop = fc.train_parallel if parallel else fc.train
    env = _FakeEnv()
    obs, info = env.reset()
    _fake_warmup(monkeypatch, env, [])
    with pytest.raises(ValueError, match="iql_pretrain"):
        loop(env, obs, info, str(tmp_path / "run"), trunk=fc.ControlTrunk(control_dir, "cpu"),
             row_of=_row_of, settings=_loop_settings(iql_pretrain=7, parallel=parallel, expert_mode="update"),
             c0_dir=control_dir, log=lambda m: None)
    assert env.actions == []


def test_the_warm_up_is_refused_with_c0s_head_before_a_server_starts():
    """AWR needs the log-probability of a dataset action, which c0's head (C0HeadActor) does not
    give. torch_rl.frl.iql refuses it, but only once the learner exists -- after CARLA is up -- so
    the settings check says it first."""
    ok = {"MONO_CHW": "1", "SPEED_OBS": "1"}
    bad = fc.frl_refusals(["Town05"], False, {**ok, "FRL_ACTOR": "c0", "FRL_EXPERT": "/e.npz",
                                              "FRL_IQL_PRETRAIN": "100"})
    assert any("FRL_IQL_PRETRAIN" in m and "FRL_ACTOR=c0" in m for m in bad)
    assert fc.warm_start_refusals("", "c0", 100, "/e.npz")
    assert fc.warm_start_refusals("", "frl", 100, "/e.npz") == []

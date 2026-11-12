"""Expert prior data for learner C (torch_rl/frl/expert.py; user 2026-11-10: "can we use c0 expert data
locally with run like fastrlap but rl collects data like now").

FastRLAP's online loop makes a second update per step on a fixed expert replay, temperature untouched. The
expert replay is built from c0's own demonstrations (the M1 shards) exactly as the online loop sees the world:
c0's trunk stepped once per DECISION (every k-th frame at action repeat k), k ticks' rewards summed, rewards
relabelled to the run's reward (idle charge added, explore junction bonus removed), the no_arrival cut, expert
episodes only, the steering probe's held-out episodes excluded."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")

from torch_rl import pixel_rows as pxr  # noqa: E402
from torch_rl.frl import carla as fc  # noqa: E402
from torch_rl.frl import expert as fx  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig  # noqa: E402

BINOC = pxr.BINOC_DIM


@pytest.fixture(scope="module")
def trunk(tmp_path_factory):
    sd = pxr.mono_state_dim(BINOC)
    cfg = GatedRecurrentSACConfig(
        mono_encoder="resnet", embed_dim=3, grid=pxr.MONO_IMG, action_dim=2, mono_head="ln_tanh",
        goal_tokens=False, prev_action=False, core_state_cols=pxr.mono_core_state_cols(BINOC),
        state_dim=sd, mode_index=0, one_critic=True, goal_cols=tuple(range(sd)), explore_cols=(),
        hidden=16, visual_dim=16, mlp=(8, 8), n_critics=2, device="cpu", compile=False, seed=0)
    d = str(tmp_path_factory.mktemp("c0x"))
    GatedRecurrentSAC(cfg).save(d)
    return fc.C0Trunk(d, "cpu")


def _episode(T, mode=0, end="crossing", terminal=True, driver="expert", seed=0):
    g = np.random.default_rng(seed)
    return {"frames": g.integers(0, 256, (T + 1, pxr.MONO_IMG, pxr.MONO_IMG, 3), dtype=np.uint8),
            "binoc": g.standard_normal((T + 1, BINOC)).astype(np.float16),
            "act_exec": g.uniform(-1, 1, (T, 2)).astype(np.float32),
            "reward": np.arange(T, dtype=np.float32) + 1.0,            # 1, 2, 3, ...
            "speed": np.full(T, 10.0, np.float32), "mode": np.array(mode),
            "terminal": np.array(terminal), "truncated": np.array(not terminal),
            "end": np.array(end), "driver": np.array(driver)}


def test_rewards_are_relabelled_to_the_runs_reward():
    r = np.array([0.3, 10.2, 0.1, 100.0, -1.0], np.float32)
    speed = np.array([1.0, 12.0, 1.9, 3.0, 2.0], np.float32)
    explore = fx.relabel_rewards(r, speed, mode=0, idle_penalty=0.5, idle_kmh=2.0, junction_bonus=10.0)
    assert np.allclose(explore, [0.3 - 0.5, 0.2, 0.1 - 0.5, 90.0, -1.0])   # bonus off, idle on
    goal = fx.relabel_rewards(r, speed, mode=1, idle_penalty=0.5, idle_kmh=2.0, junction_bonus=10.0)
    assert np.allclose(goal, [0.3 - 0.5, 10.2, 0.1 - 0.5, 100.0, -1.0])    # no junction bonus in goal mode


def test_decision_windows_cover_the_episode_once_per_phase():
    assert fx.decision_windows(7, k=2, phase=0) == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert fx.decision_windows(7, k=2, phase=1) == [(1, 3), (3, 5), (5, 7)]
    assert fx.decision_windows(6, k=1, phase=0) == [(i, i + 1) for i in range(6)]


def test_transitions_follow_the_online_loop_at_action_repeat_two(trunk):
    ep = _episode(7)
    tr = fx.episode_transitions(trunk, ep, k=2, idle_penalty=0.0, junction_bonus=0.0)
    assert len(tr["rewards"]) == 4 + 3                                   # two phases
    assert np.allclose(tr["rewards"][:4], [1 + 2, 3 + 4, 5 + 6, 7])       # k ticks summed, short last
    assert tr["masks"].tolist() == [1, 1, 1, 0, 1, 1, 0]                  # terminal only at the end
    assert np.allclose(tr["actions"][0], ep["act_exec"][0])
    assert np.allclose(tr["actions"][1], ep["act_exec"][2])
    # next observation of a decision == observation of the next decision (same GRU path)
    assert np.array_equal(tr["next_emb"][0], tr["emb"][1])
    # and the GRU is stepped once per decision, from zero at the phase's first frame -- as online
    h = trunk.zero_hidden()
    want = []
    for f in (0, 2, 4, 6):
        row = pxr.mono_row(ep["frames"][f], float(ep["mode"]), ep["binoc"][f].astype(np.float32),
                           speed_kmh=0.0 if f == 0 else float(ep["speed"][f - 1]))
        o, h = trunk.step(row, h)
        want.append(o["embeddings"][0].numpy())
    assert np.allclose(tr["emb"][:4].astype(np.float32), np.stack(want), atol=2e-3)


def test_a_reached_episode_loses_its_arrival_and_ends_truncated(trunk):
    ep = _episode(40, mode=1, end="reached", terminal=True)
    tr = fx.episode_transitions(trunk, ep, k=1, idle_penalty=0.0, junction_bonus=0.0)
    assert len(tr["rewards"]) == 10                                       # 40 - ARRIVAL_CUT 30
    assert tr["masks"].tolist() == [1] * 10                               # cut = truncated, bootstraps
    assert tr["rewards"].max() == 10.0


def test_only_expert_episodes_outside_the_probe_hold_out_are_used():
    names = [f"ep_{i:06d}.npz" for i in range(25)]
    drivers = ["expert"] * 24 + ["policy"]
    keep = fx.select_episodes(names, drivers)
    held = {names[0], names[1], names[2], names[10], names[20]}            # first 3 + every 10th
    assert set(keep) == set(names[:24]) - held


def test_expert_updates_leave_the_temperature_alone(trunk, tmp_path):
    ep = _episode(9)
    tr = fx.episode_transitions(trunk, ep, k=1, idle_penalty=0.0, junction_bonus=0.0)
    p = str(tmp_path / "x.npz")
    fx.save_replay(p, tr, k=1, c0_md5="abc", meta={"towns": ["TownXX"]})
    rb = fx.ExpertReplay.load(p, seed=0)
    assert len(rb) == 9 and rb.k == 1 and rb.c0_md5 == "abc"
    learner = fc.make_learner(trunk.emb_dim, trunk.state_dim, 2, "cpu", seed=0)
    t0 = float(learner.log_temp)
    a0 = [q.detach().clone() for q in learner.actor.parameters()]
    info = fx.expert_update(learner, rb, batch=4, utd=2)
    assert float(learner.log_temp) == t0                                  # FastRLAP: update_temperature=False
    assert any(not torch.equal(a, q) for a, q in zip(a0, learner.actor.parameters()))
    assert np.isfinite(info["critic_loss"])


def _replay(trunk, tmp_path, k=1, md5=None):
    tr = fx.episode_transitions(trunk, _episode(9), k=k, idle_penalty=0.0, junction_bonus=0.0)
    p = str(tmp_path / f"x_k{k}_{md5 or 'ok'}.npz")
    fx.save_replay(p, tr, k=k, c0_md5=md5 if md5 is not None else fc.params_md5(trunk.c0_dir),
                   meta={"towns": ["TownXX"]})
    return p


def test_the_loop_adds_one_expert_update_per_step_after_start_training(trunk, tmp_path, monkeypatch):
    from tests.test_frl_carla import _FakeEnv, _row_of
    rb = fx.ExpertReplay.load(_replay(trunk, tmp_path), seed=0)
    calls = []
    real = fx.expert_update
    monkeypatch.setattr(fx, "expert_update", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    env = _FakeEnv()
    obs, info = env.reset()
    s = fc.LoopSettings(max_steps=8, start_training=5, utd=2, batch=4, replay=100, ckpt_every=8,
                        log_every=8, device="cpu", seed=0)
    msgs = []
    fc.train(env, obs, info, str(tmp_path / "run"), trunk=trunk, row_of=_row_of, settings=s,
             c0_dir=trunk.c0_dir, log=msgs.append, expert=rb)
    assert len(calls) == 4                                                # steps 5, 6, 7, 8
    assert any("x_critic_loss=" in m for m in msgs)


def test_run_carla_refuses_an_expert_replay_built_for_another_k_or_trunk(trunk, tmp_path, monkeypatch):
    from tests.test_frl_carla import _FakeEnv, _row_of
    for k, v in {"FRL_C0": trunk.c0_dir, "FRL_DEVICE": "cpu", "FRL_MAX_STEPS": "2",
                 "FRL_START_TRAINING": "100", "FRL_PROBE_CACHE": "", "MONO_CHW": "1",
                 "SPEED_OBS": "1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("JUNCTION_RAMP_START", raising=False)
    for bad in (_replay(trunk, tmp_path, k=2),                            # built at k=2, run at k=1
                _replay(trunk, tmp_path, k=1, md5="0" * 32)):             # built on another c0
        monkeypatch.setenv("FRL_EXPERT", bad)
        monkeypatch.setenv("ACTION_REPEAT", "1")
        env = _FakeEnv()
        obs, info = env.reset()
        with pytest.raises(SystemExit):
            fc.run_carla(env, obs, info, None, str(tmp_path / "run"), row_of=_row_of,
                         to_env=fc.box_map, log=lambda m: None)

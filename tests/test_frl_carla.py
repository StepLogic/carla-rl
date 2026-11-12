"""Learner C in CARLA (torch_rl/frl/carla.py; user 2026-11-10: "Compressed to 50-d, compression
trained by the critic (FastRLAP)", "proceed build it and run locally", "move straight to learner C").

c0's frozen trunk runs online and feeds learner C [GRU | binoc] as embeddings and [mode, speed] as
states; FastRLAP's loop trains learner C. These tests pin the glue: the online features equal the
offline cache's (so the steering probe reads the critic on the features it was trained on), the
replay, the loop's FastRLAP rules, and the checkpoint the eval harness drives."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts", "bench")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")

from torch_rl import pixel_rows as pxr  # noqa: E402
from torch_rl.frl import carla as fc  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig  # noqa: E402

BINOC = pxr.BINOC_DIM


@pytest.fixture(scope="module")
def c0_dir(tmp_path_factory):
    """A small clone with c0's row layout (mono 128 px resnet, [mode | speed | binoc] state)."""
    sd = pxr.mono_state_dim(BINOC)
    cfg = GatedRecurrentSACConfig(
        mono_encoder="resnet", embed_dim=3, grid=pxr.MONO_IMG, action_dim=2, mono_head="ln_tanh",
        goal_tokens=False, prev_action=False, core_state_cols=pxr.mono_core_state_cols(BINOC),
        state_dim=sd, mode_index=0, one_critic=True, goal_cols=tuple(range(sd)), explore_cols=(),
        hidden=16, visual_dim=16, mlp=(8, 8), n_critics=2, device="cpu", compile=False, seed=0)
    agent = GatedRecurrentSAC(cfg)
    d = str(tmp_path_factory.mktemp("c0"))
    agent.save(d)
    return d


def _row(seed, mode=1.0, kmh=7.0):
    g = np.random.default_rng(seed)
    frame = g.integers(0, 256, (pxr.MONO_IMG, pxr.MONO_IMG, 3), dtype=np.uint8)
    return pxr.mono_row(frame, mode, g.standard_normal(BINOC).astype(np.float32), speed_kmh=kmh)


def test_the_online_trunk_matches_the_offline_cache_and_splits_the_layout(c0_dir):
    trunk = fc.C0Trunk(c0_dir, "cpu")
    rows = np.stack([_row(i) for i in range(4)])
    h = trunk.zero_hidden()
    feats = []
    for r in rows:
        obs, h = trunk.step(r, h)
        feats.append(obs)
    # the probe cache (scripts/probe_c0_features.episode_features): the GRU over the episode
    a = trunk.agent
    x = torch.from_numpy(rows)
    e = a._embed(a.compress, a.act_embed, x)
    seq, _ = a.core(e[None], torch.zeros(1, 1, a.cfg.hidden))
    for t, obs in enumerate(feats):
        assert torch.allclose(obs["embeddings"][0, :a.cfg.hidden], seq[0, t], atol=1e-5)
        assert torch.equal(obs["embeddings"][0, a.cfg.hidden:], x[t, -BINOC:])     # binoc
        assert torch.equal(obs["states"][0], x[t, pxr.MONO_PIX:pxr.MONO_PIX + 2])  # mode, speed
    assert trunk.emb_dim == a.cfg.hidden + BINOC and trunk.state_dim == 2
    assert not any(p.requires_grad for p in a.compress.parameters())             # frozen


def test_the_replay_keeps_each_transition_together():
    rb = fc.FeatureReplay(10, emb_dim=3, state_dim=2, action_dim=2, device="cpu", seed=0)
    for i in range(4):
        o = {"embeddings": torch.full((1, 3), float(i)), "states": torch.full((1, 2), float(i))}
        n = {"embeddings": torch.full((1, 3), i + 0.5), "states": torch.full((1, 2), i + 0.5)}
        rb.add(o, np.array([i, -i], np.float32), reward=10.0 * i, mask=float(i != 2), next_obs=n)
    b = rb.sample(64)
    i = b["observations"]["embeddings"][:, 0].round().long()
    assert len(rb) == 4 and set(i.tolist()) <= {0, 1, 2, 3}
    assert torch.allclose(b["rewards"], 10.0 * i.float())
    assert torch.allclose(b["masks"], (i != 2).float())
    assert torch.allclose(b["next_observations"]["states"][:, 0], i.float() + 0.5)
    assert torch.allclose(b["actions"][:, 1], -i.float())


def test_the_settings_are_the_benchmarks_redq_config():
    from frl_train import CONFIGS
    ref = {k: v for k, v in CONFIGS["redq"].items() if k != "learner"}
    assert fc.REDQ == ref
    assert fc.LATENT_DIM == 50


class _FakeEnv:
    """Fixed-length episodes; the 3rd step of every episode terminates, every 5th truncates."""

    def __init__(self):
        from gymnasium.spaces import Box
        self.action_space = Box(np.array([-1.0, 0.0], np.float32), np.array([1.0, 1.0], np.float32))
        self.t, self.n_reset, self.actions = 0, 0, []

    def _o(self):
        return ({"mode": np.array([1.0], np.float32),
                 "binoc": np.full(BINOC, 0.01 * self.t, np.float32)},
                {"debit_cur": np.zeros((pxr.MONO_IMG, pxr.MONO_IMG, 3), np.uint8),
                 "speed_kmh": 3.0, "binoc_encoded": True})

    def reset(self, seed=None):
        self.t, self.n_reset = 0, self.n_reset + 1
        return self._o()

    def step(self, a):
        self.actions.append(np.asarray(a, np.float32))
        self.t += 1
        o, i = self._o()
        term = self.n_reset % 2 == 1 and self.t == 3
        trunc = self.n_reset % 2 == 0 and self.t == 5
        i["goal_end"] = "reached" if term else ("clock" if trunc else None)
        return o, 1.0, term, trunc, i


def _row_of(o, i):
    return pxr.mono_row(i["debit_cur"], float(o["mode"][0]), o["binoc"], speed_kmh=i["speed_kmh"])


def test_the_loop_follows_fastrlap_and_checkpoints(c0_dir, tmp_path, monkeypatch):
    from torch_rl.junction_ramp import JunctionRamp
    monkeypatch.setenv("JUNCTION_GOAL_FRAC", "0.85")           # the ramp writes it; restored
    ramp = JunctionRamp(start=0.5, target=0.85, step=0.05, up=0.3, down=0.1, window=1)
    env = _FakeEnv()
    obs, info = env.reset()
    s = fc.LoopSettings(max_steps=12, start_training=5, utd=2, batch=4, replay=100,
                        ckpt_every=6, log_every=4, device="cpu", seed=0)
    out = str(tmp_path / "run")
    learner, rb, _ = fc.train(env, obs, info, out, trunk=fc.C0Trunk(c0_dir, "cpu"),
                              row_of=_row_of, settings=s, c0_dir=c0_dir, log=lambda m: None,
                              ramp=ramp)
    # the ramp saw the 2 goal episodes the actor drove (ends at steps 8 and 11, not the random
    # warm-up's at step 3): a time-out, then an arrival -> up one step
    assert ramp.value == pytest.approx(0.55)
    assert len(rb) == 12
    # masks: 0 only on the terminations (steps 3 and 3 + 5 + 3 = 11), truncations keep 1
    assert sorted(np.flatnonzero(rb.mask[:12].cpu().numpy() == 0).tolist()) == [2, 10]
    assert env.n_reset == 4                                     # 3 episode ends + the first
    # the target entropy anneals by -25e-7 per step from -action_dim / 2
    assert learner.target_entropy == pytest.approx(-1.0 - 12 * 25e-7)
    for step in (6, 12):
        d = os.path.join(out, f"step_{step}")
        assert fc.is_frl_checkpoint(d)
        assert json.load(open(os.path.join(d, "config.json")))["step"] == step


class _FlakyEnv(_FakeEnv):
    """Step 4 raises, as CARLA does on a sensor timeout."""

    def step(self, a):
        if len(self.actions) == 3:
            self.actions.append(None)
            raise RuntimeError("A sensor took too long to send their data")
        return super().step(a)


def test_a_failed_simulator_step_abandons_the_episode_and_stores_nothing(c0_dir, tmp_path):
    env = _FlakyEnv()
    obs, info = env.reset()
    s = fc.LoopSettings(max_steps=6, start_training=100, utd=1, batch=2, replay=100,
                        ckpt_every=6, log_every=100, device="cpu", seed=0)
    msgs = []
    _, rb, _ = fc.train(env, obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                        row_of=_row_of, settings=s, c0_dir=c0_dir, log=msgs.append)
    assert len(rb) == 5                                         # 6 steps, the failed one dropped
    assert env.n_reset == 3          # first + the abandoned episode's reset + the step-3 end
    assert any("episode abandoned" in m for m in msgs)
    assert fc.is_frl_checkpoint(str(tmp_path / "run" / "step_6"))


def test_a_checkpoint_drives_the_eval_harness_api(c0_dir, tmp_path):
    trunk = fc.C0Trunk(c0_dir, "cpu")
    learner = fc.make_learner(trunk.emb_dim, trunk.state_dim, 2, "cpu", seed=0)
    d = str(tmp_path / "step_7")
    fc.save_checkpoint(d, learner, step=7, c0_dir=c0_dir, settings=fc.LoopSettings())
    pol = fc.load_policy(d, "cpu")
    h = pol.reset_hidden(1)
    assert torch.count_nonzero(h) == 0
    a1, h1 = pol.act(_row(3), h, goal=None, deterministic=True, noise_scale=0.0)
    obs, _ = trunk.step(_row(3), trunk.zero_hidden())
    with torch.no_grad():
        want = torch.tanh(learner.actor.pi.dist_params(learner.actor.trunk(obs))[0])[0].numpy()
    assert a1.shape == (2,) and np.allclose(a1, want, atol=1e-6)
    assert torch.count_nonzero(h1) > 0                          # the GRU state moves on
    assert not fc.is_frl_checkpoint(c0_dir)                     # a clone is not learner C


# ---- review fixes (2026-11-10, /code-review high before the first CARLA run) -----------------

class _DeadAfterFault(_FakeEnv):
    """Step 4 raises; the reset after it raises too (the actors are gone): only a rebuild helps."""

    def step(self, a):
        if len(self.actions) == 3:
            self.actions.append(None)
            self.dead = True
            raise RuntimeError("A sensor took too long to send their data")
        return super().step(a)

    def reset(self, seed=None):
        if getattr(self, "dead", False):
            raise RuntimeError("actor registry destroyed")
        return super().reset(seed)


def test_a_failed_reset_rebuilds_the_env_and_the_run_goes_on(c0_dir, tmp_path):
    env = _DeadAfterFault()
    obs, info = env.reset()
    built = []

    def rebuild():
        built.append(_FakeEnv())
        return built[-1]

    s = fc.LoopSettings(max_steps=6, start_training=100, utd=1, batch=2, replay=100,
                        ckpt_every=6, log_every=100, device="cpu", seed=0)
    msgs = []
    _, rb, live = fc.train(env, obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                           row_of=_row_of, settings=s, c0_dir=c0_dir, log=msgs.append,
                           rebuild_env=rebuild)
    assert len(built) == 1 and live == [built[0]]               # the caller closes the live env
    assert len(rb) == 5 and any("rebuilding" in m for m in msgs)


def test_without_a_rebuild_a_failed_reset_still_stops_the_run(c0_dir, tmp_path):
    env = _DeadAfterFault()
    obs, info = env.reset()
    s = fc.LoopSettings(max_steps=6, start_training=100, utd=1, batch=2, replay=100,
                        ckpt_every=6, log_every=100, device="cpu", seed=0)
    with pytest.raises(RuntimeError, match="registry"):
        fc.train(env, obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                 row_of=_row_of, settings=s, c0_dir=c0_dir, log=lambda m: None)


def test_settings_that_would_be_silently_ignored_or_wrong_are_refused():
    ok = {"MONO_CHW": "1", "SPEED_OBS": "1"}
    assert fc.frl_refusals(["Town05"], False, ok) == []
    assert fc.frl_refusals(["Town05", "Town01"], False, ok) == []   # towns in turn (test_frl_towns)
    assert any("PARALLEL" in m for m in fc.frl_refusals(["Town05"], True, ok))
    for k, v in (("INIT_FROM", "/x"), ("IL_PRELOAD_ROOT", "/y"), ("HER", "1"),
                 ("DET_EPISODE_FRAC", "0.15"), ("MONO_CHW", "0"), ("SPEED_OBS", "0")):
        assert any(k in m for m in fc.frl_refusals(["Town05"], False, {**ok, k: v})), k
    assert fc.frl_refusals(["Town05"], False, {**ok, "HER": "0", "DET_EPISODE_FRAC": "0"}) == []


def test_device_and_seed_follow_the_trainer_unless_frl_overrides():
    s = fc.LoopSettings.from_env({"TORCH_DEVICE": "cuda:1", "SEED": "7"})
    assert (s.device, s.seed) == ("cuda:1", 7)
    s = fc.LoopSettings.from_env({"TORCH_DEVICE": "cuda:1", "SEED": "7", "FRL_DEVICE": "cpu",
                                  "FRL_SEED": "3", "FRL_MAX_STEPS": "1500"})
    assert (s.device, s.seed, s.max_steps) == ("cpu", 3, 1500)


def test_an_eval_refuses_a_different_c0_trunk(c0_dir, tmp_path, monkeypatch):
    import shutil
    trunk = fc.C0Trunk(c0_dir, "cpu")
    learner = fc.make_learner(trunk.emb_dim, trunk.state_dim, 2, "cpu", seed=0)
    d = str(tmp_path / "step_1")
    fc.save_checkpoint(d, learner, step=1, c0_dir=c0_dir, settings=fc.LoopSettings())
    cfg = json.load(open(os.path.join(c0_dir, "config.json")))
    other = GatedRecurrentSAC(GatedRecurrentSACConfig(**{**cfg, "seed": 5}))
    odir = str(tmp_path / "other_clone")
    other.save(odir)                                            # same dims, other weights
    monkeypatch.setenv("FRL_C0", odir)
    with pytest.raises(ValueError, match="c0"):
        fc.load_policy(d, "cpu")
    same = str(tmp_path / "c0_copy")
    shutil.copytree(c0_dir, same)                               # the same trunk elsewhere is fine
    monkeypatch.setenv("FRL_C0", same)
    assert fc.load_policy(d, "cpu").trunk.c0_dir == same


def test_run_carla_starts_the_ramp_before_the_first_episode_and_closes_the_env(
        c0_dir, tmp_path, monkeypatch):
    for k, v in {"FRL_C0": c0_dir, "FRL_DEVICE": "cpu", "FRL_MAX_STEPS": "4",
                 "FRL_START_TRAINING": "100", "FRL_CKPT_EVERY": "4", "FRL_LOG_EVERY": "100",
                 "FRL_PROBE_CACHE": "", "JUNCTION_RAMP_START": "0.5",
                 "JUNCTION_GOAL_FRAC": "0.85", "MONO_CHW": "1", "SPEED_OBS": "1"}.items():
        monkeypatch.setenv(k, v)
    seen = []

    class _Env(_FakeEnv):
        def reset(self, seed=None):
            seen.append(os.environ["JUNCTION_GOAL_FRAC"])
            return super().reset(seed)

        def close(self):
            self.closed = True

    env = _Env()
    obs, info = env.reset()                                     # main's first reset (0.85)
    fc.run_carla(env, obs, info, None, str(tmp_path / "run"), row_of=_row_of,
                 to_env=fc.box_map, rebuild_env=None, log=lambda m: None)
    assert seen[:2] == ["0.85", "0.50"]                         # re-drawn at the ramp's start
    assert getattr(env, "closed", False)

"""Data-flow audit of learner C's collection loops (user 2026-11-12: "ensure the data colleciton is not buggy -
review data follow from environment to replay and data flow from model to environments").

Several towns, each env TAGGING its observations (binoc and camera frame) and rewards (100 * town + t) with
its own id. The actor is replaced by a fixed function of its input embedding, and every env's own stream
is replayed through an INDEPENDENT copy of c0's trunk (fresh GRU state at each of that env's resets).
Both loops must then satisfy:
  model -> env:   each env received exactly box_map(f(its own current embedding)), step by step;
  env -> replay:  each stored row is one of that env's transitions, in its order: (embedding, action in
                  [-1, 1], reward, mask 0 only on termination, next embedding) -- so no row mixes envs,
                  the GRU state is per env, the terminal observation is the next one, and resets restart
                  the GRU."""
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
from tests.test_frl_carla import _FakeEnv, _row_of, c0_dir  # noqa: E402,F401  (the tiny c0 fixture)


class _TaggedEnv(_FakeEnv):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.stream = []            # ("reset", row) / ("step", row, reward, term, trunc)

    def _o(self):
        o, i = super()._o()
        o["binoc"] = np.full(pxr.BINOC_DIM, 0.01 * self.t + 0.3 * self.k, np.float32)
        i["debit_cur"] = np.full((pxr.MONO_IMG, pxr.MONO_IMG, 3), (40 * self.k + 9 * self.t) % 256,
                                 np.uint8)
        return o, i

    def reset(self, seed=None):
        o, i = super().reset(seed)
        self.stream.append(("reset", _row_of(o, i)))
        return o, i

    def step(self, a):
        o, _, term, trunc, i = super().step(a)
        r = float(100 * self.k + self.t)
        self.stream.append(("step", _row_of(o, i), r, term, trunc))
        return o, r, term, trunc, i


def _actor(obs):
    """A fixed function of each row's embedding (stands in for the policy)."""
    e = obs["embeddings"].float()
    return torch.stack([torch.tanh(e.mean(1) * 40.0), torch.tanh(e.std(1) * 8.0)], 1)


@pytest.mark.parametrize("parallel", [0, 1])
def test_each_env_gets_its_own_action_and_each_row_is_its_own_transition(
        c0_dir, tmp_path, monkeypatch, parallel):  # noqa: F811
    from torch_rl.frl.learner import SACLearner
    monkeypatch.setattr(SACLearner, "sample_actions", lambda self, obs: _actor(obs).cpu().numpy())
    envs = [_TaggedEnv(k) for k in range(3)]
    obs, info = envs[0].reset()
    s = fc.LoopSettings(max_steps=30, start_training=10 ** 6, utd=1, batch=2, replay=100,
                        ckpt_every=30, log_every=10 ** 6, device="cpu", seed=0, parallel=parallel,
                        random_start=0)
    loop = fc.train_parallel if parallel else fc.train
    _, rb, _ = loop(envs[0], obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                    row_of=_row_of, settings=s, c0_dir=c0_dir, log=lambda m: None,
                    extra_envs=[(e, None) for e in envs[1:]], towns=["A", "B", "C"])
    box = envs[0].action_space
    ref = fc.C0Trunk(c0_dir, "cpu")             # replays each env's own stream on its own
    expected = {k: [] for k in range(3)}
    for k, e in enumerate(envs):
        cur, hidden, n_act = None, None, 0
        for ev in e.stream:
            if ev[0] == "reset":
                cur, hidden = ref.step(ev[1], ref.zero_hidden())
                continue
            _, row, r, term, trunc = ev
            nxt, hidden = ref.step(row, hidden)
            a = _actor(cur).numpy()[0]
            # MODEL -> ENV: this env received the action computed from its own observation
            np.testing.assert_allclose(e.actions[n_act], fc.box_map(a, box), atol=1e-5)
            expected[k].append((cur, a, r, 0.0 if term else 1.0, nxt))
            cur, n_act = nxt, n_act + 1
        assert n_act == len(e.actions)
    # ENV -> REPLAY: every row is one env's transition, in that env's order
    assert len(rb) == sum(len(v) for v in expected.values()) == 30
    rows = {k: [j for j in range(len(rb)) if int(rb.rew[j]) // 100 == k] for k in range(3)}
    for k in range(3):
        assert len(rows[k]) == len(expected[k]) > 0
        for j, (cur, a, r, mask, nxt) in zip(rows[k], expected[k]):
            assert float(rb.rew[j]) == r and float(rb.mask[j]) == mask
            np.testing.assert_allclose(rb.act[j].numpy(), a, atol=1e-6)
            torch.testing.assert_close(rb.emb[j].float(), cur["embeddings"][0].half().float())
            torch.testing.assert_close(rb.st[j], cur["states"][0])
            torch.testing.assert_close(rb.next_emb[j].float(), nxt["embeddings"][0].half().float())
            torch.testing.assert_close(rb.next_st[j], nxt["states"][0])

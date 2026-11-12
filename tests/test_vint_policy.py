"""VintPolicy: ViNT inference for the driver (one goal) and for the topological
map (many nodes at once): the car's context against many node images, and many
node contexts against one goal image. Node images are stored as ViNT's own
85x64 crop (before normalisation), so thousands of them are cheap to score."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from torch_rl.vint_driver import ViNTDriver, VintPolicy  # noqa: E402

META = {"image_size": [85, 64], "metric_step_m": 0.5, "stride": 2, "context_size": 5}


class FakeViNT(torch.nn.Module):
    """dist = 10 x mean(goal) + mean(obs); waypoints x = 1..5 (normalised units)."""

    def forward(self, obs, goal):
        assert obs.shape[1:] == (18, 64, 85) and goal.shape[1:] == (3, 64, 85)
        dist = 10.0 * goal.mean(dim=(1, 2, 3)) + obs.mean(dim=(1, 2, 3))
        act = torch.zeros(obs.shape[0], 5, 4)
        act[:, :, 0] = torch.arange(1, 6, dtype=torch.float32)
        return dist[:, None], act


def _crop(img):
    from torchvision.transforms import functional as TF
    return TF.to_tensor(img.resize((85, 64)))


def _policy(chunk=2):
    p = VintPolicy(FakeViNT(), META, device="cpu", crop_fn=_crop, chunk=chunk)
    p.normalize = lambda t: t                      # keep the fake model's arithmetic readable
    return p


def test_crop85_is_vints_input_crop_as_uint8():
    p = _policy()
    c = p.crop85(np.full((128, 128, 3), 77, np.uint8))
    assert c.shape == (64, 85, 3) and c.dtype == np.uint8 and int(c[0, 0, 0]) == 77


def test_scoring_many_nodes_keeps_their_order_across_chunks_and_scales_waypoints():
    p = _policy(chunk=2)
    ctx = [np.zeros((64, 85, 3), np.uint8)] * 6
    goals = np.stack([np.full((64, 85, 3), v, np.uint8) for v in (0, 51, 102, 153, 204)])
    d, w = p.dists_to_goals(ctx, goals)
    assert d.shape == (5,) and np.all(np.diff(d) > 0)
    assert w.shape == (5, 5, 2) and w[0, :, 0] == pytest.approx(np.arange(1, 6) * 1.0)


def test_many_node_contexts_against_one_goal():
    p = _policy(chunk=3)
    ctxs = np.stack([np.full((6, 64, 85, 3), v, np.uint8) for v in (0, 102, 204, 51)])
    d = p.dists_from_contexts(ctxs, np.zeros((128, 128, 3), np.uint8))
    assert d.shape == (4,) and np.argsort(d).tolist() == [0, 3, 1, 2]


def test_pair_dists_scores_each_context_against_its_own_goal():
    p = _policy(chunk=2)
    ctxs = np.stack([np.full((6, 64, 85, 3), v, np.uint8) for v in (0, 0, 51)])
    goals = np.stack([np.full((64, 85, 3), v, np.uint8) for v in (102, 0, 0)])
    d = p.pair_dists(ctxs, goals)
    want = [10 * 0.4, 0.0, 0.2]                    # 10 x mean(goal) + mean(obs), /255 scale
    assert d.shape == (3,) and d == pytest.approx(want, abs=1e-5)


def test_a_single_call_accepts_raw_frames_or_stored_crops():
    p = _policy()
    raw = [np.full((128, 128, 3), 51, np.uint8)] * 6
    crops = [np.full((64, 85, 3), 51, np.uint8)] * 6
    d1, w1 = p(raw, np.full((128, 128, 3), 102, np.uint8))
    d2, w2 = p(crops, np.full((64, 85, 3), 102, np.uint8))
    assert d1 == pytest.approx(d2) and np.allclose(w1, w2)


class _Fixed:
    def __call__(self, ctx, goal):
        return 2.0, np.array([[1.1 * k, 0.4] for k in range(1, 6)])


def test_push_then_control_is_act_env():
    low, high = np.array([-1, -1], np.float32), np.array([1, 1], np.float32)
    a, b = ViNTDriver(_Fixed(), low, high), ViNTDriver(_Fixed(), low, high)
    a.reset_hidden(1)
    b.reset_hidden(1)
    f, g = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)
    for _ in range(3):
        x = a.act_env(f, g, 8.0)
        b.push(f)
        y = b.control(*_Fixed()(b.context(), g), 8.0)
        assert np.allclose(x, y)

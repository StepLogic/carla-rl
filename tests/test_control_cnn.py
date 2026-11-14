"""The CONTROL STREAM: a small trainable CNN on ONE frame plus a GRU (user 2026-11-13: "lets start
with small CNN - pretrained with already collected expert data from c0 and c1 - then finetune with
SAC", "add gru so it learns speed  do not stack froms").

WHY THIS SHAPE. The 2026-11-09 encoder probe says lateral offset is in c0's features (R^2 0.70-0.81)
but heading error is NOT (camera latent 0.046, binocular 0.06) -- and heading is what a steering
action changes. From raw pixels one frame gives heading R^2 0.042 and two frames 0.317, so the
information is temporal; in c0 the ONLY place it appeared was the GRU state (0.31-0.36). So the
control stream reads one frame through a trainable CNN and integrates it in a GRU, which also lets
it infer speed rather than being handed it. DEBiT stays frozen as the GOAL stream: its binocular
features are what give the policy its wrong-goal gap (+0.65, p<1e-4), and a SAC gradient has no
business in them.

Random-crop augmentation is DrQ's and is kept: our own benchmark bisection lists "no augmentation"
and "encoder trained by the actor as well as the critic" as the two classic pixel-SAC failures, and
the arm that passed DMC cartpole-from-pixels (827 vs 197) had the crop.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
torch = pytest.importorskip("torch")

from torch_rl.control_cnn import ControlCNN, ControlGRU, random_shift  # noqa: E402


def _frames(b=4, px=84):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (b, 3, px, px), generator=g, dtype=torch.uint8)


def test_the_cnn_maps_one_frame_to_a_bounded_latent():
    net = ControlCNN(latent_dim=50)
    out = net(_frames())
    assert out.shape == (4, 50)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0            # LayerNorm + tanh, the project's mono-head convention
    # uint8 and the same frames as float in [0,255] must agree: the net owns the /255 scaling
    assert torch.allclose(out, net(_frames().float()), atol=1e-5)


def test_random_shift_keeps_the_shape_and_is_identity_at_pad_zero():
    x = _frames().float()
    same = random_shift(x, pad=0)
    assert same.shape == x.shape and torch.equal(same, x)
    shifted = random_shift(x, pad=4)
    assert shifted.shape == x.shape
    assert not torch.equal(shifted, x)       # it did something
    assert torch.isfinite(shifted).all()


def test_the_gru_integrates_across_steps():
    net = ControlGRU(latent_dim=50, hidden_dim=64)
    f, speed = _frames(1), torch.tensor([[10.0]])
    h0 = net.reset_hidden(1)
    out1, h1 = net(f, speed, h0)
    out2, _ = net(f, speed, h1)              # the SAME frame again, carried state
    assert out1.shape == (1, 64) and out2.shape == (1, 64)
    assert not torch.allclose(out1, out2)    # the GRU is integrating, not restating the frame
    again, _ = net(f, speed, net.reset_hidden(1))
    assert torch.allclose(out1, again)       # a reset hidden state is reproducible


def test_speed_reaches_the_output_so_it_can_be_learned():
    net = ControlGRU(latent_dim=50, hidden_dim=64)
    f, h = _frames(1), net.reset_hidden(1)
    slow, _ = net(f, torch.tensor([[0.0]]), h)
    fast, _ = net(f, torch.tensor([[30.0]]), h)
    assert not torch.allclose(slow, fast)


def test_the_heads_give_an_action_and_the_lane_geometry():
    net = ControlGRU(latent_dim=50, hidden_dim=64)
    out, _ = net(_frames(), torch.zeros(4, 1), net.reset_hidden(4))
    act = net.action(out)
    aux = net.lane(out)
    assert act.shape == (4, 2) and act.abs().max() <= 1.0     # steer, throttle in [-1, 1]
    assert aux.shape == (4, 2) and torch.isfinite(aux).all()  # offset (m), heading error (deg)


def test_a_frame_stack_goes_in_on_the_channel_axis():
    """user 2026-11-13 "ok stack images": N frames concatenated as channels, DrQ's layout."""
    net = ControlGRU(latent_dim=50, hidden_dim=64, n_frames=3)
    stack = torch.randint(0, 256, (2, 9, 84, 84), dtype=torch.uint8)
    out, _ = net(stack, torch.zeros(2, 1), net.reset_hidden(2))
    assert out.shape == (2, 64)
    assert net.cnn.conv[0].in_channels == 9


def test_one_shift_is_shared_by_every_frame_in_the_stack():
    """THE PROPERTY THAT MATTERS: heading is visible only as motion BETWEEN the stacked frames, so
    the crop must be the same for all of them. Two identical frames must stay identical after the
    shift; a per-frame shift would fabricate motion and destroy the signal the stack is there for."""
    one = torch.randint(0, 256, (5, 3, 84, 84), dtype=torch.uint8).float()
    stack = torch.cat([one, one], dim=1)                      # frame t == frame t-1
    shifted = random_shift(stack, pad=4)
    assert torch.equal(shifted[:, :3], shifted[:, 3:])        # still identical -> one crop per sample
    assert not torch.equal(shifted, stack)                    # and it really did shift


def test_it_is_small_enough_to_train_on_one_gpu():
    net = ControlGRU(latent_dim=50, hidden_dim=64)
    n = sum(p.numel() for p in net.parameters())
    assert n < 3_000_000, n        # "small CNN": DEBiT's trunk is ~20M, c0's GRU path 512-wide

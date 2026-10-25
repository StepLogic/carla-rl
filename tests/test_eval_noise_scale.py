"""EVAL_NOISE_SCALE: the eval can roll out the policy with the training noise
scaled by any factor, to separate "the mean action is bad" from "the task is
hard". Scale 0 must reproduce the deterministic path exactly, and scale 1 must
leave training's own sampling untouched.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_rl.nets import Actor  # noqa: E402


def _actor():
    torch.manual_seed(0)
    return Actor(6, 2)


def test_scale_zero_is_the_deterministic_action():
    a, s = _actor(), torch.randn(4, 6)
    act, _ = a.sample(s, noise_scale=0.0)
    assert torch.allclose(act, a.deterministic(s), atol=0, rtol=0)


def test_scale_one_matches_unscaled_sampling():
    a, s = _actor(), torch.randn(4, 6)
    torch.manual_seed(7)
    want, want_lp = a.sample(s)
    torch.manual_seed(7)
    got, got_lp = a.sample(s, noise_scale=1.0)
    assert torch.allclose(got, want) and torch.allclose(got_lp, want_lp)


def test_half_scale_sits_between_mean_and_full_sample():
    a, s = _actor(), torch.randn(8, 6)
    mean = a.deterministic(s)
    torch.manual_seed(3)
    full, _ = a.sample(s, noise_scale=1.0)
    torch.manual_seed(3)
    half, _ = a.sample(s, noise_scale=0.5)
    # Same draw, half the step away from the mean in pre-tanh space, so the
    # half-scale action is strictly nearer the mean wherever the draw moved.
    moved = (full - mean).abs() > 1e-6
    assert moved.any()
    assert ((half - mean).abs()[moved] < (full - mean).abs()[moved]).all()


def test_negative_scale_is_refused():
    with pytest.raises(ValueError):
        _actor().sample(torch.randn(2, 6), noise_scale=-0.5)


def test_eval_out_suffix_names_the_scale():
    from train_carla_sac import _eval_suffix
    assert _eval_suffix(0.0) == ""
    assert _eval_suffix(1.0) == "_stoch"      # the folder already published
    assert _eval_suffix(0.25) == "_n25"
    assert _eval_suffix(0.5) == "_n50"

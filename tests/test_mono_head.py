"""ENCODER HEAD (user 2026-10-26, "LayerNorm + tanh").

The monocular encoder ended relu(fc(...)). Trained only by the critic, every
one of its 512 outputs went negative for every frame: v15@1.5M had 0/512 units
alive, v13 ~11%, v16 ~5% -- the policies drove without their camera. With
MONO_HEAD=ln_tanh the head is fc -> LayerNorm -> tanh: LayerNorm re-centres
the 512 values per image, so the layer cannot drift below zero together, and
tanh has no zero-gradient region. "relu" stays the default so every old
checkpoint loads unchanged. enc_alive (fraction of output units that vary
across a batch) is logged so a collapse can never go unseen again.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")

from torch_rl.gated_recurrent_tqc import Vc1Compression  # noqa: E402


def _push_negative(c):
    with torch.no_grad():
        c.fc.bias.fill_(-5.0)


def test_relu_head_dies_under_a_negative_bias_the_new_head_does_not():
    torch.manual_seed(0)
    x = torch.randn(16, 8, 4, 4)
    old = Vc1Compression(8, grid=4, out_dim=32)
    new = Vc1Compression(8, grid=4, out_dim=32, head="ln_tanh")
    new.load_state_dict({**old.state_dict(), **{k: v for k, v in new.state_dict().items()
                                                if k.startswith("out_norm")}})
    _push_negative(old)
    _push_negative(new)
    assert float(old(x).abs().max()) == 0.0                 # every unit dead
    out = new(x)
    assert float(out.std(0).mean()) > 0.05                  # still varies per image
    assert float(out.abs().max()) < 1.0                     # tanh-bounded


def test_default_head_is_the_old_one_and_loads_old_weights():
    c = Vc1Compression(8, grid=4, out_dim=32)
    assert c.head == "relu" and not hasattr(c, "out_norm")
    c2 = Vc1Compression(8, grid=4, out_dim=32)
    c2.load_state_dict(c.state_dict())                      # strict: no new keys


def test_unknown_head_is_refused():
    with pytest.raises(ValueError):
        Vc1Compression(8, grid=4, out_dim=32, head="gelu")


def test_mono_resnet_passes_the_head_through():
    from torch_rl.resnet_mono import MonoResNetEncoder
    enc = MonoResNetEncoder(128, 64, 32, 16, in_channels=3, head="ln_tanh")
    assert enc.compress.head == "ln_tanh"
    assert "ln_tanh" in enc.describe()


def test_update_reports_enc_alive():
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    for head, want_dead in (("relu", True), ("ln_tanh", False)):
        agent = GatedRecurrentSAC(_cfg(one=True, mono_head=head))
        _push_negative(agent.compress)
        out = agent.update(_chunk(agent.cfg, [1, 1], seed=0))
        alive = float(out["enc_alive"])
        assert 0.0 <= alive <= 1.0
        assert (alive == 0.0) if want_dead else (alive > 0.5), (head, alive)

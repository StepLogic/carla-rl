"""AdaptFormer adapters in DEBiT's ENCODER blocks, decoder frozen.

User 2026-10-12, reversing their first answer once the peer's argument was put to
them: adapters go in the CroCo encoder blocks ONLY; the cross-attention decoder
stays frozen. The reason is our own DEBiT probe -- the decoder IS the
correspondence mechanism, and the probe credits that correspondence for the
held-out gain (Town03 distance R2 0.81 vs the seg-VAE control's 0.64). Put
gradients into it and a later improvement cannot be told apart from having eroded
the thing that was worth keeping.

WHAT EACH TEST PROTECTS:
  * IDENTITY AT INIT. AdaptFormer zero-initialises the up-projection so the
    adapted network starts as exactly the pretrained one. Without it the first
    gradient step is taken from a randomly perturbed encoder and the pretrained
    weights are partly wasted -- and nothing downstream would look obviously wrong.
  * THE DECODER IS UNTOUCHED. The whole point of the user's decision.
  * ONLY ADAPTERS TRAIN. A stray requires_grad on the backbone turns a 0.3 M-param
    adapter run into a 55 M-param fine-tune at a different memory budget.
  * THE PARALLEL BRANCH READS THE SAME INPUT AS THE MLP. AdaptFormer is parallel to
    the MLP sublayer (on norm2(x)), not serial after it; serial is a different
    method with different published results.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
nn = torch.nn

from torch_rl import debit_adapters as da  # noqa: E402

CROCO_SRC = os.environ.get("DEBIT_SRC", "/home/robotlab/scratch/debit/src")
TINY = "/home/robotlab/scratch/debit/out/ckpt/hab_bl/imgnav/debit_tiny.pth"


def _croco_block(dim=384, heads=6):
    """A REAL croco encoder block, so the wrapper is tested against the class it
    will actually wrap rather than a stand-in that cannot drift."""
    if CROCO_SRC not in sys.path:
        sys.path.insert(0, CROCO_SRC)
    try:
        from croco.models.blocks import Block
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"croco not importable: {e}")
    return Block(dim, heads, mlp_ratio=4.0)


# ------------------------------------------------------------ identity at init --
def test_the_adapter_is_the_IDENTITY_at_initialisation():
    """THE PROPERTY THAT MATTERS MOST. Zero-initialised up-projection means the
    wrapped MLP returns the frozen MLP's output bit for bit on step 0."""
    blk = _croco_block()
    h = torch.randn(2, 49, 384)
    before = blk.mlp(h)
    wrapped = da.AdaptMLP(blk.mlp, dim=384, bottleneck=64, scale=0.1)
    after = wrapped(h)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_the_adapter_stops_being_the_identity_once_the_up_projection_moves():
    """The flip side: a zero branch that can never leave zero would be a no-op
    dressed as a method."""
    blk = _croco_block()
    h = torch.randn(2, 49, 384)
    wrapped = da.AdaptMLP(blk.mlp, dim=384, bottleneck=64, scale=0.1)
    with torch.no_grad():
        wrapped.up.weight.normal_(0.0, 0.02)
    assert not torch.allclose(wrapped(h), blk.mlp(h))


def test_the_branch_is_PARALLEL_to_the_mlp_not_serial_after_it():
    """out = mlp(h) + scale * up(act(down(h))), both reading the SAME h. Serial
    (up(act(down(mlp(h))))) is a different method; this pins the arithmetic."""
    blk = _croco_block()
    wrapped = da.AdaptMLP(blk.mlp, dim=384, bottleneck=8, scale=1.0)
    with torch.no_grad():
        wrapped.up.weight.normal_(0.0, 0.05)
        wrapped.up.bias.normal_(0.0, 0.05)
    h = torch.randn(2, 49, 384)
    expected = blk.mlp(h) + 1.0 * wrapped.up(wrapped.act(wrapped.down(h)))
    torch.testing.assert_close(wrapped(h), expected)


def test_the_scale_multiplies_only_the_adapter_branch():
    blk = _croco_block()
    h = torch.randn(1, 49, 384)
    base = blk.mlp(h)
    w1 = da.AdaptMLP(blk.mlp, dim=384, bottleneck=8, scale=1.0)
    with torch.no_grad():
        w1.up.weight.normal_(0.0, 0.05)
    branch = w1(h) - base
    w2 = da.AdaptMLP(blk.mlp, dim=384, bottleneck=8, scale=0.25)
    # Copy the adapter WEIGHTS only. load_state_dict would also copy w1's `scale`
    # buffer and overwrite the 0.25 this test is about.
    with torch.no_grad():
        for dst, src in ((w2.down, w1.down), (w2.up, w1.up)):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
    torch.testing.assert_close(w2(h) - base, 0.25 * branch, rtol=1e-4, atol=1e-6)


# -------------------------------------------------------------- what trains --
def test_only_the_adapter_parameters_require_gradient():
    blk = _croco_block()
    wrapped = da.AdaptMLP(blk.mlp, dim=384, bottleneck=64, scale=0.1)
    trainable = {n for n, p in wrapped.named_parameters() if p.requires_grad}
    assert trainable == {"down.weight", "down.bias", "up.weight", "up.bias"}, trainable


def test_the_frozen_mlp_receives_no_gradient_through_the_wrapper():
    blk = _croco_block()
    wrapped = da.AdaptMLP(blk.mlp, dim=384, bottleneck=64, scale=0.1)
    with torch.no_grad():
        wrapped.up.weight.normal_(0.0, 0.02)
    wrapped(torch.randn(2, 49, 384)).sum().backward()
    assert all(p.grad is None for p in wrapped.mlp.parameters())
    assert wrapped.down.weight.grad is not None
    assert wrapped.up.weight.grad is not None


def test_at_init_only_the_UP_projection_receives_gradient():
    """NOT A BUG -- the consequence of the zero-init, and worth pinning so nobody
    "fixes" it later. With up.weight == 0 the chain rule gives the down-projection
    exactly zero gradient on step 0; it starts learning once up has moved off
    zero. LoRA's B=0 initialisation behaves the same way. Measured on the real
    DEBiT-tiny: 16 of 32 adapter tensors carry gradient on the first backward,
    which is the 8 blocks' up.weight and up.bias."""
    blk = _croco_block()
    wrapped = da.AdaptMLP(blk.mlp, dim=384, bottleneck=64, scale=0.1)
    wrapped(torch.randn(2, 49, 384)).square().mean().backward()
    assert wrapped.up.weight.grad.abs().sum() > 0
    assert wrapped.down.weight.grad.abs().sum() == 0

    wrapped.zero_grad(set_to_none=True)
    with torch.no_grad():
        wrapped.up.weight.normal_(0.0, 0.02)
    wrapped(torch.randn(2, 49, 384)).square().mean().backward()
    assert wrapped.down.weight.grad.abs().sum() > 0


def test_the_adapter_is_small_relative_to_the_block_it_adapts():
    """Budget check: 2*d*r plus biases, versus the MLP's 8*d^2. At d=384, r=64
    that is ~49 K against ~1.2 M, so ~4%."""
    blk = _croco_block()
    mlp_params = sum(p.numel() for p in blk.mlp.parameters())
    wrapped = da.AdaptMLP(blk.mlp, dim=384, bottleneck=64, scale=0.1)
    adapter = sum(p.numel() for n, p in wrapped.named_parameters()
                  if not n.startswith("mlp."))
    assert adapter < 0.1 * mlp_params, (adapter, mlp_params)


# ------------------------------------------------- injection into the network --
class _FakeCroco(nn.Module):
    """Stands in for CroCoDownstreamBinocular's block layout: enc_blocks and
    dec_blocks, which is all the injector addresses."""

    def __init__(self, dim=384, n_enc=8, n_dec=2):
        super().__init__()
        self.enc_blocks = nn.ModuleList([_croco_block(dim) for _ in range(n_enc)])
        self.dec_blocks = nn.ModuleList([_croco_block(dim) for _ in range(n_dec)])
        self.enc_embed_dim = dim


def test_every_encoder_block_gets_an_adapter():
    net = _FakeCroco()
    da.add_encoder_adapters(net, bottleneck=64, scale=0.1)
    assert all(isinstance(b.mlp, da.AdaptMLP) for b in net.enc_blocks)


def test_NO_decoder_block_gets_an_adapter():
    """The user's decision, pinned. If this ever passes by accident the
    correspondence prior is being trained and the probe's conclusion no longer
    applies to the arm."""
    net = _FakeCroco()
    da.add_encoder_adapters(net, bottleneck=64, scale=0.1)
    assert not any(isinstance(b.mlp, da.AdaptMLP) for b in net.dec_blocks)


def test_the_whole_backbone_is_frozen_and_only_adapters_train():
    net = _FakeCroco()
    da.add_encoder_adapters(net, bottleneck=64, scale=0.1)
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    assert trainable, "nothing trains: the adapters were not registered"
    assert all(".mlp.down." in n or ".mlp.up." in n for n in trainable), trainable
    assert all(n.startswith("enc_blocks.") for n in trainable), trainable


def test_adapter_parameters_returns_exactly_the_trainable_set():
    net = _FakeCroco()
    da.add_encoder_adapters(net, bottleneck=64, scale=0.1)
    got = {id(p) for p in da.adapter_parameters(net)}
    want = {id(p) for p in net.parameters() if p.requires_grad}
    assert got == want


def test_injecting_twice_is_refused():
    """Double-wrapping would stack two adapter branches per block and silently
    double the trainable count."""
    net = _FakeCroco()
    da.add_encoder_adapters(net)
    with pytest.raises(RuntimeError, match="already"):
        da.add_encoder_adapters(net)


def test_a_network_without_encoder_blocks_is_refused():
    with pytest.raises(AttributeError, match="enc_blocks"):
        da.add_encoder_adapters(nn.Linear(4, 4))


def test_the_injected_network_is_numerically_unchanged_at_init():
    """End to end: a full forward through every wrapped encoder block must match
    the frozen network at step 0, not just one block in isolation."""
    net = _FakeCroco()
    x, xpos = torch.randn(2, 49, 384), torch.zeros(2, 49, 2, dtype=torch.long)
    net.eval()
    with torch.no_grad():
        before = x.clone()
        for b in net.enc_blocks:
            before = b(before, xpos)
    da.add_encoder_adapters(net, bottleneck=64, scale=0.1)
    with torch.no_grad():
        after = x.clone()
        for b in net.enc_blocks:
            after = b(after, xpos)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


# ------------------------------------------------------- against real weights --
@pytest.mark.skipif(not os.path.exists(TINY), reason="debit_tiny.pth not present")
def test_the_real_debit_tiny_encoder_accepts_adapters_and_is_identity_at_init():
    """The real thing: load the published DEBiT-tiny weights, inject, and check
    the binocular output is bit-identical at init while the adapters are the only
    trainable parameters."""
    from torch_rl.debit_encoder import DebitTinyEncoder
    enc = DebitTinyEncoder(TINY, device="cpu", half=False)
    cur = torch.randint(0, 255, (1, 120, 120, 3), dtype=torch.uint8).numpy()
    goal = torch.randint(0, 255, (1, 120, 120, 3), dtype=torch.uint8).numpy()
    before = enc.encode_batch(cur, goal)
    n = da.add_encoder_adapters(enc.net, bottleneck=64, scale=0.1)
    assert n == len(enc.net.enc_blocks) > 0
    after = enc.encode_batch(cur, goal)
    torch.testing.assert_close(torch.tensor(after), torch.tensor(before),
                               rtol=0, atol=0)
    trainable = [nm for nm, p in enc.net.named_parameters() if p.requires_grad]
    assert trainable and all(nm.startswith("enc_blocks.") for nm in trainable)
    assert not any("dec_blocks" in nm for nm in trainable)

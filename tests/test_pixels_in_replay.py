"""Pixels in the replay: the row layout, and DEBiT as the learner's visual stack.

User 2026-10-13: "Pixels in replay proceed". The point of it is that the adapters
chosen the day before live in DEBiT's encoder blocks, and a replay holding a
precomputed binocular vector cannot train them -- the update needs the frames.

THE DESIGN THESE TESTS PIN. The learner's existing per-chunk goal path
(`goal_tokens`) reshapes a row's visual block to (embed_dim, grid, grid),
broadcasts the goal over time and channel-stacks the two before ONE visual call.
At embed_dim=3 and grid=112 that stack IS a frame pair, so a binocular encoder
drops in and _embed, _burn_in and _roll are untouched. Each test below is one
place where that correspondence could silently break:
  * the goal ending up in the current frame's channels, or vice versa
  * a pre-normalised or uint8-typed frame reaching a path expecting raw floats
  * the adapters not actually receiving gradient (the whole purpose)
  * the frozen backbone quietly becoming trainable
  * the geometry drifting from 3x112x112, which fails deep inside croco
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
nn = torch.nn

from torch_rl import debit_adapters as da  # noqa: E402
from torch_rl import pixel_rows as pr  # noqa: E402
from torch_rl.debit_visual import IMG, DebitVisual  # noqa: E402

STATE = 8


# ------------------------------------------------------------- the row layout --
def test_a_row_is_frame_then_previous_action_then_state():
    frame = np.full((IMG, IMG, 3), 7, np.uint8)
    row = pr.pixel_row(frame, [0.5, -0.25], np.arange(STATE))
    assert row.shape == (pr.row_dim(STATE),)
    pix, pa, st = pr.split_row(row, STATE)
    assert pix.shape == (pr.PIX,) and np.all(pix == 7.0)
    np.testing.assert_allclose(pa, [0.5, -0.25])
    np.testing.assert_allclose(st, np.arange(STATE))


def test_pixels_are_stored_RAW_not_normalised():
    """The /255 and the ImageNet normalisation belong in the learner. Baking
    them in would freeze today's constants into every row in the replay and make
    a preprocessing change a buffer flush."""
    row = pr.pixel_row(np.full((IMG, IMG, 3), 255, np.uint8), [0, 0], np.zeros(STATE))
    assert pr.split_row(row, STATE)[0].max() == 255.0


def test_a_float_frame_is_refused():
    with pytest.raises(ValueError, match="uint8"):
        pr.pixel_row(np.zeros((IMG, IMG, 3), np.float32), [0, 0], np.zeros(STATE))


def test_the_env_frame_is_resized_to_DEBiTs_input_size():
    """The camera is 120x120 and DEBiT wants 112. Resizing at insert keeps the
    row width fixed and makes the stored frame exactly what the encoder sees."""
    row = pr.pixel_row(np.full((120, 120, 3), 3, np.uint8), [0, 0], np.zeros(STATE))
    assert pr.split_row(row, STATE)[0].size == pr.PIX


def test_a_frame_survives_the_round_trip():
    frame = (np.arange(IMG * IMG * 3, dtype=np.int64) % 251).astype(np.uint8)
    frame = frame.reshape(IMG, IMG, 3)
    row = pr.pixel_row(frame, [0, 0], np.zeros(STATE))
    back = pr.frame_from_block(pr.split_row(row, STATE)[0])
    np.testing.assert_array_equal(back.astype(np.uint8), frame)


def test_the_goal_block_matches_the_rows_visual_block_layout():
    """They are channel-stacked without reshaping, so the two must agree."""
    g = pr.goal_block(np.full((IMG, IMG, 3), 9, np.uint8))
    assert g.shape == (pr.PIX,)


def test_a_wrong_sized_previous_action_is_refused():
    with pytest.raises(ValueError, match="previous action"):
        pr.pixel_row(np.zeros((IMG, IMG, 3), np.uint8), [0.0], np.zeros(STATE))


# --------------------------------------------------------------- the encoder --
class _FakeNet(nn.Module):
    """Stands in for croco: records the pair it was given and returns something
    differentiable, so gradient flow can be checked without weights."""

    def __init__(self, out=3136):
        super().__init__()
        self.seen = []
        self.lin = nn.Linear(3 * IMG * IMG, out)
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

    def forward(self, cur, goal):
        self.seen.append((cur.detach().clone(), goal.detach().clone()))
        return self.lin(cur.flatten(1)) + self.lin(goal.flatten(1))


class _FakeEnc:
    def __init__(self):
        self.net = _FakeNet()
        self.output_size = 3136


def _stack(cur_fill, goal_fill, n=2):
    cur = torch.full((n, 3, IMG, IMG), float(cur_fill))
    goal = torch.full((n, 3, IMG, IMG), float(goal_fill))
    return torch.cat([cur, goal], dim=1)


def test_the_first_three_channels_are_the_CURRENT_frame():
    """Swap these and the policy is handed its goal as its view, which would
    still train -- to do the wrong task."""
    enc = _FakeEnc()
    v = DebitVisual(enc, visual_dim=16)
    v(_stack(10, 200))
    cur, goal = enc.net.seen[-1]
    # Invert the normalisation to recover the raw fill. Note it is PER CHANNEL,
    # so a constant frame is deliberately NOT constant across channels after it --
    # which is why this reads the values back rather than checking flatness.
    for got, want in ((cur, 10.0), (goal, 200.0)):
        raw = (got * v.std + v.mean) * 255.0
        assert torch.allclose(raw, torch.full_like(raw, want), atol=1e-3), want


def test_the_normalisation_happens_in_the_module():
    enc = _FakeEnc()
    v = DebitVisual(enc, visual_dim=16)
    v(_stack(255, 0))
    cur, goal = enc.net.seen[-1]
    # 255 -> (1 - mean)/std, 0 -> -mean/std: neither is the raw value.
    assert cur.max() < 3.0 and goal.min() < 0.0


def test_the_output_width_is_visual_dim():
    v = DebitVisual(_FakeEnc(), visual_dim=64)
    assert v(_stack(1, 2)).shape == (2, 64)


def test_a_stack_without_six_channels_is_refused():
    v = DebitVisual(_FakeEnc(), visual_dim=16)
    with pytest.raises(ValueError, match="6"):
        v(torch.zeros(2, 3, IMG, IMG))


def test_gradient_reaches_the_ADAPTERS_through_this_path():
    """THE WHOLE POINT. If this fails, the pixels in the replay buy nothing and
    the arm is a frozen-encoder run wearing adapter clothes."""
    import sys as _s
    _s.path.insert(0, os.path.join(ROOT, "src"))

    class _Blk(nn.Module):
        def __init__(self, dim=12):
            super().__init__()
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        def forward(self, x, xpos=None):
            return x + self.mlp(self.norm2(x))

    class _Net(nn.Module):
        def __init__(self, dim=12):
            super().__init__()
            self.enc_blocks = nn.ModuleList([_Blk(dim), _Blk(dim)])
            self.dec_blocks = nn.ModuleList([_Blk(dim)])
            self.enc_embed_dim = dim
            self.inp = nn.Linear(3 * IMG * IMG, dim)
            self.out = nn.Linear(dim, 3136)

        def forward(self, cur, goal):
            x = self.inp(cur.flatten(1)) + self.inp(goal.flatten(1))
            for b in self.enc_blocks:
                x = b(x)
            for b in self.dec_blocks:
                x = b(x)
            return self.out(x)

    net = _Net()
    n_adapted = da.add_encoder_adapters(net, bottleneck=4, scale=1.0)
    assert n_adapted == 2

    class _E:
        pass
    e = _E()
    e.net, e.output_size = net, 3136
    v = DebitVisual(e, visual_dim=8)
    # Move the up-projections off zero, else the down-projections legitimately
    # see no gradient on step 0 (the zero-init property, tested elsewhere).
    # SEEDED: the draw comes from the global RNG, so without this the magnitudes
    # depend on what ran before and the test is order-dependent -- it failed once
    # in file order and passed alone, which is exactly that.
    torch.manual_seed(0)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, da.AdaptMLP):
                m.up.weight.normal_(0.0, 0.05)
    v(_stack(50, 150)).square().mean().backward()
    grads = [p.grad is not None and p.grad.abs().sum().item() > 0
             for p in da.adapter_parameters(net)]
    assert grads and all(grads), f"adapter gradients: {grads}"
    # And the frozen backbone got none.
    for blk in net.dec_blocks:
        assert all(p.grad is None for p in blk.parameters())


def test_the_projection_is_trainable_and_the_backbone_is_not():
    net = _FakeNet()
    for p in net.parameters():
        p.requires_grad_(False)

    class _E:
        pass
    e = _E()
    e.net, e.output_size = net, 3136
    v = DebitVisual(e, visual_dim=8)
    trainable = {n for n, p in v.named_parameters() if p.requires_grad}
    assert trainable == {"proj.weight", "proj.bias"}, trainable


# ------------------------------------------------------- the config contract --
def _cfg(**kw):
    from torch_rl.gated_recurrent_sac import GatedRecurrentSACConfig
    base = dict(embed_dim=3, grid=112, goal_tokens=True, mono_encoder="debit",
                state_dim=STATE, mode_index=STATE - 1, goal_cols=(0, 1),
                explore_cols=(2,), hidden=16, visual_dim=16, mlp=(8, 8),
                burn_in=2, device="cpu", compile=False, seed=0)
    base.update(kw)
    return GatedRecurrentSACConfig(**base)


def test_the_debit_geometry_is_enforced():
    _cfg()                                   # the valid combination
    with pytest.raises(ValueError, match="embed_dim 3"):
        _cfg(embed_dim=16)
    with pytest.raises(ValueError, match="grid 112"):
        _cfg(grid=8)
    with pytest.raises(ValueError, match="goal_tokens"):
        _cfg(goal_tokens=False)


def test_build_visual_refuses_debit_without_the_encoder_instance():
    """The config cannot carry a live module, and the rollout and the update MUST
    share one set of adapters -- so the instance is passed, not rebuilt."""
    from torch_rl.gated_recurrent_tqc import build_visual
    with pytest.raises(ValueError, match="debit_encoder"):
        build_visual(_cfg(), in_ch=6)


def test_build_visual_returns_the_debit_stack_when_given_one():
    from torch_rl.gated_recurrent_tqc import build_visual
    v = build_visual(_cfg(debit_encoder=_FakeEnc()), in_ch=6)
    assert isinstance(v, DebitVisual)
    assert "ENCODER only" in v.describe()


# ------------------------------------------- the learner on real pixel rows --
def test_the_LEARNER_updates_from_pixel_rows_and_trains_the_adapters():
    """END TO END through the real learner: a chunk of pixel rows plus a
    per-chunk goal frame, through _embed -> _roll -> both critic heads and the
    actor, with the adapters receiving gradient.

    This is the test that says the feature works. Everything above checks a
    piece; this one checks the pieces still mean the same thing once the
    learner's own row geometry, goal broadcast and target copy are involved --
    which is where a per-chunk goal of silently the wrong shape, or a visual
    stack that quietly bypasses the encoder, would show up.
    """
    from torch_rl import debit_visual
    from torch_rl.gated_recurrent_sac import (GatedRecurrentSAC,
                                              GatedRecurrentSACConfig)

    class _Blk(nn.Module):
        def __init__(self, dim=12):
            super().__init__()
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                     nn.Linear(dim, dim))

        def forward(self, x, xpos=None):
            return x + self.mlp(self.norm2(x))

    class _Net(nn.Module):
        """A stand-in for croco with the same contract: two frames in, a 3136-d
        vector out, encoder blocks the adapters can wrap."""

        def __init__(self, dim=12):
            super().__init__()
            self.enc_blocks = nn.ModuleList([_Blk(dim)])
            self.dec_blocks = nn.ModuleList([_Blk(dim)])
            self.enc_embed_dim = dim
            self.inp = nn.Linear(3 * IMG * IMG, dim)
            self.out = nn.Linear(dim, 3136)

        def forward(self, cur, goal):
            x = self.inp(cur.flatten(1)) + self.inp(goal.flatten(1))
            for b in self.enc_blocks:
                x = b(x)
            for b in self.dec_blocks:
                x = b(x)
            return self.out(x)

    class _E:
        pass
    enc = _E()
    enc.net, enc.output_size = _Net(), 3136
    da.add_encoder_adapters(enc.net, bottleneck=4, scale=1.0)
    with torch.no_grad():            # off the zero-init, so down/ trains too
        for m in enc.net.modules():
            if isinstance(m, da.AdaptMLP):
                m.up.weight.normal_(0.0, 0.05)

    cfg = GatedRecurrentSACConfig(
        embed_dim=3, grid=IMG, goal_tokens=True, mono_encoder="debit",
        debit_encoder=enc, state_dim=STATE, mode_index=STATE - 1,
        goal_cols=(0, 1), explore_cols=(2,), hidden=16, visual_dim=16,
        mlp=(8, 8), burn_in=2, device="cpu", compile=False, seed=0,
        explore_actor_warmup=0, n_critics=10, subset_size=2)
    agent = GatedRecurrentSAC(cfg)
    assert isinstance(agent.compress, debit_visual.DebitVisual)

    # A chunk built the way the rollout will build it: rows from pixel_row, one
    # goal block per chunk entry.
    B, K, T = 2, cfg.burn_in, 4
    rng = np.random.default_rng(0)
    rows = np.stack([
        np.stack([pr.pixel_row(
            rng.integers(0, 255, (120, 120, 3), dtype=np.uint8),
            rng.uniform(-1, 1, 2), rng.normal(size=STATE))
            for _ in range(K + T + 1)]) for _ in range(B)])
    assert rows.shape[-1] == pr.row_dim(STATE) == cfg.obs_dim, (
        rows.shape[-1], cfg.obs_dim)
    goals = np.stack([pr.goal_block(
        rng.integers(0, 255, (IMG, IMG, 3), dtype=np.uint8)) for _ in range(B)])

    chunk = {"obs": torch.from_numpy(rows).float(),
             "keep": torch.ones(B, K + 1),
             "actions": torch.from_numpy(
                 rng.uniform(-1, 1, (B, T, 2)).astype(np.float32)),
             "rewards": torch.randn(B, T),
             "masks": torch.ones(B, T),
             "valid": torch.ones(B, T),
             "mode": torch.tensor([1.0, 0.0]),      # one goal row, one explore
             "goal": torch.from_numpy(goals).float()}

    before = [p.detach().clone() for p in da.adapter_parameters(agent.compress.net)]
    out = agent.update(chunk)
    assert out and "critic_loss_g" in out
    after = list(da.adapter_parameters(agent.compress.net))
    assert any(not torch.equal(a, b) for a, b in zip(after, before)), \
        "the adapters did not move: the pixels in the replay bought nothing"
    assert all(not p.requires_grad for blk in agent.compress.net.dec_blocks
               for p in blk.parameters())


# ------------------------------------ the binocular + monocular row layout --
def test_the_MONO_frame_is_stored_scaled_to_zero_one():
    """NOT a free choice, and the opposite of the DEBiT row above:
    MonoResNetEncoder applies the ImageNet mean/std to what it is handed and
    REFUSES anything outside [0, 1]. The first pilot of this stack died at step
    283 on exactly that guard, with a range of [39, 246]."""
    row = pr.mono_row(np.full((120, 120, 3), 255, np.uint8), [0, 0], 1,
                      np.zeros(pr.BINOC_DIM))
    pix = row[:pr.MONO_PIX]
    assert pix.max() == pytest.approx(1.0) and pix.min() == pytest.approx(1.0)


def test_the_mono_row_is_frame_then_action_then_mode_then_binoc():
    binoc = np.arange(pr.BINOC_DIM, dtype=np.float32)
    row = pr.mono_row(np.zeros((128, 128, 3), np.uint8), [0.25, -0.5], 1, binoc)
    assert row.shape == (pr.mono_row_dim(),)
    i = pr.MONO_PIX
    np.testing.assert_allclose(row[i:i + 2], [0.25, -0.5])
    assert row[i + 2] == 1.0                       # the mode bit
    np.testing.assert_allclose(row[i + 3:], binoc)


def test_the_mode_bit_is_the_FIRST_state_column():
    """The learner's mode_index is 0 and core_state_cols is everything after
    it, so the critic reads the mode and the GRU reads the binoc."""
    assert pr.mono_core_state_cols()[0] == 1
    assert pr.mono_core_state_cols()[-1] == pr.BINOC_DIM
    assert len(pr.mono_core_state_cols()) == pr.BINOC_DIM


def test_a_wrong_width_binoc_is_refused():
    """A zero or short binoc would train the policy goal-blind."""
    with pytest.raises(ValueError, match="binoc"):
        pr.mono_row(np.zeros((128, 128, 3), np.uint8), [0, 0], 1, np.zeros(10))


def test_the_mono_frame_is_resized_to_128():
    """MonoResNetEncoder requires a multiple of 32, and DEBiT's 112 is not one."""
    row = pr.mono_row(np.zeros((120, 120, 3), np.uint8), [0, 0], 0,
                      np.zeros(pr.BINOC_DIM))
    assert row[:pr.MONO_PIX].size == 128 * 128 * 3

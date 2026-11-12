"""AGENT_LATENT_DIMS: per-key projection widths for PixelMultiplexer.

WHY THIS EXISTS. PixelMultiplexer maps every observation key through
Dense(width) -> LayerNorm -> tanh, and the width defaults to the learner's
latent_dim, which jaxrl2 sets to 50. train_carla_tqc passes AGENT_LATENT_DIM
only when AGENT_ENCODER != "none", and every arm here runs AGENT_ENCODER=none --
so AGENT_LATENT_DIM=512 in four snapshots was read by nothing, and v13's
checkpoint really does hold encoder_latent/kernel = (3840, 50).

The lesson these tests pin is the one that cost those runs: a width that is
silently ignored is worse than one that raises. Malformed input must fail loudly,
and the DEFAULT must stay "pass nothing" so existing arms are byte-identical.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def clean_env(monkeypatch):
    monkeypatch.delenv("AGENT_LATENT_DIMS", raising=False)
    return monkeypatch


def _fn():
    from train_carla_tqc import _latent_dims
    return _latent_dims


def test_unset_is_empty(clean_env):
    """Unset -> (), which the caller reads as 'do not pass the kwarg'."""
    assert _fn()() == ()


def test_blank_and_whitespace_are_empty(clean_env):
    for raw in ("", "   ", ",", " , "):
        clean_env.setenv("AGENT_LATENT_DIMS", raw)
        assert _fn()() == (), f"{raw!r} should be empty"


def test_the_users_case(clean_env):
    """User 2026-09-26 'reduce latent to 768': one key, the rest untouched."""
    clean_env.setenv("AGENT_LATENT_DIMS", "latent:768")
    assert _fn()() == (("latent", 768),)


def test_multiple_keys_keep_order(clean_env):
    clean_env.setenv("AGENT_LATENT_DIMS", "latent:768,state:32")
    assert _fn()() == (("latent", 768), ("state", 32))


def test_spaces_are_tolerated(clean_env):
    clean_env.setenv("AGENT_LATENT_DIMS", " latent : 768 , state : 32 ")
    assert _fn()() == (("latent", 768), ("state", 32))


@pytest.mark.parametrize("raw", ["latent", "latent=768", "768"])
def test_missing_colon_raises(clean_env, raw):
    """A pair with no width must NOT be silently dropped."""
    clean_env.setenv("AGENT_LATENT_DIMS", raw)
    with pytest.raises(ValueError, match="key:width"):
        _fn()()


def test_empty_key_raises(clean_env):
    clean_env.setenv("AGENT_LATENT_DIMS", ":768")
    with pytest.raises(ValueError, match="empty key"):
        _fn()()


@pytest.mark.parametrize("raw", ["latent:0", "latent:-5"])
def test_non_positive_width_raises(clean_env, raw):
    clean_env.setenv("AGENT_LATENT_DIMS", raw)
    with pytest.raises(ValueError, match="must be positive"):
        _fn()()


def test_non_integer_width_raises(clean_env):
    clean_env.setenv("AGENT_LATENT_DIMS", "latent:big")
    with pytest.raises(ValueError):
        _fn()()


def test_the_concatenated_width_is_what_we_think(clean_env):
    """v13's four keys at 768/50/50/50 = 918 into the (256,256) MLP, against
    4 x 50 = 200 before. Pinned so the arithmetic in the commit message and the
    code comment cannot drift apart from the parser."""
    clean_env.setenv("AGENT_LATENT_DIMS", "latent:768")
    widths = dict(_fn()())
    keys = ("latent", "goal", "mode", "state")
    assert sum(widths.get(k, 50) for k in keys) == 918


# ---- STACKS AND SHARED SPECS (user 2026-09-26: "768 from vc1 -> 512 -> 256",
# then "make image latent goal latent same size") --------------------------


def test_a_stack_becomes_a_tuple(clean_env):
    """Two widths -> a tuple, which PixelMultiplexer reads as two layers."""
    clean_env.setenv("AGENT_LATENT_DIMS", "latent:512:256")
    assert _fn()() == (("latent", (512, 256)),)


def test_single_width_stays_an_int(clean_env):
    """The one-layer case must NOT become a 1-tuple: an int keeps the original
    parameter names, which is what lets old checkpoints load."""
    clean_env.setenv("AGENT_LATENT_DIMS", "latent:256")
    (key, spec), = _fn()()
    assert spec == 256 and isinstance(spec, int)


def test_shared_spec_makes_two_keys_identical(clean_env):
    """'same size' holds by construction, not by two numbers being typed alike."""
    clean_env.setenv("AGENT_LATENT_DIMS", "latent+goal:512:256")
    got = dict(_fn()())
    assert got == {"latent": (512, 256), "goal": (512, 256)}
    assert got["latent"] is got["goal"] or got["latent"] == got["goal"]


def test_shared_spec_with_a_single_width(clean_env):
    clean_env.setenv("AGENT_LATENT_DIMS", "latent+goal:256")
    assert dict(_fn()()) == {"latent": 256, "goal": 256}


def test_mixed_shared_and_individual(clean_env):
    clean_env.setenv("AGENT_LATENT_DIMS", "latent+goal:512:256,state:32")
    assert _fn()() == (("latent", (512, 256)), ("goal", (512, 256)),
                       ("state", 32))


def test_duplicate_key_raises(clean_env):
    """Two specs for one key would make the effective width depend on dict
    ordering, which is exactly the kind of silence this parser refuses."""
    clean_env.setenv("AGENT_LATENT_DIMS", "latent:256,latent:512")
    with pytest.raises(ValueError, match="set twice"):
        _fn()()
    clean_env.setenv("AGENT_LATENT_DIMS", "latent+goal:256,goal:512")
    with pytest.raises(ValueError, match="set twice"):
        _fn()()


def test_empty_key_in_a_shared_spec_raises(clean_env):
    clean_env.setenv("AGENT_LATENT_DIMS", "latent+:256")
    with pytest.raises(ValueError, match="empty key"):
        _fn()()


def test_empty_width_in_a_stack_raises(clean_env):
    clean_env.setenv("AGENT_LATENT_DIMS", "latent:512:")
    with pytest.raises(ValueError, match="empty width"):
        _fn()()


def test_the_requested_recipe_widths(clean_env):
    """VC-1 cls is 768; the image and goal latents both compress 768->512->256,
    so each contributes 256 and they are equal. mode and state keep 50."""
    clean_env.setenv("AGENT_LATENT_DIMS", "latent+goal:512:256")
    w = dict(_fn()())
    final = lambda s: s[-1] if isinstance(s, tuple) else s
    assert final(w["latent"]) == final(w["goal"]) == 256
    total = sum(final(w.get(k, 50)) for k in ("latent", "goal", "mode", "state"))
    assert total == 256 + 256 + 50 + 50 == 612

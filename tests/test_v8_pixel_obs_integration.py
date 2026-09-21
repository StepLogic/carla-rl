"""v8 pixel-obs agent, integration level: the REAL learners under the REAL flags.

WHAT THIS CLOSES. Tasks 1-5 each tested their own piece with stubs. This file
wires them end to end with no simulator and no stub: the production
`build_agent` route (agent_kwargs -> agent_view -> TQCLearner /
GatedTQCLearner) under the exact env flags train_v8.sh exports, on the v8
observation space

    {'goal': Box(4,), 'pixels': Box((7, 7, 768)), 'mode': Box((1,)),
     'state': Box((9,))}

and asserts the three things a unit test cannot see together:

1. the conv compressor is really in BOTH learners' actor AND critic params
   (`encoder_pixels`), and a fake forward through the documented trunk --
   PixelMultiplexer with the v8 conv stack -- produces the documented width:
   vision 512 + one 512 projection per flat key = the 2048 concat;
2. raw VC-1 token magnitudes are survivable. In the jaxrl2 this repo pins
   (carla-rl/dependencies/jaxrl2, submodule 56eb813 -- the copy jaxrl2.__file__
   resolves to), every normalise-if-max>1 -> /255 branch is COMMENTED OUT:
   pixel_multiplexer.py:113-117 and the sibling multiplexers carry the dead
   cond, and D4PGEncoder.__call__ has no /255 at all -- verified against the
   LOADED BYTECODE, which contains no 255 constant, and against a scaling
   experiment: standalone, the conv stack is linear (a 2550x input scales its
   output 2550x). Tokens therefore reach the conv stack at raw magnitude.
   (End to end the image branch is still input-scale-invariant -- a 2550x
   input produced the same output -- but through the post-conv
   Dense+LayerNorm, not through any /255.) The obs fixture carries raw
   magnitudes (well above 1) and every output and loss is asserted finite.
   This pins finiteness at production magnitudes, NOT that /255 normalisation
   happens;
3. a one-update smoke on a mixed batch (goal rows first + explore rows) runs
   on the real jitted path with `augument is False` (AGENT_AUGMENT_PIXELS=0),
   and every loss it reports is finite.

Each learner's init lives in its own test/fixture (measured ~3.5 s plain,
~7 s gated, CPU); the fixtures are module-scoped so one init serves both the
construction assertions and the update smoke.

THE COHERENCE SANITY (not a duplicate of Task 5's dedicated tests): the same
v8 flag set passes both gates -- gated_config.check_gated_run (env-dict form)
and vae_goal_wrapper.check_gated_coherence (module constants under a reload).
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

jax = pytest.importorskip("jax")
pytest.importorskip("jaxrl2")
pytest.importorskip("gymnasium")

from flax import linen as nn  # noqa: E402
from flax.core.frozen_dict import freeze  # noqa: E402
from gymnasium import spaces  # noqa: E402

import train_carla_tqc  # noqa: E402

# The v8 exports, exactly as train_v8.sh sets them (Task 5). GATED is added
# only on the gated route, so the plain-TQC construction exercises the same
# kwargs dict minus the GatedTQCLearner extras.
V8_FLAGS = {
    "AGENT_ENCODER": "d4pg",
    "CNN_FEATURES": "64,128,512",
    "CNN_FILTERS": "3,3,1",
    "CNN_STRIDES": "2,2,1",
    "AGENT_LATENT_DIM": "512",
    "AGENT_AUGMENT_PIXELS": "0",
    "BUFFER_PIXELS_FP16": "1",
    # the obs layout: the (7,7,768) pixels key, the 9-d state block, no
    # vector block, the mode key on (gating needs it in the observation)
    "OBS_LATENT_AS_PIXELS": "1",
    "OBS_STATE_COLLISION": "0",
    "OBS_WITH_STATE": "1",
    "OBS_WITH_VECTOR": "0",
    "OBS_WITH_MODE": "1",
}

# Everything this module touches, so the ambient environment can neither leak
# IN (a stale GATED=1 or OBS_WITH_COLLISION=1 would change the obs layout or
# the learner route) nor leak OUT (OBS_LATENT_AS_PIXELS=1 in particular must
# not survive into the files that follow).
_CONTROLLED = sorted(set(V8_FLAGS) | {
    "GATED", "GATED_ONE_CRITIC", "OBS_WITH_GOAL", "OBS_WITH_COLLISION",
    "DISCOUNT", "EXPLORE_DISCOUNT", "GATED_ALLOW_DISCOUNTED_SORB",
    "GATED_GOAL_REWARD", "PERCEPTION", "GATED_ONE_CRITIC",
    "TQC_TARGET_MIN", "TQC_TARGET_MAX", "TEMP_LR", "TARGET_ENTROPY",
    "NUM_QS", "N_QUANTILES", "TOP_QUANTILES_DROP",
})

_TRUNK = 2048  # vision 512 + one 512 projection per flat key (goal/mode/state)


def _obs_space():
    return spaces.Dict({
        "goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "pixels": spaces.Box(-np.inf, np.inf, (7, 7, 768), np.float32),
        "mode": spaces.Box(0.0, 1.0, (1,), np.float32),
        "state": spaces.Box(-np.inf, np.inf, (9,), np.float32)})


def _act_space():
    return spaces.Box(np.array([-1.0, 0.0], np.float32),
                      np.array([1.0, 1.0], np.float32))


def _set_v8_env(extra=None):
    """The v8 flags and nothing else: pop the controlled keys first so the
    ambient environment cannot leak into the learner under construction."""
    for k in _CONTROLLED:
        os.environ.pop(k, None)
    os.environ.update(V8_FLAGS)
    if extra:
        os.environ.update(extra)


@pytest.fixture(scope="module")
def v8_env_guard():
    """Save and restore the ambient env around the whole module."""
    saved = {k: os.environ.get(k) for k in _CONTROLLED}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def plain_agent(v8_env_guard):
    """TQCLearner, built by the production build_agent under the v8 flags."""
    _set_v8_env()
    yield train_carla_tqc.build_agent(_obs_space(), _act_space(), 0)


@pytest.fixture(scope="module")
def gated_agent(v8_env_guard):
    """GatedTQCLearner, the trainer's GATED=1 route, same flags + GATED."""
    _set_v8_env(extra={"GATED": "1"})
    yield train_carla_tqc.build_agent(_obs_space(), _act_space(), 0)


# --------------------------------------------------------------- helpers ----

def _kernel_shapes(params):
    """(name, shape) of every parameter array under a flax param tree."""
    out = []

    def walk(tree, prefix=""):
        for k, v in tree.items():
            if hasattr(v, "keys"):
                walk(v, prefix + "/" + k)
            else:
                out.append((prefix + "/" + k, tuple(np.asarray(v).shape)))

    walk(params)
    return out


def _assert_compressor_params(agent, critics, action_dim):
    """The conv compressor exists in the actor AND every critic's params, with
    the documented widths read from real parameter shapes -- not hard-coded
    numbers that drift, but the shapes the v8 geometry states: last conv out
    512, the post-conv Dense (512, 512), one (key_dim, 512) per flat key.

    The CRITIC's downstream ensemble is vmapped over the num_qs heads, so its
    first Dense kernel is (num_qs, in, out): the in-features live on the
    SECOND-to-last axis, and they are trunk + actions (the critic consumes the
    concatenated obs features AND the action).
    """
    for kind in ("_actor",) + tuple(critics):
        params = getattr(agent, kind).params
        names = [n for n, _ in _kernel_shapes(params)]
        assert "encoder_pixels" in " ".join(names), kind
        px = dict((name, shape) for name, shape in _kernel_shapes(params))
        # vision trunk: last conv emits 512 channels; the projection after it
        # is (512, 512)
        assert px["/encoder_pixels/Conv_2/kernel"] == (1, 1, 128, 512)
        assert px["/encoder_pre_latent/kernel"] == (512, 512)
        # one 512-wide projection per flat key, sized by the key's own dim
        assert px["/encoder_goal/kernel"] == (4, 512)
        assert px["/encoder_mode/kernel"] == (1, 512)
        assert px["/encoder_state/kernel"] == (9, 512)
        # THE CONCAT: the widest kernel under /network must take all trunk
        # inputs -- 2048 obs features for the actor, 2048 + actions for the
        # critic. This is the trunk width read off the production params, not
        # a restatement of the spec.
        net_widths = [s[-2] for n, s in _kernel_shapes(params["network"])
                      if len(s) >= 2 and n.endswith("/kernel")]
        want = _TRUNK + (0 if kind == "_actor" else action_dim)
        assert max(net_widths) == want, (kind, sorted(set(net_widths)))


def _batch(n=8, seed=0):
    """A mixed frozen batch, goal rows FIRST (the mode-split convention)."""
    rng = np.random.default_rng(seed)

    def obs(n_rows):
        # Raw VC-1 token magnitudes: O(1), so well above the 0-1 range a
        # /255-normalised image would carry -- see module docstring point 2.
        pixels = (rng.normal(size=(n_rows, 7, 7, 768)) * 2.0).astype(np.float32)
        return {"goal": rng.normal(size=(n_rows, 4)).astype(np.float32),
                "pixels": pixels,
                "mode": None,
                "state": rng.normal(size=(n_rows, 9)).astype(np.float32)}

    # goal rows FIRST, then explore -- the mode-split buffer's layout
    modes = np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)])
    o, o_next = obs(n), obs(n)
    o["mode"] = modes.reshape(n, 1).astype(np.float32)
    o_next["mode"] = modes.reshape(n, 1).astype(np.float32)
    return freeze({
        "observations": o, "next_observations": o_next,
        "actions": rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32),
        "rewards": (rng.normal(size=(n,)) * 5.0).astype(np.float32),
        "masks": np.ones((n,), np.float32),
        "dones": np.zeros((n,), np.float32)})


def _assert_finite_info(info, *required):
    for key in required:
        assert key in info, key
    for key, v in info.items():
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        assert np.isfinite(x), f"{key}={x} is not finite"


# --------------------------------------------------- Step 1: the agents ----

def test_tqc_learner_initialises_with_the_compressor_under_the_v8_flags(
        plain_agent):
    """Test A, plain route: TQCLearner under the v8 flags -- compressor in
    actor AND critic, raw token magnitudes forward to finite outputs, and the
    documented action shape."""
    from jaxrl2.agents import TQCLearner

    agent = plain_agent
    assert type(agent) is TQCLearner
    # the brief's pin, on the path the trainer actually uses
    assert agent.augument is False
    _assert_compressor_params(agent, ("_critic",), _act_space().shape[0])

    # ONE FAKE FORWARD, both heads, at raw token magnitudes (no /255 exists in
    # this jaxrl2's live path -- see module docstring point 2, so this is what
    # the conv stack really receives). Actions must match the action space;
    # the critic's quantiles must be finite.
    obs = train_carla_tqc.agent_view(_obs_space().sample())
    act = agent.eval_actions(obs)
    assert act.shape == _act_space().shape
    assert np.all(np.isfinite(np.asarray(act)))
    sample = agent.sample_actions(obs)
    assert sample.shape == _act_space().shape
    assert np.all(np.isfinite(np.asarray(sample)))
    batch = train_carla_tqc.agent_view(_batch(n=1))
    qs = agent._critic.apply_fn({"params": agent._critic.params},
                                batch["observations"], batch["actions"])
    # one row in -> the vmapped ensemble's (num_qs, B, n_quantiles) grid out
    assert np.asarray(qs).shape[1] == 1
    assert np.all(np.isfinite(np.asarray(qs)))


def test_a_fake_forward_through_the_documented_trunk_produces_2048():
    """The brief's trunk assertion, as a forward, not just param shapes: the
    exact PixelMultiplexer/D4PGEncoder config the v8 kwargs select, with a
    pass-through network, applied to one obs. Output width must be the
    documented concat -- vision 512 + one 512 per flat key (goal, mode,
    state)."""
    from functools import partial

    from jaxrl2.networks.encoders.d4pg_encoder import D4PGEncoder
    from jaxrl2.networks.pixel_multiplexer import PixelMultiplexer

    class _Identity(nn.Module):
        @nn.compact
        def __call__(self, x, *args, **kwargs):
            return x

    # The class, not an instance: PixelMultiplexer names each key's encoder
    # submodule by calling the class with `name=`, exactly as TQCLearner
    # passes its partial'd D4PGEncoder.
    trunk = PixelMultiplexer(
        encoder=partial(D4PGEncoder, features=(64, 128, 512),
                        filters=(3, 3, 1), strides=(2, 2, 1)),
        network=_Identity(), latent_dim=512)
    obs = train_carla_tqc.agent_view(_obs_space().sample())
    params = trunk.init(jax.random.PRNGKey(0), obs)
    # one unbatched obs in -> one 2048-wide trunk vector out
    y = trunk.apply(params, obs)
    assert np.asarray(y).shape == (_TRUNK,)
    assert np.all(np.isfinite(np.asarray(y)))


def test_gated_learner_initialises_with_the_compressor_under_the_v8_flags(
        gated_agent):
    """Test A, gated route: GatedTQCLearner, same v8 flags + GATED=1. Both
    critics must carry the compressor; the mode bit is the obs key "mode",
    which the compressor's per-key projections receive like any flat key."""
    from vae_nav.gated_tqc_learner import GatedTQCLearner

    agent = gated_agent
    assert type(agent) is GatedTQCLearner
    assert agent.augument is False
    _assert_compressor_params(agent, ("_critic", "_critic_e"),
                              _act_space().shape[0])

    obs = train_carla_tqc.agent_view(_obs_space().sample())
    act = agent.eval_actions(obs)
    assert act.shape == _act_space().shape
    assert np.all(np.isfinite(np.asarray(act)))


# ------------------------------------------ Step 2: the one-update smoke ----

def test_plain_update_smoke_on_a_mixed_batch(plain_agent):
    """One real jitted update, goal rows + explore rows mixed, through the
    trainer's own boundary (agent_view then update). Every reported loss
    finite, and augument False on the path actually used."""
    agent = plain_agent
    assert agent.augument is False
    info = agent.update(train_carla_tqc.agent_view(_batch()))
    _assert_finite_info(info, "critic_loss", "actor_loss")
    # the smoke executed a real update step and reports per-sample TD errors
    # (this asserts the step ran and returned diagnostics -- NOT that any
    # parameter moved, which a single update's random-init gradient could not
    # prove either way)
    assert np.asarray(info["td_errors"]).size > 0


def test_gated_update_smoke_on_a_mixed_batch(gated_agent):
    """The gated route's smoke: the same mixed batch through GatedTQCLearner.
    Both critics report finite losses, the mode mask sees both modes, and
    augument is False on this path too."""
    agent = gated_agent
    assert agent.augument is False
    info = agent.update(train_carla_tqc.agent_view(_batch()))
    _assert_finite_info(info, "critic_loss", "critic_loss_e", "actor_loss")
    assert float(info["frac_goal"]) == 0.5  # the mixed batch really is mixed
    assert agent._updates == 1


# ---------------------------------------- the coherence sanity (optional) ----

def test_the_v8_env_passes_both_coherence_checks(v8_env_guard):
    """The v8 flag set is a PERMITTED gated configuration: both gates accept
    it. (The dedicated refusal tests live in Task 5's files; this only pins
    that the config this whole stack builds actually passes them.)

    check_gated_run takes an env dict, so no reload is needed there.
    check_gated_coherence reads module constants, so it needs a reload under
    the v8 obs flags -- restored to defaults afterwards, since this file sorts
    before the test files that hold their own imports of the module.
    """
    import vae_nav.gated_config as gated_config
    import vae_nav.vae_goal_wrapper as wrapper

    run_env = dict(V8_FLAGS)
    run_env.update(GATED="1", PERCEPTION="vc1",
                   DISCOUNT="0.99", GATED_ALLOW_DISCOUNTED_SORB="1")
    gated_config.check_gated_run(env=run_env)  # no raise

    # the wrapper constants are import-time state: reload under the v8 obs
    # flags, assert, then put the defaults back.
    _set_v8_env()
    try:
        m = importlib.reload(wrapper)
        m.check_gated_coherence()  # no raise
        assert m.WITH_STATE and not m.STATE_COLLISION
    finally:
        for k in V8_FLAGS:
            os.environ.pop(k, None)
        importlib.reload(wrapper)
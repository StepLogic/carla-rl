"""TQC (+ optional episode HER) on CARLA, with a VAE latent for perception.

Configured entirely by ENVIRONMENT VARIABLES, matching the peer Habitat
trainer's convention so configs transfer by name:

    RL_ALGO=tqc            tqc | drq
    EPISODE_HER=1          opt-in hindsight relabelling
    VAE_CKPT=path.pkl      frozen lane+obstacle VAE; empty = zero-latent ablation
    TOWN_LIST=path.json    frozen train/held-out split (REQUIRED, no default glob)
    SEED=42  MAX_STEPS=200000  POLICY_FOLDER=...
    K_GOAL=600  GOAL_THRESHOLD=2.0  K_COLLISION=5  K_LIVING=0

TWO PLACES THIS DELIBERATELY DIVERGES FROM THE SETUP IT COPIES:

1. EVAL IS SEPARATELY SEEDED. Their run_evaluation reuses the SAME env object
   and calls reset() with no seed, so eval episodes are drawn from the same
   running RNG as training. Two independent peers flagged this class of bug --
   the other measured two eval paths disagreeing by up to 0.6 success rate on
   partially-learned tasks. Here eval builds its own env and replays a FIXED
   list of (town, start, goal, seed) tuples, so the same checkpoint scores the
   same number twice and two checkpoints are compared on identical episodes.

2. HELD-OUT TOWNS ARE STRUCTURALLY UNREACHABLE FROM TRAINING. Their trainer
   globbed every scene when SCENE_LIST was unset, and trained on its own eval
   maps with nothing in the logs to show it. load_split() has no default and
   assert_not_heldout() guards every town selection.
"""
import os
import pickle
import sys
import time
from datetime import datetime

# DIAGNOSTIC (2026-09-27): every v18 / v18ap Town01 block dies with
#   terminate called after throwing an instance of 'carla::client::TimeoutException'
# -- an ABORT raised from a C++ thread, which leaves no Python traceback, and
# apport discards the core. faulthandler prints every thread's Python stack on
# SIGABRT/SIGSEGV, so the next occurrence names the call that was waiting on the
# simulator. Changes nothing else.
import faulthandler
faulthandler.enable(all_threads=True)

import numpy as np
from collections.abc import Mapping

# distrax 0.1.5 against jax 0.6.2: distrax's jacobian inference reaches for
# jax.core.Var, which JAX removed in 0.6.0. Only the rescaled-tanh policy path
# touches it -- i.e. exactly the path taken when action_space is passed, which
# is the path we need. Aliasing the symbol is enough; the class moved, it did
# not change. Must run BEFORE jaxrl2 imports distrax.
import jax  # noqa: E402
import jax.extend.core  # noqa: E402
if not hasattr(jax.core, "Var"):
    jax.core.Var = jax.extend.core.Var

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from vae_nav.core_state import CoreStateWrapper  # noqa: E402
from vae_nav.episode_her import EpisodeRelabelBuffer  # noqa: E402
from vae_nav.eval_video import EvalVideoRecorder  # noqa: E402
from vae_nav.expert_oracle import ExpertOracle, BasicAgentExpert, should_use_expert  # noqa: E402
from vae_nav.latent_source import LatentSource  # noqa: E402
from vae_nav.latent_source import from_env as latent_from_env  # noqa: E402
from vae_nav.sparse_reward import (K_COLLISION, K_DIST,  # noqa: E402
                                   K_DIST_GATE_MODE,
                                   K_DIST_TARGET_SPEED, K_GOAL, K_LANE,
                                   K_DIST_DELTA_MODE,
                                   K_SPEED, SPEED_LIMIT_KMH,
                                   K_LIVING, GOAL_THRESHOLD, SparseGoalReward,
                                   tqc_kappa)
from vae_nav.town_split import assert_not_heldout, load_split  # noqa: E402
from vae_nav.vec_env import ThreadedVecEnv  # noqa: E402
from vae_nav.policy_io import (agent_state, load_agent_state,  # noqa: E402
                               param_fingerprint)
from vae_nav.vae_goal_wrapper import VAEGoalWrapper  # noqa: E402

SEED = int(os.environ.get("SEED", 42))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 200_000))
MAX_EPISODE_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", 1000))
# GATED=1 selects the third ablation arm: one policy serving both the
# goal-reaching and the goal-free task, switched by a commanded mode bit.
# Read once here because the replay insert site is on the hot path.
GATED = os.environ.get("GATED", "0") == "1"
START_TRAINING = int(os.environ.get("START_TRAINING", 5000))
EVAL_INTERVAL = int(os.environ.get("EVAL_INTERVAL", 25_000))
N_EVAL_EPISODES = int(os.environ.get("N_EVAL_EPISODES", 60))
UTD_RATIO = int(os.environ.get("UTD_RATIO", 4))
MINIBATCH = int(os.environ.get("MINIBATCH_SIZE", 256))
POLICY_FOLDER = os.environ.get("POLICY_FOLDER", "carla_tqc_vae")

# Fraction of EPISODES driven by the scripted route-following oracle. Under a
# sparse reward this is not a nicety: a random policy in CARLA essentially never
# reaches a goal tens of metres away, so without expert episodes the buffer
# holds only zero-reward rows and the critic has nothing to fit. Their Habitat
# trainer carries the same 0.2 default for an easier task.
EXPERT_PROB = float(os.environ.get("EXPERT_PROB", 0.2))

# One env per TRAINING TOWN, each pinned to its own CARLA server. The ports are
# supplied by the queue, which starts as many servers as the card can hold and
# reports what it actually managed -- so N_ENVS follows the hardware rather than
# a hardcoded hope.
CARLA_PORTS = [int(p) for p in
               os.environ.get("CARLA_PORTS", os.environ.get("CARLA_PORT", "2000")).split(",")
               if p.strip()]


def make_expert(env, action_space):
    """Build an oracle against the live CARLA actors, or None if unavailable.

    Returns None rather than raising: an expert that cannot be constructed
    should cost the run its seeding, not abort it. The message is loud because
    silently training with EXPERT_PROB=0.2 and no expert would look like a
    learning failure rather than a configuration one.
    """
    core = getattr(getattr(env, "unwrapped", env), "core", None)
    hero = getattr(core, "hero", None)
    if hero is None:
        print("[Expert] WARNING: no hero actor reachable; expert disabled. "
              "Under a sparse reward this usually means zero successful "
              "episodes ever enter the buffer.")
        return None
    try:
        # THE ORACLE'S SPEED TARGET MUST BE REACHABLE, or its PID saturates.
        # ExpertOracle defaults to 25 km/h and this passed nothing, while the
        # env runs TARGET_SPEED=20 at MAX_THROTTLE=0.4 -- a cap the config
        # measures at 22.4 km/h when throttle is 0.6, so 25 is unreachable and
        # the longitudinal PID never lifts for a corner. Measured with the
        # default: oracle collided on 36% of episodes, 0.000 success.
        _tgt = float(os.environ.get("EXPERT_TARGET_SPEED",
                                    os.environ.get("TARGET_SPEED", 20.0)))
        # EXPERT_KIND=basic uses CARLA's own BasicAgent (see BasicAgentExpert
        # for the benchmark that justified it); anything else keeps the
        # hand-rolled ExpertOracle so existing runs are unchanged.
        _cls = (BasicAgentExpert
                if os.environ.get("EXPERT_KIND", "oracle").lower() == "basic"
                else ExpertOracle)
        # NAME THE EXPERT IN THE LOG. Which class drove is otherwise only
        # inferable from its failure rates, and that inference was wrong three
        # times tonight.
        print(f"[Expert] kind={_cls.__name__} target_speed={_tgt:g} km/h",
              flush=True)
        return _cls(hero, hero.get_world().get_map(), action_space,
                    target_speed_kmh=_tgt)
    except Exception as e:  # noqa: BLE001
        print(f"[Expert] WARNING: could not build oracle ({e}); expert disabled.")
        return None


def wants_one_critic():
    """THE ONE-CRITIC ROUTE: GATED=1 AND GATED_ONE_CRITIC=1.

    The 2-mode image-goal pairing (IMAGE_GOAL_TASK=1 AND GATED=1, which
    check_gated_run requires to run with GATED_ONE_CRITIC=1) trains a UNIFIED
    reward through ONE critic, so the two-critic GatedTQCLearner is the wrong
    learner for it and the ModeSplitBuffer is the wrong buffer. Extracted as a
    function so the branch condition is pinned by a test
    (tests/test_gated_one_critic_route.py) and checkable without a simulator,
    exactly as _two_mode_image_goal is. Reads the environment LIVE rather than
    the module-level GATED constant: the routing must follow the environment
    the caller sees, not the one at import.
    """
    return (os.environ.get("GATED", "0") == "1"
            and os.environ.get("GATED_ONE_CRITIC", "0") == "1")


def agent_encoder():
    """The conv-encoder selection, D2 of the v8 design.

    "none" (the default) is today's behaviour exactly: no encoder kwarg is
    passed, so TQCLearner keeps its own default and every existing run is
    unchanged. "d4pg" routes image-like obs keys (the (7, 7, 768) VC-1 token
    grid published under `pixels`) through jaxrl2's D4PG conv trunk. Anything
    else exits: a typo here would otherwise silently fall back to "none" --
    TQCLearner's `else` branch -- and the run would look healthy while
    training a placeholder for perception.
    """
    enc = os.environ.get("AGENT_ENCODER", "none").strip().lower()
    if enc not in ("none", "d4pg"):
        raise SystemExit(
            f"AGENT_ENCODER={enc!r} is not supported; use 'none' or 'd4pg'.")
    return enc


def _int_list(env_name, default):
    """A comma-separated int list from the environment, spaces stripped."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return tuple(default)
    return tuple(int(part) for part in raw.split(",") if part.strip())


def _latent_dims(env_name="AGENT_LATENT_DIMS"):
    """Per-key projection widths for PixelMultiplexer, as a tuple of pairs.

    "latent:768"           -> (("latent", 768),)            one Dense
    "latent:512:256"       -> (("latent", (512, 256)),)      a stack
    "latent:768,state:32"  -> (("latent", 768), ("state", 32))

    SHARED SPECS (user 2026-09-26: "make image latent goal latent same size").
    Keys joined with '+' get ONE spec, expanded to an identical entry each:

    "latent+goal:512:256" -> (("latent", (512, 256)), ("goal", (512, 256)))

    Writing the width once is the point: "the image latent and the goal latent
    are the same size" is then true BY CONSTRUCTION, not because two numbers
    were typed to match and stayed matched through later edits.

    Empty or unset returns (), which the caller reads as "pass nothing", so the
    learner keeps its own latent_dim for every key and existing runs are
    unchanged. Order is preserved so the log line is deterministic.

    Raises on a malformed entry rather than falling back: a silently ignored
    width is how AGENT_LATENT_DIM=512 came to mean 50 for four whole runs.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return ()
    out, seen = [], set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"{env_name}={raw!r}: expected 'key:width' pairs, "
                             f"got {part!r}")
        keyspec, _, widthspec = part.partition(":")
        keys = [k.strip() for k in keyspec.split("+")]
        if not all(keys):
            raise ValueError(f"{env_name}={raw!r}: empty key in {part!r}")
        widths = []
        for tok in widthspec.split(":"):
            tok = tok.strip()
            if not tok:
                raise ValueError(f"{env_name}={raw!r}: empty width in {part!r}")
            w = int(tok)
            if w <= 0:
                raise ValueError(f"{env_name}={raw!r}: width must be positive, "
                                 f"got {w} in {part!r}")
            widths.append(w)
        # A single width stays an INT, so the one-layer case keeps the original
        # parameter names and old checkpoints still load.
        spec = widths[0] if len(widths) == 1 else tuple(widths)
        for k in keys:
            if k in seen:
                raise ValueError(f"{env_name}={raw!r}: key {k!r} set twice")
            seen.add(k)
            out.append((k, spec))
    return tuple(out)


def _target_entropy(action_space=None):
    """TARGET_ENTROPY, optionally restated in the UNIT action space.

    THE TARGET IS COMPARED AGAINST log_prob IN THE RESCALED SPACE. jaxrl2's
    TanhMultivariateNormalDiag chains an affine rescale onto the tanh
    (normal_tanh_policy.py), and its forward_log_det_jacobian contributes
        J = sum_i log(0.5 * (high_i - low_i))
    to every log_prob. So an entropy measured here is H_unit + J, and a target
    of -2.0 is NOT SAC's -dim(A) heuristic unless J == 0 -- which holds only
    for the [-1, 1]^d box.

    For this stack's box, [-0.5, 0.5] x [0, 0.4], J = log(0.5) + log(0.2)
    = -2.303, so the shipped -2.0 asks for H_unit = +0.30 where -dim(A) asks
    for -2.0. MEASURED consequence: training-time steer std 0.14 of the +-0.5
    range against 0.045 at the conventional target, i.e. ~3x the exploration
    noise, on a task whose dominant failure is lane crossings.

    TARGET_ENTROPY_UNIT=1 reads TARGET_ENTROPY as a UNIT-space target and adds
    J, so -2.0 means -dim(A) as intended. Default 0 keeps every existing run
    byte-identical. Needs the action space; without one it is a no-op, which
    is the eval path that never tunes a temperature.
    """
    te = float(os.environ.get("TARGET_ENTROPY", -2.0))
    if os.environ.get("TARGET_ENTROPY_UNIT", "0") != "1":
        return te
    if action_space is None:
        print("[TargetEntropy] TARGET_ENTROPY_UNIT=1 but no action space here; "
              f"leaving target_entropy={te:g} unchanged", flush=True)
        return te
    lo = np.asarray(action_space.low, np.float64)
    hi = np.asarray(action_space.high, np.float64)
    jac = float(np.sum(np.log(0.5 * (hi - lo))))
    print(f"[TargetEntropy] TARGET_ENTROPY_UNIT=1: unit-space {te:g} + "
          f"tanh-rescale Jacobian {jac:.3f} -> target_entropy={te + jac:.3f} "
          f"(action box low={lo.tolist()} high={hi.tolist()})", flush=True)
    return te + jac


def agent_kwargs(action_space=None):
    """The TQC build kwargs, shared by every learner route.

    THE GATED EXTRAS ARE ROUTE-CONDITIONAL. The two-critic route passes the
    kwargs only GatedTQCLearner understands (both discounts, the warm-start
    knobs, the value bounds). The one-critic route passes NONE of them:
    jaxrl2's TQCLearner takes no **kwargs and would reject every one, and its
    critic bounds arrive instead through TQC_TARGET_MIN/MAX, read at import in
    critic_updater. Everything else -- the peer's config of record, kappa
    derived from the reward -- is identical on every route.
    """
    kwargs = dict(
        actor_lr=3e-4,
        critic_lr=3e-4,
        # Their config's 1e-5 wins over the learner's own 3e-4 default.
        temp_lr=float(os.environ.get("TEMP_LR", 1e-5)),
        hidden_dims=(256, 256),
        # DISCOUNT, env-var configurable. 0.99 remains the default so every
        # other run script is unchanged.
        #
        # Under the minimum-time reward (-0.1 per step until arrival) the critic
        # IS the cost-to-go, and gamma decides whether that reading survives.
        # gamma=0.99 implies a 1/(1-gamma) = 100-step horizon, so Q saturates
        # at -10 and cannot separate distances beyond it:
        #
        #   steps to goal    undiscounted   gamma=0.99   gamma=1.0
        #            100         -10.00        -6.34       -10.00
        #            400         -40.00        -9.82       -40.00
        #           1000        -100.00       -10.00      -100.00
        #
        # Q(400) and Q(1000) differ by 0.18 at gamma=0.99 -- the critic cannot
        # tell "nearly there" from "hopeless" over exactly the range the
        # 5-113 m curriculum drives it into (a 113 m goal is ~600 steps at the
        # measured 0.189 m/step). It also explains the previous run's q = -8.70
        # against returns of -141 to -150: -10 was the ceiling, not a failure
        # to learn.
        #
        # gamma=1.0 makes Q exactly -0.1 * steps-to-goal. It is safe here ONLY
        # because every episode terminates or truncates within TTG_MAX_STEPS,
        # so the undiscounted return is bounded by construction. It would
        # diverge in a genuinely non-terminating task.
        discount=float(os.environ.get("DISCOUNT", 0.99)),
        tau=0.005,
        init_temperature=0.2,
        # Explicit, not auto. It happens to equal -dim(A) for a 2-D action
        # space, but it is pinned so a change to the action space cannot move
        # it silently.
        target_entropy=_target_entropy(action_space),
        backup_entropy=True,
        num_qs=int(os.environ.get("NUM_QS", 5)),
        num_quantiles=int(os.environ.get("N_QUANTILES", 25)),
        top_quantiles_to_drop_per_net=int(os.environ.get("TOP_QUANTILES_DROP", 2)),
        # DERIVED from the reward scale -- see sparse_reward.tqc_kappa. At
        # kappa=1 against k_goal=600 every arrival sits in Huber's linear
        # regime and Q stalls.
        kappa=float(os.environ.get("TQC_KAPPA", tqc_kappa())),
        # MUST be passed. NormalTanhPolicy defaults to low=-1, high=+1 on every
        # dimension; our action space is Box([-1, 0], [1, 1]) because throttle
        # has no reverse. Without this the policy emits negative throttle half
        # the time, the env clips it to zero, and the agent idles by
        # construction -- which under a sparse reward looks exactly like "TQC
        # cannot learn this task" rather than like a misconfigured head.
        action_space=action_space,
    )
    # THE V8 CONV COMPRESSOR IS A CONFIG, NOT A NEW MODULE (design D2).
    # jaxrl2's PixelMultiplexer already applies the conv D4PGEncoder to
    # image-like keys and per-key Dense+LayerNorm to flat keys, so the
    # compressor is exactly these kwargs. The keys are added ONLY when
    # AGENT_ENCODER selects an encoder: TQCLearner takes no **kwargs on the
    # one-critic route, and the default dict must stay byte-identical to
    # today's for every existing arm.
    _encoder = agent_encoder()
    if _encoder != "none":
        # cnn_padding stays TQCLearner's "VALID" default -- the spec's
        # 7x7x768 -> 1x1x512 geometry assumes it.
        kwargs.update(
            encoder=_encoder,
            cnn_features=_int_list("CNN_FEATURES", (32, 32, 32, 32)),
            cnn_filters=_int_list("CNN_FILTERS", (3, 3, 3, 3)),
            cnn_strides=_int_list("CNN_STRIDES", (2, 1, 1, 1)),
            latent_dim=int(os.environ.get("AGENT_LATENT_DIM", 50)),
        )
    # PER-KEY PROJECTION WIDTH (user 2026-09-26: "reduce latent to 768").
    #
    # WHAT WAS WRONG. PixelMultiplexer maps EVERY observation key through
    # Dense(width) -> LayerNorm -> tanh, where width defaults to the learner's
    # own latent_dim -- and jaxrl2's default is 50
    # (jaxrl2/agents/tqc/tqc_learner.py:185). AGENT_LATENT_DIM is passed ONLY
    # inside the branch above, which needs AGENT_ENCODER != "none". Every run
    # here has AGENT_ENCODER=none, so AGENT_LATENT_DIM=512 in the snapshots was
    # read by nothing. CONFIRMED from v13's own checkpoint parameters:
    # encoder_latent/kernel is (3840, 50), beside encoder_goal (4,50),
    # encoder_mode (1,50) and encoder_state (9,50) -- the 3840-d VC-1 pyramid
    # went through a 50-wide bottleneck and the 1-d mode bit was given the same
    # 50 dims as the pyramid.
    #
    # WHY PER-KEY AND NOT ONE GLOBAL WIDTH. Raising latent_dim alone would give
    # the 1-d mode bit and the 4-d goal block 768 dims each, concatenating to
    # 3072 before a (256,256) MLP. latent_dims overrides ONE key and leaves the
    # rest at 50, so "reduce the 3840-d latent to 768" is exactly what happens:
    # 768 + 50 + 50 + 50 = 918 into the MLP.
    #
    # OFF BY DEFAULT, so every existing arm stays byte-identical: unset means
    # the kwarg is never passed and the learner keeps its own default. The name
    # starts with AGENT_, which the train_v10.sh snapshot regex already matches,
    # so unlike TARGET_ENTROPY_UNIT this one lands in the run's .env record.
    _ldims = _latent_dims()
    if _ldims:
        kwargs["latent_dims"] = _ldims
        print(f"[AgentLatent] per-key projection widths {dict(_ldims)}; "
              f"every other key keeps the learner default "
              f"{kwargs.get('latent_dim', 50)}", flush=True)
    if os.environ.get("GATED", "0") == "1" and not wants_one_critic():
        # TWO CRITICS, and NOT for the reason the Habitat agent has two. There
        # the split was a discount mismatch, 1.0 against 0.99. Both lane arms
        # here run at 0.99, so EXPLORE_DISCOUNT defaults to the same value and
        # the split is justified instead by RETURN SCALE: the goal arm's shaped
        # reward produces returns of order 100 and the goal-free arm's order
        # 10, and kappa above is derived from the reward scale. One critic
        # fitted to both is dominated by the larger.
        # VALUE BOUNDS FOR THE GOAL CRITIC, DERIVED FROM THE REWARD so the two
        # cannot disagree. Under the SORB minimum-time reward every goal return
        # lies in [-step_cost * max_steps, 0] BY CONSTRUCTION: every step pays
        # -step_cost, arrival pays 0, and the episode cannot exceed max_steps.
        # A target outside that is unrealisable, not merely unlikely, so
        # clipping adds information rather than bias -- and it is what makes
        # gamma=1.0 safe, since V(s) = -c + V(s') has no bounded fixed point on
        # a state the agent never escapes.
        #
        # NOT the global TQC_TARGET_MIN/MAX: those are read at import in
        # critic_updater and would clip the EXPLORE critic's positive targets
        # to zero as well.
        floor = ceiling = None
        if os.environ.get("GATED_GOAL_REWARD", "sorb").lower() == "sorb":
            step_cost = float(os.environ.get("STEP_COST", 1.0))
            goal_cap = int(os.environ.get("GATED_GOAL_MAX_STEPS", 1000))
            floor = -step_cost * goal_cap
            ceiling = 0.0
        elif os.environ.get("GATED_GOAL_REWARD", "sorb").lower() == "progress":
            # Bono et al. 2023's progress reward: the SAME bounds validate()
            # checks the launch against, so the clip and the check cannot
            # disagree. TQC_VALUE_FLOOR/CEILING below still override.
            from vae_nav.gated_config import progress_value_bounds
            floor, ceiling = progress_value_bounds(os.environ)
        kwargs.update(
            discount_e=float(os.environ.get("EXPLORE_DISCOUNT",
                                            kwargs["discount"])),
            explore_actor_warmup=int(os.environ.get("EXPLORE_ACTOR_WARMUP", 0)),
            # ACTOR_FREEZE_STEPS: critics-only updates for this many updates
            # (= env steps at one update per step). For warm starts; see
            # GatedTQCLearner.actor_freeze_updates for the measurement.
            actor_freeze_updates=int(os.environ.get("ACTOR_FREEZE_STEPS", 0)),
            # ACTOR_Q_NORM=1: per-mode normalisation of the actor's -Q term so
            # the goal critic's ~750-wide range and the explore critic's range
            # pull the shared actor equally (option C, 2026-09-11).
            actor_q_norm=os.environ.get("ACTOR_Q_NORM", "0") == "1",
            # QNORM_CRITIC_ALPHA=1, only with ACTOR_Q_NORM=1: the critics'
            # entropy backups scale alpha by the same per-mode value spread
            # the actor divides Q by. Opt-in and off by default -- it changes
            # the critic target, so it is a separate experiment (see
            # GatedTQCLearner.qnorm_critic_alpha).
            qnorm_critic_alpha=(os.environ.get("ACTOR_Q_NORM", "0") == "1"
                                and os.environ.get("QNORM_CRITIC_ALPHA",
                                                   "0") == "1"),
            value_floor=float(os.environ.get("TQC_VALUE_FLOOR", floor))
            if (floor is not None or "TQC_VALUE_FLOOR" in os.environ) else None,
            value_ceiling=float(os.environ.get("TQC_VALUE_CEILING", ceiling))
            if (ceiling is not None or "TQC_VALUE_CEILING" in os.environ)
            else None,
        )
    return kwargs


def agent_view(obs):
    """The obs as the AGENT must see it, when the conv compressor is on.

    jaxrl2's D4PGEncoder flattens its last two input axes before its conv
    stack (`jnp.reshape(x, (*x.shape[:-2], -1))`, d4pg_encoder.py:30), so the
    (7, 7, 768) token grid would reach the convolutions as a (7, 5376)
    rank-2 array and the network construction itself dies ("conv_general_
    diluted lhs and rhs must have the same number of dimensions"). With a
    trailing singleton the flatten is a no-op and the conv stack sees exactly
    the geometry design D2 states: H=7, W=7, C=768, valid padding, strides
    2,2,1 -> 1x1x512. The view is pure (numpy adds an axis without copying)
    and IDEMPOTENT, applied at every agent boundary -- construction, acting,
    eval and update -- so the learner never sees another shape, while the
    env, the observation space and the replay buffer all keep publishing
    (7, 7, 768) exactly as design D1 specifies.

    Inert unless AGENT_ENCODER selects an encoder and the obs carries a
    `pixels` key, so every existing arm's inputs are untouched.
    """
    if agent_encoder() == "none" or not isinstance(obs, Mapping):
        return obs
    rebuilt = {}
    changed = False
    for key, value in obs.items():
        if key == "pixels":
            px = np.asarray(value)
            if px.shape[-1] != 1:
                value = px[..., None]
                changed = True
        elif isinstance(value, Mapping):
            nested = agent_view(value)
            changed = changed or nested is not value
            value = nested
        rebuilt[key] = value
    if not changed:
        return obs
    return rebuilt if isinstance(obs, dict) else type(obs)(rebuilt)


def apply_augment_override(learner):
    """The token-grid augmentation override, applied POST-CONSTRUCTION.

    TQCLearner hardcodes `self.augument = not encoder is None` inside its
    constructor, and both the plain and the Gated update paths forward that
    attribute into the jitted step (`augument=self.augument`), which applies
    DrQ's batched_random_crop to the image-like obs. Cropping a (7, 7, 768)
    ViT token grid DROPS TOKENS -- each position is a distinct semantic unit,
    not a pixel -- so with AGENT_AUGMENT_PIXELS=0 the attribute is flipped
    here, after construction, on every route that built a learner with a conv
    encoder. NO edit to the jaxrl2 repo, which is shared and never modified.

    With AGENT_AUGMENT_PIXELS unset (the default "1") this is a no-op and
    today's behaviour is preserved bit for bit.
    """
    if (agent_encoder() != "none"
            and os.environ.get("AGENT_AUGMENT_PIXELS", "1") == "0"):
        learner.augument = False
        print("[Algo] AGENT_AUGMENT_PIXELS=0: augmentation OFF -- a random "
              "crop of the (7,7,768) token grid would drop tokens")
    return learner


def fp16_pixel_obs_space(obs_space):
    """The obs space the REPLAY BUFFER stores, when BUFFER_PIXELS_FP16=1.

    Wrapping the buffer to convert at insert is NOT enough: jaxrl2's
    ReplayBuffer preallocates its storage arrays at the observation space's
    per-key dtype (_init_replay_dict, replay_buffer.py:14-25), so a float32
    pixels space would upcast every row back and the ~34 GB saving would
    silently vanish. Float16 STORAGE therefore means building the buffer
    with this space -- the env, the wrapper and the agent all keep the
    float32 (7, 7, 768) space; only the buffer's preallocated arrays change.
    Only the pixels key's dtype moves; shapes and every other key are
    identical. Flag unset (the default): the same object, unchanged.
    """
    if os.environ.get("BUFFER_PIXELS_FP16", "0") != "1":
        return obs_space
    spaces = getattr(obs_space, "spaces", None)
    if not spaces or "pixels" not in spaces:
        return obs_space
    import gymnasium.spaces as gym_spaces
    px = spaces["pixels"]
    out = dict(spaces)
    out["pixels"] = gym_spaces.Box(px.low.astype(np.float16),
                                   px.high.astype(np.float16), px.shape,
                                   np.float16)
    return gym_spaces.Dict(out)


def build_agent(observation_space, action_space, seed):
    """TQC with the peer's config of record, kappa derived from the reward."""
    from jaxrl2.agents import TQCLearner

    kwargs = agent_kwargs(action_space)
    n_drop = kwargs["top_quantiles_to_drop_per_net"] * kwargs["num_qs"]
    total = kwargs["num_quantiles"] * kwargs["num_qs"]
    print(f"[Algo] TQC num_qs={kwargs['num_qs']} "
          f"n_quantiles={kwargs['num_quantiles']} "
          f"drop_per_net={kwargs['top_quantiles_to_drop_per_net']} "
          f"(truncating {n_drop} of {total}) kappa={kwargs['kappa']:g}")
    if kwargs.get("encoder"):
        # The v7 lesson: the run banner must carry every flag the trainer
        # read, so a snapshot can be checked against the live process.
        print(f"[Algo] conv compressor: encoder={kwargs['encoder']} "
              f"cnn_features={kwargs['cnn_features']} "
              f"cnn_filters={kwargs['cnn_filters']} "
              f"cnn_strides={kwargs['cnn_strides']} "
              f"latent_dim={kwargs['latent_dim']}")
        # EQUAL LENGTHS, ASSERTED NOT ASSUMED. D4PGEncoder zips the three
        # lists, so a malformed CNN_FILTERS like "3,,1" -- _int_list drops the
        # empty segment -- would SILENTLY shorten the conv stack by one layer
        # and change the compressor's geometry instead of failing the launch.
        _feats, _filts, _strs = (kwargs["cnn_features"], kwargs["cnn_filters"],
                                 kwargs["cnn_strides"])
        if not (len(_feats) == len(_filts) == len(_strs)):
            raise SystemExit(
                f"conv compressor misconfigured: cnn_features={_feats} "
                f"cnn_filters={_filts} cnn_strides={_strs} must have equal "
                "lengths. A short list silently truncates the conv stack "
                "(D4PGEncoder zips the three lists) -- fix the CNN_* env "
                "vars.")
    # The construction obs goes through the encoder view: with the conv
    # compressor on, pixels must reach D4PGEncoder as (7, 7, 768, 1) -- see
    # agent_view. Without an encoder this is the sampled obs, unchanged.
    obs_sample = agent_view(observation_space.sample())
    if wants_one_critic():
        # ONE critic, mode-conditioned via the obs dict. Bounds come from
        # TQC_TARGET_MIN/MAX (read at import in critic_updater) -- coherence
        # refuses to start the run without them. agent_kwargs already left the
        # GatedTQCLearner-only kwargs off this route; the pops stay as a guard
        # so a future kwarg added to the shared dict cannot reach the plain
        # learner silently.
        plain = dict(kwargs); plain.pop("value_floor", None); plain.pop("value_ceiling", None)
        return apply_augment_override(
            TQCLearner(seed, obs_sample,
                       action_space.sample(), **plain))
    if os.environ.get("GATED", "0") == "1":
        from vae_nav.gated_tqc_learner import GatedTQCLearner
        return apply_augment_override(
            GatedTQCLearner(seed, obs_sample,
                            action_space.sample(), **kwargs))
    return apply_augment_override(
        TQCLearner(seed, obs_sample, action_space.sample(),
                   **kwargs))


# Shared across every TRAINING env rebuild -- see the note in make_env.
# None until the explore arm asks for it, so the goal arm never builds one.
_EXPLORE_NOVELTY = None


# --- one CARLA per process, started from Python ---------------------------
_PY_SERVER = {"proc": None, "port": None, "stop": None}


def ensure_carla(port, timeout=180.0):
    """Start ONE CARLA on `port` for this process, and reuse it thereafter.

    WHY NOT CarlaCore's own start_server. init_server() does
    `self.server_port = random.randint(15000, 32000)` and IGNORES the
    configured port, so every env built in a process launches its OWN server
    on its OWN random port. eval_policy builds two envs (a throwaway probe for
    the observation/action spaces, then one per cell), so a single evaluation
    started a server, killed it on probe.close(), and started another --
    two heavyweight boots on one GPU, the second racing the first's shutdown,
    and the client's 50 s timeout firing before it answered. That is what
    "CARLA keeps restarting" was.

    Here the server is started once, on a KNOWN port, and polled until it
    actually answers rather than slept at for a fixed 45 s. Envs then attach
    with start_server=False, so no env owns or kills it, and it is torn down
    at process exit.
    """
    import atexit
    import signal
    import socket
    import subprocess

    if _PY_SERVER["port"] == port and _PY_SERVER["proc"] is not None \
            and _PY_SERVER["proc"].poll() is None:
        return port

    def answering():
        """Ready means the RPC ANSWERS, not that the port accepts TCP.

        CARLA binds and accepts connections seconds before it can serve, so a
        bare socket check returns True too early and the client then dies on
        "time-out of 50000ms while waiting for the simulator" -- which is
        exactly what the first smoke test did. get_world() is the real test.
        """
        with socket.socket() as sk:
            sk.settimeout(1.0)
            if sk.connect_ex(("localhost", int(port))) != 0:
                return False
        try:
            import carla
            c = carla.Client("localhost", int(port))
            c.set_timeout(4.0)
            c.get_world().get_map()
            return True
        except Exception:  # noqa: BLE001
            return False

    if answering():
        # Something is already serving this port. Attaching to a server this
        # process did not start is exactly the stale-server trap, so say so
        # loudly rather than silently measuring someone else's simulator.
        print(f"[CARLA] port {port} is already serving -- attaching to a "
              f"server this process did NOT start", flush=True)
        _PY_SERVER.update(proc=None, port=port)
        return port

    root = os.environ["CARLA_ROOT"]
    quality = os.environ.get("CARLA_QUALITY", "Epic")
    # -prefernvidia: CARLA picks the first active GPU (the Intel iGPU on this
    # box) unless told otherwise. The VK_ICD/PRIME env vars already force
    # NVIDIA; the flag is the documented belt to those braces.
    # CARLA_OFFSCREEN=1 -> -RenderOffScreen. An on-screen render costs extra VRAM against a headless one
    # (run_carla_queue.sh:130 says so, and offers the same switch there), and that cost decides whether a
    # MULTI-SERVER run fits at all: MEASURED windowed at Low, two servers took 4846 + 4694 MiB of a 12.3 GB card,
    # leaving 2384 MiB for a learner that needs ~2300 -- an 84 MiB margin, against the 3500 MiB the queue reserves.
    # It also drops the live-X-session dependency: windowed, CarlaUE4 EXITS rather than falling back to headless if
    # DISPLAY is wrong or the session is locked, which an unattended long run must not depend on.
    # Default stays windowed so no existing caller changes behaviour.
    _offscreen = os.environ.get("CARLA_OFFSCREEN", "0") == "1"
    cmd = [f"{root}/CarlaUE4.sh", f"-carla-port={port}",
           f"-quality-level={quality}",
           "-RenderOffScreen" if _offscreen else "-windowed",
           "-ResX=640", "-ResY=360", "-prefernvidia"]
    # TEXTURE STREAMING POOL CAP (user 2026-09-21, two-town squeeze). UE4's
    # streaming pool auto-grows into whatever VRAM is free -- MEASURED: a
    # Town10HD server idled up to 5716 MiB on a mostly-empty card against
    # 4290 during real training, and the servers boot FIRST, so they eat the
    # space the jax learner needs and the learner OOMs (RESOURCE_EXHAUSTED on
    # a 768 KB alloc, four attempts). The cap bounds the cache, not the
    # detail: at Low/640x360 the working set sits under 2 GB, so frames are
    # expected to be identical. Env var so the .env snapshot records it.
    _pool = os.environ.get("CARLA_STREAMING_POOL")
    if _pool:
        # Shell-quoted because `shell` below joins cmd with spaces: UE4 needs
        # the cvar and its value as ONE argv entry ("r.Streaming.PoolSize 2000").
        cmd.append(f'"-ExecCmds=r.Streaming.PoolSize {_pool}"')
    # THE SERVER'S OWN OUTPUT GOES TO A FILE, NOT /dev/null. When the
    # simulator stops answering mid-run, its log is the only evidence of why
    # (Vulkan device lost, GPU OOM, a crash) -- and it was being discarded.
    # The per-frame "Invalid session: no stream available" spam is dropped in
    # the pipeline, as run_gated.sh does, so the log stays readable.
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"carla_py_{port}_{int(time.time())}.log")
    # `sed -u`, not `grep --line-buffered`: grep on this box is ugrep, whose
    # line buffering does not hold in a pipe -- the log files came out 0 bytes.
    shell = (" ".join(cmd) + " 2>&1 | sed -u '/Invalid session: no stream "
             "available/d' > " + log_path)
    print(f"[CARLA] starting: {' '.join(cmd)}  (log: {log_path})", flush=True)
    proc = subprocess.Popen(shell, shell=True, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"CARLA exited during startup "
                               f"(rc={proc.returncode})")
        if answering():
            print(f"[CARLA] ready on port {port} after "
                  f"{time.time() - t0:.0f} s", flush=True)
            break
        time.sleep(1.0)
    else:
        proc.kill()
        raise RuntimeError(f"CARLA did not answer on port {port} in {timeout}s")

    _PY_SERVER.update(proc=proc, port=port)

    def _stop():
        # KILL THE PROCESS GROUP, not the process. CarlaUE4.sh is a SHELL that
        # execs the real CarlaUE4-Linux-Shipping as a child, so proc.kill()
        # reaps the wrapper and orphans the simulator -- measured: the sweep
        # died, atexit ran, and both the sh and the binary were still alive
        # 64 s later holding port 2000, ready to be silently attached to by
        # the next run. start_new_session=True gives them their own group.
        pr = _PY_SERVER["proc"]
        if pr is None or pr.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(pr.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pr.kill()
    atexit.register(_stop)
    _PY_SERVER["stop"] = _stop
    return port


def kill_carla_server(port):
    """Kill the CARLA server on `port` so the next make_env RESPAWNS it.

    Used by the town-rotation branch. Rotation rebuilds the env, whose env
    build calls load_world -- and load_world on the LONG-LIVED shared server
    is the one operation known to crash CARLA 0.9.15 (the note at the
    rotation setup, :1248; measured live 2026-09-21: the v8 Town01->Town10HD
    switch segfaulted the server, Signal 11 in logs/carla_py_2000_*.log, and
    the trainer burned its 25 reconnect attempts against a dead port). A
    FRESH server plus the env build's single load_world is the path every
    startup already survives -- attempts 1 and 2 both built their envs
    through it.

    Kills by port, not by the _PY_SERVER slot: the slot holds only the most
    recently spawned server, and ensure_carla's ready check is what turns the
    kill into a respawn -- CARLA_START_SERVER=0 (attach-only) must not use
    this, which the rotate branch guards.
    """
    import signal
    import socket
    port = int(port)
    pat = f"-carla-port={port}"
    killed = 0
    for pid_dir in os.listdir("/proc"):
        if not pid_dir.isdigit():
            continue
        try:
            with open(f"/proc/{pid_dir}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "CarlaUE4" in cmd and pat in cmd:
            try:
                os.killpg(int(pid_dir), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            killed += 1
    if _PY_SERVER["port"] == port:
        _PY_SERVER.update(proc=None, port=None)
    if not killed:
        return
    # The port must stop ANSWERING before the caller rebuilds the env, or
    # ensure_carla attaches to the dying server instead of spawning fresh.
    t0 = time.time()
    while time.time() - t0 < 15:
        with socket.socket() as sk:
            sk.settimeout(1.0)
            if sk.connect_ex(("localhost", port)) != 0:
                return
        time.sleep(1.0)


def _two_mode_image_goal():
    """THE 2-MODE PAIRING: IMAGE_GOAL_TASK=1 AND GATED=1.

    Extracted as a function so the branch condition is pinned by a test
    (tests/test_make_env_two_mode_branch.py) and make_env's branch order is
    checkable without a simulator. GATED=1 alone must NOT select this arm --
    the frozen gated arms take their own branch below.
    """
    return (os.environ.get("IMAGE_GOAL_TASK", "0") == "1"
            and os.environ.get("GATED", "0") == "1")


def make_env(town, seed, latent_source, eval_mode=False, port=None):
    """One env, one town, one server. Wrapper order is shared with eval."""
    from rlib_integration.carla_env import CarlaEnv
    from jaxrl2.wrappers.timelimit import TimeLimit

    from src.configs.train_env_config_v4_image_goal import config as base_config

    cfg = dict(base_config)
    cfg["town"] = town

    # ATTACH to the server the queue manages; do NOT let CarlaCore spawn one.
    #
    # With start_server=True (its default) CarlaCore launches a fresh CARLA on a
    # RANDOM port in 15000-32000 for every env it builds -- and this trainer
    # rebuilds the env on every episode to rotate towns, so that is one orphaned
    # UE4 server per episode. Worse, CarlaCore.close() then calls
    # kill_all_servers(), which SIGKILLs every process with "carla" in its name
    # anywhere on the box, including servers belonging to other people.
    #
    # Setting start_server=False disables both behaviours: no spawn, no global
    # kill. The queue owns exactly one server and cleans up only that one.
    # SHORT-HORIZON GOALS. carla_core.reset_hero samples the initial goal from
    #   lenghts = range(10,80,5) if is_short_horizon else [200,230,240,150,120]
    # and is_short_horizon defaults to False -- so every episode had a goal
    # 120-240 m away. At the measured 0.22 m/step that needs 545-1091 steps
    # against a 300-step limit: the goal was PHYSICALLY UNREACHABLE, which is
    # why the policy scored zero arrivals in every run and configuration.
    #
    # The config comment claiming "4-8m range" is stale and was what I built on
    # instead of reading the sampler. Short horizon gives 10-75 m, all
    # reachable inside MAX_EPISODE_STEPS with headroom.
    cfg = dict(cfg)
    _exp = dict(cfg.get("experiment", {}))
    _hero = dict(_exp.get("hero", {}))
    _hero["is_short_horizon"] = os.environ.get("SHORT_HORIZON", "1") == "1"
    _exp["hero"] = _hero
    cfg["experiment"] = _exp

    env_cfg = dict(cfg.get("env_config", {}))
    carla_cfg = dict(env_cfg.get("carla", {}))
    # THE TOWN MUST BE SET ON THE CARLA BLOCK, NOT THE TOP LEVEL. CarlaCore is
    # built from cfg["env_config"]["carla"] and calls load_world on THAT dict
    # (carla_core.py:182-183), where the base config hardcodes "town":
    # "Town01". Without this every env loaded Town01 whatever was requested --
    # training and eval alike -- while the banner printed the town asked for.
    carla_cfg["town"] = town
    # START CARLA FROM PYTHON BY DEFAULT -- it is faster than launching the
    # shell script and waiting on a readiness probe, and CarlaCore then owns
    # the process: its close() kills the server it started (kill_own_server),
    # so a run cleans up after itself instead of leaving one behind. Was
    # hardcoded False here, which forced every caller to launch a server first.
    # CARLA_START_SERVER=0 keeps the old behaviour for attaching to one that is
    # already up.
    _port = int(port if port is not None else CARLA_PORTS[0])
    # The PYTHON side starts CARLA (faster than a shell launch plus a fixed
    # sleep), but ONCE per process on a known port -- see ensure_carla. The env
    # then attaches, so no env owns the server and none can kill it mid-run.
    if os.environ.get("CARLA_START_SERVER", "1") == "1":
        ensure_carla(_port)
    carla_cfg["start_server"] = False
    # BASE_CORE_CONFIG defaults quality_level to "Low" (carla_core.py:30), and
    # the frozen VAE was trained on EPIC frames -- at Low it is handed
    # out-of-distribution images with nothing in the logs to say so. Note
    # carla_core only passes the flag for Town01/Town04/None, so other towns
    # take the simulator's own default (Epic); this makes the intent explicit
    # wherever the flag is actually used.
    carla_cfg["quality_level"] = os.environ.get("CARLA_QUALITY", "Epic")
    carla_cfg["port"] = _port
    # CARLA_TIMEOUT: the client RPC timeout (config default 50 s). Town12's
    # load_world alone takes minutes, so the large-map stages pass 300.
    if os.environ.get("CARLA_TIMEOUT"):
        carla_cfg["timeout"] = float(os.environ["CARLA_TIMEOUT"])
    env_cfg["carla"] = carla_cfg
    cfg["env_config"] = env_cfg

    env = CarlaEnv(config=cfg)
    # FIRST: CarlaEnv returns an empty info dict; everything below reads plain
    # keys, and this is the only place that reaches into core/experiment.
    env = CoreStateWrapper(env)
    # LANE FOLLOWING uses the env's own per-step distance reward instead of the
    # sparse +600 arrival. LANE_FOLLOW=1 selects it; the sparse path is
    # untouched so existing goal-navigation runs are bit-identical.
    # THE 2-MODE IMAGE-GOAL ARM: IMAGE_GOAL_TASK=1 AND GATED=1. One reward
    # (TwoModeImageGoalReward) serves both the image-goal task and the
    # goal-free explore task, selected by the mode bit drawn at reset. Placed
    # BEFORE the single-mode IMAGE_GOAL_TASK branch -- which returns early --
    # so this pairing can never be captured by it; and before the GATED
    # branch, so the frozen two-critic gated stack (GatedReward over
    # LaneFollowReward, the mode-split learner) cannot be half-applied to an
    # arm whose design is ONE bounded critic under one reward.
    if _two_mode_image_goal():
        from vae_nav.image_goal_reward import (TwoModeImageGoalReward,
                                               image_goal_coherence)
        from vae_nav.gated_config import check_gated_run

        # BOTH GUARDS RUN HERE, before a server start and before any GPU
        # memory is taken. image_goal_coherence dispatches on the pairing and
        # demands every flag the 2-mode learner depends on (OBS_WITH_STATE,
        # OBS_WITH_MODE, VC-1, one critic, HER, the value bounds) and refuses
        # the single-mode arm's flags; check_gated_run then checks the
        # run-level split -- skipping its OBS_WITH_VECTOR requirement for this
        # pairing, whose velocity lives in the `state` block.
        image_goal_coherence()
        check_gated_run()

        env = TwoModeImageGoalReward(
            env,
            explore_prob=float(os.environ.get("EXPLORE_PROB", 0.5)),
            force_mode=(int(os.environ.get("EVAL_FORCE_MODE", 1)) if eval_mode else None),
            max_steps=(10 ** 9 if os.environ.get("NO_TIMELIMIT", "0") == "1"
                       else int(os.environ.get("MAX_EPISODE_STEPS", 1000))),
            seed=seed)
        env = VAEGoalWrapper(env, source=latent_source)
        return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    # IMAGE-GOAL ARM. One reward, one mode, VC-1 features: reach the goal
    # image's viewpoint and stop there. Placed before the GATED branch and
    # returning early, for the same reason the gated and explorer branches
    # are: so no part of either existing stack can be half-applied.
    #
    # Since 2026-09-20 the IMAGE_GOAL_TASK=1 + GATED=1 pairing is taken ABOVE
    # by the 2-mode arm; this branch is the single-mode arm only.
    if os.environ.get("IMAGE_GOAL_TASK", "0") == "1":
        # IG_MAX_STEPS (imported aliased: this file's own module-level
        # MAX_STEPS is the 200k training-step budget, a different thing).
        from vae_nav.image_goal_reward import (ImageGoalReward,
                                               image_goal_coherence)
        from vae_nav.image_goal_reward import MAX_STEPS as IG_MAX_STEPS
        image_goal_coherence()
        # The Phase 1 reward owns termination (success / simulator_invalid /
        # MAX_STEPS timeout), latches the collision flag, and publishes every
        # info key the coll key and the relabel buffer read. The env's own
        # `done` is discarded inside ImageGoalReward.
        #
        # THE REWARD'S OWN CLOCK MUST GO WITH THE TIMELIMIT, OR NO_TIMELIMIT IS
        # HALF-APPLIED. ImageGoalReward truncates at its own max_steps
        # (default IG_MAX_STEPS=1000) INDEPENDENT of the TimeLimit wrapper, so
        # under NO_TIMELIMIT=1 the banner below would claim "no step cap" while
        # every episode still died at 1000 steps. Same pattern as the GATED
        # branch: remove TimeLimit AND set the reward's cap to 10**9. Normal
        # runs keep IG_MAX_STEPS (default 1000) unchanged.
        env = ImageGoalReward(
            env,
            max_steps=(10 ** 9 if os.environ.get("NO_TIMELIMIT", "0") == "1"
                       else IG_MAX_STEPS))
        env = VAEGoalWrapper(env, source=latent_source)
        if os.environ.get("NO_TIMELIMIT", "0") == "1":
            print("[Env] IMAGE_GOAL + NO_TIMELIMIT=1: no step cap (trace "
                  "collection)", flush=True)
            return env
        return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    # GATED ARM. One policy serving BOTH the goal-reaching and the goal-free
    # task, switched by a bit drawn once per episode. Placed before the
    # goal-reward branches and returning early, for the same reason the
    # explorer branch is: so no part of the goal-conditioned stack can be
    # half-applied.
    #
    # THIS IS THE THIRD LEG OF THE ABLATION, not a replacement for either arm.
    # It exists to be compared against carla_lanefollow (goal-only) and
    # carla_lanefollow_nogoal (goal-free), so everything except the gate is
    # deliberately identical to the goal arm.
    if os.environ.get("GATED", "0") == "1":
        from vae_nav.gated_config import check_gated_run
        from vae_nav.gated_reward import GOAL, GatedReward
        from vae_nav.lane_follow_reward import LaneFollowReward

        # Refuse an incoherent run here, before a CARLA server is started and
        # before any GPU memory is taken.
        check_gated_run()

        # LaneFollowReward sits BELOW GatedReward and ITS REWARD IS DISCARDED.
        # It is present for two things GatedReward reads but does not compute:
        # info["is_success"], which carries the arrival bonus, and the goal
        # arm's own termination rule, which goal mode honours unchanged. Both
        # rewards are then pure functions of info, computed side by side.
        env = LaneFollowReward(env)
        env = GatedReward(
            env,
            explore_prob=float(os.environ.get("EXPLORE_PROB", 0.5)),
            # EVAL FORCES A MODE, GOAL BY DEFAULT. Letting the bit be drawn
            # during evaluation would report a success rate averaged over two
            # different tasks, one of which has no goal and cannot succeed at
            # reaching it.
            #
            # EVAL_FORCE_MODE=0 EVALUATES THE EXPLORE HALF INSTEAD, which a
            # gated arm needs: it is supposed to serve BOTH modes, so measuring
            # only the goal half can report an arm as better while its explore
            # behaviour has degraded, invisibly. Success is meaningless in that
            # mode and the harness refuses to report it.
            force_mode=(int(os.environ.get("EVAL_FORCE_MODE", GOAL))
                        if eval_mode else None),
            # NO_TIMELIMIT=1 (trace collection only, user decision 2026-09-12):
            # no clock on explore episodes -- they end on a crash, a crossing,
            # the 100 m arrival or the env's idle guard, never on a step cap.
            max_steps=(10 ** 9 if os.environ.get("NO_TIMELIMIT", "0") == "1"
                       else int(os.environ.get("GATED_EXPLORE_MAX_STEPS", 1000))),
            seed=seed)
        env = VAEGoalWrapper(env, source=latent_source)
        # The GOAL arm's cap. GatedReward truncates explore episodes itself at
        # its own, shorter cap, so the two modes keep the episode lengths their
        # separate arms use and the comparison stays honest.
        if os.environ.get("NO_TIMELIMIT", "0") == "1":
            print("[Env] NO_TIMELIMIT=1: TimeLimit wrapper and explore cap removed "
                  "(trace collection)", flush=True)
            return env
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        return env

    # GOAL-FREE EXPLORER. A separate arm entirely: no goal in the reward, no
    # goal in the observation, and its own memory of where it has been. Placed
    # before the goal-reward branches and returning early, so none of the
    # goal-conditioned stack can be half-applied to it.
    if os.environ.get("LANE_EXPLORE", "0") == "1":
        from vae_nav.explore_obs_wrapper import ExploreObsWrapper
        from vae_nav.image_novelty import ImageNoveltyBuffer
        from vae_nav.lane_explore_reward import (LANE_EXPLORE_MAX_STEPS,
                                                 LaneExploreReward)
        # THE BUFFER MUST OUTLIVE THE ENV, NOT JUST THE EPISODE.
        #
        # NOVELTY_PERSIST=1 stops it clearing on env.reset(). It cannot stop
        # the buffer OBJECT being replaced, and make_env is called again for
        # every eval env and on any env rebuild -- so a fresh buffer here threw
        # the memory away roughly every eval. MEASURED before this fix: it grew
        # 52 -> 1259, reset to 35, grew to 1130, reset to 9, grew to 1358,
        # reset. Three wipes in 480 episodes.
        #
        # That silently removed the whole mechanism this arm depends on: turn
        # diversity comes from the critic preferring a branch whose road is
        # unseen, and "unseen" only means anything if the memory survives.
        #
        # Eval envs get their OWN buffer: sharing the training memory would let
        # eval inherit whatever the policy happened to have explored, so two
        # evals of the same checkpoint would not score the same.
        novelty = None
        if os.environ.get("K_NOVELTY", "0.2") != "0":
            if eval_mode:
                novelty = ImageNoveltyBuffer()
            else:
                global _EXPLORE_NOVELTY
                if _EXPLORE_NOVELTY is None:
                    _EXPLORE_NOVELTY = ImageNoveltyBuffer()
                novelty = _EXPLORE_NOVELTY
        env = LaneExploreReward(env, novelty=novelty, encoder=latent_source)
        env = ExploreObsWrapper(env, source=latent_source)
        # The explorer's own step cap, not MAX_EPISODE_STEPS: the goal arm's
        # cap is tied to its horizon and means nothing here.
        env = TimeLimit(env, max_episode_steps=LANE_EXPLORE_MAX_STEPS)
        return env

    if os.environ.get("TIME_TO_GOAL", "0") == "1":
        # MINIMUM-TIME task: -0.1 per step, 0 on arrival, and any terminal
        # failure bills every remaining step so it costs exactly the same as
        # timing out. No arrival bonus, no shaping, no coefficients to balance.
        from vae_nav.time_to_goal_reward import TimeToGoalReward
        env = TimeToGoalReward(env)
    elif os.environ.get("LANE_FOLLOW", "0") == "1":
        from vae_nav.lane_follow_reward import LaneFollowReward
        env = LaneFollowReward(env)
    else:
        env = SparseGoalReward(env, k_goal=K_GOAL, goal_threshold=GOAL_THRESHOLD,
                               k_collision=K_COLLISION, k_living=K_LIVING,
                               max_episode_steps=MAX_EPISODE_STEPS)
    # Built through the SAME call in training and eval so the observation
    # cannot drift between them -- the failure their map_harness.py has to
    # guard against by staying byte-identical to the wrapper.
    env = VAEGoalWrapper(env, source=latent_source)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env


def build_eval_episodes(heldout_towns, n_episodes, seed=0):
    """A FIXED episode set: same towns, same seeds, every time.

    Frozen from a dedicated RNG that is never touched by training, so an eval
    at step 50k and one at 200k are the same episodes and the comparison is
    about the policy rather than about the draw.
    """
    rng = np.random.default_rng(seed)
    per_town = max(1, n_episodes // max(1, len(heldout_towns)))
    eps = []
    for town in heldout_towns:
        for k in range(per_town):
            eps.append({"town": town, "seed": int(rng.integers(0, 2 ** 31 - 1)),
                        "index": len(eps)})
    print(f"[Eval] fixed set: {len(eps)} episodes over {heldout_towns} "
          f"(rng seed {seed}, never shared with training)")
    return eps


def run_explore_evaluation(agent, town, n_episodes, latent_source, step=0,
                           seed0=7000, max_steps=None):
    """Evaluate the EXPLORE half. Returns (mean_dist_m, ends, rows).

    WHY THIS EXISTS. The periodic eval forces GOAL, so a gated arm was judged
    on half of what it does, and an explore arm can degrade invisibly -- v5's
    learned to cross a line at ~step 37 to cap a terminal loss while its goal
    numbers still looked fine. There is NO success rate here: explore has no
    goal. What is reported is what the explore reward pays for -- distance
    driven, steps survived, and WHY the episode ended.

    EVAL_FORCE_MODE is set to EXPLORE around make_env, which reads it at call
    time, and restored afterwards so the goal eval is unaffected.
    """
    import random as _random
    from collections import Counter

    from vae_nav.gated_reward import EXPLORE, EXPLORE_GOAL_DIST_M

    prev = os.environ.get("EVAL_FORCE_MODE")
    os.environ["EVAL_FORCE_MODE"] = str(EXPLORE)
    env, rows = None, []
    try:
        env = make_env(town, seed0, latent_source, eval_mode=True)
        cap = int(max_steps or os.environ.get("GATED_EXPLORE_MAX_STEPS", 1000))
        for i in range(n_episodes):
            sd = seed0 + i
            _random.seed(sd)
            np.random.seed(sd % (2 ** 32))
            obs, info = env.reset(seed=sd)
            done, steps = False, 0
            while not done and steps < cap:
                obs, _r, term, trunc, info = env.step(agent.eval_actions(agent_view(obs)))
                steps += 1
                done = bool(term or trunc)
            rows.append({"steps": steps,
                         "dist_m": float(info.get("explore_dist_m", 0.0) or 0.0),
                         "end": str(info.get("explore_end", "?"))})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:                      # noqa: BLE001
                pass
        if prev is None:
            os.environ.pop("EVAL_FORCE_MODE", None)
        else:
            os.environ["EVAL_FORCE_MODE"] = prev

    dists = [r["dist_m"] for r in rows]
    ends = Counter(r["end"] for r in rows)
    mean_d = float(np.mean(dists)) if dists else 0.0
    reached = ends.get("reached", 0) / max(1, len(rows))
    print(f"[Eval] EXPLORE n={len(rows)} dist mean={mean_d:.1f} m "
          f"max={(max(dists) if dists else 0.0):.1f} m "
          f"steps mean={(np.mean([r['steps'] for r in rows]) if rows else 0):.0f} "
          f"reached_{int(EXPLORE_GOAL_DIST_M)}m={reached:.2f} "
          f"ends={dict(ends)}", flush=True)
    return mean_d, dict(ends), rows


def run_evaluation(agent, episodes, latent_source, identity, step=0,
                   out_dir=None):
    """Evaluate on the held-out towns. Never called with a training env."""
    LatentSource.assert_matches(identity, latent_source)

    recorder = None
    if out_dir and os.environ.get("EVAL_VIDEO", "1") == "1":
        recorder = EvalVideoRecorder(out_dir, step, latent_source=latent_source)

    by_town, results = {}, []
    env, current = None, None
    try:
        for ep in episodes:
            if ep["town"] != current:
                if env is not None:
                    env.close()
                env = make_env(ep["town"], ep["seed"], latent_source,
                               eval_mode=True)
                current = ep["town"]
            # THE SEED MUST REACH THE GOAL SAMPLER. CarlaEnv.reset drops its
            # kwargs and carla_core.reset_hero draws spawn and goal from Python's
            # GLOBAL `random`, so env.reset(seed=...) alone fixed nothing and
            # every eval was a fresh draw of 60 goals. Measured 2026-09-11:
            # consecutive evals drew median routes of 40/26/25/38 m and scored
            # 0.45/0.73/0.78/0.13 -- the distance draw, not the policy, moved.
            import random as _random
            _random.seed(int(ep["seed"]))
            np.random.seed(int(ep["seed"]) % (2 ** 32))
            obs, info = env.reset(seed=ep["seed"])
            if recorder is not None:
                recorder.start_episode(ep["index"], ep["town"])
                recorder.record(info.get("rgb"), info)
            done, steps, success = False, 0, False
            while not done and steps < MAX_EPISODE_STEPS:
                action = agent.eval_actions(agent_view(obs))
                obs, _r, term, trunc, info = env.step(action)
                done, steps = bool(term or trunc), steps + 1
                success = success or bool(info.get("is_success", False))
                if recorder is not None:
                    recorder.record(info.get("rgb"), info)
            if recorder is not None:
                recorder.end_episode(success)
            results.append(success)
            by_town.setdefault(ep["town"], []).append(success)
    finally:
        if env is not None:
            env.close()

    rate = float(np.mean(results)) if results else 0.0
    # n is reported alongside the rate deliberately. The peer measured a
    # 71.7% vs 60.0% gap at n=60 that looked like a win and was parity
    # (z=1.35, CI spans zero); resolving 11.7 points at 80% power needs ~259
    # episodes per arm. A success rate without its n invites that mistake.
    line = "  ".join(f"{t}={np.mean(v):.3f}(n={len(v)})" for t, v in by_town.items())
    tag = "IN-DIST" if os.environ.get("EVAL_TOWNS") else "held-out"
    # THE GOAL BAND GOES IN THE EVAL LINE.
    #
    # Eval samples goals from GOAL_DIST_MIN/MAX, the same variables the
    # curriculum rewrites, so eval DIFFICULTY moves with the curriculum and two
    # evals from one run are not necessarily comparable. That confound already
    # produced a misreading: a jump from 0.683 to 0.800 looked like the 113 m
    # ceiling breaking, when the curriculum had stepped back to 90.4 m eleven
    # minutes earlier and the policy was simply being asked easier questions.
    # Recording the band makes every eval self-describing.
    _band = (f"{os.environ.get('GOAL_DIST_MIN', '?')}-"
             f"{os.environ.get('GOAL_DIST_MAX', '?')}m")
    print(f"[Eval] {tag} success={rate:.3f} (n={len(results)}) "
          f"goals={_band}  {line}")
    return rate, by_town


def _route_provider_for(get_env):
    """A zero-arg callable returning an XYRoute on the env's CURRENT map, or None
    before the env has connected. `get_env` is called EACH TIME (v18): a town
    rotation replaces venv.envs[i] outright, so a provider holding the original
    env object would keep planning on the previous town's roads. Walks the
    wrapper chain by `.env` for the object owning `core` (the gated wrappers
    return themselves from .unwrapped). One planner per town name."""
    cache = {}

    def provider():
        node, core = get_env(), None
        for _ in range(16):
            if node is None:
                break
            if getattr(node, "core", None) is not None:
                core = node.core
                break
            node = getattr(node, "env", None)
        world_map = getattr(core, "map", None)
        if world_map is None:
            return None
        name = getattr(world_map, "name", "?")
        if name not in cache:
            from vae_nav.route_potential import XYRoute
            cache.clear()
            cache[name] = XYRoute(world_map)
            print(f"[EpisodeHER] route planner for hindsight progress on {name}",
                  flush=True)
        return cache[name]
    return provider


def main():
    from jaxrl2.data import ReplayBuffer
    from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics

    train_towns, heldout = load_split()
    for t in train_towns:
        assert_not_heldout(t, heldout)

    # from_env() honours PERCEPTION (vae | mobilenet) and is the ONLY
    # place a backend is chosen, in training and eval alike.
    latent_source = latent_from_env()
    # The VAE must not have seen the held-out towns either: a perception module
    # trained on Town03 leaks it just as surely as a policy trained on it.
    leaked = set(latent_source.trained_on_towns) & set(heldout)
    if leaked:
        raise SystemExit(
            f"VAE checkpoint was trained on held-out town(s) {sorted(leaked)}. "
            "The latent would carry information from the evaluation maps, so "
            "held-out results would be meaningless. Retrain the VAE on the "
            "training towns only.")

    # THE EXPERT COIN GETS ITS OWN STREAM. GatedReward draws its mode bit from
    # default_rng(env seed) = default_rng(SEED) for env 0, one draw per reset;
    # this coin also drew once per episode from default_rng(SEED). Same seed,
    # same cadence, same uniform: u < 0.4 armed the oracle and u < 0.5 made
    # the episode explore, so EVERY oracle episode was an explore episode and
    # goal mode never received one oracle row (14/14 armed episodes at
    # mode=0, in a run whose [Why] line reported them as the goal oracle).
    rng = np.random.default_rng([SEED, 0x0E5])

    # ONE ENV PER TOWN, one server each. Replaces the previous scheme of
    # rebuilding the env every episode to rotate towns, which reloaded the CARLA
    # world each time -- tens of seconds for a few hundred steps of data. Now
    # each world loads once for the whole run and every batch is mixed across
    # towns, which is what a held-out-town claim actually needs.
    # TOWN PRIORITY, because the fleet is usually smaller than the split.
    #
    # An Epic CARLA server costs ~5.9 GB, so a 12 GB card holds two -- not the
    # four the split contains. Taking the first N of the split would have
    # picked Town01 and Town02, whose lane channels measure 0.24% and 0.03%:
    # the two towns carrying almost no lane signal at all. The policy would
    # have trained on obstacle geometry and essentially no lane markings, then
    # been evaluated on held-out maps that have them.
    #
    # Default order is by measured lane density (Town05 4.21%, Town04 2.13%,
    # Town01 0.24%, Town02 0.03%), so a truncated fleet keeps the informative
    # towns. Override with ENV_TOWNS to test a different mix.
    priority = os.environ.get("ENV_TOWNS", "Town05,Town04,Town01,Town02").split(",")
    ordered = ([t for t in priority if t in train_towns] +
               [t for t in train_towns if t not in priority])
    n_envs = min(len(CARLA_PORTS), len(ordered))
    env_towns = ordered[:n_envs]
    if n_envs < len(train_towns):
        dropped = [t for t in ordered[n_envs:]]
        print(f"[Vec] WARNING: {len(CARLA_PORTS)} server(s) for "
              f"{len(train_towns)} training towns. Using {env_towns} "
              f"(highest lane density first); NOT training on {dropped}.")
    for t in env_towns:
        assert_not_heldout(t, heldout)
    print(f"[Vec] {n_envs} envs: " +
          ", ".join(f"{t}@{p}" for t, p in zip(env_towns, CARLA_PORTS)))

    venv = ThreadedVecEnv(
        [(lambda t=t, p=p, i=i: make_env(t, SEED + i, latent_source, port=p))
         for i, (t, p) in enumerate(zip(env_towns, CARLA_PORTS))],
        names=env_towns)

    agent = build_agent(venv.observation_space, venv.action_space, SEED)
    _capacity = int(os.environ.get("REPLAY_CAPACITY", 1_000_000))
    # FP16 PIXELS STORAGE (v8). The buffer's obs space carries the pixels key
    # as float16 when BUFFER_PIXELS_FP16=1 -- fp16_pixel_obs_space returns the
    # agent's space untouched otherwise. Every row then reaches storage
    # through the Fp16PixelsBuffer wrapper below, which recasts pixels at the
    # shared record path (both HER subclasses and the non-HER path) and
    # restores float32 on every sample.
    _buf_obs_space = fp16_pixel_obs_space(venv.observation_space)
    # N-STEP RETURNS (v11, user picked n-step over a gamma change for the
    # far-field saturation). 1 = every earlier run, byte-identical. n > 1
    # merges each sampled row with up to n-1 same-segment successors at sample
    # time (mode_split_buffer) and the learner raises its bootstrap discount to
    # gamma**n'. Segment ids come from the HER buffers, so n > 1 requires the
    # gated two-critic route WITH EPISODE_HER=1; anything else is refused
    # rather than silently degraded.
    N_STEP = int(os.environ.get("N_STEP", "1"))
    if GATED and not wants_one_critic():
        from vae_nav.mode_split_buffer import ModeSplitBuffer
        if N_STEP > 1 and os.environ.get("EPISODE_HER", "") != "1":
            raise SystemExit(
                f"N_STEP={N_STEP} needs EPISODE_HER=1: the n-step segments are "
                "assigned by the per-env relabel buffers (episode ends, leg "
                "ends, hindsight-goal runs), and a non-HER caller has no "
                "boundaries to offer -- silent 1-step behaviour would be a "
                "flag that does nothing.")
        # MODE-STRATIFIED, and the reason is measured rather than theoretical.
        # A uniformly sampled shared buffer gives each critic whatever share of
        # the batch the environment happened to produce. The peer measured one
        # mode reaching ~90% of env steps, the other critic receiving 10-37% of
        # each batch, and that mode's evaluation collapsing from 40-65% to 15%.
        # It was starved, not broken.
        #
        # THE SKEW RUNS THE OTHER WAY HERE: the goal cap is 2000 steps against
        # 500 for explore, so GOAL rows dominate. Fixing composition on SAMPLE
        # rather than correcting a known skew on insert is exactly what makes
        # the direction irrelevant.
        #
        # THE DISCOUNTS COME FROM THE AGENT, not re-read from the environment:
        # the merged reward sum must be accumulated in exactly the frame the
        # learner's gamma**n' bootstrap assumes, and the agent is the single
        # source of both gammas.
        buffer = ModeSplitBuffer(
            _buf_obs_space, venv.action_space, _capacity,
            goal_frac=float(os.environ.get("GOAL_BATCH_FRAC", 0.5)),
            n_step=N_STEP, discount_goal=float(agent.discount),
            discount_explore=float(agent.discount_e))
        if N_STEP > 1:
            print(f"[Buffer] n-step returns: N_STEP={N_STEP}, "
                  f"gamma_goal={agent.discount} gamma_explore={agent.discount_e}"
                  " -- merged reward sums + gamma**n' bootstrap; the TD fixed "
                  "point is unchanged, the path to it is shorter")
    else:
        if N_STEP > 1:
            raise SystemExit(
                f"N_STEP={N_STEP} is the gated two-critic route's feature: the "
                "one-critic/plain route samples a plain jaxrl2 ReplayBuffer and "
                "its learner cannot read the per-row gamma**n' bootstrap. A "
                "silent no-op would be a flag that lies.")
        # THE ONE-CRITIC ROUTE AND THE PLAIN ROUTE ALIKE. A single plain
        # ReplayBuffer: the unified-reward arm has ONE critic, so there is no
        # second critic to stratify a batch for, and the mode bit already rides
        # INSIDE the observation dict (OBS_WITH_MODE), where the one critic
        # reads it. No GOAL_BATCH_FRAC split.
        buffer = ReplayBuffer(_buf_obs_space, venv.action_space,
                              _capacity)
    if _buf_obs_space is not venv.observation_space:
        from vae_nav.episode_her import Fp16PixelsBuffer
        buffer = Fp16PixelsBuffer(buffer)
        # CAPACITY-DERIVED ESTIMATE, NOT A HARDCODED NUMBER: fp16 pixels are
        # 2 bytes per element, and the ReplayBuffer preallocates a
        # next_observations mirror besides, hence the factor 2. The banner
        # must track REPLAY_CAPACITY so a changed capacity cannot silently
        # stale the printed size (the "~34 GB at 450k" figure was written
        # against a different capacity and drifted).
        _px_el = int(np.prod(_buf_obs_space["pixels"].shape))
        _fp16_gb = _capacity * _px_el * 2 * 2 / 2 ** 30
        print(f"[Buffer] pixels stored as float16 (BUFFER_PIXELS_FP16=1, "
              f"REPLAY_CAPACITY={_capacity}, fp16 ~{_fp16_gb:.0f} GB incl. "
              f"the next_observations mirror; fp32 would be ~{_fp16_gb * 2:.0f} GB), "
              "restored to float32 on sample")

    # HER IS PER ENV. Environments finish episodes at different times, and a
    # shared relabel buffer would splice transitions from different towns into
    # one hindsight trajectory -- a "future" goal drawn from another map.
    hers = None
    _recost = os.environ.get("HER_RECOST", "0") == "1"
    if _recost and os.environ.get("IMAGE_GOAL_TASK", "0") == "1":
        # The image-goal arm's hindsight rows keep the flown reward (it is
        # goal-independent), so there is no counting regime to mirror and the
        # recost kwarg does not exist on that class.
        raise SystemExit("HER_RECOST=1 is the point-goal arm's counting-regime "
                         "recost and is incompatible with IMAGE_GOAL_TASK=1")
    if os.environ.get("EPISODE_HER", "") == "1":
        hers = []
        for e in venv.envs:
            w = e
            while w is not None and not isinstance(w, VAEGoalWrapper):
                w = getattr(w, "env", None)
            if w is None:
                raise SystemExit("EPISODE_HER=1 needs VAEGoalWrapper in the stack")
            # THE SELECTION. The image-goal arm's hindsight buffer differs in
            # CLASS, not just coefficients: it keeps the flown reward on every
            # relabelled row (the image-goal reward is goal-independent) and
            # sets done only at the hindsight entry, so it takes none of the
            # k_* coefficients and neither time_to_goal nor gated -- those
            # kwargs do not exist on the variant. Its goal_radius is the same
            # 2.0 m the success test uses, not the goal arm's GOAL_THRESHOLD.
            if os.environ.get("IMAGE_GOAL_TASK", "0") == "1":
                from vae_nav.image_goal_her import ImageGoalRelabelBuffer
                hers.append(ImageGoalRelabelBuffer(
                    buffer, w.goal_block,
                    goal_radius=float(os.environ.get(
                        "IG_GOAL_DISTANCE_THRESHOLD_M", 2.0)),
                    n_relabel=int(os.environ.get("EPISODE_HER_N", 1)),
                    min_disp=float(os.environ.get("HER_MIN_DISP", 5.0)),
                    seed=SEED + len(hers)))
                continue
            hers.append(EpisodeRelabelBuffer(
                buffer, w.goal_block, goal_radius=GOAL_THRESHOLD, k_goal=K_GOAL,
                k_collision=K_COLLISION, k_lane=K_LANE,
                k_dist=K_DIST,
                k_dist_target_speed=K_DIST_TARGET_SPEED,
                k_dist_gate_mode=K_DIST_GATE_MODE,
                k_speed=K_SPEED, speed_limit_kmh=SPEED_LIMIT_KMH,
                k_dist_delta_mode=K_DIST_DELTA_MODE,
                k_living=K_LIVING,
                # N-STEP SEGMENT TAG: this env's index in the high bits, so two
                # envs' ids can never alias and two towns' trajectories can
                # never merge into one n-step segment.
                segment_tag=len(hers),
                # Under TIME_TO_GOAL every sparse coefficient above is dead and
                # the relabelled reward is 0-on-arrival / -step_cost otherwise.
                # Passing the flag is MANDATORY, not optional: without it the
                # hindsight rows carry +600 arrivals into a buffer whose
                # on-policy rows pay 0, and the critic fits two reward
                # functions over one set of states.
                # THE GATED ARM'S GOAL REWARD WAS ALSO MINIMUM TIME, under a
                # different flag. GATED_GOAL_REWARD=sorb pays -STEP_COST per
                # step and 0 on arrival exactly as TIME_TO_GOAL does, so the
                # same recosting applies and omitting it here would put +k_goal
                # arrivals into a buffer whose on-policy rows pay 0.
                # THAT HELD UNTIL v9's COUNTING REGIME, which pays -fines on
                # fault steps, +arrival on arrival and truncates at fault-step
                # counts -- the min-time mirror fell behind (measured ~50% of
                # the goal side of the v9mb buffer, fines diluted ~2x), so
                # HER_RECOST=1 switches the mirror to the counting branch
                # below.
                # progress (Bono et al. 2023) takes the same recost path: its
                # rows are re-costed by gated_reward, never by the legacy sparse
                # coefficients, so it needs time_to_goal=True exactly as sorb.
                time_to_goal=(os.environ.get("TIME_TO_GOAL", "") == "1"
                              or (GATED and os.environ.get(
                                  "GATED_GOAL_REWARD", "sorb").lower()
                                  in ("sorb", "progress"))),
                step_cost=float(os.environ.get("STEP_COST", 0.1)),
                # Routes on-policy rows by their own mode and forces every
                # hindsight copy to GOAL.
                gated=GATED,
                n_relabel=int(os.environ.get("EPISODE_HER_N", 1)),
                min_disp=float(os.environ.get("HER_MIN_DISP", GOAL_THRESHOLD)),
                max_disp=float(os.environ.get("HER_MAX_DISP", np.inf)),
                seed=SEED + len(hers),
                # RECOST UNDER THE COUNTING REGIME (user 2026-09-22): without
                # this the relabelled rows carry the old min-time reward -- no
                # fines, 0 on arrival -- next to on-policy rows that pay
                # -100-per-fault-step and +10 on arrival. Measured on v9mb:
                # hindsight was ~50% of the goal side, so the fines were
                # diluted ~2x. The buffer reads every coefficient from
                # vae_nav.gated_reward itself, so the two cannot drift.
                recost_counting=_recost,
                # ROAD DISTANCE TO HINDSIGHT GOALS (2026-09-26). Needed by any
                # goal reward with an along-road progress term (progress, or
                # sorb + POTENTIAL_K); resolved lazily from THIS env's map.
                route_provider=_route_provider_for(
                    lambda ix=len(hers): venv.envs[ix])))
        print(f"[EpisodeHER] on -- {len(hers)} per-env buffers, "
              f"n_relabel={hers[0].n_relabel} min_disp={hers[0].min_disp} "
              f"goal_radius={hers[0].goal_radius}"
              + (f" | RECOST on: fines {hers[0]._fine_offroad}/"
                 f"{hers[0]._fine_lane}/{hers[0]._fine_collision} "
                 f"(offroad/lane/collision), arrival {hers[0]._arrival_bonus}, "
                 f"truncate at {hers[0]._trunc_offroad}/"
                 f"{hers[0]._trunc_crossing}/{hers[0]._trunc_collision} counts"
                 + (f", velocity bonus {hers[0]._speed_bonus}"
                    if getattr(hers[0], "_speed_bonus", 0.0) else "")
                 # Stated outright (2026-09-26): without it the only evidence
                 # that hindsight legs are cut at fatal crossings was inferring
                 # it from the [Ep] cross= share.
                 + (", LANE TERMINAL: legs cut at the first fatal crossing"
                    if getattr(hers[0], "_lane_terminal", False) else "")
                 if getattr(hers[0], "recost", False) else ""))

    identity = dict(latent_source.identity())
    identity.update({"train_towns": env_towns, "heldout_towns": heldout,
                     "seed": SEED, "k_goal": K_GOAL,
                     "goal_threshold": GOAL_THRESHOLD, "n_envs": n_envs,
                     "goal_dim": 12})

    # EVAL_TOWNS overrides the held-out set. Used for the OVERFITTING TEST:
    # train and evaluate on the SAME town, to answer "can this policy learn the
    # task at all" separately from "does it transfer". A number produced this
    # way is IN-DISTRIBUTION and is not a generalisation result -- the banner
    # below says so on every run, because a 0.4 success rate here would be very
    # easy to quote later as though it were held-out.
    _eval_towns = [t for t in os.environ.get("EVAL_TOWNS", "").split(",") if t.strip()]
    if _eval_towns:
        in_dist = sorted(set(_eval_towns) & set(env_towns))
        print(f"[Eval] *** OVERFITTING TEST *** evaluating on {_eval_towns}; "
              f"{in_dist} also in the TRAINING set -- these numbers are "
              f"IN-DISTRIBUTION, not held out.")
        eval_episodes = build_eval_episodes(_eval_towns, N_EVAL_EPISODES, seed=0)
    else:
        eval_episodes = build_eval_episodes(heldout, N_EVAL_EPISODES, seed=0)
    out_dir = os.path.join("checkpoints", POLICY_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Setup] {datetime.now():%Y-%m-%d %H:%M:%S} towns={env_towns} "
          f"steps={MAX_STEPS} backend='{latent_source.name}'")

    # TOWN ROTATION. Only one CARLA server fits alongside the learner on a 12 GB
    # card (measured ~4.9-5.5 GB per server, essentially independent of quality
    # or headless), so simultaneous multi-town training is not available here.
    # Instead each env cycles through the town list every ROTATE_EPISODES
    # episodes, rebuilding the env to trigger the world load.
    #
    # The cost is honest and worth stating: within a rotation block the buffer
    # fills with one town, so batches are town-correlated rather than mixed.
    # Over a run the buffer still spans all four towns, which is what the
    # held-out claim needs -- but it is weaker than true simultaneous mixing.
    ROTATE_EPISODES = int(os.environ.get("TOWN_ROTATE_EPISODES", 20))

    # AN EXPLICIT ENV_TOWNS CONSTRAINS THE ROTATION, not just the starting map.
    #
    # This was `rotation = ordered`, the FULL training-town list, while
    # env_towns held only the towns asked for. Setting ENV_TOWNS=Town02 for a
    # single-town overfitting test therefore picked Town02 to START on and then
    # cycled through Town05, Town04 and Town01 anyway, twenty episodes at a
    # time. The run was silently a four-town run: its logs said
    # `town=Town05` while its banner said Town02.
    #
    # It also killed the run. Every rotation rebuilds the env, which calls
    # load_world on the shared server -- the one operation known to crash CARLA
    # 0.9.15 -- and the trainer died with "Cannot connect to server" after 25
    # reconnect attempts. Honouring the override removes the rotation entirely
    # for a single-town run, so there are no world reloads at all.
    if os.environ.get("ENV_TOWNS"):
        # ROTATION FOLLOWS THE REQUESTED TOWNS, NOT THE TRUNCATED ENV LIST.
        #
        # env_towns is ordered[:n_envs], and n_envs is capped by how many CARLA servers fit -- ONE on a 12 GB card.
        # Reading rotation off env_towns therefore collapsed ENV_TOWNS=Town01,Town10HD to rotation=['Town01'] and
        # printed "[Rotate] disabled -- single town", so the run trained on ONE of the two towns asked for while its
        # config, its .env snapshot and its banner all said two. A silent half-run is the worst shape of this bug.
        #
        # The truncation is correct for how many towns run SIMULTANEOUSLY; it is wrong for which towns the single env
        # CYCLES through, which is precisely what rotation exists to do when the fleet is smaller than the request.
        # Falls back to env_towns when the override names nothing trainable, so a bad ENV_TOWNS cannot empty the list.
        rotation = [t for t in priority if t in train_towns] or list(env_towns)
        print(f"[Rotate] ENV_TOWNS is set; rotation restricted to {rotation}")
    else:
        rotation = ordered          # full town list, richest lane density first
    rot_idx = list(range(n_envs))   # which town each env currently holds
    if len(rotation) > 1:
        print(f"[Rotate] every {ROTATE_EPISODES} episodes per env, "
              f"cycling {rotation}")
    else:
        print(f"[Rotate] disabled -- single town {rotation}, no world reloads")

    # EXPERT ORACLE, PER ENV. This was present in the single-env trainer and I
    # dropped it when rewriting the loop for vectorised envs -- EXPERT_PROB was
    # silently dead for a whole run. Measured cost of its absence: 100 episodes,
    # zero successes, every reward 0.0, so HER had no real arrivals to relabel
    # around and the critic saw no positive return anywhere.
    experts = [make_expert(e, venv.action_space) for e in venv.envs]
    is_expert_ep = [False] * n_envs
    # Why the oracle declined an episode. It has been arming at ~0.65 of its
    # configured probability, and "coin came up tails" is indistinguishable
    # from "route planning failed" without this.
    arm_failures = {}

    obs, infos = venv.reset(seed=SEED)

    def _arm_expert(i, info):
        """Decide whether env i's NEW episode is oracle-driven, and plan it."""
        # THE ORACLE MUST HOLD THE LIVE HERO. The vec env rebuilds an env (new
        # world actors) after a failed reset/step, and a destroyed actor does
        # not raise -- it reports the origin. Measured: from episode ~118 every
        # [ExpertPlan] read hero=(0.0,0.0) and the oracle drove 300-700-node
        # routes from a phantom pose, lane_any/orc climbing from 0 to 0.09
        # while the rows kept entering the buffer. Rebuild whenever the hero
        # actor the oracle was built on is no longer the env's hero.
        _core = getattr(getattr(venv.envs[i], "unwrapped", venv.envs[i]), "core", None)
        _hero = getattr(_core, "hero", None)
        _held = getattr(experts[i], "_vehicle", None) if experts[i] is not None else None
        if _hero is not None and (experts[i] is None or
                                  getattr(_held, "id", None) != _hero.id):
            print(f"[Expert] env{i}: hero actor is "
                  f"{getattr(_held, 'id', None)} -> {_hero.id}; rebuilding oracle",
                  flush=True)
            experts[i] = make_expert(venv.envs[i], venv.action_space)
        if experts[i] is None or not should_use_expert(rng, EXPERT_PROB):
            is_expert_ep[i] = False
            return
        goal = (info or {}).get("goal_location")
        if goal is None:
            arm_failures["no_goal_in_info"] = arm_failures.get("no_goal_in_info", 0) + 1
            is_expert_ep[i] = False
            return
        # EXPERT_MODES: which gated episodes the oracle may drive. "both"
        # (default) is the documented setup -- 40% of ALL episodes; "goal"
        # keeps it to goal-mode episodes, where a route-following driver is
        # the task; in explore mode it drives to a goal explore does not
        # have, then parks. A non-gated env has no mode and counts as goal.
        _mode = (info or {}).get("mode", 1)
        _allowed = {"both": (0, 1), "goal": (1,), "explore": (0,)}[
            os.environ.get("EXPERT_MODES", "both")]
        if _mode not in _allowed:
            is_expert_ep[i] = False
            return
        print(f"[ExpertArm] env{i} mode={info.get('mode')} "
              f"goal=({goal.x:.1f},{goal.y:.1f})", flush=True)
        is_expert_ep[i] = bool(experts[i].reset(goal) > 0)
        if not is_expert_ep[i]:
            r = getattr(experts[i], "last_plan_error", None) or "unknown"
            arm_failures[r] = arm_failures.get(r, 0) + 1

    for i in range(n_envs):
        _arm_expert(i, infos[i] if i < len(infos) else {})

    ep_done = np.zeros(n_envs, np.int64)
    n_expert_eps = 0

    # PER-EPISODE STATISTICS. Without these a run that terminates every episode
    # on step 1 looks exactly like one that is learning: the step counter still
    # advances, the buffer still fills, nothing errors. The first live run did
    # 20 episodes in 70 seconds and there was no way to tell instant collisions
    # from instant successes. Success and collision rates belong side by side --
    # a rising episode length with a flat success rate means the agent learned
    # to survive without arriving, which is a different failure from not
    # learning at all.
    # GOAL-DISTANCE CURRICULUM.
    #
    # Advances on DEMONSTRATED SUCCESS, not on step count. A step-based
    # schedule widens the band whether or not the agent can handle the current
    # one, which under a sparse reward means it can be moved out to distances
    # it never solved and lose the only positive signal it had. Requiring a
    # success rate before each widening makes the curriculum follow the agent.
    #
    # carla_core.reset_hero reads these from the environment at EVERY reset, so
    # updating os.environ here takes effect on the next episode. The threaded
    # vec env runs in this process, so the write is visible to it.
    CURR_MIN = float(os.environ.get("GOAL_DIST_MIN", 5.0))
    CURR_START = float(os.environ.get("GOAL_DIST_START", 5.0))
    CURR_TARGET = float(os.environ.get("GOAL_DIST_TARGET", 20.0))
    CURR_STEP = float(os.environ.get("GOAL_DIST_ADVANCE", 2.5))
    # ADDITIVE OR MULTIPLICATIVE ADVANCE.
    #
    # Additive is right over a narrow range. Over a wide one it is not: 5 m to
    # 200 m in 2.5 m steps is 78 stages, and each stage needs at least
    # CURRICULUM_MIN_EPISODES (50) episodes to clear, so the schedule alone
    # would consume ~3,900 episodes -- most of the step budget -- before the
    # agent ever saw the target distance, and that assumes every stage passes
    # on the first attempt.
    #
    # Multiplicative keeps the DIFFICULTY increment roughly constant instead of
    # the metric one: going 5 -> 6.25 m and 160 -> 200 m are both x1.25, and
    # the whole range is ~17 stages. GOAL_DIST_MUL is the factor; it takes
    # precedence over GOAL_DIST_ADVANCE when set above 1.
    CURR_MUL = float(os.environ.get("GOAL_DIST_MUL", 1.0))
    CURR_RATE = float(os.environ.get("CURRICULUM_SUCCESS", 0.5))
    CURR_DEMOTE = float(os.environ.get("CURRICULUM_DEMOTE", 0.15))
    CURR_STALL = int(os.environ.get("CURRICULUM_STALL_EPISODES", 400))
    CURR_MIN_EPS = int(os.environ.get("CURRICULUM_MIN_EPISODES", 50))
    curr_max = CURR_START
    os.environ["GOAL_DIST_MIN"] = str(CURR_MIN)
    os.environ["GOAL_DIST_MAX"] = str(curr_max)
    last_advance_ep = 0
    _adv = (f"x{CURR_MUL}" if CURR_MUL > 1.0 else f"+{CURR_STEP} m")
    print(f"[Curriculum] goals {CURR_MIN}-{curr_max} m, advancing {_adv} to "
          f"{CURR_TARGET} m once policy success >= {CURR_RATE} over "
          f"{CURR_MIN_EPS} episodes (demote at <= {CURR_DEMOTE})")

    from collections import deque
    recent = {"success": deque(maxlen=100), "length": deque(maxlen=100),
              "return": deque(maxlen=100), "collision": deque(maxlen=100),
              "success_oracle": deque(maxlen=100),
              "success_policy": deque(maxlen=100),
              "goals": deque(maxlen=100),
              # PER-MODE, for the gated arm. A single mean over both modes is
              # unreadable: the goal arm pays -1 per step (returns of order
              # -1000) while the explore arm pays a progress bonus (order +10),
              # so their average is a number describing neither. The gated arm
              # exists to be compared against two specialists, and a blended
              # statistic cannot support that comparison.
              "ret_goal": deque(maxlen=100), "ret_explore": deque(maxlen=100),
              "len_goal": deque(maxlen=100), "len_explore": deque(maxlen=100),
              "succ_goal": deque(maxlen=100),
              "coll_goal": deque(maxlen=100), "coll_explore": deque(maxlen=100),
              # HOW FAR explore actually drives, and WHY it stops. Added
              # 2026-09-11: the explore arm was being judged on return and
              # length alone, and neither says whether it ended on a lane
              # crossing, off-road, a collision, the step cap, or the new
              # 100 m arrival.
              "dist_explore": deque(maxlen=100),
              "end_explore": deque(maxlen=100),
              # Goal-side end reasons (user 2026-09-23): which counting cap
              # the on-policy GOAL episode tripped, from the HER buffer's
              # fault counters (episode_her.last_end_reason). Agent-driven
              # episodes only, like succ_goal.
              "end_goal": deque(maxlen=100)}
    LOG_EVERY = int(os.environ.get("LOG_EVERY_EPISODES", 10))
    total_eps = 0
    # Histogram of WHY episodes end. Without it "return 0.0" is the only
    # symptom and every diagnosis is a guess.
    end_reasons = {}
    # Denominators for the oracle/policy split below. A tagged count divided by
    # total_eps would be wrong by ~5x for the oracle, whose episodes are only
    # EXPERT_PROB of the total -- the exact class of mis-normalised metric this
    # split exists to expose.
    ep_by_driver = {"orc": 0, "pol": 0}
    last_update_info = {}

    # `step` counts ENV TRANSITIONS, not vec-steps, so MAX_STEPS and
    # START_TRAINING keep the same meaning they had with one env.
    # RESUME FROM THE NEWEST CHECKPOINT.
    #
    # MEASURED COST OF NOT HAVING THIS: a run reached 374,625 steps at 0.950
    # success, hit a CARLA actor-registry fault, and the supervisor's restart
    # loop began again at step 0 -- discarding four hours while
    # checkpoint_375000.pkl sat on disk holding the exact policy. The restart
    # loop protected the session and threw away the training.
    #
    # Weights only, deliberately: optimiser moments and the replay buffer are
    # NOT restored. The buffer is the real loss (a resumed run refills it from
    # a competent policy rather than a random one, which is a different
    # distribution than an uninterrupted run would have had), so this is
    # crash recovery, not a substitute for a clean run. Stated rather than
    # hidden, because a resumed curve is not identical to an unbroken one.
    #
    # RESUME=0 disables it, for when a fresh run is what is wanted.
    step = 0
    if os.environ.get("RESUME", "1") == "1" and os.path.isdir(out_dir):
        import re as _re
        cks = [(int(m.group(1)), f) for f in os.listdir(out_dir)
               for m in [_re.match(r"checkpoint_(\d+)\.pkl$", f)] if m]
        # WALK BACKWARDS THROUGH CHECKPOINTS, newest first.
        #
        # Trying only the newest one and giving up is not enough: the newest is
        # precisely the one most likely to be damaged, because it is the one
        # being written when something goes wrong. MEASURED -- the root
        # filesystem filled to 100% mid-write, leaving a 0-byte
        # checkpoint_275000.pkl. Every earlier checkpoint was intact and
        # carried curriculum_max=113.0, but the resume saw one UnpicklingError
        # and restarted from scratch, discarding 275k steps that were sitting
        # on disk one file away.
        for best_step, best_file in sorted(cks, reverse=True):
            try:
                with open(os.path.join(out_dir, best_file), "rb") as f:
                    ck = pickle.load(f)
                if "agent" not in ck:
                    print(f"[Resume] {best_file} predates weight saving; "
                          "trying older", flush=True)
                    continue
                load_agent_state(agent, ck["agent"])
                step = best_step
                if ck.get("curriculum_max"):
                    curr_max = float(ck["curriculum_max"])
                    os.environ["GOAL_DIST_MAX"] = str(curr_max)
                    print(f"[Resume] curriculum restored to "
                          f"{CURR_MIN}-{curr_max} m", flush=True)
                print(f"[Resume] {best_file} -> continuing from step {step} "
                      f"(eval_success was {ck.get('eval_success')}); "
                      "replay buffer and optimiser state start fresh",
                      flush=True)
                break
            except (Exception, SystemExit) as e:  # noqa: BLE001
                # SystemExit is caught DELIBERATELY. load_agent_state raises it
                # on a pytree mismatch -- correct for a direct load, fatal here.
                # SystemExit is not an Exception subclass, so it propagated
                # straight through this handler and killed the process.
                # MEASURED: flipping OBS_WITH_VECTOR from 1 to 0 made every
                # checkpoint in the folder incompatible, and the supervisor
                # burned its attempts restarting into the identical crash.
                # A shape mismatch should fall through to the next-older
                # checkpoint and ultimately to a fresh start, never abort.
                print(f"[Resume] {best_file} unusable ({type(e).__name__}: "
                      f"{str(e)[:110]}); trying the next older checkpoint",
                      flush=True)
        else:
            print("[Resume] no loadable checkpoint; starting from scratch",
                  flush=True)

    # WARM START (INIT_FROM=<checkpoint>): weights from another run's
    # checkpoint, but the STEP COUNTER, REPLAY BUFFER, OPTIMISER STATE and
    # CURRICULUM all start fresh -- this is initialisation, not a resume, and
    # the new run keeps its own name and folder. RESUME (same folder) wins if
    # it found a checkpoint. User decision 2026-09-11: v5 starts from v4.
    init_from = os.environ.get("INIT_FROM", "")
    if init_from and step == 0:
        with open(init_from, "rb") as f:
            _ck0 = pickle.load(f)
        _state0 = dict(_ck0["agent"])
        # INIT_FROM_FRESH_EXPLORE_CRITIC=1: do NOT restore the source run's
        # explore critic/temperature. A critic fitted to a reward of a
        # different scale is confidently wrong from step 0 (v4's explore
        # returns ~-10 against v5's ~+500), and the shared actor followed it.
        if os.environ.get("INIT_FROM_FRESH_EXPLORE_CRITIC", "0") == "1":
            for k in ("critic_e", "target_critic_e", "temp_e"):
                _state0.pop(k, None)
            print("[Init] explore critic + temperature start FRESH "
                  "(INIT_FROM_FRESH_EXPLORE_CRITIC=1)", flush=True)
        load_agent_state(agent, _state0)
        print(f"[Init] weights warm-started from {init_from} "
              f"(that run's step {_ck0.get('step')}, eval_success "
              f"{_ck0.get('eval_success')}); step counter, replay buffer, "
              "optimiser and curriculum start fresh", flush=True)
    while step < MAX_STEPS:
        if step < START_TRAINING:
            actions = np.stack([venv.action_space.sample() for _ in range(n_envs)])
        else:
            # np.array(..., copy=True), NOT np.asarray. asarray on a JAX device
            # array yields a READ-ONLY zero-copy view, and the expert override
            # below writes into this buffer:
            #     ValueError: assignment destination is read-only
            # It only triggers once training has started AND an oracle episode
            # is active, so it survived 5000 steps before firing.
            actions = np.array(agent.sample_actions(agent_view(obs)),
                               dtype=np.float32,
                               copy=True)
        # Oracle overrides the policy for whole episodes. Its actions live in
        # the agent's own action space, so these are ordinary off-policy rows
        # that the buffer and the HER relabeller cannot distinguish.
        for i in range(n_envs):
            if is_expert_ep[i] and experts[i] is not None:
                a_exp = experts[i].act()
                if a_exp is None:        # route exhausted or unplannable
                    is_expert_ep[i] = False
                else:
                    actions[i] = a_exp

        # The chain-cap ramp in sparse_reward.max_goals_now reads this. Shared
        # by training and eval deliberately: their habitat_wrappers notes that
        # a train/eval mismatch in termination rules "cost a whole run".
        os.environ["TRAIN_STEP"] = str(step)

        next_obs, rewards, dones, step_infos, final_infos = venv.step(actions)

        for i in range(n_envs):
            # A transition produced by a simulator fault carries no valid
            # observation or reward; recording it would inject noise the critic
            # cannot distinguish from real experience.
            if step_infos[i].get("env_error"):
                if hers is not None:
                    hers[i]._reset_episode()      # discard the partial episode
                continue
            o_i = ThreadedVecEnv.unstack(obs, i)
            n_i = ThreadedVecEnv.unstack(next_obs, i)
            inf = step_infos[i]
            # A time-limit truncation must still bootstrap; only a real
            # termination cuts the value backup.
            mask = 0.0 if inf.get("terminated", False) else 1.0
            tr = dict(observations=o_i, actions=actions[i],
                      rewards=np.float32(rewards[i]), masks=np.float32(mask),
                      dones=np.float32(float(dones[i])), next_observations=n_i)
            if hers is not None:
                # THE CALL SITE IS SHARED BETWEEN BUFFER CLASSES: `hers` holds
                # an ImageGoalRelabelBuffer only when IMAGE_GOAL_TASK=1 and a
                # plain EpisodeRelabelBuffer otherwise. `off_road` exists only
                # on the image-goal subclass (its flush-time dirty cut), so it
                # is passed ONLY there -- the base record() has no such kwarg
                # and would raise TypeError on every step. Same class-attribute
                # dispatch the flush site below uses.
                from vae_nav.gated_reward import (
                    goal_lane_departed as _goal_lane_departed,
                    goal_lane_fine as _goal_lane_fine,
                    goal_lane_fault as _goal_lane_fault)
                _rec_kwargs = dict(
                    speed_kmh=float(inf.get("speed_kmh", 0.0) or 0.0),
                    target_speed_kmh=float(inf.get("target_speed_kmh", 0.0)
                                           or 0.0),
                    # THE FAULT FLAGS THE COUNTING REGIME BILLS, staged per
                    # step for HER_RECOST. `lane` must read the SAME key the
                    # wrapper's counters read (gated_reward.py) -- the env
                    # publishes `lane_invasion_any` (core_state.py), never a
                    # bare `lane_invasion`, so the old read here staged False
                    # on every step and any fine keyed on it would have been
                    # silently free. `off_road` was image-goal-only because
                    # only that subclass staged it; the base record() now
                    # takes it too, so both arms stage the same flags.
                    lane=bool(inf.get("lane_any",
                                      inf.get("lane_invasion_any", False))),
                    off_road=bool(inf.get("off_road", False)),
                    # THE LANE-TERMINAL VERDICT (user 2026-09-26), from the SAME
                    # function goal_reward_sorb uses, so a hindsight row is cut
                    # at exactly the crossing that billed and ended the
                    # on-policy episode. Staged always; read only when
                    # GATED_GOAL_LANE_TERMINAL=1 under HER_RECOST.
                    lane_fatal=_goal_lane_departed(inf),
                    # Height, so hindsight route plans start on the right road.
                    z=float(inf.get("hero_z", 0.0) or 0.0),
                    # GRADED LANE FINES (v18): the reward's own verdicts, so the
                    # relabeller fines and counts exactly what the reward did.
                    lane_fine=_goal_lane_fine(inf),
                    lane_fault=_goal_lane_fault(inf),
                    # ROUTES THE ON-POLICY ROW. Without this the relabel buffer
                    # inserts with no mode and the split buffer defaults every
                    # row to GOAL, filing every explore transition as goal data.
                    # Hindsight copies are forced to GOAL inside the buffer.
                    mode=int(inf.get("mode", 1)))
                hers[i].record(
                    tr,
                    fields={"hero_pos": inf["hero_pos"], "hero_yaw": inf["hero_yaw"],
                            "goal_pos": inf["goal_pos"],
                            "action_history": inf.get("action_history",
                                                      np.zeros((4, 2)))},
                    pos=inf["hero_pos"], hit=bool(inf.get("collision", False)),
                    **_rec_kwargs)
                # CLOSE THE LEG ON ARRIVAL. Must come AFTER record(), so the
                # arriving transition belongs to the leg it completed rather
                # than opening the next one. Without this a chained episode is
                # one relabelling unit and hindsight goals cross goal
                # boundaries -- see EpisodeRelabelBuffer.end_leg.
                if inf.get("arrived_this_step"):
                    hers[i].end_leg()
            else:
                if GATED and not wants_one_critic():
                    # Routed by the mode this transition was collected under.
                    # Defaults to GOAL if absent, matching the observation
                    # wrapper, so a missing GatedReward degrades to the
                    # familiar single-mode behaviour rather than silently
                    # filing every row as exploration.
                    buffer.insert(tr, mode=int(inf.get("mode", 1)))
                else:
                    # The one-critic route's plain ReplayBuffer takes no mode
                    # kwarg -- the mode rides inside the observation dict.
                    buffer.insert(tr)

            if final_infos[i] is not None:
                ep_done[i] += 1
                total_eps += 1
                fi = final_infos[i]
                # SUCCESS SPLIT BY DRIVER. A single success rate conflates the
                # scripted oracle with the learned policy, and the oracle drives
                # a third of episodes -- so "success 0.16" was consistent with
                # the policy contributing nothing. Only the policy figure
                # answers whether TQC is learning.
                _succ = float(bool(fi.get("is_success", False)))
                recent["success"].append(_succ)
                (recent["success_oracle"] if is_expert_ep[i]
                 else recent["success_policy"]).append(_succ)
                recent["length"].append(float(fi.get("episode_length", 0)))
                recent["return"].append(float(fi.get("episode_return", 0.0)))
                recent["collision"].append(float(bool(fi.get("collision", False))))
                recent["goals"].append(float(fi.get("goals_reached", 0)))

                if GATED:
                    # Routed by the mode the episode was actually run under.
                    # Success is recorded for GOAL episodes ONLY: in explore
                    # mode it would measure arrival at a goal the policy cannot
                    # see and is not paid for, which is exactly the meaningless
                    # zero the goal-free arm reports over 60 episodes.
                    _ep_len = float(fi.get("episode_length", 0))
                    _ep_ret = float(fi.get("episode_return", 0.0))
                    _coll = float(bool(fi.get("collision", False)))
                    if int(fi.get("mode", 1)) == 1:
                        recent["ret_goal"].append(_ep_ret)
                        recent["len_goal"].append(_ep_len)
                        # AGENT-DRIVEN SR ONLY (user 2026-09-21): the expert
                        # drives a fixed share of episodes at 100% success,
                        # so mixing it in inflates the goal success rate by
                        # construction. succ_goal counts policy episodes
                        # ONLY; oracle-driven episodes are excluded here (the
                        # [Ep] line still reports the oracle split).
                        if not is_expert_ep[i]:
                            recent["succ_goal"].append(_succ)
                        recent["coll_goal"].append(_coll)
                    else:
                        recent["ret_explore"].append(_ep_ret)
                        recent["len_explore"].append(_ep_len)
                        recent["coll_explore"].append(_coll)
                        recent["dist_explore"].append(
                            float(fi.get("explore_dist_m", 0.0) or 0.0))
                        recent["end_explore"].append(
                            str(fi.get("explore_end", "?")))

                # Widen the band when the POLICY (never the oracle) is solving
                # the current one. Requires a fresh window of episodes since
                # the last advance, so two widenings cannot fire back to back
                # off the same evidence.
                if (len(recent["success_policy"]) >= CURR_MIN_EPS
                        and total_eps - last_advance_ep >= CURR_MIN_EPS):
                    _rate = float(np.mean(recent["success_policy"]))
                    if curr_max < CURR_TARGET and _rate >= CURR_RATE:
                        _next = (curr_max * CURR_MUL if CURR_MUL > 1.0
                                 else curr_max + CURR_STEP)
                        curr_max = min(CURR_TARGET, round(_next, 2))
                        os.environ["GOAL_DIST_MAX"] = str(curr_max)
                        last_advance_ep = total_eps
                        recent["success_policy"].clear()
                        print(f"[Curriculum] policy success {_rate:.2f} >= "
                              f"{CURR_RATE} -> goals now {CURR_MIN}-{curr_max} m "
                              f"(episode {total_eps})", flush=True)
                    elif (curr_max > CURR_START and CURR_STALL > 0
                          and curr_max < CURR_TARGET
                          and total_eps - last_advance_ep >= CURR_STALL
                          and _rate < CURR_RATE):
                        # `curr_max < CURR_TARGET` IS LORE, NOT TIDINESS.
                        #
                        # The stall exists to unblock a curriculum wedged at an
                        # INTERMEDIATE stage. At the target there is nothing to
                        # unblock: the advance branch above caps at
                        # min(CURR_TARGET, ...), so no amount of success can
                        # move the band, `last_advance_ep` therefore never
                        # updates, and the stall fires on a timer.
                        #
                        # ON A RESUMED RUN THAT IS AUTOMATIC. Resume restores
                        # curr_max from the checkpoint -- already at the target
                        # -- while last_advance_ep starts at 0, so the counter
                        # runs from episode 0 of the new process and trips at
                        # exactly CURR_STALL every time. MEASURED today across
                        # several restarts: "STALLED 400 episodes at 0.38 ->
                        # stepping back to 20.0-90.4 m (episode 400)", fired
                        # while the fixed-set eval was RISING 0.283 -> 0.400 at
                        # that very band.
                        #
                        # That is the failure this branch's own comment warns
                        # about -- "the rate was RISING through that window, so
                        # an aggressive demote would have been wrong" -- and at
                        # the target it was guaranteed to happen rather than
                        # merely possible. The demote branch below still
                        # catches genuine collapse at <= CURRICULUM_DEMOTE.
                        # STALL. The promote/demote pair leaves a DEAD BAND
                        # between them, and a run can sit in it indefinitely.
                        # MEASURED: at 113.55 m the policy held 0.29-0.44 for
                        # 300+ episodes -- under the 0.50 promote threshold so
                        # it could not advance, and far above the 0.15 demote
                        # threshold so it could not step back either. The
                        # curriculum simply stopped.
                        #
                        # Note the rate was RISING through that window
                        # (0.29 -> 0.44), so an aggressive demote would have
                        # been wrong; this is why the stall window is generous
                        # rather than tight. It only fires when a stage has had
                        # a genuinely long chance and still not cleared.
                        _back = (curr_max / CURR_MUL if CURR_MUL > 1.0
                                 else curr_max - CURR_STEP)
                        curr_max = max(CURR_START, round(_back, 2))
                        os.environ["GOAL_DIST_MAX"] = str(curr_max)
                        last_advance_ep = total_eps
                        recent["success_policy"].clear()
                        print(f"[Curriculum] STALLED {CURR_STALL} episodes at "
                              f"{_rate:.2f} -> stepping back to {CURR_MIN}-"
                              f"{curr_max} m (episode {total_eps})", flush=True)
                    elif curr_max > CURR_START and _rate <= CURR_DEMOTE:
                        # DEMOTE. Widening can outrun the policy, and without a
                        # way back the run sticks at a range it cannot learn
                        # from -- it then has no positive signal at all, which
                        # under a sparse reward is unrecoverable. Recommended
                        # by the peer, whose GoalRangeCurriculum promotes at
                        # 0.6 and demotes at 0.15.
                        _back = (curr_max / CURR_MUL if CURR_MUL > 1.0
                                 else curr_max - CURR_STEP)
                        curr_max = max(CURR_START, round(_back, 2))
                        os.environ["GOAL_DIST_MAX"] = str(curr_max)
                        last_advance_ep = total_eps
                        recent["success_policy"].clear()
                        print(f"[Curriculum] policy success {_rate:.2f} <= "
                              f"{CURR_DEMOTE} -> DEMOTE to {CURR_MIN}-{curr_max} m "
                              f"(episode {total_eps})", flush=True)
                # ORACLE-DRIVEN AND POLICY-DRIVEN EPISODES END FOR DIFFERENT
                # REASONS, AND THE POOLED HISTOGRAM CANNOT SHOW IT.
                #
                # MEASURED on the 100-200 m point-goal run: the scripted expert
                # -- GlobalRoutePlanner route, PID control, competent by
                # construction -- reached the goal on only 0.52 of its episodes
                # while the policy managed 0.126. A driver that good failing
                # half the time means either the wedge rule is too strict for
                # ANY driver, or the expert is worse than its description. The
                # pooled [Why] histogram is identical under both, so it cannot
                # separate them; splitting by who drove does.
                #
                # Measurement only. Nothing here changes what wedges, what
                # terminates, or what the agent is trained on.
                _who = "orc" if is_expert_ep[i] else "pol"
                ep_by_driver[_who] += 1
                for rname, fired in (fi.get("done_reasons") or {}).items():
                    if fired:
                        end_reasons[rname] = end_reasons.get(rname, 0) + 1
                        _kw = f"{rname}/{_who}"
                        end_reasons[_kw] = end_reasons.get(_kw, 0) + 1
                # WHICH MARKING THE WEDGE FIRED ON. Measurement only -- nothing
                # here changes what wedges.
                #
                # The wedge triggers on `diff_lane`, which is
                # `'lane_invasion' in sensor_data` and so fires on ANY crossing,
                # including a legal dashed line at a junction that a
                # route-following expert cannot avoid (the config says exactly
                # this three lines below the wedge). With JUNCTION_GOAL_FRAC=0.6
                # routing 60% of episodes through junctions and lane_any
                # measured at 0.52, "the wedge is firing on legal crossings" is
                # a hypothesis the aggregate CANNOT distinguish from "the agent
                # drives badly" -- both produce one lane_any count.
                #
                # Splitting by marking type does distinguish them: Broken and
                # NONE are legal to cross, Solid is not.
                # Tagged with WHERE the crossing happened, because that is what
                # the wedge's exemption turns on: the same Broken marking is a
                # legal turn inside a junction and a drift out of lane outside
                # one. Untagged counts cannot tell those apart.
                _where = "@junc" if fi.get("in_junction") else "@road"
                for _mt in (fi.get("lane_marking_types") or []):
                    _k = "mark:" + str(_mt).split(".")[-1] + _where
                    end_reasons[_k] = end_reasons.get(_k, 0) + 1
                # OFF-ROAD TAGGED BY LOCATION, for the same reason the markings
                # are. off_road is get_waypoint(project_to_road=False) is None,
                # and the env config records that probe as LEAST RELIABLE inside
                # junctions -- "lane polygons meet with gaps there, and a vehicle
                # mid-turn can fall between them while driving the route
                # correctly". With JUNCTION_GOAL_FRAC=0.7 most episodes pass
                # through exactly that geometry, so "the policy left the road"
                # and "the probe glitched mid-turn" produce the same count and
                # the aggregate cannot tell them apart. Measurement only.
                if (fi.get("done_reasons") or {}).get("off_road"):
                    _ok = "off_road" + _where
                    end_reasons[_ok] = end_reasons.get(_ok, 0) + 1
                    _okw = f"{_ok}/{_who}"
                    end_reasons[_okw] = end_reasons.get(_okw, 0) + 1
                if total_eps % LOG_EVERY == 0:
                    tot = max(1, total_eps)
                    def _denom(k):
                        # A "/orc" or "/pol" key is a rate over THAT driver's
                        # episodes, not over all of them.
                        if k.endswith("/orc"):
                            return max(1, ep_by_driver["orc"])
                        if k.endswith("/pol"):
                            return max(1, ep_by_driver["pol"])
                        return tot
                    hist = "  ".join(
                        f"{k}={v / _denom(k):.2f}" for k, v in
                        sorted(end_reasons.items(), key=lambda kv: -kv[1])
                        if not k.startswith("mark:"))
                    split = "  ".join(
                        f"{k}={v / _denom(k):.2f}" for k, v in
                        sorted(end_reasons.items())
                        if k.endswith(("/orc", "/pol"))
                        and not k.startswith("mark:"))
                    print(f"[Why {total_eps:5d}] {hist}  | goals "
                          f"{CURR_MIN:.1f}-{curr_max:.1f}m", flush=True)
                    # THE TWO UNCALIBRATED NUMBERS IN THE EXPLORE ARM.
                    #
                    # admit_rate: the novelty threshold (0.9) was calibrated on
                    # SYNTHETIC images, and road frames are far more
                    # self-similar. A rate near 0 means the buffer never grows
                    # and the bonus is inert; near 1 means everything is stored
                    # and max-similarity saturates. Either way the reward term
                    # is not doing what it looks like it is doing.
                    #
                    # drift: DR_WHEELBASE_M and DR_MAX_STEER_RAD have never been
                    # fitted for this vehicle, and an unfitted gain manufactures
                    # drift that looks like physics. If this grows fast, the
                    # visit ring is being queried in a frame that no longer
                    # matches where the vehicle is.
                    if os.environ.get("LANE_EXPLORE", "0") == "1":
                        _d = fi.get("dr_drift") or {}
                        print(f"[Explore {total_eps:5d}] "
                              f"novelty_buf={fi.get('novelty_buffer', 0)} "
                              f"novelty_raw={fi.get('novelty_raw', 0.0):.3f} "
                              f"drift_m={_d.get('position_m', float('nan')):.2f} "
                              f"drift_rad={_d.get('heading_rad', float('nan')):.3f} "
                              f"over {_d.get('steps', 0)} steps", flush=True)
                    if split:
                        print(f"[Who {total_eps:5d}] {split}  | "
                              f"orc_eps={ep_by_driver['orc']} "
                              f"pol_eps={ep_by_driver['pol']}", flush=True)
                    marks = "  ".join(
                        f"{k}={v / tot:.2f}" for k, v in
                        sorted(end_reasons.items(), key=lambda kv: -kv[1])
                        if k.startswith("mark:") and not k.endswith(("/orc", "/pol")))
                    if marks:
                        print(f"[Mark {total_eps:5d}] {marks}", flush=True)
                    if last_update_info:
                        q = last_update_info.get("q", float("nan"))
                        cl = last_update_info.get("critic_loss", float("nan"))
                        td = last_update_info.get("td_errors", float("nan"))
                        ent = last_update_info.get("target_actor_entropy",
                                                   float("nan"))
                        flag = ""
                        if abs(q) > 2000:
                            flag = "  <-- Q FAR OUTSIDE [-5, 600]: critic diverging"
                        # q_scale_* (logging only): the per-mode value spread
                        # ACTOR_Q_NORM divides the actor's -Q by. Present in
                        # the update info only when the normalisation is on;
                        # its drift is what QNORM_CRITIC_ALPHA couples the
                        # critics' entropy backups to.
                        _qs = last_update_info.get("q_scale_goal")
                        print(f"[Critic{total_eps:5d}] q={q:10.2f} "
                              f"critic_loss={cl:12.2f} td={td:9.3f} "
                              f"actor_entropy={ent:7.3f}{flag}"
                              + ("" if _qs is None else f" q_scale={_qs:.3f}"),
                              flush=True)
                        if GATED:
                            # THE SECOND CRITIC IS INVISIBLE IN THE LINE ABOVE,
                            # which reports only the goal critic. A starved or
                            # diverging explore critic would leave no trace.
                            #
                            # WHAT GOOD LOOKS LIKE under SORB: goal q descends
                            # toward -(steps to goal), so it should approach the
                            # goal `ret` in the [GatedEp] line. If it sits near
                            # zero while ret is -500, the goal critic has not
                            # learnt the scale. Per-mode entropy falling toward
                            # the target (-2) is normal; collapsing far below it
                            # means that mode's policy has stopped exploring.
                            if wants_one_critic():
                                # THE ONE-CRITIC ROUTE HAS NO SECOND CRITIC
                                # (and no per-mode split to report): the plain
                                # TQCLearner's update_info carries only the
                                # mode-blind keys -- q, critic_loss, td_errors,
                                # target_actor_entropy, entropy -- all printed
                                # in the [Critic] line above. The q_e /
                                # critic_loss_e / td_errors_e / entropy_goal /
                                # entropy_explore / frac_goal keys this line
                                # once read are emitted only by the two-critic
                                # GatedTQCLearner's update, so under
                                # GATED_ONE_CRITIC=1 they printed NaN forever.
                                # Splitting the one critic's q by mode would
                                # need learner surgery (the batch's mode bits
                                # never leave the jitted update), so the line
                                # says so instead.
                                print(f"[CriticE{total_eps:4d}] N/A under "
                                      f"GATED_ONE_CRITIC (single mode-blind "
                                      f"critic line above)", flush=True)
                            else:
                                _g = last_update_info
                                _qe = _g.get("q_scale_explore")
                                print(f"[CriticE{total_eps:4d}] "
                                      f"q_e={_g.get('q_e', float('nan')):10.2f} "
                                      f"critic_loss_e="
                                      f"{_g.get('critic_loss_e', float('nan')):10.2f} "
                                      f"td_e={_g.get('td_errors_e', float('nan')):8.3f} "
                                      f"| ent_goal="
                                      f"{_g.get('entropy_goal', float('nan')):7.3f} "
                                      f"ent_explore="
                                      f"{_g.get('entropy_explore', float('nan')):7.3f} "
                                      f"frac_goal="
                                      f"{_g.get('frac_goal', float('nan')):.2f}"
                                      + ("" if _qe is None
                                         else f" q_scale_e={_qe:.3f}"),
                                      flush=True)
                                # ASYMMETRIC (OBS_PRIV): the privileged goal
                                # critic the actor follows. The [Critic] line
                                # above is the IMAGE-ONLY goal critic (the map
                                # distance); the two should track each other.
                                if "q_p" in _g:
                                    print(f"[CriticP{total_eps:4d}] "
                                          f"q_p={float(_g['q_p']):10.2f} "
                                          f"critic_loss_p="
                                          f"{float(_g['critic_loss_p']):10.2f} "
                                          f"td_p={float(_g['td_errors_p']):8.3f} "
                                          f"(privileged, drives the actor; "
                                          f"[Critic] = image-only)", flush=True)
                    # success= IS AGENT-ONLY (user 2026-09-24, "report only
                    # agent success rate"). The old value averaged EVERY
                    # episode -- expert-driven goal episodes (~0.9 success),
                    # agent goal episodes, and gated explore episodes, whose
                    # success is ~0 by construction -- so it tracked the
                    # expert share and the mode mix, not the policy. Gated
                    # runs report agent-driven GOAL episodes (succ_goal);
                    # single-mode runs report agent-driven episodes. Logging
                    # only: the curriculum reads the success_policy deque,
                    # never this text. policy= (agent, both modes) and
                    # oracle= keep their old meaning so existing log
                    # comparisons still line up.
                    _agent = recent["succ_goal"] if GATED else recent["success_policy"]
                    print(f"[Ep {total_eps:5d}] step={step:7d} town={venv.names[i]:8s} "
                          f"success={np.mean(_agent) if _agent else float('nan'):.3f}"
                          f"(n={len(_agent)}) "
                          f"collision={np.mean(recent['collision']):.3f} "
                          f"goals={np.mean(recent['goals']) if recent['goals'] else 0.0:.2f} "
                          f"len={np.mean(recent['length']):6.1f} "
                          f"ret={np.mean(recent['return']):8.1f} "
                          f"| policy={np.mean(recent['success_policy']) if recent['success_policy'] else float('nan'):.3f}"
                          f"(n={len(recent['success_policy'])}) "
                          # GOAL-ARM LANE CROSSINGS (user 2026-09-23): share of
                          # agent-driven goal episodes in the window ended by
                          # the crossing cap (episode_her.last_end_reason).
                          f"cross={(sum(1 for _r in recent['end_goal'] if _r == 'crossing') / max(1, len(recent['end_goal']))) if recent['end_goal'] else float('nan'):.3f}"
                          f"(n={len(recent['end_goal'])}) "
                          f"oracle={np.mean(recent['success_oracle']) if recent['success_oracle'] else float('nan'):.3f}"
                          f"(n={len(recent['success_oracle'])}) buffer={len(buffer)}"
                          + (f" her={hers[i].stats()['her_relabelled']}"
                             f" paying={hers[i].stats()['her_paying']}"
                             f"({hers[i].stats()['her_paying_frac']:.3f})"
                             + (f" entry_done={hers[i].stats()['her_entry_done']}"
                                if getattr(hers[i], "IMAGE_GOAL", False) else "")
                             if hers else ""),
                          flush=True)
                    if GATED:
                        # THE LINE THAT IS ACTUALLY READABLE FOR THIS ARM.
                        # The [Ep] line above averages over both modes, whose
                        # returns differ in sign and by two orders of magnitude,
                        # so its `ret` and `success` describe neither task.
                        #
                        # goal ret tracks the STEPS taken: under the SORB
                        # reward the return IS minus the time taken, and under
                        # this arm's 2026-09-20-evening reward it is -1/step
                        # plus the saturating speed bonus (up to +0.2/step),
                        # so -412 still means roughly 412 goal steps. Success
                        # is reported for goal episodes ONLY.
                        def _m(key, default=float("nan")):
                            return np.mean(recent[key]) if recent[key] else default
                        n_g, n_e = len(recent["ret_goal"]), len(recent["ret_explore"])
                        frac_g = n_g / max(1, n_g + n_e)
                        # GOAL-ARM LANE CROSSINGS (user 2026-09-23): share of
                        # agent-driven goal episodes in the window that ended
                        # on the crossing cap.
                        from collections import Counter as _Cg
                        _ge = _Cg(recent["end_goal"])
                        _gt = max(1, sum(_ge.values()))
                        _cross = _ge.get("crossing", 0) / _gt
                        # On the ONE-CRITIC route the batch mix is NOT fixed:
                        # the single plain buffer samples uniformly, so the
                        # GOAL_BATCH_FRAC parenthetical would be a lie there.
                        _mix = ("" if wants_one_critic() else
                                " (batch mix is FIXED at GOAL_BATCH_FRAC "
                                "regardless)")
                        print(f"[GatedEp{total_eps:5d}] "
                              f"GOAL   n={n_g:3d} succ(agent-only)={_m('succ_goal'):.3f} "
                              f"cross={_cross:.2f} "
                              f"ret={_m('ret_goal'):8.1f} (~ -steps) "
                              f"len={_m('len_goal'):6.1f} "
                              f"coll={_m('coll_goal'):.3f}", flush=True)
                        from collections import Counter as _C
                        _ends = _C(recent["end_explore"])
                        _tot = max(1, sum(_ends.values()))
                        _endstr = " ".join(
                            f"{k}={v / _tot:.2f}" for k, v in
                            sorted(_ends.items(), key=lambda kv: -kv[1])[:4])
                        print(f"[GatedEp{total_eps:5d}] "
                              f"EXPLORE n={n_e:3d} "
                              f"ret={_m('ret_explore'):8.1f} "
                              f"len={_m('len_explore'):6.1f} "
                              f"dist={_m('dist_explore'):6.1f}m "
                              f"ends[{_endstr}] "
                              f"coll={_m('coll_explore'):.3f} "
                              f"| episode mix goal={frac_g:.2f}"
                              f"{_mix}", flush=True)
                if hers is not None:
                    # THE DISPATCH IS ON A CLASS ATTRIBUTE, not an isinstance
                    # import: the base call site stays `flush_episode()` bare
                    # and only the image-goal buffer accepts `truncated=`.
                    # `fi` is the terminal step's info and DOES carry a real
                    # truncation key -- vec_env.step stuffs inf["truncated"]
                    # (and inf["terminated"]) into the step info it appends to
                    # final_infos. A truncated timeout's last frame is a clock
                    # artefact, not a destination, so the buffer must see it.
                    if getattr(hers[i], "IMAGE_GOAL", False):
                        hers[i].flush_episode(
                            truncated=bool(fi.get("truncated", False)))
                    else:
                        hers[i].flush_episode()
                    # GOAL-SIDE END REASON (user 2026-09-23): the buffer just
                    # classified this episode from its fault counters. Record
                    # agent-driven goal episodes only, matching succ_goal.
                    if (int(fi.get("mode", 1)) == 1 and not is_expert_ep[i]):
                        recent["end_goal"].append(
                            str(getattr(hers[i], "last_end_reason", "?")))

                # New episode -> re-roll the oracle coin and replan.
                # Uses the POST-RESET info, not `fi`: `fi` is the finished
                # episode's terminal info and carries the OLD goal, so arming
                # from it would plan a route to a goal that no longer exists.
                new_info = (venv.reset_infos[i] or {}) if hasattr(venv, "reset_infos") else {}
                _arm_expert(i, new_info)
                n_expert_eps += int(is_expert_ep[i])
                if total_eps % LOG_EVERY == 0:
                    fails = "  ".join(f"{k[:52]}={v}" for k, v in
                                      sorted(arm_failures.items(),
                                             key=lambda kv: -kv[1])[:2])
                    print(f"[Expert] {n_expert_eps}/{total_eps} oracle-driven "
                          f"({n_expert_eps / max(1, total_eps):.2f}, target "
                          f"{EXPERT_PROB})" +
                          (f" | plan fails: {fails}" if fails else ""), flush=True)

                # Rotate this env onto the next town. Done at an episode
                # boundary and AFTER the HER flush, so no hindsight trajectory
                # ever spans two maps.
                if (ROTATE_EPISODES > 0 and len(rotation) > 1
                        and ep_done[i] % ROTATE_EPISODES == 0):
                    # Advance by n_envs so envs never collide on the same town.
                    rot_idx[i] = (rot_idx[i] + n_envs) % len(rotation)
                    new_town = rotation[rot_idx[i]]
                    assert_not_heldout(new_town, heldout)
                    port = CARLA_PORTS[i]
                    print(f"[Rotate] env{i}: -> {new_town} after "
                          f"{ep_done[i]} episodes (world reload ~15s)", flush=True)
                    # RESPAWN, DON'T RELOAD. load_world on the long-lived
                    # shared server is the one operation known to crash CARLA
                    # 0.9.15 (:1248) -- the v8 Town01->Town10HD switch
                    # segfaulted the server and 25 reconnect attempts died
                    # against the port. Killing the server here turns the
                    # switch into a fresh boot plus the env build's ONE
                    # load_world, the path every startup survives;
                    # ensure_carla (called inside make_env below) respawns.
                    # Guarded on CARLA_START_SERVER because with =0 there is
                    # no respawn path and this would leave no server at all.
                    if os.environ.get("CARLA_START_SERVER", "1") == "1":
                        kill_carla_server(port)
                        print(f"[Rotate] env{i}: server killed for a fresh "
                              f"spawn on {new_town}", flush=True)
                    new_obs, _ = venv.replace(
                        i, lambda t=new_town, p=port, ix=i: make_env(
                            t, SEED + step + ix, latent_source, port=p),
                        name=new_town)
                    for k in next_obs:
                        next_obs[k][i] = new_obs[k]
                    if hers is not None:
                        w = venv.envs[i]
                        while w is not None and not isinstance(w, VAEGoalWrapper):
                            w = getattr(w, "env", None)
                        if w is not None:
                            hers[i].state_fn = w.goal_block
                    # New world, new actors: the old oracle holds a destroyed
                    # vehicle handle.
                    experts[i] = make_expert(venv.envs[i], venv.action_space)
                    # Arm from the REBUILT env's reset info, not an empty dict.
                    # With {} there is no goal_location, so the oracle could
                    # never take the first episode after a town rotation --
                    # surfaced by the diagnostic as no_goal_in_info.
                    try:
                        _rst = venv.envs[i].reset()
                        _rinf = _rst[1] if isinstance(_rst, tuple) else {}
                    except Exception:  # noqa: BLE001
                        _rinf = {}
                    _arm_expert(i, _rinf)

        obs = next_obs
        step += n_envs

        if step >= START_TRAINING and len(buffer) > MINIBATCH * UTD_RATIO:
            # jaxrl2 implements UTD by SPLITTING one sampled batch into
            # utd_ratio minibatches -- NOT by drawing utd_ratio fresh batches.
            batch = buffer.sample(MINIBATCH * UTD_RATIO)
            # KEEP the learner's own metrics. update() returns critic_loss, q
            # and td_errors and the trainer was discarding all of it -- so four
            # held-out evaluations at 0.000 were interpreted with no visibility
            # into whether the critic had learned anything at all.
            #
            # tqc_learner.py's own docstring records that DrQ's min failed to
            # contain overestimation three times in this project, reaching
            # Q +5,500 and critic_loss 2.8e7 against a ~600 return ceiling.
            # With k_goal=600 the sane range for Q is roughly [-5, 600]; values
            # far outside it mean the critic is diverging rather than learning,
            # which no amount of reward or HER tuning would fix.
            _info = agent.update(agent_view(batch), utd_ratio=UTD_RATIO)
            if _info is not None:
                last_update_info = {k: float(np.asarray(v).mean())
                                    for k, v in _info.items()
                                    if np.asarray(v).size}

        if step % EVAL_INTERVAL < n_envs and step >= START_TRAINING:
            # N_EVAL_EPISODES=0 SKIPS THE EVAL ENTIRELY. build_eval_episodes floors the count to at least one episode
            # per town, so "0" still built an eval env -- and that env attaches to CARLA_PORTS[0], the server a
            # training env is running on, and reloads its world. Measured 2026-09-16 (v7, 2 envs): the server
            # segfaulted at the 25k eval, the trainer hung reconnecting, and since the checkpoint is written after
            # the eval nothing was saved. Skipping keeps the checkpoint cadence; evaluate checkpoints offline.
            if N_EVAL_EPISODES > 0:
                rate, by_town = run_evaluation(agent, eval_episodes, latent_source,
                                               identity, step=step, out_dir=out_dir)
            else:
                rate, by_town = 0.0, {}
                print(f"[Eval] skipped at step {step} (N_EVAL_EPISODES=0); checkpoint still written", flush=True)
            # THE EXPLORE HALF, measured at every eval. N_EVAL_EXPLORE=0
            # turns it off; it costs one env build and N short episodes.
            _n_expl = int(os.environ.get("N_EVAL_EXPLORE", 10))
            _expl_dist, _expl_ends = 0.0, {}
            if _n_expl > 0:
                try:
                    _expl_dist, _expl_ends, _ = run_explore_evaluation(
                        agent, env_towns[0], _n_expl, latent_source, step=step)
                except Exception as _e:            # noqa: BLE001
                    print(f"[Eval] EXPLORE eval failed: "
                          f"{type(_e).__name__}: {_e}", flush=True)
            # THE WEIGHTS GO IN THE FILE. Every checkpoint before this one
            # held run metadata only -- step, success rate, HER counters -- so
            # a completed run left nothing that could be replayed, watched, or
            # re-evaluated. The policy existed solely inside a process that
            # then exited.
            agent_st = agent_state(agent)
            with open(os.path.join(out_dir, f"checkpoint_{step}.pkl"), "wb") as f:
                pickle.dump({"step": step, "identity": identity,
                             "eval_success": rate,
                             "eval_explore_dist_m": _expl_dist,
                             "eval_explore_ends": _expl_ends,
                             "eval_by_town": {k: float(np.mean(v))
                                              for k, v in by_town.items()},
                             "episodes_per_env": ep_done.tolist(),
                             "agent": agent_st,
                             "fingerprint": param_fingerprint(agent_st),
                             # CURRICULUM STATE. Without it a resume restarts
                             # the schedule at GOAL_DIST_START while holding a
                             # policy trained far past it -- MEASURED on the
                             # first resume: the run came back at 5 m with a
                             # policy that had been working at 113 m, and had
                             # to re-clear ~17 stages at 50 episodes each
                             # before seeing the target again.
                             "curriculum_max": float(curr_max),
                             "her": [h.stats() for h in hers] if hers else None}, f)

    venv.close()
    print("[Done] episodes per env:", ep_done.tolist(),
          "| HER:", [h.stats() for h in hers] if hers else "off")


if __name__ == "__main__":
    main()

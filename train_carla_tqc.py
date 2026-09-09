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
from datetime import datetime

import numpy as np

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
from vae_nav.expert_oracle import ExpertOracle, should_use_expert  # noqa: E402
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
        return ExpertOracle(hero, hero.get_world().get_map(), action_space)
    except Exception as e:  # noqa: BLE001
        print(f"[Expert] WARNING: could not build oracle ({e}); expert disabled.")
        return None


def build_agent(observation_space, action_space, seed):
    """TQC with the peer's config of record, kappa derived from the reward."""
    from jaxrl2.agents import TQCLearner

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
        target_entropy=float(os.environ.get("TARGET_ENTROPY", -2.0)),
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
    n_drop = kwargs["top_quantiles_to_drop_per_net"] * kwargs["num_qs"]
    total = kwargs["num_quantiles"] * kwargs["num_qs"]
    print(f"[Algo] TQC num_qs={kwargs['num_qs']} "
          f"n_quantiles={kwargs['num_quantiles']} "
          f"drop_per_net={kwargs['top_quantiles_to_drop_per_net']} "
          f"(truncating {n_drop} of {total}) kappa={kwargs['kappa']:g}")
    if os.environ.get("GATED", "0") == "1":
        from vae_nav.gated_tqc_learner import GatedTQCLearner
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
        kwargs.update(
            discount_e=float(os.environ.get("EXPLORE_DISCOUNT",
                                            kwargs["discount"])),
            explore_actor_warmup=int(os.environ.get("EXPLORE_ACTOR_WARMUP", 0)),
            value_floor=float(os.environ.get("TQC_VALUE_FLOOR", floor))
            if (floor is not None or "TQC_VALUE_FLOOR" in os.environ) else None,
            value_ceiling=float(os.environ.get("TQC_VALUE_CEILING", ceiling))
            if (ceiling is not None or "TQC_VALUE_CEILING" in os.environ)
            else None,
        )
        return GatedTQCLearner(seed, observation_space.sample(),
                               action_space.sample(), **kwargs)
    return TQCLearner(seed, observation_space.sample(), action_space.sample(),
                      **kwargs)


# Shared across every TRAINING env rebuild -- see the note in make_env.
# None until the explore arm asks for it, so the goal arm never builds one.
_EXPLORE_NOVELTY = None


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
    carla_cfg["start_server"] = False
    carla_cfg["port"] = int(port if port is not None else CARLA_PORTS[0])
    env_cfg["carla"] = carla_cfg
    cfg["env_config"] = env_cfg

    env = CarlaEnv(config=cfg)
    # FIRST: CarlaEnv returns an empty info dict; everything below reads plain
    # keys, and this is the only place that reaches into core/experiment.
    env = CoreStateWrapper(env)
    # LANE FOLLOWING uses the env's own per-step distance reward instead of the
    # sparse +600 arrival. LANE_FOLLOW=1 selects it; the sparse path is
    # untouched so existing goal-navigation runs are bit-identical.
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
            max_steps=int(os.environ.get("GATED_EXPLORE_MAX_STEPS", 500)),
            seed=seed)
        env = VAEGoalWrapper(env, source=latent_source)
        # The GOAL arm's cap. GatedReward truncates explore episodes itself at
        # its own, shorter cap, so the two modes keep the episode lengths their
        # separate arms use and the comparison stays honest.
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
            obs, info = env.reset(seed=ep["seed"])
            if recorder is not None:
                recorder.start_episode(ep["index"], ep["town"])
                recorder.record(info.get("rgb"), info)
            done, steps, success = False, 0, False
            while not done and steps < MAX_EPISODE_STEPS:
                action = agent.eval_actions(obs)
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

    rng = np.random.default_rng(SEED)

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
    if os.environ.get("GATED", "0") == "1":
        from vae_nav.mode_split_buffer import ModeSplitBuffer
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
        buffer = ModeSplitBuffer(
            venv.observation_space, venv.action_space, _capacity,
            goal_frac=float(os.environ.get("GOAL_BATCH_FRAC", 0.5)))
    else:
        buffer = ReplayBuffer(venv.observation_space, venv.action_space,
                              _capacity)

    # HER IS PER ENV. Environments finish episodes at different times, and a
    # shared relabel buffer would splice transitions from different towns into
    # one hindsight trajectory -- a "future" goal drawn from another map.
    hers = None
    if os.environ.get("EPISODE_HER", "") == "1":
        hers = []
        for e in venv.envs:
            w = e
            while w is not None and not isinstance(w, VAEGoalWrapper):
                w = getattr(w, "env", None)
            if w is None:
                raise SystemExit("EPISODE_HER=1 needs VAEGoalWrapper in the stack")
            hers.append(EpisodeRelabelBuffer(
                buffer, w.goal_block, goal_radius=GOAL_THRESHOLD, k_goal=K_GOAL,
                k_collision=K_COLLISION, k_lane=K_LANE,
                k_dist=K_DIST,
                k_dist_target_speed=K_DIST_TARGET_SPEED,
                k_dist_gate_mode=K_DIST_GATE_MODE,
                k_speed=K_SPEED, speed_limit_kmh=SPEED_LIMIT_KMH,
                k_dist_delta_mode=K_DIST_DELTA_MODE,
                k_living=K_LIVING,
                # Under TIME_TO_GOAL every sparse coefficient above is dead and
                # the relabelled reward is 0-on-arrival / -step_cost otherwise.
                # Passing the flag is MANDATORY, not optional: without it the
                # hindsight rows carry +600 arrivals into a buffer whose
                # on-policy rows pay 0, and the critic fits two reward
                # functions over one set of states.
                # THE GATED ARM'S GOAL REWARD IS ALSO MINIMUM TIME, under a
                # different flag. GATED_GOAL_REWARD=sorb pays -STEP_COST per
                # step and 0 on arrival exactly as TIME_TO_GOAL does, so the
                # same recosting applies and omitting it here would put +k_goal
                # arrivals into a buffer whose on-policy rows pay 0.
                time_to_goal=(os.environ.get("TIME_TO_GOAL", "") == "1"
                              or (GATED and os.environ.get(
                                  "GATED_GOAL_REWARD", "sorb").lower() == "sorb")),
                step_cost=float(os.environ.get("STEP_COST", 0.1)),
                # Routes on-policy rows by their own mode and forces every
                # hindsight copy to GOAL.
                gated=GATED,
                n_relabel=int(os.environ.get("EPISODE_HER_N", 1)),
                min_disp=float(os.environ.get("HER_MIN_DISP", GOAL_THRESHOLD)),
                max_disp=float(os.environ.get("HER_MAX_DISP", np.inf)),
                seed=SEED + len(hers)))
        print(f"[EpisodeHER] on -- {len(hers)} per-env buffers, "
              f"n_relabel={hers[0].n_relabel} min_disp={hers[0].min_disp} "
              f"goal_radius={hers[0].goal_radius}")

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
        rotation = [t for t in env_towns]
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
        if experts[i] is None or not should_use_expert(rng, EXPERT_PROB):
            is_expert_ep[i] = False
            return
        goal = (info or {}).get("goal_location")
        if goal is None:
            arm_failures["no_goal_in_info"] = arm_failures.get("no_goal_in_info", 0) + 1
            is_expert_ep[i] = False
            return
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
              "coll_goal": deque(maxlen=100), "coll_explore": deque(maxlen=100)}
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
            actions = np.array(agent.sample_actions(obs), dtype=np.float32,
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
                hers[i].record(
                    tr,
                    fields={"hero_pos": inf["hero_pos"], "hero_yaw": inf["hero_yaw"],
                            "goal_pos": inf["goal_pos"],
                            "action_history": inf.get("action_history",
                                                      np.zeros((4, 2)))},
                    pos=inf["hero_pos"], hit=bool(inf.get("collision", False)),
                    lane=bool(inf.get("lane_invasion", False)),
                    speed_kmh=float(inf.get("speed_kmh", 0.0) or 0.0),
                    target_speed_kmh=float(inf.get("target_speed_kmh", 0.0) or 0.0),
                    # ROUTES THE ON-POLICY ROW. Without this the relabel buffer
                    # inserts with no mode and the split buffer defaults every
                    # row to GOAL, filing every explore transition as goal data.
                    # Hindsight copies are forced to GOAL inside the buffer.
                    mode=int(inf.get("mode", 1)))
                # CLOSE THE LEG ON ARRIVAL. Must come AFTER record(), so the
                # arriving transition belongs to the leg it completed rather
                # than opening the next one. Without this a chained episode is
                # one relabelling unit and hindsight goals cross goal
                # boundaries -- see EpisodeRelabelBuffer.end_leg.
                if inf.get("arrived_this_step"):
                    hers[i].end_leg()
            else:
                if GATED:
                    # Routed by the mode this transition was collected under.
                    # Defaults to GOAL if absent, matching the observation
                    # wrapper, so a missing GatedReward degrades to the
                    # familiar single-mode behaviour rather than silently
                    # filing every row as exploration.
                    buffer.insert(tr, mode=int(inf.get("mode", 1)))
                else:
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
                        recent["succ_goal"].append(_succ)
                        recent["coll_goal"].append(_coll)
                    else:
                        recent["ret_explore"].append(_ep_ret)
                        recent["len_explore"].append(_ep_len)
                        recent["coll_explore"].append(_coll)

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
                        print(f"[Critic{total_eps:5d}] q={q:10.2f} "
                              f"critic_loss={cl:12.2f} td={td:9.3f} "
                              f"actor_entropy={ent:7.3f}{flag}", flush=True)
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
                            _g = last_update_info
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
                                  f"{_g.get('frac_goal', float('nan')):.2f}",
                                  flush=True)
                    print(f"[Ep {total_eps:5d}] step={step:7d} town={venv.names[i]:8s} "
                          f"success={np.mean(recent['success']):.3f} "
                          f"collision={np.mean(recent['collision']):.3f} "
                          f"goals={np.mean(recent['goals']) if recent['goals'] else 0.0:.2f} "
                          f"len={np.mean(recent['length']):6.1f} "
                          f"ret={np.mean(recent['return']):8.1f} "
                          f"| policy={np.mean(recent['success_policy']) if recent['success_policy'] else float('nan'):.3f}"
                          f"(n={len(recent['success_policy'])}) "
                          f"oracle={np.mean(recent['success_oracle']) if recent['success_oracle'] else float('nan'):.3f}"
                          f"(n={len(recent['success_oracle'])}) buffer={len(buffer)}"
                          + (f" her={hers[i].stats()['her_relabelled']}"
                             f" paying={hers[i].stats()['her_paying']}"
                             f"({hers[i].stats()['her_paying_frac']:.3f})"
                             if hers else ""),
                          flush=True)
                    if GATED:
                        # THE LINE THAT IS ACTUALLY READABLE FOR THIS ARM.
                        # The [Ep] line above averages over both modes, whose
                        # returns differ in sign and by two orders of magnitude,
                        # so its `ret` and `success` describe neither task.
                        #
                        # goal ret is in STEPS: under the SORB reward the return
                        # IS minus the time taken, so -412 means the goal took
                        # 412 steps or ran out the clock. Success is reported
                        # for goal episodes ONLY.
                        def _m(key, default=float("nan")):
                            return np.mean(recent[key]) if recent[key] else default
                        n_g, n_e = len(recent["ret_goal"]), len(recent["ret_explore"])
                        frac_g = n_g / max(1, n_g + n_e)
                        print(f"[GatedEp{total_eps:5d}] "
                              f"GOAL   n={n_g:3d} succ={_m('succ_goal'):.3f} "
                              f"ret={_m('ret_goal'):8.1f} (= -steps) "
                              f"len={_m('len_goal'):6.1f} "
                              f"coll={_m('coll_goal'):.3f}", flush=True)
                        print(f"[GatedEp{total_eps:5d}] "
                              f"EXPLORE n={n_e:3d} "
                              f"ret={_m('ret_explore'):8.1f} "
                              f"len={_m('len_explore'):6.1f} "
                              f"coll={_m('coll_explore'):.3f} "
                              f"| episode mix goal={frac_g:.2f} "
                              f"(batch mix is FIXED at GOAL_BATCH_FRAC "
                              f"regardless)", flush=True)
                if hers is not None:
                    hers[i].flush_episode()

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
            _info = agent.update(batch, utd_ratio=UTD_RATIO)
            if _info is not None:
                last_update_info = {k: float(np.asarray(v).mean())
                                    for k, v in _info.items()
                                    if np.asarray(v).size}

        if step % EVAL_INTERVAL < n_envs and step >= START_TRAINING:
            rate, by_town = run_evaluation(agent, eval_episodes, latent_source,
                                           identity, step=step, out_dir=out_dir)
            # THE WEIGHTS GO IN THE FILE. Every checkpoint before this one
            # held run metadata only -- step, success rate, HER counters -- so
            # a completed run left nothing that could be replayed, watched, or
            # re-evaluated. The policy existed solely inside a process that
            # then exited.
            agent_st = agent_state(agent)
            with open(os.path.join(out_dir, f"checkpoint_{step}.pkl"), "wb") as f:
                pickle.dump({"step": step, "identity": identity,
                             "eval_success": rate,
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

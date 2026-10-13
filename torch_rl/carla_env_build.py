#!/usr/bin/env python3
"""The CARLA env for the torch recurrent-SAC arm, with no JAX anywhere in it.

User 2026-10-12: park TQC, move the recurrent SAC policy to the cluster. The
cluster arm is torch end to end (user 2026-10-04, "torch only with
optimizations"), and the ACG cluster's working environment is ONE conda env
holding the CARLA 0.9.16 client and torch 2.5.1 -- there is no jaxlib in it and
there is not going to be.

THIS IS NOT A FORK OF make_env. The env stack it builds is the same one the v52
lineage trains on, wrapper for wrapper, and every wrapper it uses is imported
from src/vae_nav unchanged. What could not be reused is train_carla_tqc.make_env
itself: that module imports jax at line 51, so importing it anywhere in this arm
would drag jaxlib into the cluster env. Only the ~40 lines of config assembly are
restated here, and each one that matters carries the reason it is not a default.

WHAT MUST NOT DRIFT FROM make_env, each of these having cost a run:
 * THE TOWN GOES ON THE CARLA BLOCK. CarlaCore loads the world from
   cfg["env_config"]["carla"]["town"], where the base config hardcodes Town01.
   Setting only the top-level "town" made every env load Town01 whatever was
   asked, while the banner printed the town requested -- every result before
   2026-09-10 evening describes Town01.
 * start_server=False. With CarlaCore's default the core spawns a fresh server on
   a RANDOM port per env and its close() SIGKILLs every process with "carla" in
   its name anywhere on the machine, including other users'. On a shared cluster
   node that is not a tidiness problem, it is someone else's job dying.
 * is_short_horizon. The sampler's long-horizon branch draws goals 120-240 m
   away, which at the measured speed cannot be reached inside the step cap: that
   is why an entire early lineage scored zero arrivals in every configuration.

WHAT IS DELIBERATELY DIFFERENT ON THE CLUSTER:
 * NO SERVER IS STARTED FROM PYTHON. Locally ensure_carla() starts one and the
   run owns it; in a Slurm job the batch script starts CARLA inside apptainer
   and the trainer attaches. CARLA_START_SERVER is therefore refused rather than
   honoured -- silently ignoring it would leave a second simulator competing for
   the same GPU.
 * RENDER QUALITY COMES FROM THE SERVER'S COMMAND LINE, not from this config.
   With start_server=False the client's quality_level is inert, and CarlaCore
   only ever passed it for a couple of towns anyway. The sbatch script owns it.
 * PERCEPTION IS DEBiT, and the jax perception backends are refused: the seg-VAE
   and the lane VAE are flax modules that cannot load here at all.
"""
import os
from typing import Callable, Optional

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src")

JAX_PERCEPTIONS = ("segvae", "vae", "vc1")


def env_config(town: str, port: int, base: Optional[dict] = None,
               timeout: Optional[float] = None) -> dict:
    """The rllib config dict for one env on one town, attached to one server.

    Pure: no CARLA, no server, no import of the trainer. Tested directly, because
    every defect this function can have is a silent one -- a wrong town or a
    self-spawning core both produce a run that looks healthy and measures
    something else.
    """
    import copy
    if base is None:
        import sys
        if SRC not in sys.path:
            sys.path.insert(0, SRC)
        from src.configs.train_env_config_v4_image_goal import config as base
    cfg = copy.deepcopy(dict(base))

    exp = dict(cfg.get("experiment", {}))
    hero = dict(exp.get("hero", {}))
    hero["is_short_horizon"] = os.environ.get("SHORT_HORIZON", "1") == "1"
    exp["hero"] = hero
    cfg["experiment"] = exp
    cfg["town"] = town                      # provenance only; CarlaCore ignores it

    env_cfg = dict(cfg.get("env_config", {}))
    carla_cfg = dict(env_cfg.get("carla", {}))
    carla_cfg["town"] = town                # THE one CarlaCore reads
    carla_cfg["port"] = int(port)
    carla_cfg["start_server"] = False
    if timeout is None and os.environ.get("CARLA_TIMEOUT"):
        timeout = float(os.environ["CARLA_TIMEOUT"])
    if timeout is not None:
        carla_cfg["timeout"] = float(timeout)
    env_cfg["carla"] = carla_cfg
    cfg["env_config"] = env_cfg
    return cfg


def check_cluster_env() -> None:
    """Refuse the combinations that cannot work in this arm, before a GPU is taken."""
    if os.environ.get("CARLA_START_SERVER", "0") == "1":
        raise SystemExit(
            "CARLA_START_SERVER=1 is refused in the cluster arm: the Slurm "
            "script starts CARLA inside apptainer and the trainer attaches. "
            "Starting a second simulator would put two of them on one GPU.")
    perception = os.environ.get("PERCEPTION", "debit").lower()
    if perception in JAX_PERCEPTIONS:
        raise SystemExit(
            f"PERCEPTION={perception} is a JAX/flax backend and there is no "
            "jaxlib in the cluster environment. This arm's perception IS the "
            "DEBiT binocular encoder (PERCEPTION=debit).")
    if os.environ.get("GATED", "0") != "1":
        raise SystemExit("this arm is the gated two-mode policy: set GATED=1")


def build_env(town: str, seed: int, encoder=None, port: int = 2000,
              eval_mode: bool = False, max_episode_steps: Optional[int] = None,
              env_factory: Optional[Callable] = None, base: Optional[dict] = None):
    """CarlaEnv -> CoreState -> LaneFollow -> Gated -> DebitGoal -> TimeLimit.

    THE ORDER IS THE V52 ORDER with one substitution: DebitGoalWrapper stands
    where VAEGoalWrapper stood, because the goal is now an image rather than a
    pose. Everything below it -- the reward, the mode draw, the fault rules -- is
    the stack v52 trained with, so a result here differs from v52 in the goal
    representation and the learner, and not in the task.

    `env_factory` is injected so the order can be tested without a simulator.
    """
    import sys
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from gymnasium.wrappers import TimeLimit        # identical to jaxrl2's copy

    from vae_nav.core_state import CoreStateWrapper
    from vae_nav.debit_goal_wrapper import DebitGoalWrapper
    from vae_nav.gated_reward import GOAL, GatedReward
    from vae_nav.lane_follow_reward import LaneFollowReward

    check_cluster_env()
    if env_factory is None:
        from rlib_integration.carla_env import CarlaEnv
        env_factory = CarlaEnv

    env = env_factory(config=env_config(town, port, base=base))
    env = CoreStateWrapper(env)
    env = LaneFollowReward(env)
    env = GatedReward(
        env,
        explore_prob=float(os.environ.get("EXPLORE_PROB", 0.5)),
        force_mode=(int(os.environ.get("EVAL_FORCE_MODE", GOAL))
                    if eval_mode else None),
        max_steps=int(os.environ.get("GATED_EXPLORE_MAX_STEPS", 1000)),
        freeze_steps=0 if eval_mode else None,
        seed=seed)
    env = DebitGoalWrapper(
        env, encoder=encoder,
        wrong_goal=(os.environ.get("WRONG_GOAL_MODE", "0") == "1"),
        seed=seed)
    cap = int(max_episode_steps if max_episode_steps is not None
              else os.environ.get("MAX_EPISODE_STEPS", 2000))
    return TimeLimit(env, max_episode_steps=cap)

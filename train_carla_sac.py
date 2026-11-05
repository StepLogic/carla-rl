#!/usr/bin/env python3
"""The torch recurrent-SAC training loop for the CARLA image-goal arm.

User 2026-10-13: move the recurrent SAC policy to the cluster; pixels in the
replay; proceed. This is the loop that drives everything built for it:

  env         torch_rl/carla_env_build.build_env -- v52's wrapper chain with
              DebitGoalWrapper where VAEGoalWrapper stood, GIVEN the encoder
  perception  FROZEN DEBiT-tiny binocular (view, goal) -> 3136-d, precomputed in
              the env; plus a half-width ResNet-18 over the frame, TRAINED FROM
              SCRATCH, which is the only visual path carrying gradient
  replay      DeviceEpisodeBuffer of [frame | prev action | mode | binoc] rows
  learner     GatedRecurrentSAC, ONE critic for both reward modes, REDQ
              min-over-2, no TQC

WHY THE ENV ENCODES AND THE LEARNER DOES NOT (user 2026-10-14, "keep debit
frozen"). A frozen encoder's output is a fixed function of its inputs, so it can
be computed once per step and stored. The previous arm put trainable adapters
inside DEBiT, which forced the learner to re-encode a whole chunk twice per
update at 209 ms; frozen, that cost disappears and only the small ResNet needs
pixels and gradient.

WHAT THE ROW HOLDS: [frame 128x128x3 | mode | binoc 3136] -- no previous
action (user 2026-10-14), which closes the last path from the policy's own
output back into its input. The
state block is the MODE BIT AND THE BINOC, nothing else (user: "get rid of all
state vectors except mode bit"): no steer, no throttle, no speed. The actor used
to receive its own last steer and throttle, which is a feedback path a policy can
latch onto, and the run that collapsed sat at full left lock for 800k steps. The
binoc columns reach the GRU through core_state_cols; the single critic's view is
the mode bit alone.

THE UPDATE CADENCE. With DEBiT frozen the update is the cheap one again (~20 ms
measured, against 209 ms when the encoder had to be re-run), so UPDATE_EVERY can
come down from 20; it stays configurable and is worth re-measuring on the first
run of this stack rather than assumed.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


RUN_NAME = os.environ.get("RUN_NAME", "carla_sac_debit_v1")
MAX_STEPS = env_int("MAX_STEPS", 1_000_000)
UPDATE_EVERY = env_int("UPDATE_EVERY", 20)
UPDATES_PER = env_int("UPDATES_PER", 1)
LEARNING_STARTS = env_int("LEARNING_STARTS", 5_000)
CHUNK_B = env_int("CHUNK_B", 8)
CHUNK_T = env_int("CHUNK_T", 40)
BURN_IN = env_int("BURN_IN", 8)
CAPACITY = env_int("REPLAY_CAPACITY", 150_000)
# REPLAY IN HOST RAM (user 2026-10-19: "move to main memory"). "cuda" (default)
# is the old GPU-resident ring; "cpu" keeps it in the node's RAM (500 GB on the
# L40 nodes) and ships only each sampled chunk to the GPU. REPLAY_U8=1 stores
# the /255 camera frame as uint8 -- exact, and ~55 KB a row instead of ~105 KB.
REPLAY_DEVICE = os.environ.get("REPLAY_DEVICE", "cuda")
# EVALUATION MODE. EVAL_FROM=<checkpoint dir> loads that agent into the SAME
# construction the trainer uses (so the observation, encoder and action mapping
# cannot drift from training), runs EVAL_EPISODES goal-mode episodes with the
# DETERMINISTIC action, records every one, and exits without training.
# Faults END the episode (USER RULE, eval always fault-ends): a painted lane
# crossing on open road, off-road, or a collision -- the same crossing rule the
# training reward fines, so "crossing" means the same thing in both.
EVAL_FROM = os.environ.get("EVAL_FROM", "")
# INIT_FROM (user 2026-11-01, "resume v30 and train for longer"): start TRAINING
# from a checkpoint -- every module and target, both temperatures, all four
# optimisers and the update counter. The replay buffer is NOT in a checkpoint,
# and updating at once on a near-empty buffer destroyed goal steering in an
# earlier warm start (corr 0.57 -> 0.04 by 50k), so INIT_FROM requires
# LEARNING_STARTS >= INIT_MIN_LEARNING_STARTS: the loaded policy refills the
# buffer before the first update.
INIT_FROM = os.environ.get("INIT_FROM", "")
INIT_MIN_LEARNING_STARTS = env_int("INIT_MIN_LEARNING_STARTS", 20_000)
if INIT_FROM and EVAL_FROM:
    raise ValueError("INIT_FROM (resume training) and EVAL_FROM (evaluate) are exclusive")
# IMITATION-FIRST FINE-TUNE (plan Task 10, 2026-11-05): with IL_PRELOAD_ROOT set, a
# training run (normally INIT_FROM a clone) first fills its replay with the
# recorded episodes' executed steps and an expert buffer with their labels,
# warms the critic up for IL_CRITIC_WARMUP_UPDATES updates with the encoder, GRU
# and actor frozen, then after every RL update takes one cloning step at a
# weight fading from IL_BC_WEIGHT0 to 0 over IL_BC_DECAY_UPDATES
# (torch_rl/bc_trainer.il_finetune_prepare). The online cloning batch is
# IL_BC_CHUNKS x IL_BC_LEN, training-sized by default.
IL_PRELOAD_ROOT = os.environ.get("IL_PRELOAD_ROOT", "")
IL_PRELOAD_TOWNS = os.environ.get("IL_PRELOAD_TOWNS", "Town01 Town04 Town05 Town10HD").split()
IL_CRITIC_WARMUP_UPDATES = env_int("IL_CRITIC_WARMUP_UPDATES", 5000)
IL_BC_WEIGHT0 = env_float("IL_BC_WEIGHT0", 1.0)
IL_BC_DECAY_UPDATES = env_int("IL_BC_DECAY_UPDATES", 10000)
IL_BC_CHUNKS = env_int("IL_BC_CHUNKS", CHUNK_B)
IL_BC_LEN = env_int("IL_BC_LEN", CHUNK_T)
EVAL_EPISODES = env_int("EVAL_EPISODES", 40)
# EVAL_CHAIN=1 (imitation-first plan, Task 11): after each arrival the same car
# gets the next goal (GatedReward's soft reset, GOAL_CHAIN_STYLE) until the chain
# has driven GOAL_CHAIN_M, hits a fault or runs out of clock; each row records
# chain_goals, chain_m, chain_end and chain_complete.
EVAL_CHAIN = os.environ.get("EVAL_CHAIN", "0") == "1"
EVAL_SEED = env_int("EVAL_SEED", 10_000)
# EVAL_STOCHASTIC=1 samples actions from the policy (training noise) instead
# of tanh(mean): measures how much of the train/eval gap is the noise alone.
# Torch is seeded per episode so the sampled run repeats too.
EVAL_STOCHASTIC = os.environ.get("EVAL_STOCHASTIC", "0") == "1"
# EVAL_LANE_DEBUG=1 prints the lane block and the lane-centre shaping each
# 5th step and on every crossing, to check the signal on a real simulator.
EVAL_LANE_DEBUG = os.environ.get("EVAL_LANE_DEBUG", "0") == "1"
# EVAL_NOISE_SCALE scales that noise: 0 = tanh(mean), 1 = training's own
# sampling. A sweep over it says whether the mean action is worse than the
# samples that earned the training return. EVAL_STOCHASTIC=1 means scale 1.
EVAL_NOISE_SCALE = float(os.environ.get(
    "EVAL_NOISE_SCALE", "1.0" if EVAL_STOCHASTIC else "0.0"))


# EVAL_EXPLORE=1 (user 2026-10-25: "can also explore when given random goal
# image"): score EXPLORE-mode episodes instead -- the mode bit says explore and
# the goal image is unrelated to the reward. Success = the explore arrival
# (EXPLORE_GOAL_DIST_M of net distance) with the same fault rule. Needs
# EVAL_FORCE_MODE=0 so the env draws explore episodes.
EVAL_EXPLORE = os.environ.get("EVAL_EXPLORE", "0") == "1"
EVAL_WRONG_GOAL = os.environ.get("WRONG_GOAL_MODE", "0") == "1"
# THE GIBSON REPLICATION (2026-11-01; see vae_nav.carla_benchmark).
# EVAL_PROBE=1: reset each seed in EVAL_SEED..+EVAL_EPISODES, take ONE zero
#   step to read the planner's route distance, and write probe.json -- no
#   policy, no video. Selection turns probes into the fixed episode set.
# EVAL_EPISODES_FILE: drive exactly the seeds listed for this town, carrying
#   their bin and path type into eval.json. Replaces EVAL_SEED/EVAL_EPISODES.
# EVAL_VIDEO_EVERY: keep a video of every Nth episode (1 = all, as before);
#   1,800 episodes per arm of videos would be several GB.
EVAL_PROBE = os.environ.get("EVAL_PROBE", "0") == "1"
EVAL_EPISODES_FILE = os.environ.get("EVAL_EPISODES_FILE", "")
EVAL_VIDEO_EVERY = env_int("EVAL_VIDEO_EVERY", 1)
# BENCH_BAND (easy|medium|hard): which generator band this process runs
# under (set by slurm/bench_<band>.env with its GOAL_DIST_MIN/MAX). Probes
# record it; an episode-file eval replays only that band's episodes, since a
# seed replayed under a different band is a different episode.
BENCH_BAND = os.environ.get("BENCH_BAND") or None
# EXPLORATION TRACE (user 2026-11-01: "first drive policy in exploration mode
# and store trace") -- the trace the map-based arms of the Gibson replication
# are built from. EVAL_TRACE=1 with EVAL_FORCE_MODE=0 drives EXPLORE episodes
# with SAMPLED actions and keeps a sample every TRACE_SPACING_M metres, until
# TRACE_COVERAGE of the town's drivable waypoints lie within 5 m of a sample,
# or TRACE_DIST_M metres are driven, or TRACE_MAX_STEPS steps. See run_trace.
EVAL_TRACE = os.environ.get("EVAL_TRACE", "0") == "1"
# MAP ARMS (user 2026-11-02; vae_nav.image_map). EVAL_MAP_DIR holds
# <Town>.json + <Town>_frames.npy built from an exploration trace. Each episode
# is localised by TRUE POSE: start node = nearest aligned node within
# EVAL_MAP_START_M, goal node = the reachable node within EVAL_MAP_GOAL_M of the
# true goal minimising route + remaining distance. The policy is shown each
# subgoal node's frame in turn, switching within EVAL_MAP_SWITCH_M, and the
# episode's own goal image after the goal node. An episode with no start node,
# no goal node or no route is recorded end="unroutable" (a failure) without
# being driven. EVAL_MAP_RESET_HIDDEN=1 resets the GRU at each switch.
EVAL_MAP_DIR = os.environ.get("EVAL_MAP_DIR", "")
# SGM PAIRS (map arm 4): EVAL_SGM_TRACE=<trace dir> reads the goal critic
# between trace frames instead of driving -- candidates every EVAL_SGM_CAND_M
# (directional uniform thinning), every aligned (<90 deg) ordered pair within
# EVAL_SGM_PAIR_M -- and writes EVAL_SGM_OUT (.npz). See run_sgm_pairs.
EVAL_SGM_TRACE = os.environ.get("EVAL_SGM_TRACE", "")
EVAL_SGM_OUT = os.environ.get("EVAL_SGM_OUT", "")
EVAL_SGM_CAND_M = float(os.environ.get("EVAL_SGM_CAND_M", 4.0))
EVAL_SGM_PAIR_M = float(os.environ.get("EVAL_SGM_PAIR_M", 25.0))
EVAL_MAP_SWITCH_M = float(os.environ.get("EVAL_MAP_SWITCH_M", 4.0))
EVAL_MAP_START_M = float(os.environ.get("EVAL_MAP_START_M", 8.0))
EVAL_MAP_GOAL_M = float(os.environ.get("EVAL_MAP_GOAL_M", 8.0))
EVAL_MAP_RESET_HIDDEN = os.environ.get("EVAL_MAP_RESET_HIDDEN", "0") == "1"
# EXPERT RECORDING (imitation-first plan, 2026-11-04; see run_record and
# torch_rl.expert_record). EXPERT_RECORD=1 drives CARLA's BasicAgent (or, with
# EXPERT_DRIVER=policy and EVAL_FROM, the policy, relabelled by the expert:
# DAgger) through the unchanged env stack, one .npz per episode under
# EXPERT_RECORD_ROOT/<town>/. EXPERT_RECORD_DRY=1 records nothing and only
# reports the M0 gate (arrivals, metres between failures).
EXPERT_RECORD = os.environ.get("EXPERT_RECORD", "0") == "1"
EXPERT_DRIVER = os.environ.get("EXPERT_DRIVER", "expert")
EXPERT_RECORD_DRY = os.environ.get("EXPERT_RECORD_DRY", "0") == "1"
EXPERT_RECORD_EPISODES = env_int("EXPERT_RECORD_EPISODES", 200)
EXPERT_SEED0 = env_int("EXPERT_SEED0", 100000)
EXPERT_TARGET_KMH = env_float("EXPERT_TARGET_KMH", 15.0)
EXPERT_EXPLORE_MIN_M = env_float("EXPERT_EXPLORE_MIN_M", 100.0)
EXPERT_EXPLORE_MAX_M = env_float("EXPERT_EXPLORE_MAX_M", 200.0)
EXPERT_REPLAN_M = env_float("EXPERT_REPLAN_M", 3.0)
EXPERT_RECORD_ROOT = os.environ.get(
    "EXPERT_RECORD_ROOT", os.path.expanduser("~/carla-rl/checkpoints/il_data"))
# HIDE_MODE_BIT (v34, user 2026-11-02): the NETWORK always sees mode = 1. With
# FOREIGN_GOAL_EXPLORE, explore episodes carry another town's goal image, so the
# image -- not the bit -- has to say whether the goal is reachable here. The
# TRUE mode still drives the reward, the replay flag and HER: they read it from
# the observation and info, never from the row. Rows built in training, eval
# and traces all go through _net_mode, so a v34 checkpoint never sees a mode
# value it was not trained on.
HIDE_MODE_BIT = os.environ.get("HIDE_MODE_BIT", "0") == "1"


def _net_mode(mode):
    """The mode value the network sees: 1 when HIDE_MODE_BIT, else the true one."""
    return np.ones_like(np.asarray(mode, np.float32)) if HIDE_MODE_BIT else mode
TRACE_DIST_M = env_float("TRACE_DIST_M", 6000.0)
TRACE_COVERAGE = env_float("TRACE_COVERAGE", 0.90)
TRACE_SPACING_M = env_float("TRACE_SPACING_M", 2.0)
TRACE_MAX_STEPS = env_int("TRACE_MAX_STEPS", 60000)
# TRACE_NOISE_SCALE (user 2026-11-01: "redo and switch to deterministic
# actions"): 0 = tanh(mean), as the eval drives. With the training noise (1.0)
# the Town02 trace faulted every ~11 m -- 148 segments in 2 km, 86 crossings,
# 49 off-road, 13 collisions, at 7.5 km/h -- and the map should reflect how the
# NAVIGATING policy drives. The noise only existed so a retried corner would
# not repeat; the stuck rule handles that directly.
# TRACE_STUCK_LIMIT: restarts within 5 m before jumping 25 m on (a
# deterministic retry usually repeats the failure, so 3 rather than 4).
# TRACE_STALL_M: metres driven with no coverage gain before the next segment
# restarts at the nearest UNCOVERED waypoint -- a deterministic car that never
# faults could otherwise circle one block for the whole budget.
TRACE_NOISE_SCALE = env_float("TRACE_NOISE_SCALE", 1.0)
TRACE_STUCK_LIMIT = env_int("TRACE_STUCK_LIMIT", 4)
TRACE_STALL_M = env_float("TRACE_STALL_M", 1500.0)
# TRACE_END_ON_CROSSING=0 (user 2026-11-01): a lane crossing is COUNTED and the
# car drives on; off-road and collisions still end the segment. Pair it with
# slurm/trace_nocross.env, which stops the ENV terminating on crossings too.
TRACE_END_ON_CROSSING = os.environ.get("TRACE_END_ON_CROSSING", "1") == "1"


def chain_summary(goals_reached, info, last_end, chain_m_target) -> dict:
    """The chain eval's row fields: goals reached, metres the chain drove (the
    reward wrapper's chain_m), how it ended ("complete" once chain_m reaches the
    target, else the last leg's end), and whether it completed."""
    m = float(info.get("chain_m", 0.0) or 0.0)
    complete = m >= float(chain_m_target)
    return {"chain_goals": int(goals_reached), "chain_m": round(m, 2),
            "chain_end": "complete" if complete else last_end,
            "chain_complete": bool(complete)}


def no_label_end(mode, steps) -> str:
    """Why a recording ended when the expert gave no label: an explore plan ran
    out ("route_done"), the car passed a goal episode's goal so the spent route
    has no correction left ("passed_goal"), or there was never a plan
    ("no_plan", nothing recorded)."""
    if int(steps) <= 0:
        return "no_plan"
    return "route_done" if int(mode) == 0 else "passed_goal"


def ramp_needs_parallel(ramp_start, parallel, eval_from, expert_record) -> bool:
    """True when a junction-goal ramp is configured for a TRAINING run that is
    not on the parallel loop, the only loop that implements it. Eval and
    expert recording keep the overlay's fixed junction fraction, so they are
    exempt (recording tripped this in its first smoke, job 463137)."""
    return (ramp_start is not None and not parallel and not eval_from
            and not expert_record)


def use_parallel(parallel_towns: bool, eval_from: str, n_towns: int) -> bool:
    """Whether training runs through torch_rl.parallel_loop. ONE town counts
    (user 2026-10-27, local single-town v18): that loop is the only one with
    image HER, noise-free episodes and the enc_alive diagnostic, so requiring
    two towns silently gave a one-town run a weaker recipe. Eval never does."""
    return bool(parallel_towns) and not eval_from and n_towns >= 1


def _eval_tag_suffix(scale: float, explore: bool, wrong: bool) -> str:
    """Folder suffix: noise scale, then explore mode, then a wrong goal image,
    so no two kinds of eval of one checkpoint ever share a folder."""
    return (_eval_suffix(scale) + ("_explore" if explore else "")
            + ("_wrong" if wrong else ""))


def _eval_suffix(scale: float) -> str:
    """Eval output folder suffix, so one checkpoint's scales never collide.
    Scale 1.0 keeps the `_stoch` name already published for v13's Town02 run."""
    if scale <= 0.0:
        return ""
    return "_stoch" if scale == 1.0 else f"_n{int(round(scale * 100))}"
# The env's arrival test: straight-line distance to the goal < 2 m
# (train_env_config_v4_image_goal, passed_chckp). SPL measures to this circle.
GOAL_RADIUS_M = env_float("GOAL_RADIUS_M", 2.0)
REPLAY_U8 = os.environ.get("REPLAY_U8", "0") == "1"
LOG_EVERY = env_int("LOG_EVERY", 10)
CKPT_EVERY = env_int("CKPT_EVERY", 25_000)
# DEBUGGING VIDEOS (user 2026-10-14). On by default and nearly free: both
# frames are already in memory each step for the replay, so recording adds no
# render and no GPU work. One episode in VIDEO_EVERY, capped and pruned.
VIDEO_EVERY = env_int("VIDEO_EVERY", 50)
VIDEO_MAX_FRAMES = env_int("VIDEO_MAX_FRAMES", 1200)
# USER 2026-10-14: "Store only 5 episodes at a time" -- the newest 5 mp4s
# are kept and older ones deleted as each new video is written.
VIDEO_KEEP = env_int("VIDEO_KEEP", 5)
# ALL TOWNS EXCEPT Town02 AND Town03 (user 2026-10-14), which stay held out --
# they are also the pair the earlier explore traces used, so the transfer read
# is comparable. One CARLA server reloads the map every TOWN_EVERY episodes
# rather than several servers running at once: six simultaneous simulators fit
# in VRAM but not in eight CPU cores, and the ROLLOUT being blocked by town does
# not make the LEARNING blocked -- the replay spans many blocks, so every update
# still mixes towns.
TOWNS = [t for t in os.environ.get(
    "ENV_TOWNS", "Town01,Town04,Town05,Town06,Town07,Town10HD").split(",") if t]
TOWN_EVERY = env_int("TOWN_EVERY", 25)
# ALL TOWNS AT ONCE (user 2026-10-21: Town05 and Town01, "do not switch
# towns", "use existing code"): one CARLA server per town, stepped together by
# vae_nav.vec_env.ThreadedVecEnv -- the vector env the two-town JAX runs used.
# Replaces the TOWN_EVERY rotation when on. Ports are PORT + 50*i so each
# server's RPC and streaming ports (port, +1, +2) never overlap.
PARALLEL_TOWNS = os.environ.get("PARALLEL_TOWNS", "0") == "1"
# NOISE-FREE TRAINING EPISODES (user 2026-10-25, v15): this share of episodes,
# drawn per env per episode, acts with tanh(mean) -- the action the eval scores.
# v14 parked at full lock with ~0 throttle on 7 of 21 Town02 eval failures, a
# state the noisy rollouts almost never visited, so the critic never priced it.
# Parallel loop only. 0 = off (every earlier run).
DET_EPISODE_FRAC = float(os.environ.get("DET_EPISODE_FRAC", "0") or 0.0)
# HINDSIGHT RELABELLING (user 2026-10-21, "add HER"): torch_rl.image_her.
HER = os.environ.get("HER", "0") == "1"
HER_RELABEL = env_int("HER_RELABEL", 2)
HER_EXPLORE = os.environ.get("HER_EXPLORE", "1") == "1"
# JUNCTION RAMP (v27, user 2026-10-30): start the goals' junction fraction
# here and raise it to JUNCTION_GOAL_FRAC as training goal success rises.
# Unset = off, every earlier run unchanged.
JUNCTION_RAMP_START = (float(os.environ["JUNCTION_RAMP_START"])
                       if os.environ.get("JUNCTION_RAMP_START") else None)
PORT = env_int("CARLA_PORT", 2000)
SEED = env_int("SEED", 0)
DEVICE = os.environ.get("TORCH_DEVICE", "cuda")
TINY = os.environ.get("DEBIT_WEIGHTS",
                      os.path.join(ROOT, "dependencies", "debit_weights",
                                   "debit_tiny.pth"))
OUT = os.path.join(os.environ.get("CKPT_DIR",
                                  os.path.join(ROOT, "checkpoints")), RUN_NAME)


def to_env(a, box):
    """Actor action in [-1, 1] -> the env's box, per dimension."""
    lo, hi = box.low.astype(np.float32), box.high.astype(np.float32)
    return lo + 0.5 * (np.clip(a, -1.0, 1.0) + 1.0) * (hi - lo)


def to_actor(e, box):
    """Inverse of to_env: an env-box action -> the actor's [-1, 1]."""
    lo, hi = box.low.astype(np.float32), box.high.astype(np.float32)
    return (2.0 * (np.asarray(e, np.float32) - lo) / (hi - lo) - 1.0).astype(np.float32)


def _speed_kmh(info):
    """The car's speed for the observation row (SPEED_OBS): info["speed_kmh"],
    0 when absent -- at a reset the car stands still, which is also what the
    recorded episodes pair with their first frame."""
    return float(info.get("speed_kmh", 0.0) or 0.0)


def _fault(info):
    """The eval fault rule: collision, off-road, or a lane crossing as the
    training reward reads one -- solid lines, curbs and grass anywhere, broken
    lines on open road. A broken line inside a junction (the turn) and an
    unpainted road edge are not faults."""
    import vae_nav.gated_reward as G
    if info.get("collision"):
        return "collision"
    if info.get("off_road"):
        return "off_road"
    lane = bool(info.get("lane_any", info.get("lane_invasion_any", False)))
    if lane and G.explore_lane_fatal(lane, info.get("lane_marking_types"), False,
                                     in_junction=bool(info.get("in_junction", False)),
                                     off_road_fatal=False):
        return "crossing"
    return None


def _carla_env(env):
    """The CarlaEnv under the wrapper stack (it has no gym `unwrapped`)."""
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def run_trace(env, agent, pxr, server, out_root):
    """An EXPLORE-mode trace of one town for map building.

    The design is the v13/v43 traces' (user-approved then), ported to this torch
    arm so the trace runs through the same env, observations and actions as the
    eval:
      * EXPLORE MODE, no 100 m explore arrival: an episode runs until a fault
        ends it (the eval's _fault: collision, off-road, a painted crossing
        outside a junction) or the explore cap truncates it.
      * CONTINUATION (the user's rule): the next episode starts at the nearest
        drivable waypoint 3 m ahead of where the last one stopped, so the trace
        walks the town instead of sampling disconnected islands. If 4 restarts
        in a row land within 5 m (a corner the policy keeps failing), the
        restart jumps 25 m on, recorded per segment as `jumped`.
      * ACTIONS at TRACE_NOISE_SCALE: 0 = deterministic, as the eval drives
        (the user's choice, 2026-11-01). The v43 design sampled at the training
        noise so a retried corner would not repeat; on this arm that faulted
        every ~11 m, and the stuck rule now covers repeats directly. A COVERAGE
        STALL (TRACE_STALL_M driven with no gain) restarts the next segment at
        the nearest uncovered waypoint, so a car that never faults cannot circle
        one block. Torch is seeded per segment, so a sampled run repeats too.
      * KEPT EVERY TRACE_SPACING_M metres: the frame the policy saw
        (info["debit_cur"]), true pose and yaw, speed, action, reward, and
        act_with_stats's policy entropy and critic-ensemble mean/spread -- the
        signals entropy maps are placed on.
      * RESUMABLE: one .npz per segment plus index.json after every segment, so
        a killed job continues where it stopped instead of starting over.
    """
    import json
    import random as _random
    import torch
    import carla
    from vae_nav.explore_trace import CoverageGrid, CrossingCounter, StuckDetector

    tag = os.path.basename(os.path.normpath(EVAL_FROM))
    out = os.path.join(out_root, f"trace_{tag}")
    os.makedirs(out, exist_ok=True)
    base = _carla_env(env)
    if not hasattr(base, "set_start_transform"):
        raise RuntimeError(f"{type(base).__name__} has no set_start_transform: "
                           f"the continuation rule needs CarlaEnv")
    core = base.core
    box = env.action_space
    wps = np.array([[w.transform.location.x, w.transform.location.y]
                    for w in core.map.generate_waypoints(2.0)], np.float64)
    np.save(os.path.join(out, "map_waypoints.npy"), wps)
    cov = CoverageGrid(wps, radius=5.0)
    idx_path = os.path.join(out, "index.json")
    index = (json.load(open(idx_path)) if os.path.exists(idx_path) else
             {"town": TOWNS[0], "checkpoint": EVAL_FROM, "spacing_m": TRACE_SPACING_M,
              "segments": []})
    total_m = steps = 0.0
    for sg in index["segments"]:            # resume: coverage and budget so far
        with np.load(os.path.join(out, sg["file"])) as z:
            for xy in z["pos"]:
                cov.add(xy)
        total_m += sg["metres"]
        steps += sg["steps"]
    restart = index["segments"][-1]["stop_xy"] if index["segments"] else None
    stuck = StuckDetector(radius=5.0, limit=TRACE_STUCK_LIMIT)
    stall_m, last_cov = 0.0, cov.fraction()
    seg_i = len(index["segments"])
    print(f"=== TRACE {EVAL_FROM} | {TOWNS[0]} | explore mode, "
          + ("DETERMINISTIC actions" if TRACE_NOISE_SCALE <= 0
             else f"SAMPLED actions at noise x{TRACE_NOISE_SCALE:g}")
          + f", stuck after {TRACE_STUCK_LIMIT} restarts, stall jump after "
          f"{TRACE_STALL_M:g} m without coverage, "
          + ("crossings END a segment, " if TRACE_END_ON_CROSSING
             else "crossings are COUNTED and do not end a segment, ")
          + f"keep every {TRACE_SPACING_M:g} m | stop at coverage {TRACE_COVERAGE:.0%} "
          f"or {TRACE_DIST_M:g} m or {TRACE_MAX_STEPS} steps | {len(wps)} map waypoints"
          + (f" | RESUMING at segment {seg_i}, {total_m:.0f} m, coverage "
             f"{cov.fraction():.1%}" if seg_i else "") + f" -> {out}", flush=True)

    while (total_m < TRACE_DIST_M and cov.fraction() < TRACE_COVERAGE
           and steps < TRACE_MAX_STEPS):
        jumped = False
        if restart is not None and stall_m >= TRACE_STALL_M:
            # COVERAGE STALLED: restart where the map has not been yet.
            far = cov.nearest_uncovered(restart)
            if far is not None:
                restart, jumped, stall_m = far.tolist(), "stall", 0.0
        if restart is not None:
            wp = core.map.get_waypoint(carla.Location(x=float(restart[0]), y=float(restart[1]),
                                                      z=0.0),
                                       project_to_road=True, lane_type=carla.LaneType.Driving)
            ahead = 0.0 if jumped == "stall" else 3.0
            if jumped != "stall" and stuck.restart(restart):
                ahead, jumped = 25.0, "stuck"
            nxt = wp.next(ahead) if ahead > 0 else None
            wp = nxt[0] if nxt else wp
            dst = wp.next(30.0)
            base.set_start_transform(wp.transform.location)
            base.set_destination_transform((dst[0] if dst else wp).transform.location)
        _random.seed(SEED + seg_i)
        np.random.seed(SEED + seg_i)
        torch.manual_seed(SEED + seg_i)
        obs, info = env.reset(seed=SEED + seg_i)
        mode = int(round(float(np.asarray(obs["mode"]).ravel()[0])))
        if mode != 0:
            raise RuntimeError(f"trace episode is in mode {mode}: set EVAL_FORCE_MODE=0")
        hidden = agent.reset_hidden(1)
        goal_blk = pxr.goal_block(info["debit_goal"])
        start = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        rec = {k: [] for k in ("frame", "pos", "yaw", "speed", "action", "reward",
                               "entropy", "q_mean", "q_std", "t", "on_line")}
        prev, last_kept, metres, t, end = start.copy(), None, 0.0, 0, None
        # CROSSING EVENTS, not steps (see CrossingCounter): the first count put
        # Town02's segments at 4 and 8 "crossings", about four steps a line.
        crossing = CrossingCounter()
        while end is None:
            row = pxr.mono_row(info["debit_cur"], _net_mode(obs["mode"]), obs["binoc"],
                               speed_kmh=_speed_kmh(info))
            action, hidden, st = agent.act_with_stats(
                row, hidden, goal=goal_blk, deterministic=TRACE_NOISE_SCALE <= 0,
                noise_scale=max(TRACE_NOISE_SCALE, 0.0))
            a_env = to_env(np.asarray(action, np.float32), box)
            obs, r, term, trunc, info = env.step(a_env)
            t += 1
            pos = np.asarray(info.get("hero_pos", prev), np.float64)
            metres += float(np.linalg.norm(pos - prev))
            prev = pos
            # THIS step's fault, before the sample is kept: the sample records
            # whether the car is on a line NOW (computing it after the keep
            # block referenced a value from no step at all on the first one).
            fault = _fault(info)
            on_line = fault == "crossing"
            crossing.update(on_line)
            if last_kept is None or np.linalg.norm(pos - last_kept) >= TRACE_SPACING_M:
                last_kept = pos
                rec["frame"].append(np.asarray(info["debit_cur"], np.uint8))
                rec["pos"].append(pos[:2])
                rec["yaw"].append(float(info.get("hero_yaw", 0.0) or 0.0))
                rec["speed"].append(float(info.get("speed_kmh", 0.0) or 0.0))
                rec["action"].append(np.asarray(a_env, np.float32))
                rec["reward"].append(float(r))
                rec["entropy"].append(st["entropy"])
                rec["q_mean"].append(st["q_mean"])
                rec["q_std"].append(st["q_std"])
                rec["t"].append(t)
                rec["on_line"].append(on_line)
                cov.add(pos)
            if on_line and not TRACE_END_ON_CROSSING:
                # Counted, not fatal: a trace scores nothing, and ending here
                # is what kept segments at ~11 m.
                fault = None
            end = fault
            if end is None and (term or trunc):
                end = str(info.get("explore_end", "cap"))
            if end is None and steps + t >= TRACE_MAX_STEPS:
                end = "budget"
        fname = f"seg_{seg_i:05d}.npz"
        arrays = {k: np.asarray(v) for k, v in rec.items()}
        np.savez(os.path.join(out, fname), **arrays)
        index["segments"].append({
            "seg": seg_i, "file": fname, "start_xy": start[:2].round(2).tolist(),
            "stop_xy": prev[:2].round(2).tolist(), "end": end, "steps": t,
            "metres": round(metres, 2), "kept": len(rec["t"]), "jumped": jumped,
            "crossings": crossing.events, "line_steps": crossing.steps,
            "coverage": round(cov.fraction(), 4)})
        tmp = idx_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(index, fh, indent=1)
        os.replace(tmp, idx_path)
        total_m += metres
        steps += t
        restart = prev[:2].tolist()
        now = cov.fraction()
        stall_m = 0.0 if now > last_cov + 1e-9 else stall_m + metres
        last_cov = now
        print(f"[Trace {seg_i:4d}] end={end} steps={t} {metres:.0f} m kept={len(rec['t'])}"
              f"{f' crossings={crossing.events}' if not TRACE_END_ON_CROSSING else ''}"
              f"{f' JUMPED({jumped})' if jumped else ''} | total {total_m:.0f} m, {int(steps)} steps, "
              f"coverage {cov.fraction():.1%}", flush=True)
        seg_i += 1
    kept = sum(sg["kept"] for sg in index["segments"])
    print(f"[TraceDone] {TOWNS[0]}: {seg_i} segments, {total_m:.0f} m, {int(steps)} steps, "
          f"{kept} samples, coverage {cov.fraction():.1%} -> {out}", flush=True)
    env.close()


def run_sgm_pairs(agent, enc, pxr):
    """Q_g(view i, goal view j) for SGM placement, offline from a trace.

    Each pair is a one-step context exactly as an eval's first step builds it:
    row = mono_row(frame_i, goal-mode bit, DEBiT binoc(frame_i, frame_j)), zero
    GRU state, the actor's deterministic action. Also Q(i, i) for every
    candidate (the arrived value)."""
    import time
    from vae_nav import image_map
    tr = image_map.load_trace(EVAL_SGM_TRACE)
    cand = image_map.place_uniform(tr, EVAL_SGM_CAND_M)
    xy, yaw = tr["pos"][cand], tr["yaw"][cand]
    frames = image_map.node_frames(tr, cand)
    pairs = []
    for a in range(len(cand)):
        d = np.hypot(*(xy - xy[a]).T)
        ok = (d <= EVAL_SGM_PAIR_M) & image_map._aligned(yaw, yaw[a], np.radians(90.0))
        pairs.extend((a, int(b)) for b in np.flatnonzero(ok))   # includes (a, a)
    pairs = np.asarray(pairs, int)
    mode = _net_mode(np.ones(1, np.float32))
    q = np.empty(len(pairs), np.float32)
    t0, B = time.time(), 256
    for k in range(0, len(pairs), B):
        pa = pairs[k:k + B]
        binoc = enc.encode_batch(frames[pa[:, 0]], frames[pa[:, 1]])
        rows = np.stack([pxr.mono_row(frames[i], mode, b, speed_kmh=0.0)   # first step: at rest
                         for i, b in zip(pa[:, 0], binoc)])
        q[k:k + B] = agent.q_batch(rows)
        if (k // B) % 200 == 0:
            print(f"[SGMPairs] {k + len(pa)}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(EVAL_SGM_OUT)), exist_ok=True)
    np.savez(EVAL_SGM_OUT, cand=cand, pairs=pairs, q=q, xy=xy, yaw=yaw,
             cand_m=EVAL_SGM_CAND_M, pair_m=EVAL_SGM_PAIR_M, trace=EVAL_SGM_TRACE)
    print(f"[SGMPairsDone] {len(cand)} candidates, {len(pairs)} pairs, q "
          f"[{q.min():.1f}, {np.median(q):.1f}, {q.max():.1f}] -> {EVAL_SGM_OUT}", flush=True)


def run_record(env, agent, pxr, server, out_root):
    """Record expert (or DAgger) episodes for imitation; see EXPERT_RECORD.

    Each episode: the mode and recovery-noise level come from episode_plan(seed);
    goal episodes drive to the env's own goal, explore episodes follow
    expert_route.explore_route for EXPERT_EXPLORE_MIN_M..MAX_M metres. Every
    step stores the 128 px frame and binoc the policy would see, the executed
    action and the expert's label (actor space), the reward, speed and pose.
    An episode ends on the first eval-rule fault, an arrival, the env's own
    end, or (explore) the expert's plan running out. Writes record_summary.json
    (per-episode list + this town's M0 gate) and prints [RecordDone]."""
    import json
    import random as _random
    import torch
    from torch_rl.expert_data import save_episode
    from torch_rl.expert_record import (assert_recordable_town, effective_sigma,
                                        episode_plan, gate_report, keep_episode,
                                        record_step)
    from vae_nav.expert_oracle import BasicAgentExpert
    from vae_nav.expert_route import explore_route
    from vae_nav.gated_reward import GatedReward

    town = TOWNS[0]
    assert_recordable_town(town)
    if EXPERT_DRIVER not in ("expert", "policy"):
        raise ValueError(f"EXPERT_DRIVER must be expert|policy, got {EXPERT_DRIVER!r}")
    gated = env
    while not isinstance(gated, GatedReward):
        gated = gated.env
    core = getattr(env, "unwrapped", env).core
    box = env.action_space
    low, high = box.low.astype(np.float32), box.high.astype(np.float32)
    root = os.path.join(EXPERT_RECORD_ROOT, town)
    rng = np.random.default_rng(EXPERT_SEED0)
    print(f"=== RECORD {town} | driver={EXPERT_DRIVER} | {EXPERT_RECORD_EPISODES} episodes "
          f"from seed {EXPERT_SEED0} | expert {EXPERT_TARGET_KMH:g} km/h, explore "
          f"{EXPERT_EXPLORE_MIN_M:g}-{EXPERT_EXPLORE_MAX_M:g} m | "
          + ("DRY (gate only)" if EXPERT_RECORD_DRY else f"-> {root}"), flush=True)

    expert, hero_id = None, None
    episodes, total, kept = [], 0, 0
    for k in range(EXPERT_RECORD_EPISODES):
        seed = EXPERT_SEED0 + k
        plan = episode_plan(seed)
        sigma = effective_sigma(plan["sigma"], EXPERT_RECORD_DRY)
        gated.force_mode = int(plan["mode"])
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs, info = env.reset(seed=seed)
        mode = int(gated.mode)
        if expert is None or core.hero.id != hero_id:
            expert = BasicAgentExpert(core.hero, core.map, box,
                                      target_speed_kmh=EXPERT_TARGET_KMH)
            hero_id = core.hero.id
        if mode == 1:
            expert.reset(info["goal_location"])
        else:
            length = float(rng.uniform(EXPERT_EXPLORE_MIN_M, EXPERT_EXPLORE_MAX_M))
            expert.set_explore_plan(explore_route(
                core.map.get_waypoint(core.hero.get_location()), length))
        hidden = goal_blk = None
        if EXPERT_DRIVER == "policy":
            hidden = agent.reset_hidden(1)
            goal_blk = pxr.goal_block(info["debit_goal"])
        rec = {"frames": [pxr.to_frame(info["debit_cur"], pxr.MONO_IMG).astype(np.uint8)],
               "binoc": [np.asarray(obs["binoc"], np.float16)],
               "act_exec": [], "act_label": [], "reward": [], "speed": [], "pos": []}
        goal_frame = info["debit_goal"]
        prev = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]
        path_m, end, terminal, truncated = 0.0, None, False, False
        # Goal episodes: how close the expert got and how fast it was going there
        # (the same EpisodeTracker the eval uses) -- the M0 gate's diagnosis.
        tracker = None
        if mode == 1 and info.get("goal_pos") is not None:
            from vae_nav.carla_benchmark import EpisodeTracker
            tracker = EpisodeTracker(info["goal_pos"], prev, max_speed=float(
                os.environ.get("GOAL_ARRIVAL_MAX_SPEED", 10.0)),
                radius=GOAL_RADIUS_M, radius_2m=2.0)
        while end is None:
            if EXPERT_DRIVER == "policy":
                expert.replan_if_off_route(EXPERT_REPLAN_M)
            a_exp = expert.act()
            if a_exp is None:
                end = no_label_end(mode, len(rec["act_exec"]))
                truncated = True
                break
            a_pol = None
            if EXPERT_DRIVER == "policy":
                row = pxr.mono_row(info["debit_cur"], _net_mode(obs["mode"]), obs["binoc"],
                               speed_kmh=_speed_kmh(info))
                a_act, hidden = agent.act(row, hidden, goal=goal_blk, deterministic=True)
                a_pol = to_env(np.asarray(a_act, np.float32), box)
            a_exec, a_label = record_step(EXPERT_DRIVER, a_pol, a_exp, sigma,
                                          rng, low, high)
            obs, r, term, trunc, info = env.step(a_exec)
            pos = np.asarray(info.get("hero_pos", prev), np.float64)[:2]
            path_m += float(np.linalg.norm(pos - prev))
            prev = pos
            if tracker is not None:
                tracker.step(pos, float(info.get("speed_kmh", 0.0) or 0.0))
            rec["act_exec"].append(to_actor(a_exec, box))
            rec["act_label"].append(to_actor(a_label, box))
            rec["reward"].append(float(r))
            rec["speed"].append(float(info.get("speed_kmh", 0.0) or 0.0))
            rec["pos"].append(pos.astype(np.float32))
            rec["frames"].append(pxr.to_frame(info["debit_cur"], pxr.MONO_IMG).astype(np.uint8))
            rec["binoc"].append(np.asarray(obs["binoc"], np.float16))
            fault = _fault(info)
            if fault:
                end, terminal = fault, True
            elif info.get("is_success") or info.get("goal_end") == "reached":
                end, terminal = "reached", True
            elif term:
                end = str(info.get("goal_end") or info.get("explore_end") or "term")
                terminal = True
            elif trunc:
                end, truncated = "clock", True
        n = len(rec["act_exec"])
        total += n
        approach = {}
        if tracker is not None:
            fin = tracker.finish(reached=end == "reached", t_end=n, route_m=None)
            approach = {k: fin[k] for k in ("closest_m", "speed_at_closest_kmh",
                                            "entry_speed_kmh", "min_speed_inside_kmh")}
        episodes.append({"seed": seed, "mode": mode, "sigma": sigma, "end": end,
                         "path_m": round(path_m, 2), "steps": n, **approach})
        if keep_episode(n) and not EXPERT_RECORD_DRY:
            save_episode(os.path.join(root, f"ep_{seed}.npz"), {
                "frames": np.stack(rec["frames"]), "binoc": np.stack(rec["binoc"]),
                "act_exec": np.asarray(rec["act_exec"], np.float32),
                "act_label": np.asarray(rec["act_label"], np.float32),
                "reward": np.asarray(rec["reward"], np.float32),
                "speed": np.asarray(rec["speed"], np.float32),
                "pos": np.asarray(rec["pos"], np.float32).reshape(-1, 2),
                "mode": np.int8(mode), "terminal": terminal, "truncated": truncated,
                "goal": pxr.goal_block(goal_frame), "sigma": np.float32(sigma),
                "driver": EXPERT_DRIVER, "end": end, "town": town, "seed": np.int64(seed)})
            kept += n
        print(f"[Record {k + 1}/{EXPERT_RECORD_EPISODES}] seed={seed} "
              f"{'goal' if mode == 1 else 'explore'} sigma={sigma:g} end={end} "
              f"steps={n} driven={path_m:.1f} m"
              + (f" closest={approach['closest_m']} m at {approach['speed_at_closest_kmh']} km/h"
                 f" entry={approach['entry_speed_kmh']}" if approach else "")
              + f" | total {total} steps", flush=True)

    gate = gate_report([e for e in episodes if e["mode"] == 1],
                       [e for e in episodes if e["mode"] == 0])
    summary = {"town": town, "driver": EXPERT_DRIVER, "dry": EXPERT_RECORD_DRY,
               "seed0": EXPERT_SEED0, "target_kmh": EXPERT_TARGET_KMH,
               "steps": total, "kept_steps": kept, "gate": gate, "episodes": episodes}
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "record_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("[RecordDone] " + json.dumps({"town": town, "steps": total, "kept_steps": kept,
                                        "gate": gate}), flush=True)
    env.close()
    server.stop()


def _poses(info):
    """Start pose and goal position at reset, for offline map analysis."""
    hp, gp = info.get("hero_pos"), info.get("goal_pos")
    return {"start_xy": None if hp is None else [round(float(v), 3) for v in np.asarray(hp)[:2]],
            "start_yaw": round(float(info.get("hero_yaw", 0.0) or 0.0), 4),
            "goal_xy": None if gp is None else [round(float(v), 3) for v in np.asarray(gp)[:2]]}


def run_eval(env, agent, pxr, server, out_root):
    """EVAL_EPISODES goal episodes, deterministic actions, faults terminal,
    every episode recorded. Writes eval.json next to the videos."""
    import json
    import torch
    from torch_rl.episode_video import EpisodeVideo

    tag = os.path.basename(os.path.normpath(EVAL_FROM))
    out = os.path.join(out_root, f"eval_{tag}"
                       + _eval_tag_suffix(EVAL_NOISE_SCALE, EVAL_EXPLORE, EVAL_WRONG_GOAL)
                       + (f"_map-{os.path.basename(os.path.normpath(EVAL_MAP_DIR))}"
                          if EVAL_MAP_DIR else ""))
    from vae_nav.carla_benchmark import EpisodeTracker
    gmap = map_frames = goal_wrapper = None
    if EVAL_MAP_DIR:
        from vae_nav import image_map
        from vae_nav.debit_goal_wrapper import DebitGoalWrapper
        gmap = image_map.Graph.load(os.path.join(EVAL_MAP_DIR, f"{TOWNS[0]}.json"))
        map_frames = np.load(os.path.join(EVAL_MAP_DIR, f"{TOWNS[0]}_frames.npy"))
        if len(map_frames) != len(gmap.nodes):
            raise RuntimeError(f"{EVAL_MAP_DIR}: {len(map_frames)} frames for "
                               f"{len(gmap.nodes)} nodes")
        goal_wrapper = env
        while not isinstance(goal_wrapper, DebitGoalWrapper):
            goal_wrapper = goal_wrapper.env
        print(f"[Map] {EVAL_MAP_DIR} {TOWNS[0]}: {len(gmap.nodes)} nodes, "
              f"{len(gmap.edges)} edges, placement={gmap.meta.get('placement')} | "
              f"start<= {EVAL_MAP_START_M:g} m, goal<= {EVAL_MAP_GOAL_M:g} m, "
              f"switch {EVAL_MAP_SWITCH_M:g} m, reset_hidden={EVAL_MAP_RESET_HIDDEN}",
              flush=True)
    if EVAL_EPISODES_FILE:
        with open(EVAL_EPISODES_FILE) as fh:
            episodes = [e for e in json.load(fh) if e["town"] == TOWNS[0]
                        and (BENCH_BAND is None or e.get("band") == BENCH_BAND)]
        if not episodes:
            raise RuntimeError(f"{EVAL_EPISODES_FILE} lists no episodes for "
                               f"{TOWNS[0]} band {BENCH_BAND}")
    else:
        episodes = [{"seed": EVAL_SEED + k} for k in range(EVAL_EPISODES)]
    n_eps = len(episodes)
    vid = EpisodeVideo(os.path.join(out, "videos"), every=EVAL_VIDEO_EVERY,
                       max_frames=1200,
                       keep=max(n_eps // max(1, EVAL_VIDEO_EVERY) + 1, 1))
    box = env.action_space
    print(f"=== EVAL {EVAL_FROM} | {n_eps} goal episodes, "
          + (f"from {EVAL_EPISODES_FILE}" if EVAL_EPISODES_FILE else
             f"seeds {EVAL_SEED}..{EVAL_SEED + EVAL_EPISODES - 1}")
          + (" | PROBE: one zero step each, no policy" if EVAL_PROBE else "")
          + f" | arrival radius {GOAL_RADIUS_M:g} m (2 m also scored), "
          f"{'DETERMINISTIC' if EVAL_NOISE_SCALE <= 0 else f'SAMPLED at noise x{EVAL_NOISE_SCALE:g}'} actions, "
          f"faults END the episode (crossing/off-road/collision) | town "
          f"{','.join(TOWNS)} -> {out}", flush=True)
    import random as _random
    rows = []
    probes = []
    for k, ep in enumerate(episodes):
        seed = int(ep["seed"])
        # THE EPISODE SET IS FIXED BY SEEDING THE GLOBAL RNGs. The start and the
        # goal are drawn with Python's module-level `random` inside rlib's
        # carla_core (random.choice over waypoints / junctions / the distance
        # band), which reset(seed=...) never touches: two evals of the same
        # seeds shared 1 start-goal distance in 40 (2026-10-20). Seeding here,
        # immediately before reset, pins them -- eval only; training is unseeded
        # as before.
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs, info = env.reset(seed=seed)
        if EVAL_PROBE:
            # One zero step: the planner publishes route_remaining_m on steps,
            # not at reset. Zero throttle from rest moves the car ~0 m.
            p0 = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
            g0 = info.get("goal_pos")
            zero = np.clip(np.zeros(box.shape, np.float32), box.low, box.high)
            _o, _r, _te, _tr, inf1 = env.step(zero)
            p1 = np.asarray(inf1.get("hero_pos", p0), np.float64)
            rem = inf1.get("route_remaining_m")
            probes.append({
                "town": TOWNS[0], "seed": seed, "band": BENCH_BAND,
                **_poses(info),
                "route_m": (None if rem is None
                            else round(float(rem) + float(np.linalg.norm(p1 - p0)), 3)),
                "euclid_m": (None if g0 is None else round(float(np.linalg.norm(
                    np.asarray(g0, np.float64)[:2] - p0[:2])), 3))})
            if (k + 1) % 50 == 0:
                print(f"[Probe {k + 1}/{n_eps}] seed={seed} route={probes[-1]['route_m']} "
                      f"euclid={probes[-1]['euclid_m']}", flush=True)
            continue
        if EVAL_LANE_DEBUG and "lane_priv" in info and info.get("goal_pos") is not None:
            # Where the goal sits in the START lane's frame: lateral metres
            # (LEFT +, lane_priv's convention) and metres ahead. ~3.5 m sideways
            # on a straight road = the goal is in the next lane over.
            lp0 = np.asarray(info["lane_priv"], np.float64)
            lane_yaw = float(info.get("hero_yaw", 0.0)) - float(np.arctan2(lp0[2], lp0[3]))
            d = (np.asarray(info["goal_pos"], np.float64)
                 - np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64))
            cy, sy = np.cos(lane_yaw), np.sin(lane_yaw)
            g_lat = float(lp0[0]) - float(-d[0] * sy + d[1] * cy)
            g_fwd = float(d[0] * cy + d[1] * sy)
            print(f"[LaneStart] ep={k} goal_lat={g_lat:+.2f}m goal_fwd={g_fwd:+.1f}m "
                  f"lane_w={lp0[4]:.2f} left_illegal={int(lp0[7])} "
                  f"right_illegal={int(lp0[8])}", flush=True)
        mode = int(round(float(np.asarray(obs["mode"]).ravel()[0])))
        if mode != (0 if EVAL_EXPLORE else 1):
            raise RuntimeError(
                f"eval episode is in mode {mode}: EVAL_FORCE_MODE must be "
                f"{0 if EVAL_EXPLORE else 1} for {'an explore' if EVAL_EXPLORE else 'a goal'} eval")
        hidden = agent.reset_hidden(1)
        goal_blk = pxr.goal_block(info["debit_goal"])
        start_poses = _poses(info)
        nav = map_row = None
        if gmap is not None:
            hp = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]
            hy = float(info.get("hero_yaw", 0.0) or 0.0)
            s_node = gmap.nearest(hp, hy, radius=EVAL_MAP_START_M)
            g_node = (gmap.goal_node(s_node, info["goal_pos"], radius=EVAL_MAP_GOAL_M)
                      if info.get("goal_pos") is not None else None)
            route = gmap.path(s_node, g_node)
            map_row = {"map_start_node": s_node, "map_goal_node": g_node,
                       "map_hops": None if route is None else len(route) - 1,
                       "map_route_m": None if route is None else round(gmap.length(route), 2)}
            if route is None:
                why = ("no start node" if s_node is None else "no goal node"
                       if g_node is None else "no route")
                rows.append({"episode": k, "seed": seed, "town": TOWNS[0],
                             "end": "unroutable", "unroutable": why, **start_poses,
                             **{key: ep[key] for key in ("bin", "kind", "band") if key in ep},
                             "reached": False, "reached_2m": False, "spl": 0.0,
                             "spl_2m": 0.0, "steps": 0, "return": 0.0,
                             "shortest_m": None, "shortest_src": "none",
                             "straight_m": None, "path_m": 0.0, "video": None,
                             **map_row})
                print(f"[Eval {k + 1:3d}/{n_eps}] seed={seed} UNROUTABLE ({why})", flush=True)
                continue
            final_frame = goal_wrapper.latched_goal()
            nav = image_map.Navigator(gmap, route, switch_m=EVAL_MAP_SWITCH_M)

            def _aim():
                f = final_frame if nav.final else map_frames[nav.current()]
                obs["binoc"] = goal_wrapper.retarget(f, info)
                return pxr.goal_block(info["debit_goal"])
            goal_blk = _aim()
        vid.start(k)
        ret, t, end = 0.0, 0, None
        # SPL bookkeeping. p = metres actually driven; l = the shortest ROUTE
        # to the goal, from the planner the reward's potential term uses. The
        # planner publishes route_remaining_m on steps, not reliably at reset,
        # so l = (route left after step 1) + (metres moved on step 1). The
        # straight-line start distance is kept as a fallback: it is <= the
        # route, so using it can only make SPL stricter, never kinder.
        pos0 = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        prev_pos = pos0.copy()
        gpos = info.get("goal_pos")
        l_straight = (float(np.linalg.norm(np.asarray(gpos, np.float64) - pos0))
                      if gpos is not None else None)
        l_route, path_m = None, 0.0
        # Scores our training radius (2 m) inside a run at GOAL_RADIUS_M.
        tracker = (EpisodeTracker(gpos, pos0, max_speed=float(
            os.environ.get("GOAL_ARRIVAL_MAX_SPEED", 10.0)), radius=GOAL_RADIUS_M,
            radius_2m=2.0) if gpos is not None else None)
        chain_goals = 0
        while True:
            while end is None:
                row = pxr.mono_row(info["debit_cur"], _net_mode(obs["mode"]), obs["binoc"],
                                   speed_kmh=_speed_kmh(info))
                action, hidden = agent.act(row, hidden, goal=goal_blk,
                                           deterministic=EVAL_NOISE_SCALE <= 0,
                                           noise_scale=EVAL_NOISE_SCALE)
                a_env = to_env(np.asarray(action, np.float32), box)
                vid.add(info["debit_cur"], info["debit_goal"], t, mode, 0.0, a_env, ret)
                obs, r, term, trunc, info = env.step(a_env)
                ret += float(r)
                t += 1
                if EVAL_LANE_DEBUG and "lane_priv" in info:
                    lp = np.asarray(info["lane_priv"], np.float64)
                    cross = bool(info.get("lane_any", info.get("lane_invasion_any", False)))
                    if cross or t % 5 == 1:
                        print(f"[Lane] ep={k} t={t} off={lp[0]:+.2f}m "
                              f"lane_w={lp[4]:.2f} herr={np.degrees(np.arctan2(lp[2], lp[3])):+.0f}deg "
                              f"junc={int(lp[11])} c={info.get('lane_potential', float('nan')):.3f} "
                              f"F={info.get('lane_shaping', float('nan')):+.3f} "
                              f"half_car={info.get('hero_half_width')} cross={int(cross)} "
                              f"marks={[str(m).split('.')[-1] for m in (info.get('lane_marking_types') or [])]} "
                              f"a=({a_env[0]:+.2f},{a_env[1]:.2f})", flush=True)
                pos = np.asarray(info.get("hero_pos", prev_pos), np.float64)
                if tracker is not None:
                    tracker.step(pos, float(info.get("speed_kmh", 0.0) or 0.0))
                if nav is not None and nav.update(pos, float(info.get("hero_yaw", 0.0) or 0.0)):
                    goal_blk = _aim()
                    if EVAL_MAP_RESET_HIDDEN:
                        hidden = agent.reset_hidden(1)
                moved = float(np.linalg.norm(pos - prev_pos))
                path_m += moved
                prev_pos = pos
                if t == 1 and info.get("route_remaining_m") is not None:
                    l_route = float(info["route_remaining_m"]) + moved
                _end_key = "explore_end" if EVAL_EXPLORE else "goal_end"
                if info.get("is_success") or info.get(_end_key) == "reached":
                    end = "reached"
                else:
                    end = _fault(info)
                if end is None and (term or trunc):
                    end = str(info.get(_end_key, "timeout"))
            if EVAL_CHAIN and end == "reached" and info.get("chain_continue"):
                chain_goals += 1
                arrived_info = info
                obs, info = env.reset()   # soft reset: the same car, the next goal
                if not info.get("chain_soft_reset"):
                    # No goal could be placed and the wrapper reset fully, which
                    # also zeroed its chain_m: report the chain as it stood.
                    end, info = "dead_end", arrived_info
                    break
                hidden = agent.reset_hidden(1)   # a chained goal is a new episode
                goal_blk = pxr.goal_block(info["debit_goal"])
                end = None
                continue
            if EVAL_CHAIN and end == "reached":
                chain_goals += 1                 # the arrival that completed the chain
            break
        path = vid.save(end)
        ell = l_route if l_route is not None else l_straight
        reached = end == "reached"
        # SHORTEST PATH TO THE SUCCESS REGION, not to the goal point: arrival
        # counts within GOAL_RADIUS_M (2 m, the env's passed_chckp test), so a
        # 5 m goal is reached after ~3.5 m and l / max(p, l) was 1.0 for every
        # success -- SPL equalled SR by construction (2026-10-20 evals).
        ell_eff = None if ell is None else max(ell - GOAL_RADIUS_M, 0.1)
        spl_i = (float(reached) * ell_eff / max(path_m, ell_eff)
                 if ell_eff is not None else float("nan"))
        two = (tracker.finish(reached=reached, t_end=t, route_m=ell)
               if tracker is not None else {"reached_2m": reached, "spl_2m": spl_i})
        rows.append({"episode": k, "seed": seed, "town": TOWNS[0], "end": end,
                     **start_poses,
                     **{key: ep[key] for key in ("bin", "kind", "band") if key in ep},
                     "reached_2m": bool(two["reached_2m"]),
                     # APPROACH: how close it got to the episode's goal and how
                     # fast it was going there (EpisodeTracker; None if absent).
                     **{key: two.get(key) for key in (
                         "closest_m", "speed_at_closest_kmh", "entry_speed_kmh",
                         "min_speed_inside_kmh")},
                     "spl_2m": float(two["spl_2m"]),
                     "steps": t, "return": round(ret, 2), "reached": reached,
                     "shortest_m": None if ell is None else round(ell, 2),
                     "shortest_src": "route" if l_route is not None else "straight",
                     "straight_m": None if l_straight is None else round(l_straight, 2),
                     "path_m": round(path_m, 2), "spl": spl_i, "video": path,
                     **({} if nav is None else {**map_row, "map_switches": nav.switches,
                                                "map_final": bool(nav.final)}),
                     **(chain_summary(chain_goals, info, end,
                                      float(os.environ.get("GOAL_CHAIN_M", 200.0) or 200.0))
                        if EVAL_CHAIN else {})})
        sr = np.mean([x["reached"] for x in rows])
        spl = np.nanmean([x["spl"] for x in rows])
        print(f"[Eval {k + 1:3d}/{n_eps}] seed={seed} end={end} "
              f"steps={t} shortest={ell if ell is None else round(ell, 1)} m "
              f"driven={path_m:.1f} m return={ret:.1f} | SR {sr:.3f} SPL {spl:.3f}",
              flush=True)

    if EVAL_PROBE:
        os.makedirs(out, exist_ok=True)
        pf = os.path.join(out, "probe.json")
        with open(pf, "w") as fh:
            json.dump(probes, fh, indent=1)
        routed = sum(1 for p in probes if p["route_m"] is not None)
        print(f"[ProbeDone] {len(probes)} seeds, {routed} routed -> {pf}", flush=True)
        env.close()
        return
    n = len(rows)
    sr = float(np.mean([x["reached"] for x in rows])) if n else float("nan")
    # Wilson 95% interval: at n=40 the binomial noise is about +-0.15.
    z = 1.96
    centre = (sr + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(sr * (1 - sr) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    ends = {}
    for x in rows:
        ends[x["end"]] = ends.get(x["end"], 0) + 1
    spl = float(np.nanmean([x["spl"] for x in rows])) if n else float("nan")
    sr_2m = float(np.mean([x["reached_2m"] for x in rows])) if n else float("nan")
    n_route = sum(1 for x in rows if x["shortest_src"] == "route")
    summary = {"checkpoint": EVAL_FROM, "episodes": n, "success_rate": sr,
               "spl": spl, "spl_shortest_from_route": n_route,
               "arrival_radius_m": GOAL_RADIUS_M, "success_rate_2m": sr_2m,
               "episodes_file": EVAL_EPISODES_FILE or None,
               "map_dir": EVAL_MAP_DIR or None,
               "unroutable": sum(1 for x in rows if x["end"] == "unroutable"),
               "wilson95": [round(centre - half, 3), round(centre + half, 3)],
               "ends": ends, "fault_ends": "crossing/off_road/collision",
               "deterministic": EVAL_NOISE_SCALE <= 0,
               "noise_scale": EVAL_NOISE_SCALE, "towns": TOWNS, "seed0": EVAL_SEED,
               "episodes_detail": rows}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[EvalDone] SR={sr:.3f} (95% CI {summary['wilson95'][0]:.2f}-"
          f"{summary['wilson95'][1]:.2f}, n={n}) SPL={spl:.3f} "
          f"(shortest path from the route planner in {n_route}/{n}, straight "
          f"line otherwise) SR@2m={sr_2m:.3f} radius={GOAL_RADIUS_M:g}m "
          f"ends={ends} -> {out}", flush=True)
    env.close()
    server.stop()


def build_agent(binoc_dim, act_dim):
    """The learner exactly as main() configures it, with DEBiT's pretrained
    monocular trunk loaded into both the online and the target encoder.
    Factored out of main() (2026-11-04, imitation-first plan) so behaviour
    cloning can build the same network without starting CARLA. The trunk
    load report is attached as agent.mono_report for the preflight line."""
    from torch_rl import pixel_rows as pxr
    from torch_rl.debit_mono import load_debit_mono
    from torch_rl.gated_recurrent_sac import (GatedRecurrentSAC,
                                              GatedRecurrentSACConfig)
    state_dim = pxr.mono_state_dim(binoc_dim)
    ACT_DIM = int(act_dim)
    cfg = GatedRecurrentSACConfig(
        # THE BINOCULAR + MONOCULAR STACK. The trainable visual path is a
        # half-width ResNet-18 from scratch over the 128 px frame
        # (mono_encoder="resnet"); the FROZEN DEBiT binocular vector reaches the
        # GRU through core_state_cols, which is what the peer's own vc3 layout
        # does. goal_tokens is OFF: that path stacks a goal IMAGE onto the
        # frame, and here the goal arrives already encoded.
        mono_encoder="resnet", embed_dim=3, grid=pxr.MONO_IMG,
        # THE ACTOR IS SIZED FROM THE ENV (STOP_ACTION adds a third dimension;
        # the default 2 then dies on "operands could not be broadcast together
        # with shapes (2,) (3,)", which a simulator probe hit on 2026-10-28).
        action_dim=ACT_DIM,
        # Encoder output head (user 2026-10-26): ln_tanh = fc -> LayerNorm ->
        # tanh, which cannot die; relu (default) is what v13-v16 ran.
        mono_head=os.environ.get("MONO_HEAD", "relu"),
        goal_tokens=False,
        # NO PREVIOUS ACTION IN THE OBSERVATION (user 2026-10-14). The
        # learner then builds no action embedding and the GRU reads
        # [visual | binoc] alone.
        prev_action=False,
        mono_baseplanes=env_int("MONO_BASEPLANES", 32),
        mono_ngroups=env_int("MONO_NGROUPS", 16),
        core_state_cols=pxr.mono_core_state_cols(binoc_dim),
        state_dim=state_dim, mode_index=0,
        # ONE CRITIC FOR BOTH REWARD MODES, and the mode bit is its whole view:
        # the two modes differ only in what the env pays, so without that column
        # the critic would learn the average of two different returns.
        one_critic=os.environ.get("ONE_CRITIC", "1") == "1",
        # THE CRITIC SEES EVERYTHING THE ACTOR SEES (user 2026-10-14):
        # the whole state block, mode bit and binoc alike. The actor is
        # handed the state block in full, so anything less here makes the
        # critic judge actions on strictly less information than the
        # actor chose them with -- and the mode bit, which one_critic
        # requires, is column 0 of it.
        goal_cols=tuple(range(state_dim)), explore_cols=(),
        hidden=env_int("HIDDEN", 512),
        visual_dim=env_int("VISUAL_DIM", 512), mlp=(256, 256), burn_in=BURN_IN,
        device=DEVICE, compile=os.environ.get("TORCH_COMPILE", "0") == "1",
        seed=SEED, n_critics=env_int("N_CRITICS", 10),
        # LR drives the critic and the actor; ALPHA_LR the temperatures
        # (0 = follow LR). Under a high replay ratio both have to come
        # down: see GatedRecurrentSACConfig.alpha_lr.
        lr=env_float("LR", 3e-4), alpha_lr=env_float("ALPHA_LR", 0.0),
        # DroQ: dropout + LayerNorm per critic hidden layer. REDQ (the min over
        # a random subset) was already here; these two are what let the replay
        # ratio go up without the critic running away.
        critic_dropout=env_float("CRITIC_DROPOUT", 0.0),
        critic_layer_norm=os.environ.get("CRITIC_LAYERNORM", "0") == "1",
        subset_size=env_int("SUBSET_SIZE", 2),
        discount=env_float("DISCOUNT", 0.99),
        discount_e=env_float("EXPLORE_DISCOUNT", 0.99),
        explore_actor_warmup=env_int("EXPLORE_ACTOR_WARMUP", 5000),
        value_floor=env_float("TQC_VALUE_FLOOR", -200.0),
        value_ceiling=env_float("TQC_VALUE_CEILING", 150.0),
        value_floor_e=env_float("EXPLORE_VALUE_FLOOR", -200.0),
        value_ceiling_e=env_float("EXPLORE_VALUE_CEILING", 150.0),
        # THE ENTROPY TARGET the temperature tunes -log pi (sampled, AFTER the
        # tanh squash) towards. -2 (the default, -action_dim) let alpha_g fall
        # to ~5e-4 in v10/v11 while the actor's raw outputs drifted past +-1:
        # v10's deterministic eval sat at steer +-0.99 (SR 0.10). A higher
        # target keeps alpha alive, and because the squash's Jacobian makes a
        # pinned action improbable-looking, the only way to meet it is to keep
        # actions off the limits. User 2026-10-20, "try 2": v12 sets 0.
        target_entropy=env_float("TARGET_ENTROPY", -2.0),
        actor_pretanh_l2=env_float("ACTOR_PRETANH_L2", 0.0),
        actor_pretanh_free=env_float("ACTOR_PRETANH_FREE", 1.5))
    agent = GatedRecurrentSAC(cfg)
    # DEBiT'''S PRETRAINED MONOCULAR TRUNK (user 2026-10-14, "I want debit Mono +
    # binocular setup"). Loaded AFTER the learner builds the module and copied
    # into the target as well: the learner makes its frozen target copy at
    # construction, so loading only into the online branch would leave the
    # target encoding with random weights.
    mono_report = load_debit_mono(agent.compress, TINY)
    agent.compress_targ.load_state_dict(agent.compress.state_dict())
    agent.mono_report = mono_report
    return agent


def main():
    import torch

    from torch_rl import pixel_rows as pxr
    from torch_rl.carla_env_build import build_env
    from torch_rl.carla_server import CarlaServer, connect_with_restarts
    from torch_rl.debit_encoder import DebitTinyEncoder
    from torch_rl.debit_mono import describe_load, load_debit_mono
    from torch_rl.debit_visual import IMG
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.episode_video import EpisodeVideo
    from torch_rl.gated_recurrent_sac import (GatedRecurrentSAC,
                                              GatedRecurrentSACConfig)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA: this arm is GPU-resident, so this is fatal")
    os.makedirs(OUT, exist_ok=True)

    # BEHAVIOUR CLONING (imitation-first plan, Task 6): BC_TRAIN=1 clones the
    # recorded expert into this learner and returns. No simulator and no DEBiT
    # encoder: the frozen binocular features are already in the shards.
    if os.environ.get("BC_TRAIN", "0") == "1":
        from torch_rl.bc_trainer import run_bc_train
        return run_bc_train(OUT, make_agent=build_agent)

    # DEBiT IS FROZEN (user 2026-10-14), so no adapters are added and its 3136-d
    # binocular output is a fixed function of (view, goal). That lets the ENV
    # compute it once per step and store it in the row, instead of the learner
    # re-encoding the whole chunk twice per update at 209 ms. The trainable
    # visual path is the monocular ResNet, from scratch, which is why the row
    # still carries pixels.
    enc = DebitTinyEncoder(TINY, device=DEVICE, half=False)

    # THE TRAINER OWNS THE SIMULATOR, so a town change can restart it. Closing
    # the env and rebuilding against a live server leaves the old sensors'
    # streams dangling and the next client cannot attach (job 453315: the first
    # map change worked, the next rebuild hung while the server flooded "Invalid
    # session: no stream available"). A restart cannot inherit that state.
    town_i = 0
    parallel = use_parallel(PARALLEL_TOWNS, EVAL_FROM, len(TOWNS)) and not EXPERT_RECORD
    # JUNCTION RAMP (torch_rl.junction_ramp) lives in the parallel loop only,
    # and never in eval, whose goal set must stay at the overlay's fraction.
    if ramp_needs_parallel(JUNCTION_RAMP_START, parallel, EVAL_FROM, EXPERT_RECORD):
        raise ValueError("JUNCTION_RAMP_START needs PARALLEL_TOWNS=1: the "
                         "serial loop has no junction ramp")
    ports = [PORT + 50 * i for i in range(len(TOWNS))] if parallel else [PORT]
    servers = []

    def _start_server(i, port):
        srv = CarlaServer(port, quality=os.environ.get("CARLA_QUALITY", "Epic"),
                          log_dir=os.path.join(OUT, "carla", TOWNS[i]))
        srv.start()
        print(f"  carla[{TOWNS[i]}]: {srv.describe()} ready in "
              f"{srv.wait_ready():.0f}s", flush=True)
        servers.append(srv)
        return srv

    if not parallel:
        _start_server(0, ports[0])
    server = servers[0] if servers else None
    venv = None
    if parallel:
        import threading
        from vae_nav.vec_env import ThreadedVecEnv
        # BUILT ONE AT A TIME, STEPPED TOGETHER. ThreadedVecEnv constructs its
        # envs concurrently; with two Epic servers that made one env's first
        # sensor read time out while the other map was still loading (job
        # 455086, "A sensor took too long to send their data"). The lock
        # serialises construction -- and any later rebuild -- while the steps
        # still overlap on threads.
        _build_lock = threading.Lock()

        # EACH SERVER STARTS RIGHT BEFORE ITS ENV, inside the lock (job 455090).
        # A server with no client renders its default map free-running at Epic
        # until an env connects and puts it in synchronous mode; started all
        # up front, the idle second server starved the first of GPU and its
        # first sensor read missed rlib's 10 s timeout. Every single-server
        # smoke (Town05/Town01, either goal camera) passed. The JAX trainer
        # also started each server inside its env factory, staggered.
        def _make(t, p, k):
            with _build_lock:
                if not any(sv.port == p for sv in servers):
                    _start_server(k, p)
                srv = next(sv for sv in servers if sv.port == p)
                return connect_with_restarts(
                    lambda: build_env(t, seed=SEED + k, encoder=enc, port=p, state_keys=()),
                    srv)
        venv = ThreadedVecEnv(
            [(lambda t=t, p=p, k=k: _make(t, p, k))
             for k, (t, p) in enumerate(zip(TOWNS, ports))],
            names=list(TOWNS))
        env = venv.envs[0]
        server = servers[0]
    else:
        env = connect_with_restarts(
            lambda: build_env(TOWNS[town_i], seed=SEED, encoder=enc, port=PORT, state_keys=(), eval_mode=bool(EVAL_FROM) or EXPERT_RECORD),
            server)
    obs, info = env.reset(seed=SEED)
    # THE STATE BLOCK IS THE MODE BIT AND THE BINOC, NOTHING ELSE (user
    # 2026-10-14: "get rid of all state vectors except mode bit"). No steer, no
    # throttle, no speed: the actor used to receive its own last steer and
    # throttle, a feedback path a policy can latch onto, and the run that
    # collapsed sat at full left lock for 800k steps.
    binoc_dim = int(np.shape(obs["binoc"])[0])
    state_dim = pxr.mono_state_dim(binoc_dim)
    row_dim = pxr.mono_row_dim(binoc_dim)

    # THE ACTOR AND THE REPLAY ARE SIZED FROM THE ENV (STOP_ACTION adds a third
    # dimension). The defaults were 2 in both places; a simulator probe hit the
    # first ("shapes (2,) (3,)") and the smoke run the second ("the expanded
    # size of the tensor (2) must match the existing size (3)").
    ACT_DIM = int(env.action_space.shape[0])
    agent = build_agent(binoc_dim, ACT_DIM)
    mono_report = agent.mono_report
    cfg = agent.cfg          # the banner and the [Ep] line read the learner's config
    if EXPERT_RECORD:
        if EXPERT_DRIVER == "policy":
            if not EVAL_FROM:
                raise ValueError("EXPERT_DRIVER=policy needs EVAL_FROM: the policy to drive")
            agent.load(EVAL_FROM)
        return run_record(env, agent, pxr, server, OUT)
    if EVAL_FROM:
        agent.load(EVAL_FROM)          # strict: every module, name and shape
        if EVAL_SGM_TRACE:
            run_sgm_pairs(agent, enc, pxr)
            env.close()
            server.stop()
            return
        if EVAL_TRACE:
            return run_trace(env, agent, pxr, server, OUT)
        return run_eval(env, agent, pxr, server, OUT)
    if INIT_FROM:
        # After load_debit_mono above, so the checkpoint's trained encoder wins.
        if LEARNING_STARTS < INIT_MIN_LEARNING_STARTS and not IL_PRELOAD_ROOT:
            raise ValueError(
                f"INIT_FROM={INIT_FROM} with LEARNING_STARTS={LEARNING_STARTS}: the "
                f"replay buffer starts empty, and updating on a near-empty one "
                f"destroyed goal steering before. Set LEARNING_STARTS >= "
                f"{INIT_MIN_LEARNING_STARTS} so the loaded policy refills it first.")
        agent.load(INIT_FROM)
        print(f"  WARM START from {INIT_FROM}: modules, targets, temperatures, "
              f"optimisers and update counter restored; replay buffer EMPTY, so no "
              f"update until {LEARNING_STARTS} transitions from the loaded policy",
              flush=True)
    buf = DeviceEpisodeBuffer(row_dim, ACT_DIM, 1000 if EVAL_FROM else CAPACITY, DEVICE, seed=SEED,
                              goal_dim=pxr.PIX,
                              goal_batch_frac=env_float("GOAL_BATCH_FRAC", 0.5),
                              store_device=REPLAY_DEVICE,
                              u8_cols=pxr.MONO_PIX if REPLAY_U8 else 0)

    # PREFLIGHT, printed before a single step (user rule, after four restarts):
    # the label, the checkpoint directory, the town, the seeds and the action
    # mode, so a run's identity is in its own log rather than inferred later.
    print(f"=== [{RUN_NAME}] preflight ===", flush=True)
    print(f"  towns={','.join(TOWNS)} (rotating every {TOWN_EVERY} episodes; "
          f"Town02/Town03 HELD OUT) port={PORT} seed={SEED} device={DEVICE}",
          flush=True)
    print(f"  {agent.describe()}", flush=True)
    # THE REPLAY RATIO AND THE RATES THAT HAVE TO MATCH IT (v31). Samples per
    # new transition is what destabilises a single critic, not "updates", so
    # the banner states it: CHUNK_B*CHUNK_T samples every UPDATE_EVERY vec
    # steps over len(TOWNS) envs.
    _per_tr = CHUNK_B * CHUNK_T * UPDATES_PER / max(1, UPDATE_EVERY * len(TOWNS))
    print(f"  replay ratio: {UPDATES_PER} update(s) of {CHUNK_B}x{CHUNK_T} every "
          f"{UPDATE_EVERY} vec step(s) = {_per_tr:.0f} replayed samples per new "
          f"transition | lr={cfg.lr:g} alpha_lr={cfg.alpha_lr or cfg.lr:g}"
          + ("" if cfg.alpha_lr else " (follows lr)"), flush=True)
    print(f"  monocular (TRAINED): {agent.compress.describe()}", flush=True)
    print(f"  {describe_load(mono_report)}", flush=True)
    print(f"  binocular (FROZEN, precomputed in the env): {binoc_dim}-d -> GRU "
          f"via core_state_cols[{cfg.core_state_cols[0]}..{cfg.core_state_cols[-1]}]",
          flush=True)
    print(f"  state block: mode bit ONLY at index {cfg.mode_index} "
          f"(no steer/throttle/speed) | critic view = the FULL state block "
          f"({len(cfg.goal_cols)} cols: mode + binoc), same as the actor", flush=True)
    print(f"  weights: {enc.describe()}", flush=True)
    print(f"  row={row_dim} (frame {pxr.MONO_PIX} + mode 1 + "
          f"binoc {binoc_dim}; NO prev action)"
          f" | replay {CAPACITY} rows = {buf.nbytes / 2 ** 30:.1f} GiB on "
          f"{buf.store_device} (frames {'uint8' if REPLAY_U8 else 'fp16'}), "
          f"batches delivered to {buf.device}",
          flush=True)
    print(f"  update: every {UPDATE_EVERY} steps x{UPDATES_PER}, chunk "
          f"B={CHUNK_B} T={CHUNK_T} burn_in={BURN_IN}, starts at {LEARNING_STARTS}",
          flush=True)
    # THE SPACE ITSELF, not the environment variables that were meant to shape
    # it. The banner used to print "throttle 0-1.0" from MAX_THROTTLE while the
    # space was [-1, 1] with negative meaning reverse -- a preflight that states
    # the intent rather than the fact cannot catch the case where they differ,
    # which is the only case it is for.
    _box = env.action_space
    print(f"  env action box: steer [{_box.low[0]:+.2f}, {_box.high[0]:+.2f}]  "
          f"throttle [{_box.low[1]:+.2f}, {_box.high[1]:+.2f}] "
          f"(actor tanh rescaled onto it; no reverse)  "
          f"incremental={os.environ.get('INCREMENTAL_ACTIONS')}"
          + (f"  | DECLARE ARRIVAL: dim 2, > 0 and slow = 'I am at the goal'; "
             f"declaring elsewhere ends the episode unpaid"
             if _box.shape[0] > 2 else ""), flush=True)
    print(f"  reward: GATED_GOAL_REWARD={os.environ.get('GATED_GOAL_REWARD')} "
          f"bonus={os.environ.get('GOAL_EXPLORE_ARRIVAL_BONUS')} "
          f"band={os.environ.get('GOAL_DIST_MIN')}-{os.environ.get('GOAL_DIST_MAX')} m",
          flush=True)
    print(f"  checkpoints -> {OUT}", flush=True)
    vid = EpisodeVideo(os.path.join(OUT, "videos"), every=VIDEO_EVERY,
                       max_frames=VIDEO_MAX_FRAMES, keep=VIDEO_KEEP)
    print(f"  videos: 1 episode in {VIDEO_EVERY}, <={VIDEO_MAX_FRAMES} frames, "
          f"keep {VIDEO_KEEP} -> {vid.out_dir} "
          "(camera | goal, with step/mode/reward/action burnt in)", flush=True)

    def row_of(o, i):
        """[mono frame | mode | binoc] for this step.

        The binoc comes from the observation, where the wrapper put it after
        encoding (view, goal) through the FROZEN DEBiT -- so it is a stored
        feature, not something the learner recomputes."""
        # THE ENCODER MUST HAVE RUN. With no encoder the wrapper serves a ZERO
        # binoc and flags it; stored, that trains the policy goal-blind and
        # nothing downstream looks wrong -- the run simply never learns to use
        # the goal, which is the failure this whole arm exists to detect.
        if not i.get("binoc_encoded", False):
            raise RuntimeError(
                "info['binoc_encoded'] is False: the env wrapper has no DEBiT "
                "encoder, so the binoc column would be zeros.")
        return pxr.mono_row(i["debit_cur"], _net_mode(o["mode"]), o["binoc"],
                            speed_kmh=_speed_kmh(i))

    il_hook = None
    if IL_PRELOAD_ROOT:
        from torch_rl.bc_trainer import il_finetune_prepare
        il_aug = None
        if os.environ.get("BC_AUG", "") == "1":
            il_aug = {"gen": torch.Generator().manual_seed(SEED),
                      "bright": env_float("BC_AUG_BRIGHT", 0.4),
                      "contrast": env_float("BC_AUG_CONTRAST", 0.4),
                      "color": env_float("BC_AUG_COLOR", 0.1)}
        il_hook = il_finetune_prepare(
            agent, buf, IL_PRELOAD_ROOT, IL_PRELOAD_TOWNS,
            warmup_updates=IL_CRITIC_WARMUP_UPDATES, chunk_b=CHUNK_B, chunk_t=CHUNK_T,
            w0=IL_BC_WEIGHT0, decay_updates=IL_BC_DECAY_UPDATES, bc_chunks=IL_BC_CHUNKS,
            bc_len=IL_BC_LEN, aug=il_aug, log=lambda m: print(m, flush=True))

    if parallel:
        from torch_rl.parallel_loop import run_parallel
        print(f"  PARALLEL: {len(TOWNS)} envs at once, one server per town "
              f"({', '.join(f'{t}:{p}' for t, p in zip(TOWNS, ports))}); no "
              f"town switching", flush=True)
        print(f"  NOISE-FREE EPISODES: {DET_EPISODE_FRAC:.0%} of training episodes "
              f"act with tanh(mean), drawn per env per episode"
              + ("" if DET_EPISODE_FRAC else " -- OFF"), flush=True)
        vids = [EpisodeVideo(os.path.join(OUT, "videos", t), every=VIDEO_EVERY,
                             max_frames=VIDEO_MAX_FRAMES, keep=VIDEO_KEEP)
                for t in TOWNS]
        her = None
        if HER:
            import vae_nav.gated_reward as _G
            from torch_rl.image_her import ImageGoalHER
            her = ImageGoalHER(
                buf, enc, pxr, n_relabel=HER_RELABEL,
                min_m=env_float("GOAL_DIST_MIN", 5.0),
                max_m=env_float("GOAL_DIST_MAX", 25.0), radius=GOAL_RADIUS_M,
                max_speed=float(_G.GOAL_ARRIVAL_MAX_SPEED or 1e9),
                goal_bonus=float(_G.GOAL_EXPLORE_ARRIVAL_BONUS),
                explore_bonus=float(_G.EXPLORE_ARRIVAL_BONUS),
                potential_k=float(_G.POTENTIAL_K),
                lane_fault=lambda inf: _fault(inf) == "crossing",
                relabel_explore=HER_EXPLORE, seed=SEED,
                speed_gate_kmh=(env_float("TARGET_SPEED", 20.0)
                                if (_G.PROGRESS_SPEED_GATE
                                    and (_G.GOAL_ROUTE_ONLY or _G.GOAL_ZONE_M > 0))
                                else None),
                # Under the ratchet nothing is charged for moving away; along a
                # path the car drove, distance to a hindsight goal only falls,
                # so new-ground pay equals plain progress there.
                retreat_scale=float(0.0 if _G.ROUTE_PROGRESS_RATCHET else _G.ROUTE_RETREAT_SCALE),
                zone_only=_G.GOAL_ZONE_M > 0)
            print(f"  HER: {HER_RELABEL} hindsight legs per episode, "
                  f"{env_float('GOAL_DIST_MIN', 5.0):g}-{env_float('GOAL_DIST_MAX', 25.0):g} m, "
                  f"arrival {GOAL_RADIUS_M:g} m at <= {_G.GOAL_ARRIVAL_MAX_SPEED:g} km/h, "
                  f"+{_G.GOAL_EXPLORE_ARRIVAL_BONUS:g}; progress K={_G.POTENTIAL_K:g} "
                  f"to the hindsight goal; clean prefix only; explore episodes "
                  f"{'relabelled' if HER_EXPLORE else 'skipped'}", flush=True)
        ramp = None
        if JUNCTION_RAMP_START is not None:
            from torch_rl.junction_ramp import JunctionRamp
            # The ramp's TARGET is the overlay's JUNCTION_GOAL_FRAC, the
            # fraction the evals use; read before the ramp overwrites it.
            ramp = JunctionRamp(
                start=JUNCTION_RAMP_START,
                target=env_float("JUNCTION_GOAL_FRAC", 0.85),
                step=env_float("JUNCTION_RAMP_STEP", 0.05),
                up=env_float("JUNCTION_RAMP_UP", 0.3),
                down=env_float("JUNCTION_RAMP_DOWN", 0.1),
                window=int(env_float("JUNCTION_RAMP_WINDOW", 100)))
            print(f"  {ramp.describe()}", flush=True)
        t_start = time.time()
        steps, episodes = run_parallel(
            venv, agent, buf, row_of, _box, to_env=to_env, to_actor=to_actor,
            goal_block=pxr.goal_block, max_steps=MAX_STEPS,
            learning_starts=LEARNING_STARTS, update_every=UPDATE_EVERY,
            updates_per=UPDATES_PER, chunk_b=CHUNK_B, chunk_t=CHUNK_T,
            burn_in=BURN_IN, ckpt_every=CKPT_EVERY, out_dir=OUT,
            log_every=LOG_EVERY, vids=vids, her=her,
            log=lambda m: print(m, flush=True),
            det_frac=DET_EPISODE_FRAC, det_seed=SEED, junction_ramp=ramp,
            # A warm start refills the empty buffer with its OWN policy.
            random_warmup=not INIT_FROM, after_update=il_hook)
        venv.close()
        for srv in servers:
            srv.stop()
        print(f"[Done] {steps} steps, {episodes} episodes in "
              f"{(time.time() - t_start) / 3600:.2f} h -> {OUT}", flush=True)
        return

    hidden = agent.reset_hidden(1)
    goal_blk = pxr.goal_block(info["debit_goal"])   # the buffer's episode goal
    row = row_of(obs, info)
    mode = int(round(float(np.asarray(obs["mode"]).ravel()[0])))

    ep_ret, ep_len, episodes = 0.0, 0, 0
    # Executed-action symptoms since the last log line: steering at full lock
    # (v7, v10) and throttle near zero (v8, v10 parking). Logged only.
    roll = {"n": 0, "lock": 0, "idle_thr": 0}
    vid.start(0)
    recent = {"goal": [], "explore": [], "policy": [], "ends": {}}
    t_start = time.time()
    metrics = {}

    for step in range(1, MAX_STEPS + 1):
        # `action` is the ACTOR's, in tanh space [-1, 1]: that is what replay
        # stores and the learner trains on. The env gets it rescaled onto its
        # own box (throttle [0, max] since 2026-10-19).
        if step <= LEARNING_STARTS:
            action = to_actor(env.action_space.sample().astype(np.float32), _box)
        else:
            action, hidden = agent.act(row, hidden, goal=goal_blk)
            action = np.asarray(action, np.float32)

        _env_a = to_env(action, _box)
        roll["n"] += 1
        roll["lock"] += int(abs(float(action[0])) > 0.95)
        roll["idle_thr"] += int(float(_env_a[1]) < 0.05)
        nobs, reward, terminated, truncated, ninfo = env.step(_env_a)
        nrow = row_of(nobs, ninfo)
        buf.add(row, action, float(reward), bool(terminated), bool(truncated),
                nrow, mode, goal=goal_blk)
        # The frames the policy just acted on, beside the goal it was given.
        vid.add(info["debit_cur"], info["debit_goal"], ep_len, mode,
                float(reward), to_env(action, _box), ep_ret)
        ep_ret += float(reward)
        ep_len += 1
        obs, info, row = nobs, ninfo, nrow

        if terminated or truncated:
            buf.end_episode()
            episodes += 1
            end = str(info.get("goal_end", info.get("explore_end", "?")))
            saved = vid.save(end)
            if saved:
                print(f"[Video] {saved}", flush=True)
            recent["ends"][end] = recent["ends"].get(end, 0) + 1
            arrived = end == "reached"
            recent["policy"].append(float(arrived))
            recent["goal" if mode == 1 else "explore"].append(
                dict(ret=ep_ret, len=ep_len, arrived=float(arrived), end=end))
            for k in ("goal", "explore", "policy"):
                recent[k] = recent[k][-100:]

            if episodes % LOG_EVERY == 0:
                pol = float(np.mean(recent["policy"])) if recent["policy"] else float("nan")
                g, e = recent["goal"], recent["explore"]
                sr = float(np.mean([x["arrived"] for x in g])) if g else float("nan")
                per_h = step / max(1e-9, (time.time() - t_start)) * 3600
                # LEAD WITH policy= (user rule 2026-09-25): the arm serves two
                # modes and the goal-arm rate alone can read well while the
                # explore half degrades.
                print(f"[Ep {episodes:5d}] step={step:8d} "
                      f"policy={pol:.3f}(n={len(recent['policy'])}) "
                      f"goal={sr:.3f}(n={len(g)}) "
                      f"goal_len={np.mean([x['len'] for x in g]):.0f} "
                      f"explore_len={np.mean([x['len'] for x in e]):.0f} "
                      f"ends={recent['ends']} "
                      f"buffer={buf.size} {per_h / 1000:.0f}k steps/h "
                      f"| {metrics.get('critic_loss_g', float('nan')):.3f} "
                      f"q_g={metrics.get('q_g', float('nan')):.2f} "
                      f"q_e={metrics.get('q_e', float('nan')):.2f} "
                      f"alpha_g={metrics.get('alpha_g', float('nan')):.4f}",
                      flush=True)
                # The saturation diagnostics, on their own line so the [Ep] line
                # every existing parser reads is unchanged.
                _m = lambda k: float(metrics.get(k, float("nan")))
                print(f"[Diag] step={step} -logpi={_m('neg_logp_g'):.2f} "
                      f"(target {cfg.target_entropy:g}) "
                      f"log_std={_m('log_std_g'):.2f} "
                      f"|u| steer={_m('mean_abs_u_steer'):.2f} "
                      f"thr={_m('mean_abs_u_throttle'):.2f} "
                      f"sat(replay) steer={_m('sat_steer'):.2f} "
                      f"thr={_m('sat_throttle'):.2f} | rollout: "
                      f"steer_lock={roll['lock'] / max(1, roll['n']):.2f} "
                      f"throttle<0.05={roll['idle_thr'] / max(1, roll['n']):.2f}",
                      flush=True)
                roll = {"n": 0, "lock": 0, "idle_thr": 0}
                recent["ends"] = {}

            # ROTATE THE TOWN. One server, so this reloads the map inside the
            # same simulator; the env is rebuilt because CarlaCore reads the
            # town from the config it was constructed with.
            if TOWN_EVERY > 0 and episodes % TOWN_EVERY == 0 and len(TOWNS) > 1:
                town_i = (town_i + 1) % len(TOWNS)
                env.close()
                took = server.restart()
                env = connect_with_restarts(
                    lambda: build_env(TOWNS[town_i], seed=SEED + episodes, port=PORT, encoder=enc, state_keys=()),
                    server)
                print(f"[Town] -> {TOWNS[town_i]} at episode {episodes} "
                      f"(server restarted in {took:.0f}s)", flush=True)
            obs, info = env.reset()
            vid.start(episodes)
            hidden = agent.reset_hidden(1)
            goal_blk = pxr.goal_block(info["debit_goal"])   # the buffer's episode goal
            row = row_of(obs, info)
            mode = int(round(float(np.asarray(obs["mode"]).ravel()[0])))
            ep_ret, ep_len = 0.0, 0

        if (step > LEARNING_STARTS and step % UPDATE_EVERY == 0
                and buf.sampleable >= CHUNK_T + BURN_IN + 1):
            for _ in range(UPDATES_PER):
                chunk = buf.sample_chunks(CHUNK_B, CHUNK_T + BURN_IN, BURN_IN)
                metrics = agent.update(chunk)
                if il_hook is not None:
                    il_hook()

        if step % CKPT_EVERY == 0:
            d = os.path.join(OUT, f"step_{step}")
            os.makedirs(d, exist_ok=True)
            agent.save(d)
            print(f"[Ckpt] {d}", flush=True)

    d = os.path.join(OUT, f"step_{MAX_STEPS}")
    os.makedirs(d, exist_ok=True)
    agent.save(d)
    env.close()
    server.stop()
    print(f"[Done] {MAX_STEPS} steps, {episodes} episodes in "
          f"{(time.time() - t_start) / 3600:.2f} h -> {d}", flush=True)


if __name__ == "__main__":
    main()

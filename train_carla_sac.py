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
EVAL_EPISODES = env_int("EVAL_EPISODES", 40)
EVAL_SEED = env_int("EVAL_SEED", 10_000)
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


def run_eval(env, agent, pxr, server, out_root):
    """EVAL_EPISODES goal episodes, deterministic actions, faults terminal,
    every episode recorded. Writes eval.json next to the videos."""
    import json
    from torch_rl.episode_video import EpisodeVideo

    tag = os.path.basename(os.path.normpath(EVAL_FROM))
    out = os.path.join(out_root, f"eval_{tag}")
    vid = EpisodeVideo(os.path.join(out, "videos"), every=1, max_frames=1200,
                       keep=max(EVAL_EPISODES, 1))
    box = env.action_space
    print(f"=== EVAL {EVAL_FROM} | {EVAL_EPISODES} goal episodes, seeds "
          f"{EVAL_SEED}..{EVAL_SEED + EVAL_EPISODES - 1}, DETERMINISTIC actions, "
          f"faults END the episode (crossing/off-road/collision) | town "
          f"{','.join(TOWNS)} -> {out}", flush=True)
    rows = []
    for k in range(EVAL_EPISODES):
        obs, info = env.reset(seed=EVAL_SEED + k)
        mode = int(round(float(np.asarray(obs["mode"]).ravel()[0])))
        if mode != 1:
            raise RuntimeError("eval episode is not in GOAL mode: EVAL_FORCE_MODE "
                               "must be the goal mode")
        hidden = agent.reset_hidden(1)
        goal_blk = pxr.goal_block(info["debit_goal"])
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
        while end is None:
            row = pxr.mono_row(info["debit_cur"], obs["mode"], obs["binoc"])
            action, hidden = agent.act(row, hidden, deterministic=True, goal=goal_blk)
            a_env = to_env(np.asarray(action, np.float32), box)
            vid.add(info["debit_cur"], info["debit_goal"], t, mode, 0.0, a_env, ret)
            obs, r, term, trunc, info = env.step(a_env)
            ret += float(r)
            t += 1
            pos = np.asarray(info.get("hero_pos", prev_pos), np.float64)
            moved = float(np.linalg.norm(pos - prev_pos))
            path_m += moved
            prev_pos = pos
            if t == 1 and info.get("route_remaining_m") is not None:
                l_route = float(info["route_remaining_m"]) + moved
            if info.get("is_success") or info.get("goal_end") == "reached":
                end = "reached"
            else:
                end = _fault(info)
            if end is None and (term or trunc):
                end = str(info.get("goal_end", "timeout"))
        path = vid.save(end)
        ell = l_route if l_route is not None else l_straight
        reached = end == "reached"
        spl_i = (float(reached) * ell / max(path_m, ell)
                 if (ell is not None and ell > 0) else float("nan"))
        rows.append({"episode": k, "seed": EVAL_SEED + k, "end": end,
                     "steps": t, "return": round(ret, 2), "reached": reached,
                     "shortest_m": None if ell is None else round(ell, 2),
                     "shortest_src": "route" if l_route is not None else "straight",
                     "straight_m": None if l_straight is None else round(l_straight, 2),
                     "path_m": round(path_m, 2), "spl": spl_i, "video": path})
        sr = np.mean([x["reached"] for x in rows])
        spl = np.nanmean([x["spl"] for x in rows])
        print(f"[Eval {k + 1:3d}/{EVAL_EPISODES}] seed={EVAL_SEED + k} end={end} "
              f"steps={t} shortest={ell if ell is None else round(ell, 1)} m "
              f"driven={path_m:.1f} m return={ret:.1f} | SR {sr:.3f} SPL {spl:.3f}",
              flush=True)

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
    n_route = sum(1 for x in rows if x["shortest_src"] == "route")
    summary = {"checkpoint": EVAL_FROM, "episodes": n, "success_rate": sr,
               "spl": spl, "spl_shortest_from_route": n_route,
               "wilson95": [round(centre - half, 3), round(centre + half, 3)],
               "ends": ends, "fault_ends": "crossing/off_road/collision",
               "deterministic": True, "towns": TOWNS, "seed0": EVAL_SEED,
               "episodes_detail": rows}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[EvalDone] SR={sr:.3f} (95% CI {summary['wilson95'][0]:.2f}-"
          f"{summary['wilson95'][1]:.2f}, n={n}) SPL={spl:.3f} "
          f"(shortest path from the route planner in {n_route}/{n}, straight "
          f"line otherwise) ends={ends} -> {out}", flush=True)
    env.close()
    server.stop()


def main():
    import torch

    from torch_rl import pixel_rows as pxr
    from torch_rl.carla_env_build import build_env
    from torch_rl.carla_server import CarlaServer
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
    server = CarlaServer(PORT, quality=os.environ.get("CARLA_QUALITY", "Epic"),
                         log_dir=os.path.join(OUT, "carla"))
    server.start()
    print(f"  carla: {server.describe()} ready in "
          f"{server.wait_ready():.0f}s", flush=True)
    env = build_env(TOWNS[town_i], seed=SEED, encoder=enc, port=PORT,
                    state_keys=(), eval_mode=bool(EVAL_FROM))
    obs, info = env.reset(seed=SEED)
    # THE STATE BLOCK IS THE MODE BIT AND THE BINOC, NOTHING ELSE (user
    # 2026-10-14: "get rid of all state vectors except mode bit"). No steer, no
    # throttle, no speed: the actor used to receive its own last steer and
    # throttle, a feedback path a policy can latch onto, and the run that
    # collapsed sat at full left lock for 800k steps.
    binoc_dim = int(np.shape(obs["binoc"])[0])
    state_dim = pxr.mono_state_dim(binoc_dim)
    row_dim = pxr.mono_row_dim(binoc_dim)

    cfg = GatedRecurrentSACConfig(
        # THE BINOCULAR + MONOCULAR STACK. The trainable visual path is a
        # half-width ResNet-18 from scratch over the 128 px frame
        # (mono_encoder="resnet"); the FROZEN DEBiT binocular vector reaches the
        # GRU through core_state_cols, which is what the peer's own vc3 layout
        # does. goal_tokens is OFF: that path stacks a goal IMAGE onto the
        # frame, and here the goal arrives already encoded.
        mono_encoder="resnet", embed_dim=3, grid=pxr.MONO_IMG,
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
        target_entropy=env_float("TARGET_ENTROPY", -2.0))
    agent = GatedRecurrentSAC(cfg)
    # DEBiT'''S PRETRAINED MONOCULAR TRUNK (user 2026-10-14, "I want debit Mono +
    # binocular setup"). Loaded AFTER the learner builds the module and copied
    # into the target as well: the learner makes its frozen target copy at
    # construction, so loading only into the online branch would leave the
    # target encoding with random weights.
    mono_report = load_debit_mono(agent.compress, TINY)
    agent.compress_targ.load_state_dict(agent.compress.state_dict())
    if EVAL_FROM:
        agent.load(EVAL_FROM)          # strict: every module, name and shape
        return run_eval(env, agent, pxr, server, OUT)
    buf = DeviceEpisodeBuffer(row_dim, 2, 1000 if EVAL_FROM else CAPACITY, DEVICE, seed=SEED,
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
          f"incremental={os.environ.get('INCREMENTAL_ACTIONS')}", flush=True)
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
        return pxr.mono_row(i["debit_cur"], o["mode"], o["binoc"])

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
                env = build_env(TOWNS[town_i], seed=SEED + episodes, port=PORT,
                                encoder=enc, state_keys=())
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

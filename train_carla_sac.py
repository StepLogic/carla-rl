#!/usr/bin/env python3
"""The torch recurrent-SAC training loop for the CARLA image-goal arm.

User 2026-10-13: move the recurrent SAC policy to the cluster; pixels in the
replay; proceed. This is the loop that drives everything built for it:

  env         torch_rl/carla_env_build.build_env -- v52's wrapper chain with
              DebitGoalWrapper where VAEGoalWrapper stood, and NO encoder passed,
              because the learner encodes (see THE ENCODER IS SHARED below)
  perception  DEBiT-tiny with AdaptFormer adapters in the ENCODER blocks only
  replay      DeviceEpisodeBuffer of pixel rows, goal frame once per episode
  learner     GatedRecurrentSAC: 10 critics per head, REDQ min-over-2, no TQC

THE ENCODER IS SHARED, AND THE ENV DOES NOT USE IT. DebitGoalWrapper can encode
the binocular vector into the observation, and this loop passes encoder=None so
it does not: the policy's own act() encodes the same frame pair through the same
adapters a moment later, and encoding twice would cost 7 ms a step for a vector
nothing reads. The wrapper still publishes both frames as pixels, which is what
the replay stores and what the learner re-encodes with gradient.

WHAT THE ROW HOLDS: [current frame 112x112x3 | previous action | state | mode].
The goal frame rides the buffer's per-episode goal field. The state block ends
with the mode bit at cfg.mode_index, and the critic column views are empty --
with goal_tokens the goal is an image the GRU has already folded into the hidden
state, so no state column carries it.

THE UPDATE CADENCE IS A MEASUREMENT, NOT A GUESS. One pixel update costs 209 ms
at B=8 T=40 on an L40 while the env runs at ~48 steps/s, so the two cannot both
run flat out on one GPU. UPDATE_EVERY defaults to 20 env steps: ~2.4 updates/s,
about half the GPU for the learner, and each update still learns from 320
transitions, so every new transition is replayed ~16 times.
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
TOWN = os.environ.get("ENV_TOWNS", "Town05").split(",")[0]
PORT = env_int("CARLA_PORT", 2000)
SEED = env_int("SEED", 0)
DEVICE = os.environ.get("TORCH_DEVICE", "cuda")
TINY = os.environ.get("DEBIT_WEIGHTS",
                      os.path.join(ROOT, "dependencies", "debit_weights",
                                   "debit_tiny.pth"))
OUT = os.path.join(os.environ.get("CKPT_DIR",
                                  os.path.join(ROOT, "checkpoints")), RUN_NAME)


def main():
    import torch

    from torch_rl import debit_adapters as da
    from torch_rl import pixel_rows as pxr
    from torch_rl.carla_env_build import build_env
    from torch_rl.debit_encoder import DebitTinyEncoder
    from torch_rl.debit_visual import IMG
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.episode_video import EpisodeVideo
    from torch_rl.gated_recurrent_sac import (GatedRecurrentSAC,
                                              GatedRecurrentSACConfig)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA: this arm is GPU-resident, so this is fatal")
    os.makedirs(OUT, exist_ok=True)

    enc = DebitTinyEncoder(TINY, device=DEVICE, half=False)
    n_blocks = da.add_encoder_adapters(
        enc.net, bottleneck=env_int("ADAPTER_BOTTLENECK", 64),
        scale=env_float("ADAPTER_SCALE", 0.1))

    # NO ENCODER FOR THE WRAPPER: see the module docstring. It still publishes
    # the frames, which is all this loop takes from it.
    env = build_env(TOWN, seed=SEED, encoder=None, port=PORT)
    obs, info = env.reset(seed=SEED)
    # THE STATE BLOCK IS obs["vector"] + the mode bit, not obs["state"]: the
    # env's "state" key holds its own 5400-d MobileNet features, which this arm
    # does not use. The pilot run built a 43,035-wide row out of them before
    # failing on the missing mode key.
    state_dim = int(np.shape(obs["vector"])[0]) + 1
    row_dim = pxr.row_dim(state_dim)

    cfg = GatedRecurrentSACConfig(
        embed_dim=3, grid=IMG, goal_tokens=True, mono_encoder="debit",
        debit_encoder=enc, state_dim=state_dim, mode_index=state_dim - 1,
        # ONE CRITIC FOR BOTH REWARD MODES (user 2026-10-13), and its view is
        # the MODE BIT: the two modes differ only in what the env pays, so
        # without that column the critic would predict two different returns
        # from otherwise identical inputs and learn their average. There is no
        # goal column besides it -- with goal_tokens the goal is an image the
        # GRU has already folded into the hidden state.
        one_critic=os.environ.get("ONE_CRITIC", "1") == "1",
        goal_cols=(state_dim - 1,), explore_cols=(),
        hidden=env_int("HIDDEN", 512),
        visual_dim=env_int("VISUAL_DIM", 512), mlp=(256, 256), burn_in=BURN_IN,
        device=DEVICE, compile=os.environ.get("TORCH_COMPILE", "0") == "1",
        seed=SEED, n_critics=env_int("N_CRITICS", 10),
        subset_size=env_int("SUBSET_SIZE", 2),
        discount=env_float("DISCOUNT", 0.99),
        # discount_e and the explore value bounds are inert under one_critic
        # (nothing trains a second head); kept so a two-critic comparison run is
        # one flag away and differs in nothing else.
        discount_e=env_float("EXPLORE_DISCOUNT", 0.99),
        explore_actor_warmup=env_int("EXPLORE_ACTOR_WARMUP", 5000),
        value_floor=env_float("TQC_VALUE_FLOOR", -200.0),
        value_ceiling=env_float("TQC_VALUE_CEILING", 150.0),
        value_floor_e=env_float("EXPLORE_VALUE_FLOOR", -200.0),
        value_ceiling_e=env_float("EXPLORE_VALUE_CEILING", 150.0))
    agent = GatedRecurrentSAC(cfg)
    buf = DeviceEpisodeBuffer(row_dim, 2, CAPACITY, DEVICE, seed=SEED,
                              goal_dim=pxr.PIX,
                              goal_batch_frac=env_float("GOAL_BATCH_FRAC", 0.5))

    # PREFLIGHT, printed before a single step (user rule, after four restarts):
    # the label, the checkpoint directory, the town, the seeds and the action
    # mode, so a run's identity is in its own log rather than inferred later.
    print(f"=== [{RUN_NAME}] preflight ===", flush=True)
    print(f"  town={TOWN} port={PORT} seed={SEED} device={DEVICE}", flush=True)
    print(f"  {agent.describe()}", flush=True)
    print(f"  visual: {agent.compress.describe()}", flush=True)
    print(f"  adapters: {da.describe_adapters(enc.net)} on {n_blocks} blocks", flush=True)
    print(f"  weights: {enc.describe()}", flush=True)
    print(f"  row={row_dim} (frame {pxr.PIX} + prev action 2 + state {state_dim})"
          f" | replay {CAPACITY} rows ~ {CAPACITY * row_dim * 2 / 2 ** 30:.1f} GiB fp16",
          flush=True)
    print(f"  update: every {UPDATE_EVERY} steps x{UPDATES_PER}, chunk "
          f"B={CHUNK_B} T={CHUNK_T} burn_in={BURN_IN}, starts at {LEARNING_STARTS}",
          flush=True)
    print(f"  action box: steer +/-{os.environ.get('MAX_STEER')} "
          f"throttle 0-{os.environ.get('MAX_THROTTLE')} "
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

    def row_of(o, i, prev_a):
        """The replay row for the step whose observation is `o` and whose info
        is `i`. The mode bit is appended to the state block, so the learner's
        mode_index lands on it."""
        st = np.concatenate([np.asarray(o["vector"], np.float32).ravel(),
                             np.asarray(o["mode"], np.float32).ravel()])
        return pxr.pixel_row(i["debit_cur"], prev_a, st)

    hidden = agent.reset_hidden(1)
    goal_blk = pxr.goal_block(info["debit_goal"])
    prev_a = np.zeros(2, np.float32)
    row = row_of(obs, info, prev_a)
    mode = int(round(float(np.asarray(obs["mode"]).ravel()[0])))

    ep_ret, ep_len, episodes = 0.0, 0, 0
    vid.start(0)
    recent = {"goal": [], "explore": [], "policy": [], "ends": {}}
    t_start = time.time()
    metrics = {}

    for step in range(1, MAX_STEPS + 1):
        if step <= LEARNING_STARTS:
            action = env.action_space.sample().astype(np.float32)
        else:
            action, hidden = agent.act(row, hidden, goal=goal_blk)
            action = np.asarray(action, np.float32)

        nobs, reward, terminated, truncated, ninfo = env.step(action)
        nrow = row_of(nobs, ninfo, action)
        buf.add(row, action, float(reward), bool(terminated), bool(truncated),
                nrow, mode, goal=goal_blk)
        # The frames the policy just acted on, beside the goal it was given.
        vid.add(info["debit_cur"], info["debit_goal"], ep_len, mode,
                float(reward), action, ep_ret)
        ep_ret += float(reward)
        ep_len += 1
        obs, info, row, prev_a = nobs, ninfo, nrow, action

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
                recent["ends"] = {}

            obs, info = env.reset()
            vid.start(episodes)
            hidden = agent.reset_hidden(1)
            goal_blk = pxr.goal_block(info["debit_goal"])
            prev_a = np.zeros(2, np.float32)
            row = row_of(obs, info, prev_a)
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
    print(f"[Done] {MAX_STEPS} steps, {episodes} episodes in "
          f"{(time.time() - t_start) / 3600:.2f} h -> {d}", flush=True)


if __name__ == "__main__":
    main()

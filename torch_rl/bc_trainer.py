"""Behaviour cloning of the recurrent SAC learner (imitation-first plan, Task 6).

M2 of docs/superpowers/specs/2026-11-04-imitation-first-image-goal-design.md:
clone CARLA's autopilot (the recorded episodes of torch_rl/expert_record.py)
into the very network SAC later fine-tunes -- the trainable monocular encoder,
the GRU and the actor -- so the fine-tune starts from a policy that already
keeps its lane instead of one that fails every 12-32 m.

THE LOSS is the pre-tanh Gaussian negative log-likelihood of the expert's
action. The actor is a tanh-squashed Gaussian; its density is evaluated where it
is a plain Gaussian, at u = atanh(a), with the label clipped to +-0.995 so the
expert's full-lock and full-brake actions (a = +-1, u = +-inf) stay finite. The
row alignment is exactly the SAC update's (_update_impl): core states s_t of the
trained window paired with action t, burn-in rows giving the hidden state only.
Gradients reach the encoder and the GRU, not just the actor head: learning to
see the road is the point.

Rows the buffer marks invalid (padding past an episode's end) count for
nothing, even if their action is garbage.
"""
import json
import os
import time

import numpy as np
import torch

from torch_rl import pixel_rows as pxr
from torch_rl.expert_data import episode_paths, fill_buffer, load_episode, split_by_episode
from torch_rl.expert_record import assert_recordable_town

TRAINING_TOWNS = ("Town01", "Town04", "Town05", "Town10HD")
LOG_EVERY = 250
VAL_CHUNKS = 8


def _env_int(name, default):
    return int(os.environ.get(name, default))


def bc_modules(agent):
    """The online modules cloning trains: encoder, action embedding (when the
    config has one), GRU and actor. Never a critic, a target or a temperature."""
    mods = [agent.compress]
    if agent.act_embed is not None:
        mods.append(agent.act_embed)
    return mods + [agent.core, agent.actor]


def bc_loss(agent, chunk, clip=0.995):
    """(loss, stats) for one sampled chunk (the buffer's _CHUNK_KEYS). The loss
    is the valid-row mean of the pre-tanh Gaussian NLL, summed over action
    dimensions; stats are floats: bc_nll (the loss), bc_mae (mean |tanh(mean) -
    label|, the deterministic action's error) and bc_log_std."""
    cfg, dev = agent.cfg, agent.device
    obs, keep, actions, valid = (chunk[k].to(device=dev, dtype=torch.float32)
                                 for k in ("obs", "keep", "actions", "valid"))
    goal = chunk["goal"].to(device=dev, dtype=torch.float32) if cfg.goal_tokens else None
    K = int(cfg.burn_in)
    B, T = valid.shape
    N, A = B * T, int(cfg.action_dim)

    s_all = agent._roll(agent.compress, agent.act_embed, agent.core, obs, keep, goal)
    s = s_all[:, :-1].reshape(N, cfg.hidden)
    st = agent._state(obs[:, K:-1]).reshape(N, cfg.state_dim)
    mean, log_std = agent._actor_forward(torch.cat([s, st], -1))
    a = actions.reshape(N, A)
    u = torch.atanh(a.clamp(-clip, clip))
    nll = (0.5 * ((u - mean) / log_std.exp()) ** 2 + log_std).sum(-1)
    v = valid.reshape(N) > 0
    n = v.sum().clamp_min(1)
    # where(), not a multiply: a padded row's NaN or inf must not leak in as NaN * 0.
    loss = torch.where(v, nll, torch.zeros_like(nll)).sum() / n
    with torch.no_grad():
        mae = torch.where(v, (torch.tanh(mean) - a).abs().mean(-1), torch.zeros_like(nll)).sum() / n
        ls = torch.where(v, log_std.mean(-1), torch.zeros_like(nll)).sum() / n
        vals = torch.stack([loss.detach(), mae, ls]).tolist()
    return loss, dict(zip(("bc_nll", "bc_mae", "bc_log_std"), vals))


def make_bc_optimizer(agent, lr=3e-4):
    """Adam over the cloned modules' trainable parameters only."""
    return torch.optim.Adam([p for m in bc_modules(agent) for p in m.parameters()
                             if p.requires_grad], lr=float(lr))


def sync_targets(agent):
    """Copy the online encoder, action embedding and GRU into their targets.
    Cloning never touches the targets, so a checkpoint saved without this would
    hand the fine-tune's critic a target encoder that is still the init."""
    with torch.no_grad():
        agent.compress_targ.load_state_dict(agent.compress.state_dict())
        if agent.act_embed is not None:
            agent.act_embed_targ.load_state_dict(agent.act_embed.state_dict())
        agent.core_targ.load_state_dict(agent.core.state_dict())


def augment_frames(obs, gen=None, bright=0.4, contrast=0.4, color=0.1):
    """Photometric augmentation of the camera frame in sampled rows (B, L, D).

    User 2026-11-05: augmentation for the ResNet to help Town03, whose faults
    sit in a dark lamp-lit tunnel and on glare-washed roads -- lighting the
    training towns never show. Per CHUNK (one lighting over a stretch of
    driving, as the GRU sees it): per-colour gain 1 +- color, contrast about
    each frame's mean 1 +- contrast, brightness 1 +- bright; clipped to [0, 1]
    (the ResNet refuses anything else). Only the frame block changes -- mode,
    speed and the frozen binoc stay as recorded. Works in either frame layout
    (pixel_rows.MONO_CHW). Returns a new tensor; `obs` is not modified."""
    P, S = pxr.MONO_PIX, pxr.MONO_IMG
    B, L = obs.shape[0], obs.shape[1]

    def u(*shape):
        return (torch.rand(*shape, generator=gen) * 2.0 - 1.0).to(obs.device, obs.dtype)

    pix = obs[..., :P]
    x = (pix.reshape(B, L, 3, S * S) if pxr.MONO_CHW
         else pix.reshape(B, L, S * S, 3).transpose(-1, -2))          # (B, L, 3, HW)
    x = x * (1.0 + color * u(B, 1, 3, 1))
    m = x.mean(dim=(2, 3), keepdim=True)
    x = (x - m) * (1.0 + contrast * u(B, 1, 1, 1)) + m
    x = (x * (1.0 + bright * u(B, 1, 1, 1))).clamp(0.0, 1.0)
    back = x.reshape(B, L, P) if pxr.MONO_CHW else x.transpose(-1, -2).reshape(B, L, P)
    out = obs.clone()
    out[..., :P] = back
    return out


def bc_step(agent, opt, chunk, weight=1.0, aug=None):
    """One cloning update; returns bc_loss's stats (before the step). `aug`:
    augment_frames keyword arguments (gen, bright, contrast, color) to apply
    to this batch's frames, or None."""
    if aug is not None:
        chunk = dict(chunk, obs=augment_frames(chunk["obs"].to(agent.device, torch.float32), **aug))
    loss, stats = bc_loss(agent, chunk)
    opt.zero_grad(set_to_none=True)
    (loss * float(weight)).backward()
    opt.step()
    return stats


def bc_weight(n_updates, w0, decay_updates):
    """The fine-tune's cloning weight: w0 at the first update, falling linearly
    to 0 at decay_updates and staying there."""
    if decay_updates <= 0:
        return 0.0
    return float(w0) * max(0.0, 1.0 - float(n_updates) / float(decay_updates))


def il_bc_hook(agent, opt, expert_buf, n_chunks, length, w0, decay_updates,
               on_weight=None, aug=None, freeze_updates=0, log=None):
    """The fine-tune's per-update cloning step (imitation-first plan, Task 10):
    a callable the online loop calls after every agent.update. Each call takes
    one bc_step on an expert-label chunk (length + burn-in rows) at weight
    bc_weight(calls so far); once the weight reaches 0 it does nothing.
    `on_weight` (optional) receives each call's weight, for the log.

    ONLINE FREEZE (2026-11-05): with `freeze_updates` = N > 0 the clone stays
    frozen (agent.set_warmup(True)) for the first N online updates -- the
    critic learns from the clone's own rollouts while nothing the clone learned
    moves -- and those calls take no cloning step. The N-th call unfreezes it,
    and the cloning schedule starts from w0 at the call after. Unfreezing on the
    first live update collapsed the fine-tune (job 463440: policy= 0.76 -> 0.10
    within ~190 updates, the cloning loss at full weight)."""
    count = [0]
    K = int(agent.cfg.burn_in)
    freeze = int(freeze_updates)
    if freeze > 0:
        agent.set_warmup(True)

    def hook():
        n = count[0]
        count[0] += 1
        if n < freeze:
            if n == freeze - 1:
                agent.set_warmup(False)          # the next update trains everything
                if log is not None:
                    log(f"[IL] online freeze over after {freeze} updates: encoder, GRU "
                        f"and actor train from the next update; cloning weight {w0:g} "
                        f"fading over {int(decay_updates)} updates")
            return None
        w = bc_weight(n - freeze, w0, decay_updates)
        if on_weight is not None:
            on_weight(w)
        if w <= 0.0:
            return None
        return bc_step(agent, opt, expert_buf.sample_chunks(n_chunks, length + K, K),
                       weight=w, aug=aug)

    return hook


def il_finetune_prepare(agent, buf, root, towns, *, warmup_updates, chunk_b, chunk_t,
                        w0, decay_updates, bc_chunks, bc_len, aug=None, log=print,
                        online_freeze_updates=0, expert_replay=None):
    """Ready a clone for the RL fine-tune (imitation-first plan, Task 10), before
    the first env step:
      1. the recorded episodes fill the replay `buf` with their EXECUTED actions
         and rewards (off-policy data the critic can learn from at once), and a
         separate expert buffer with their LABELS (what the cloning loss copies);
      2. the critic warms up: `warmup_updates` updates with set_warmup(True), so
         it learns the clone's values while the clone itself stays put;
      3. returns il_bc_hook -- the per-update cloning step, weight w0 fading to 0
         over decay_updates -- for the online loop to call after every update.
    The cloning optimiser lives in the hook, never in agent.save."""
    for t in towns:
        assert_recordable_town(t)                 # never a held-out town
    paths = episode_paths(root, towns)
    if not paths:
        raise SystemExit(f"IL preload: no episodes under {root} for {towns}")
    # expert_replay (user 2026-11-05, mixed_replay.MixedReplay): the executed
    # expert steps go to a replay of their own and the agent's `buf` starts empty.
    replay = buf if expert_replay is None else expert_replay
    n_exec = fill_buffer(replay, paths, which="exec")
    act_dim = int(load_episode(paths[0])["act_label"].shape[1])
    expert_buf, n_lab = _buffer(paths, act_dim, agent.device, seed=1)
    log(f"[IL] preload {root}: {len(paths)} episodes -> replay +{n_exec} executed steps, "
        f"expert buffer {n_lab} labelled steps")
    K = int(agent.cfg.burn_in)
    t0 = time.time()
    agent.set_warmup(True)
    m = {}
    for _ in range(int(warmup_updates)):
        m = agent.update(replay.sample_chunks(chunk_b, chunk_t + K, K))
    agent.set_warmup(False)
    log(f"[IL] critic warm-up: {int(warmup_updates)} updates, encoder/GRU/actor frozen, "
        f"q_g {m.get('q_g', float('nan')):.2f} critic_loss_g "
        f"{m.get('critic_loss_g', float('nan')):.3f} ({time.time() - t0:.0f} s)")
    if online_freeze_updates:
        log(f"[IL] online freeze: the first {int(online_freeze_updates)} RL updates train "
            f"the critic only, on the clone's own rollouts")
    return il_bc_hook(agent, make_bc_optimizer(agent), expert_buf, bc_chunks, bc_len, w0,
                      decay_updates, aug=aug, freeze_updates=online_freeze_updates, log=log)


def expert_rows(root, towns):
    """Buffer rows the recorded episodes under `root` occupy (T + 1 each): the
    capacity a replay of their own needs (mixed_replay.MixedReplay)."""
    return sum(_episode_rows(p) for p in episode_paths(root, towns))


def _episode_rows(path):
    """Buffer rows an episode occupies (T transitions + 1), read from the one
    small array so the frames are not decompressed twice."""
    with np.load(path) as z:
        return int(z["act_exec"].shape[0]) + 1


def _buffer(paths, act_dim, device, seed):
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    cap = max(1, sum(_episode_rows(p) for p in paths))
    # goal_dim 0: cloning reads the goal through the frozen binocular features
    # in the row, never the stored goal image.
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(pxr.BINOC_DIM), act_dim, cap, device,
                              seed=seed, store_device=os.environ.get("REPLAY_DEVICE", "cuda"),
                              u8_cols=pxr.MONO_PIX)
    added = fill_buffer(buf, paths, which="label")
    if buf.size != added:
        raise RuntimeError(f"buffer evicted episodes: holds {buf.size} of {added} transitions")
    return buf, added


def run_bc_train(out_root, make_agent=None):
    """The BC_TRAIN=1 mode: clone the recorded expert, checkpointing to
    <out_root>/step_<n>/. `make_agent(binoc_dim, act_dim)` builds the learner;
    the trainer passes its build_agent (default: imported from train_carla_sac)."""
    root = os.environ.get("BC_DATA_ROOT", "")
    if not root:
        raise SystemExit("BC_TRAIN=1 needs BC_DATA_ROOT (the expert recording root)")
    towns = os.environ.get("BC_TOWNS", " ".join(TRAINING_TOWNS)).split()
    for t in towns:
        assert_recordable_town(t)                 # never clone a held-out town
    updates = _env_int("BC_UPDATES", 20000)
    n_chunks = _env_int("BC_CHUNKS", 32)
    length = _env_int("BC_LEN", 40)
    save_every = _env_int("BC_SAVE_EVERY", 2500)
    init_from = os.environ.get("BC_INIT_FROM", "")

    paths = episode_paths(root, towns)
    # BC_MAX_SEED: keep episodes whose seed (ep_<seed>.npz) is at most this --
    # the DAgger rounds write into the same root from seed 300000, so a round-0
    # clone sets 199999 to train on the expert's own episodes only.
    max_seed = os.environ.get("BC_MAX_SEED", "")
    if max_seed:
        paths = [p for p in paths
                 if int(os.path.basename(p)[3:-4]) <= int(max_seed)]
    train, val = split_by_episode(paths)
    if not train or not val:
        raise SystemExit(f"BC needs >= 2 episodes under {root} for {towns}: "
                         f"{len(train)} train, {len(val)} val")
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "bc_split.json"), "w") as fh:
        json.dump({"data_root": root, "towns": towns, "train": train, "val": val}, fh, indent=1)

    if make_agent is None:
        from train_carla_sac import build_agent as make_agent
    act_dim = int(load_episode(train[0])["act_label"].shape[1])
    agent = make_agent(pxr.BINOC_DIM, act_dim)
    if init_from:
        agent.load(init_from)
    K = int(agent.cfg.burn_in)
    t0 = time.time()
    train_buf, n_train = _buffer(train, act_dim, agent.device, seed=0)
    val_buf, n_val = _buffer(val, act_dim, agent.device, seed=1)
    # One fixed validation batch, so val_nll is comparable across steps.
    val_chunk = val_buf.sample_chunks(VAL_CHUNKS, length + K, K)
    print(f"[BC] data {root} towns={towns}: train {len(train)} episodes / {n_train} steps, "
          f"val {len(val)} / {n_val}; loaded in {time.time() - t0:.0f} s; "
          f"init={init_from or 'build_agent'}; {updates} updates x {n_chunks} chunks "
          f"x {length} (+{K} burn-in)", flush=True)

    opt = make_bc_optimizer(agent)
    # BC_AUG=1: photometric augmentation of every TRAINING batch's frames
    # (augment_frames); the fixed validation batch is never augmented.
    aug = None
    if os.environ.get("BC_AUG", "") == "1":
        aug = {"gen": torch.Generator().manual_seed(_env_int("SEED", 0)),
               "bright": float(os.environ.get("BC_AUG_BRIGHT", 0.4)),
               "contrast": float(os.environ.get("BC_AUG_CONTRAST", 0.4)),
               "color": float(os.environ.get("BC_AUG_COLOR", 0.1))}
        print(f"[BC] augmentation on: brightness +-{aug['bright']:g}, contrast "
              f"+-{aug['contrast']:g}, colour gain +-{aug['color']:g} (per chunk)", flush=True)
    window, t0 = [], time.time()
    for step in range(1, updates + 1):
        window.append(bc_step(agent, opt, train_buf.sample_chunks(n_chunks, length + K, K),
                              aug=aug))
        if step % LOG_EVERY == 0 or step == updates:
            with torch.no_grad():
                _, vs = bc_loss(agent, val_chunk)
            tr = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
            print(f"[BC] step={step} train_nll={tr['bc_nll']:.4f} val_nll={vs['bc_nll']:.4f} "
                  f"mae={vs['bc_mae']:.4f} train_mae={tr['bc_mae']:.4f} "
                  f"log_std={tr['bc_log_std']:.3f} ({len(window) / (time.time() - t0):.2f} upd/s)",
                  flush=True)
            window, t0 = [], time.time()
        if step % save_every == 0 or step == updates:
            sync_targets(agent)
            print(f"[BC] saved {agent.save(os.path.join(out_root, f'step_{step}'))}", flush=True)

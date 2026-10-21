"""The SAC trainer's loop over N CARLA envs stepped in parallel.

User 2026-10-21: train on Town05 AND Town01 at once, "do not switch towns",
"use existing code". The existing code is vae_nav.vec_env.ThreadedVecEnv,
which ran the two-town JAX runs (v7, v25): one CARLA server per town, the N
blocking env steps overlapped on threads, per-env auto-reset. This module is
the SAC side of it, kept out of train_carla_sac.main so it can be tested with
fake envs and a fake agent.

Per env it keeps exactly what the single-env loop keeps -- the row the policy
acts on, its episode's goal block and mode, the GRU hidden column, the video --
and hands the replay one episode per env via env_id, so transitions from two
towns are never spliced together. A finished env's NEXT row is built from the
terminal observation (vec_env's final_infos[i]["final_obs"]), not the auto-reset
one: a truncated transition bootstraps from it.

Steps are counted in TRANSITIONS (N per vec step), so MAX_STEPS, the update
cadence and the checkpoint interval mean what they meant with one env.
"""
import os
import time

import numpy as np


def _mode_of(obs):
    return int(round(float(np.asarray(obs["mode"]).ravel()[0])))


def run_parallel(venv, agent, buf, row_of, box, *, to_env, to_actor, goal_block,
                 max_steps, learning_starts, update_every, updates_per,
                 chunk_b, chunk_t, burn_in, ckpt_every, out_dir, log_every,
                 vids=None, her=None, log=print, on_episode=None):
    """Train until `max_steps` transitions. Returns (steps, episodes)."""
    n = venv.n
    names = list(venv.names)
    obs_s, infos = venv.reset()
    obs = [venv.unstack(obs_s, i) for i in range(n)]
    infos = list(infos)
    rows = [row_of(obs[i], infos[i]) for i in range(n)]
    goals = [goal_block(infos[i]["debit_goal"]) for i in range(n)]
    modes = [_mode_of(obs[i]) for i in range(n)]
    hidden = agent.reset_hidden(n)
    ep_ret = [0.0] * n
    ep_len = [0] * n
    if vids:
        for i in range(n):
            vids[i].start(0)
    if her is not None:
        for i in range(n):
            her.begin(i, infos[i], modes[i])

    recent = {"goal": [], "explore": [], "policy": [], "ends": {}}
    roll = {"n": 0, "lock": 0, "idle_thr": 0}
    metrics = {}
    step, episodes = 0, 0
    next_update, next_ckpt = update_every, ckpt_every
    t0 = time.time()

    while step < max_steps:
        if step < learning_starts:
            actions = np.stack([to_actor(box.sample().astype(np.float32), box)
                                for _ in range(n)]).astype(np.float32)
        else:
            actions, hidden = agent.act_batch(np.stack(rows), hidden,
                                              goal=np.stack(goals))
            actions = np.asarray(actions, np.float32)
        env_actions = np.stack([to_env(a, box) for a in actions])
        for a, ea in zip(actions, env_actions):
            roll["n"] += 1
            roll["lock"] += int(abs(float(a[0])) > 0.95)
            roll["idle_thr"] += int(float(ea[1]) < 0.05)

        obs_s, rewards, dones, step_infos, final_infos = venv.step(env_actions)
        for i in range(n):
            inf = step_infos[i]
            done = bool(dones[i])
            if done:
                fin = final_infos[i]
                nrow = row_of(fin["final_obs"], fin)
            else:
                nrow = row_of(venv.unstack(obs_s, i), inf)
            term = bool(inf.get("terminated", False))
            trunc = bool(inf.get("truncated", False)) or (done and not term)
            if vids:
                vids[i].add(infos[i]["debit_cur"], infos[i]["debit_goal"],
                            ep_len[i], modes[i], float(rewards[i]),
                            env_actions[i], ep_ret[i])
            buf.add(rows[i], actions[i], float(rewards[i]), term, trunc, nrow,
                    modes[i], env_id=i, goal=goals[i])
            if her is not None:
                her.record(i, rows[i], actions[i], float(rewards[i]), term,
                           trunc, nrow, infos[i], inf)
            ep_ret[i] += float(rewards[i])
            ep_len[i] += 1

            if not done:
                rows[i], infos[i] = nrow, inf
                continue

            # ---- episode end for env i -----------------------------------
            episodes += 1
            end = str(inf.get("goal_end", inf.get("explore_end", "?")))
            if vids:
                saved = vids[i].save(f"{names[i]}_{end}")
                if saved:
                    log(f"[Video] {saved}")
            if her is not None:
                her.flush(i, terminated=term, truncated=trunc)
            arrived = end == "reached"
            key = f"{names[i]}:{end}"
            recent["ends"][key] = recent["ends"].get(key, 0) + 1
            recent["policy"].append(float(arrived))
            recent["goal" if modes[i] == 1 else "explore"].append(
                dict(len=ep_len[i], arrived=float(arrived)))
            for k in ("goal", "explore", "policy"):
                recent[k] = recent[k][-100:]
            if on_episode is not None:
                on_episode(i, end, ep_ret[i], ep_len[i])

            o_new = venv.unstack(obs_s, i)
            rinf = venv.reset_infos[i] if venv.reset_infos[i] is not None else {}
            infos[i] = rinf
            rows[i] = row_of(o_new, rinf)
            goals[i] = goal_block(rinf["debit_goal"])
            modes[i] = _mode_of(o_new)
            hidden = agent.zero_hidden_columns(hidden, [i])
            ep_ret[i], ep_len[i] = 0.0, 0
            if vids:
                vids[i].start(episodes)
            if her is not None:
                her.begin(i, rinf, modes[i])

            if episodes % log_every == 0:
                g, e = recent["goal"], recent["explore"]
                pol = float(np.mean(recent["policy"]))
                sr = float(np.mean([x["arrived"] for x in g])) if g else float("nan")
                per_h = step / max(1e-9, time.time() - t0) * 3600
                _m = lambda k: float(metrics.get(k, float("nan")))
                log(f"[Ep {episodes:5d}] step={step:8d} "
                    f"policy={pol:.3f}(n={len(recent['policy'])}) "
                    f"goal={sr:.3f}(n={len(g)}) "
                    f"goal_len={np.mean([x['len'] for x in g]) if g else float('nan'):.0f} "
                    f"explore_len={np.mean([x['len'] for x in e]) if e else float('nan'):.0f} "
                    f"ends={recent['ends']} buffer={buf.size} {per_h / 1000:.0f}k steps/h "
                    f"| {_m('critic_loss_g'):.3f} q_g={_m('q_g'):.2f} "
                    f"q_e={_m('q_e'):.2f} alpha_g={_m('alpha_g'):.4f}"
                    + (f" | her={her.describe_counts()}" if her is not None else ""))
                log(f"[Diag] step={step} -logpi={_m('neg_logp_g'):.2f} "
                    f"log_std={_m('log_std_g'):.2f} "
                    f"|u| steer={_m('mean_abs_u_steer'):.2f} "
                    f"thr={_m('mean_abs_u_throttle'):.2f} "
                    f"sat(replay) steer={_m('sat_steer'):.2f} "
                    f"thr={_m('sat_throttle'):.2f} | rollout: "
                    f"steer_lock={roll['lock'] / max(1, roll['n']):.2f} "
                    f"throttle<0.05={roll['idle_thr'] / max(1, roll['n']):.2f}")
                roll = {"n": 0, "lock": 0, "idle_thr": 0}
                recent["ends"] = {}

        step += n
        # Same update-to-data ratio as one env: UPDATES_PER updates for every
        # UPDATE_EVERY transitions collected, whichever env they came from.
        while step >= next_update:
            if (step > learning_starts
                    and buf.sampleable >= chunk_t + burn_in + 1):
                for _ in range(updates_per):
                    chunk = buf.sample_chunks(chunk_b, chunk_t + burn_in, burn_in)
                    metrics = agent.update(chunk)
            next_update += update_every
        while step >= next_ckpt:
            d = os.path.join(out_dir, f"step_{next_ckpt}")
            os.makedirs(d, exist_ok=True)
            agent.save(d)
            log(f"[Ckpt] {d}")
            next_ckpt += ckpt_every

    d = os.path.join(out_dir, f"step_{max_steps}")
    os.makedirs(d, exist_ok=True)
    agent.save(d)
    return step, episodes

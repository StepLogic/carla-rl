#!/usr/bin/env python
"""Explore-mode trace WITH PIXELS, driven by the v13 policy.

USER 2026-09-26: "collect trace in exploration mode with v13 in town02 and
town03 and store all observations, store position of observations, store all
relevant data also images to enable dinov2 localisation."

WHY THIS EXISTS RATHER THAN eval_placement.collect_trace. That collector
already drives the explore arm, already thins by distance and already
implements the continuation rule, and this file REUSES all of it by import.
What it cannot do is keep the camera frame: it stores the encoder's latent and
nothing else, which is why capture_rgb_bank.py had to teleport back over every
pose afterwards to rebuild a bank for the DINOv2 localiser. Recording
info["rgb"] as the car drives makes that second pass unnecessary and, more
importantly, makes the frame the one the policy ACTUALLY saw at that pose
rather than one recaptured after a teleport (median teleport error was 0.2 m,
but the settle ticks change the vehicle's pitch and the traffic-light phase).

eval_placement.py is NOT edited: it is shared with the placement pipeline and
with other sessions' work, so the tested helpers are imported instead.

WHAT IS RECORDED, per kept sample (thinned to --candidate-spacing metres):
    rgb          (120,120,3) uint8 -- info["rgb"], i.e. experiment.obs_img,
                 the exact frame the perception stack encoded this step.
    position     (x, y) float64, and yaw in radians -- the TRUE pose, for
                 scoring a localiser against ground truth.
    obs_driven   the observation actually fed to the policy this step, mode
                 bit = EXPLORE. This is what a replay would need.
    obs_goalmode the same observation with the mode bit flipped to GOAL, which
                 is what the placement/SGM critic queries consume. Both are
                 kept because they are not interchangeable and deriving one
                 from the other later needs the wrapper's goal transform.
    action       the action taken FROM this state, reward the step returned.
    fields       wrapper._fields(info): hero_pos, hero_yaw, goal_pos, history.
    segment      episode index, so trace adjacency can be checked against
                 episode boundaries.
Per segment: the end cause, its step count and its start pose.

EXPLORE MODE IS FORCED (EVAL_FORCE_MODE=0) and the 100 m explore arrival is
DISABLED (EXPLORE_GOAL_DIST_M=0, user's choice), so an episode runs until the
explore arm's own terminal fires -- collision, off-road, or a painted crossing
outside a junction -- and the next one continues from the nearest drivable
waypoint to where it stopped. Random respawns sample the town as disconnected
islands; continuing walks it.

ACTIONS ARE SAMPLED, not taken at the distribution mode. With the continuation
rule a deterministic policy restarted at the same waypoint repeats the same
failure forever, which is a trace of one corner rather than of a town. The
sample uses the agent's own PRNG seeded at build_agent, so a fixed seed is
still reproducible.

RESUMABLE BY ARTEFACT: written after every episode, because the supervisor on
this machine kills background jobs with a spurious low-memory message.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "src"))


def _stack_obs(list_of_dicts):
    """[{k: arr}] -> {k: stacked}. Smaller and far faster to load than a list
    of 5000 dicts, and it makes the per-key widths visible in the artefact."""
    if not list_of_dicts:
        return {}
    keys = sorted(list_of_dicts[0].keys())
    return {k: np.stack([np.asarray(d[k], np.float32) for d in list_of_dicts])
            for k in keys}


def _unstack_obs(stacked):
    """Inverse of _stack_obs, for --resume. Written out rather than inlined
    because a zip-based reconstruction silently reorders keys."""
    if not stacked:
        return []
    keys = sorted(stacked)
    n = len(stacked[keys[0]])
    return [{k: np.asarray(stacked[k][i], np.float32) for k in keys}
            for i in range(n)]



# ---------------------------------------------------------------------------
# UNCERTAINTY TELEMETRY (user 2026-10-02: "record epistemic uncertain,
# aleoteric uncetaintty and policy entropy (analytical)").
#
# THE TQC CRITIC RETURNS (n_critics, batch, n_quantiles) -- the shape the
# learner's actor loss reduces with qs.mean(axis=(0, 2))
# (gated_tqc_learner._gated_actor_update). That gives the two uncertainties a
# clean, non-overlapping split:
#   EPISTEMIC  disagreement BETWEEN critics -> std over axis 0 of each critic's
#              own mean. Shrinks with data; high where the ensemble has not
#              seen this state. This is the quantity a SORB/SGM graph should
#              distrust an edge on.
#   ALEATORIC  spread WITHIN a critic's own return distribution -> std over
#              axis 2 (the quantiles), averaged over critics. Does NOT shrink
#              with data; it is the environment's own stochasticity.
# Both are recorded for BOTH heads: the explore critic (_critic_e, the one the
# actor follows in this mode) and the goal critic (_critic, which is what the
# placement/SGM queries consume via obs_goalmode).
#
# POLICY ENTROPY IS ONLY PARTLY ANALYTIC, AND THE SPLIT IS RECORDED HONESTLY.
# The policy is a tanh-squashed, affine-rescaled Normal. The BASE Normal's
# entropy is exact and closed form (sum over dims of 0.5*log(2*pi*e*sigma^2)),
# and that is `ent_gauss`. The tanh changes it by E[log|det J|], which has NO
# closed form -- so `ent_tanh_at_mean` evaluates that Jacobian at the
# distribution's mean action (one deterministic point, NOT an expectation) and
# `ent_analytic` is their sum. Treat ent_gauss as exact and the other two as
# the stated approximation. log_std is kept per dimension so any other
# estimator can be computed offline without re-running the car.
def _uncertainty(agent, obs_driven, obs_goalmode, action):
    """-> dict of float32 telemetry for one kept sample. Never raises."""
    import numpy as _np
    out = {}
    try:
        import jax, jax.numpy as jnp
        from vae_nav.lane_priv import strip_priv

        def _b(o):
            return jax.tree_util.tree_map(lambda x: _np.asarray(x)[None], o)

        a1 = _np.asarray(action, _np.float32)[None]
        for tag, head, o in (("e", "_critic_e", obs_driven),
                             ("g", "_critic", obs_goalmode)):
            c = getattr(agent, head, None)
            if c is None:
                continue
            qs = c.apply_fn({"params": c.params}, _b(strip_priv(o)), a1)
            qs = _np.asarray(qs)                      # (n_critics, 1, n_quant)
            if qs.ndim != 3:
                continue
            per_critic = qs.mean(axis=2)[:, 0]        # each critic's own mean
            out[f"q_{tag}"] = float(per_critic.mean())
            out[f"epistemic_{tag}"] = float(per_critic.std())
            out[f"aleatoric_{tag}"] = float(qs.std(axis=2)[:, 0].mean())
            out[f"n_critics_{tag}"] = int(qs.shape[0])
            out[f"n_quantiles_{tag}"] = int(qs.shape[2])

        act = getattr(agent, "_actor", None)
        if act is not None:
            # Exact closed-form Gaussian + rescale parts; tanh part by 1024-sample
            # MC (SE ~0.02 nats). See vae_nav/policy_entropy.py and its test.
            from vae_nav.policy_entropy import policy_entropy
            dist = act.apply_fn({"params": act.params}, _b(strip_priv(obs_driven)))
            out.update(policy_entropy(dist, n_mc=1024))
    except Exception as _e:                            # telemetry never fatal
        out["uncertainty_error"] = f"{type(_e).__name__}: {_e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/carla_gated_her_v13_entunit_vc1/"
                                     "checkpoint_450000.pkl")
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=5000,
                    help="KEPT samples to collect (user 2026-09-26: 5000/town). "
                         "Not simulator steps: thinning is by distance.")
    ap.add_argument("--candidate-spacing", type=float, default=2.0,
                    help="metres between kept samples; the placement default.")
    ap.add_argument("--collect-seed", type=int, default=0)
    ap.add_argument("--episode-steps", type=int, default=2000,
                    help="hard cap per episode, so one lucky run cannot spend "
                         "the whole budget in a single loop of the same block.")
    ap.add_argument("--action-mode", choices=("sample", "eval"), default="sample")
    ap.add_argument("--resume", type=int, default=1,
                    help="1 = continue an existing --out artefact.")
    a = ap.parse_args()

    # EVERY FLAG THAT THE CONFIG READS AT IMPORT MUST BE SET FIRST. This is the
    # repo's standing hazard: these modules read os.environ at module scope, so
    # setting them after the import is a silent no-op.
    os.environ["EVAL_FORCE_MODE"] = "0"          # explore half of the gated arm
    # The 100 m explore arrival would end an episode mid-street and hand the
    # continuation rule a boundary that has nothing to do with the road.
    os.environ["EXPLORE_GOAL_DIST_M"] = "0"
    # The env's own distance limit and time limit are collection artefacts here.
    os.environ.setdefault("MAX_DISTANCE_M", "1000000000")
    os.environ.setdefault("NO_TIMELIMIT", "1")
    # Zeroes the two discarded MobileNet vectors ONLY; info["rgb"] is still set
    # from experiment.obs_img, which is what this script records.
    os.environ.setdefault("ENV_SKIP_IMAGE_FEATURES", "1")

    import carla
    import train_carla_tqc as T
    from vae_nav.latent_source import from_env as latent_from_env, LatentSource
    from vae_nav.policy_io import load_agent_state
    from eval_placement import (_assert_town, clear_continue_origin,
                                collect_action, continued_spawn_ok,
                                find_attr_owner, find_wrapper, goal_mode,
                                set_continue_origin)

    latent = latent_from_env()
    env = T.make_env(a.town, seed=a.collect_seed, latent_source=latent,
                     eval_mode=True, port=a.port)
    _assert_town(env, a.town, "collect-explore")
    wrapper = find_wrapper(env, "goal_block")
    if wrapper is None:
        raise SystemExit("no VAEGoalWrapper in the chain: _fields is unavailable")

    agent = T.build_agent(env.observation_space, env.action_space, a.collect_seed)
    with open(a.ckpt, "rb") as fh:
        state = pickle.load(fh)
    # Refuse a perception mismatch before driving: a 3840-d VC-1 checkpoint read
    # under a 128-d autoencoder does not crash, it produces confident nonsense.
    if isinstance(state, dict) and "identity" in state:
        LatentSource.assert_matches(state["identity"], latent)
    load_agent_state(agent, state.get("agent", state))

    rgb_l, obs_d_l, obs_g_l, fields_l = [], [], [], []
    unc_l = []
    pos_l, yaw_l, seg_l, act_l, rew_l = [], [], [], [], []
    segments = []
    first_ep, last_end_pos = 0, None
    if a.resume and os.path.exists(a.out):
        with open(a.out, "rb") as fh:
            prev = pickle.load(fh)
        rgb_l = list(prev["rgb"])
        obs_d_l = _unstack_obs(prev["obs_driven"])
        obs_g_l = _unstack_obs(prev["obs_goalmode"])
        fields_l = list(prev["fields"])
        pos_l = [np.asarray(p, np.float64) for p in prev["positions"]]
        yaw_l = list(prev["yaws"])
        seg_l = list(prev["segments"])
        act_l = list(prev["actions"])
        rew_l = list(prev["rewards"])
        # UNCERTAINTY MUST BE RESTORED TOO or a resumed run misaligns every
        # row after the resume point (this script is resumable precisely
        # because the supervisor kills long jobs). Older artefacts have no
        # such key: pad with empty dicts so index i still means sample i.
        unc_l = (list(prev["uncertainty"]) if "uncertainty" in prev
                 else [{} for _ in rew_l])
        if len(unc_l) != len(rew_l):
            raise SystemExit(f"resume misalignment: {len(unc_l)} uncertainty "
                             f"rows vs {len(rew_l)} samples")
        segments = list(prev["segment_info"])
        first_ep = int(prev["episodes_done"])
        last_end_pos = (np.asarray(prev["last_end_pos"], np.float64)
                        if prev.get("last_end_pos") is not None else None)
        print(f"[Collect] resumed {a.out}: {len(pos_l)} samples, "
              f"{first_ep} episodes done", flush=True)

    def snapshot(eps_done):
        return {
            "rgb": np.asarray(rgb_l, np.uint8),
            "obs_driven": _stack_obs(obs_d_l),
            "obs_goalmode": _stack_obs(obs_g_l),
            "fields": fields_l,
            "positions": np.asarray(pos_l, np.float64),
            "yaws": np.asarray(yaw_l, np.float64),
            "segments": np.asarray(seg_l, dtype=int),
            "actions": np.asarray(act_l, np.float32),
            "rewards": np.asarray(rew_l, np.float32),
            "uncertainty": unc_l,
            "segment_info": segments,
            "episodes_done": int(eps_done),
            "last_end_pos": (None if last_end_pos is None
                             else np.asarray(last_end_pos, np.float64).tolist()),
            "meta": {
                "purpose": "explore-mode trace with pixels for DINOv2 localisation",
                "checkpoint": a.ckpt, "town": a.town,
                "mode": "explore (EVAL_FORCE_MODE=0)",
                "explore_goal_dist_m": os.environ.get("EXPLORE_GOAL_DIST_M"),
                "candidate_spacing_m": a.candidate_spacing,
                "collect_seed": a.collect_seed,
                "action_mode": a.action_mode,
                "episode_steps_cap": a.episode_steps,
                "perception": os.environ.get("PERCEPTION"),
                "vc1_readout": os.environ.get("VC1_READOUT"),
                "vc1_model": os.environ.get("VC1_MODEL", "vc1_vitb (default)"),
                "continuation": "nearest drivable waypoint to the previous stop",
                "rgb_source": "info['rgb'] == experiment.obs_img, the frame the "
                              "perception stack encoded on that step",
                # Channel order follows ENV_RGB_CHANNEL_FIX. parse() flips
                # BGRA->RGB and post_process_image() swaps R/B back (BGR); with
                # the fix ON, _fix_channel_order() flips again before obs_img,
                # so info['rgb'] is RGB. v43 runs with the fix ON (the shell
                # wrapper refuses otherwise) and its seg-VAE was trained on RGB
                # (collect_seg_data -> _to_array + env_frame, no swap).
                # Re-verified 2026-10-02 with a pure-blue buffer: channel 2.
                "rgb_channel_order": ("RGB" if os.environ.get(
                    "ENV_RGB_CHANNEL_FIX", "1") == "1" else "BGR"),
                "rgb_channel_order_note":
                    "Frames are byte-identical to what the policy encoded. RGB "
                    "when ENV_RGB_CHANNEL_FIX=1 (v17 onward, incl. v43); BGR only "
                    "for checkpoints run with the fix OFF (v13 and earlier).",
                "rgb_shape": (list(np.asarray(rgb_l[0]).shape) if rgb_l else None),
            },
        }

    def save(eps_done):
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(snapshot(eps_done), fh, protocol=4)
        os.replace(tmp, a.out)     # atomic: a kill mid-write cannot corrupt it

    core_owner = find_attr_owner(env, "core")
    amap = core_owner.core.map
    exp_owner = find_attr_owner(env, "experiment")
    cont_misses = 0
    t0 = time.time()
    ep = first_ep - 1

    while len(pos_l) < a.samples:
        ep += 1
        rng = np.random.default_rng((a.collect_seed, ep))
        seed = int(rng.integers(0, 2 ** 31 - 1))
        import random as _random
        _random.seed(seed)
        np.random.seed(seed % (2 ** 32))

        wp = None
        if last_end_pos is not None:
            try:
                wp = amap.get_waypoint(
                    carla.Location(x=float(last_end_pos[0]),
                                   y=float(last_end_pos[1]), z=0.0),
                    project_to_road=True)
                if wp is not None:
                    set_continue_origin(exp_owner.experiment, wp)
            except Exception as exc:                      # noqa: BLE001
                print(f"[Collect] ep{ep}: continuation failed "
                      f"({type(exc).__name__}: {exc}); using the sampler",
                      flush=True)
                wp = None

        obs, info = env.reset(seed=seed)
        if wp is not None:
            try:
                clear_continue_origin(exp_owner.experiment)
            except Exception:                             # noqa: BLE001
                pass
            spawn = np.asarray(info.get("hero_pos", (np.inf, np.inf)),
                               np.float64)[:2]
            if continued_spawn_ok(spawn, wp):
                cont_misses = 0
            else:
                cont_misses += 1
                print(f"[Collect] ep{ep}: CONTINUATION MISSED -- spawned "
                      f"({spawn[0]:.1f},{spawn[1]:.1f}), asked "
                      f"({wp.transform.location.x:.1f},"
                      f"{wp.transform.location.y:.1f})", flush=True)
                # Three in a row means the reset is not honouring the origin.
                # Recording a random-respawn trace under the continuation rule's
                # name is the exact failure that invalidated the v6n traces.
                if cont_misses >= 3:
                    save(ep)
                    raise SystemExit("[Collect] 3 continuation misses in a row; "
                                     "refusing to record a random-respawn trace "
                                     "as a continued one")

        start_pose = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]
        last_kept, why, t = None, "cap", 0
        for t in range(a.episode_steps):
            if len(pos_l) >= a.samples:
                why = "budget"
                break
            p = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]
            keep = (last_kept is None
                    or np.linalg.norm(p - last_kept) >= a.candidate_spacing)
            action = collect_action(agent, obs, a.action_mode)
            if keep:
                rgb = info.get("rgb")
                if rgb is None:
                    raise SystemExit("info['rgb'] is missing -- the env did not "
                                     "expose the camera frame; the whole point "
                                     "of this trace is the pixels")
                rgb_l.append(np.ascontiguousarray(np.asarray(rgb, np.uint8)))
                obs_d_l.append({k: np.asarray(v, np.float32).copy()
                                for k, v in obs.items()})
                obs_g_l.append({k: np.asarray(v, np.float32).copy()
                                for k, v in goal_mode(obs).items()})
                fields_l.append(wrapper._fields(info))
                pos_l.append(p.copy())
                yaw_l.append(float(info.get("hero_yaw", 0.0)))
                seg_l.append(ep)
                act_l.append(np.asarray(action, np.float32).copy())
                unc_l.append(_uncertainty(agent, obs, goal_mode(obs), action))
                last_kept = p.copy()
            obs, reward, term, trunc, info = env.step(action)
            if keep:
                rew_l.append(np.float32(reward))
            if term or trunc:
                # The explore arm's own verdict, so the cause matches the rule
                # v13 was trained under rather than a second reading of info.
                why = str(info.get("explore_end", "terminated" if term else "cap"))
                break

        last_end_pos = np.asarray(info.get("hero_pos", (0.0, 0.0)),
                                  np.float64)[:2].copy()
        segments.append({"segment": ep, "steps": int(t + 1), "end": why,
                         "start_xy": start_pose.tolist(),
                         "end_xy": last_end_pos.tolist()})
        save(ep + 1)
        print(f"[Collect] ep{ep} town={a.town} end={why} steps={t+1} "
              f"samples={len(pos_l)}/{a.samples} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    env.close()
    save(ep + 1)
    ends = {}
    for s in segments:
        ends[s["end"]] = ends.get(s["end"], 0) + 1
    why_line = "  ".join(f"{k}={v}" for k, v in
                         sorted(ends.items(), key=lambda kv: -kv[1]))
    print(f"[Collect] DONE {a.out}: {len(pos_l)} samples over {len(segments)} "
          f"episodes, rgb {np.asarray(rgb_l[0]).shape}, "
          f"spacing {a.candidate_spacing} m", flush=True)
    print(f"[Collect] episode ends: {why_line}", flush=True)


if __name__ == "__main__":
    main()

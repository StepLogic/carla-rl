"""Standalone evaluation of a trained CARLA TQC policy.

WHY THIS EXISTS SEPARATELY FROM THE TRAINER. The trainer's periodic eval reads
its goal band from GOAL_DIST_MIN/MAX -- the same variables the curriculum
rewrites -- so its numbers move with the curriculum and are not comparable to
each other. This pins the band explicitly per cell, so every number here means
what it says and two cells differ only in the thing being varied.

WHAT IT MEASURES
  * success per (town, goal-distance band), with n reported alongside
  * WHY episodes ended, which is where the actual finding is: a policy that
    fails by clipping lane markings is a different problem than one that
    crashes or one that wanders
  * distance travelled vs distance required, so "drove competently but never
    converged" is distinguishable from "never moved"

TOWNS. Town02 is the training map, so its column is IN-DISTRIBUTION and is
labelled as such. Town03 and Town10HD are the frozen held-out split and have
never been trained on -- those columns are the generalisation result.

    python eval_policy.py --ckpt checkpoints/.../checkpoint_1000000.pkl
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Bands are contiguous and cover the curriculum's whole range. Reported
# separately rather than pooled because the single most important property of
# this policy is how success decays with distance -- a pooled number hides it.
DEFAULT_BANDS = [(5, 10), (10, 20), (20, 40), (40, 70), (70, 113)]


def evaluate_cell(agent, town, lo, hi, n_episodes, latent_source, port, seed0,
                  max_steps, recorder=None):
    """One (town, band) cell, in a FRESH env. Returns per-episode records.

    Kept for the single-shot CLI. A sweep over many seeds should build the env
    ONCE and call run_episodes directly (see sweep_eval.py): rebuilding it per
    cell reloads the world and re-attaches every sensor for nothing, since the
    goal band is read from the environment at each reset rather than baked in
    at construction.
    """
    import train_carla_tqc as T

    env = T.make_env(town, seed0, latent_source, eval_mode=True, port=port)
    try:
        return run_episodes(agent, env, town, lo, hi, n_episodes, seed0,
                            max_steps, recorder=recorder)
    finally:
        env.close()


def run_episodes(agent, env, town, lo, hi, n_episodes, seed0, max_steps,
                 recorder=None, action_mode="deterministic"):
    """One (town, band) cell in an EXISTING env. Returns per-episode records.

    THE BAND IS AN ENVIRONMENT VARIABLE, NOT CONSTRUCTION STATE.
    carla_core.reset_hero reads GOAL_DIST_MIN/MAX at EVERY reset, so a cell can
    change the band on a live env -- which is what lets one env serve every
    seed and every band in a sweep.
    """
    os.environ["GOAL_DIST_MIN"] = str(float(lo))
    os.environ["GOAL_DIST_MAX"] = str(float(hi))

    # ASSERT THE SIMULATOR ACTUALLY LOADED THIS TOWN. Every field below is
    # labelled with `town` -- the loop variable -- so a make_env that quietly
    # loaded a different map produces a full results file that says Town02 and
    # measures somewhere else. That exact bug ran for weeks (make_env set the
    # town on the wrong config dict, and every env loaded Town01). Cheap check,
    # and it fails the cell rather than the paper.
    _loaded = env.unwrapped.core.world.get_map().name
    if town.split("_")[0].lower() not in _loaded.lower():
        raise RuntimeError(f"asked for {town} but the simulator has {_loaded}")
    print(f"[Eval] {town}: simulator map is {_loaded}", flush=True)

    # DRAW_GOAL=1 marks the goal in the CARLA world, visible in the spectator
    # view and in anything recording it.
    #
    # FROM info["goal_pos"], NOT carla_core.draw_goal. That method snaps the
    # point to the nearest road waypoint via get_waypoint(), so it can show a
    # position several metres from the one the policy was actually given and the
    # reward is actually scored against -- which is worse than drawing nothing,
    # because it looks authoritative. Same reasoning as watch_policy.draw_goal.
    _world = None
    if os.environ.get("DRAW_GOAL", "0") == "1":
        try:
            w = env
            while _world is None and w is not None:
                _world = getattr(getattr(w, "core", None), "world", None)
                w = getattr(w, "env", None)
        except Exception:  # noqa: BLE001
            _world = None
        print(f"[DrawGoal] {'enabled' if _world is not None else 'UNAVAILABLE'}",
              flush=True)

    def _mark(goal_xy, hero_z):
        """A vertical line at the goal, redrawn each step so it persists.

        life_time slightly exceeds one tick so the marker does not flicker
        between redraws and does not accumulate -- CARLA expires each primitive
        itself. Height comes from the HERO, not a constant: these towns are not
        flat and a fixed z buries the marker on a slope.
        """
        if _world is None:
            return
        try:
            import carla
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
            _world.debug.draw_line(
                carla.Location(x=gx, y=gy, z=hero_z),
                carla.Location(x=gx, y=gy, z=hero_z + 6.0),
                thickness=0.15, color=carla.Color(255, 0, 0), life_time=0.12)
        except Exception:  # noqa: BLE001
            pass

    records = []
    _ep_seeds = np.random.default_rng(int(seed0)).integers(0, 2 ** 31 - 1,
                                                          size=int(n_episodes))
    try:
        for i in range(n_episodes):
            # EPISODE SEEDS COME FROM A GENERATOR KEYED ON seed0, NOT seed0 + i.
            # With seed0 + i, sweep seeds 1235/1236/1237 shared 23-24 of their
            # 25 episodes (measured: identical median routes, sd 0.02 across
            # "seeds"), so four seeds were one draw. A generator keyed on
            # seed0 gives disjoint sets for different seeds and the same set
            # for the same seed. The GLOBAL random module is seeded too:
            # carla_core.reset_hero draws spawn and goal from it and
            # CarlaEnv.reset ignores its seed kwarg.
            ep_seed = int(_ep_seeds[i])
            import random as _random
            _random.seed(ep_seed)
            np.random.seed(ep_seed % (2 ** 32))
            obs, info = env.reset(seed=ep_seed)
            start = np.asarray(info["hero_pos"], np.float64)[:2]
            goal = np.asarray(info["goal_pos"], np.float64)[:2]
            required = float(np.linalg.norm(goal - start))

            # ROUTABLE FRACTION IS ITS OWN COLUMN (experiments.txt s4). An
            # episode whose goal the route planner cannot reach is not a
            # driving failure, and folding the two together lets a harness
            # that REFUSES hard episodes outscore one that attempts them.
            # route_remaining_m is None exactly when GlobalRoutePlanner
            # produced no route, so it is the routability test; it also gives
            # the ALONG-ROAD distance, which `required` (straight line) is not.
            routable, required_road = None, None

            if recorder is not None:
                recorder.start_episode(i, town)
                recorder.record(info.get("rgb"), info)

            done, steps, travelled = False, 0, 0.0
            prev = start.copy()
            goals_reached, reasons = 0, {}
            # ARRIVAL IS `arrived`/`is_success`, NOT `goals_reached`.
            #
            # goals_reached counts completed legs of a CHAIN. At
            # CHAIN_MAX_GOALS=1 the env never increments it, so a success test
            # of `goals_reached > 0` reads a counter that is identically zero
            # and reports 0.000 for every policy, however well it drives.
            #
            # MEASURED: checkpoint_75000 and checkpoint_100000 both scored
            # exactly 0.000 here on the 20-46.51 m band while the trainer's own
            # eval scored the same weights at 0.617 on the same band, and the
            # training logs print `success=0.700 ... goals=0.00` side by side
            # in one line. Held for BOTH checkpoints, which is what identified
            # the harness rather than the policies.
            #
            # Latched inside the loop rather than read at the end: the env
            # RESAMPLES the goal the moment it is reached, so by the final step
            # the flag has been cleared and the goal has moved. That is the same
            # trap documented in time_to_goal_reward.py.
            arrived = False
            # DRIVING QUALITY, not just arrival. Success rate is nearly
            # indifferent to HOW the car drives -- the goal-conditioned policy
            # scored 0.808 while accelerating at a p95 of 196 m/s^2 and
            # reversing its steering on 32.6% of steps. Path length and
            # collision count cannot see that, so it is measured directly.
            speeds, steers, throttles = [], [], []
            while not done and steps < max_steps:
                # ACTION MODE IS RECORDED, NEVER IMPLIED. "deterministic" is
                # tanh(mean); "sample" draws from the policy as trained (user
                # decision 2026-09-11 for the sweep: training-side success uses
                # samples, and the mean under-reads a wide policy).
                action = (agent.sample_actions(obs) if action_mode == "sample"
                          else agent.eval_actions(obs))
                obs, _r, term, trunc, info = env.step(action)
                now = np.asarray(info["hero_pos"], np.float64)[:2]
                # Re-read the goal every step: the env RESAMPLES it on arrival,
                # so a goal cached at reset would mark a stale point for every
                # leg after the first in a chain.
                _mark(np.asarray(info["goal_pos"], np.float64),
                      float(np.asarray(info["hero_pos"], np.float64)[2])
                      if np.asarray(info["hero_pos"]).size > 2 else 0.5)
                d = float(np.linalg.norm(now - prev))
                travelled += d
                speeds.append(float(info.get("speed_kmh", d / 0.05 * 3.6)))
                a = np.asarray(action, np.float64).reshape(-1)
                steers.append(float(a[0])); throttles.append(float(a[1]))
                prev = now
                steps += 1
                if recorder is not None:
                    recorder.record(info.get("rgb"), info)
                if routable is None:          # first step publishes the route
                    _rem = info.get("route_remaining_m", "absent")
                    if _rem != "absent":
                        routable = _rem is not None
                        required_road = (None if _rem is None
                                         else round(float(_rem), 1))
                goals_reached = max(goals_reached, int(info.get("goals_reached", 0)))
                arrived = arrived or bool(info.get("arrived", False)
                                          or info.get("is_success", False))
                done = bool(term or trunc)
                if done:
                    reasons = {k: bool(v) for k, v in
                               (info.get("done_reasons") or {}).items() if v}
                    if trunc and not term and not reasons:
                        reasons = {"step_limit": True}
            sp = np.asarray(speeds, np.float64)
            # Acceleration from the SPEED SERIES, not from commanded throttle:
            # what matters is what the car did, not what was asked of it.
            acc = np.diff(sp) / 0.05 / 3.6 if sp.size > 1 else np.zeros(1)
            st = np.asarray(steers, np.float64)
            # A steering "reversal" is a sign change in the commanded steer --
            # the direct measure of sawing at the wheel.
            flips = (np.sign(st[1:]) * np.sign(st[:-1]) < 0).mean() if st.size > 1 else 0.0
            records.append({
                "town": town, "band": f"{lo}-{hi}m", "episode": i,
                "action_mode": action_mode,
                # Either signal counts: `arrived` is the single-goal path,
                # goals_reached the chained one. Neither alone covers both.
                "success": bool(arrived or goals_reached > 0),
                "arrived": bool(arrived),
                "goals_reached": goals_reached,
                "required_m": round(required, 1),
                # None when ROUTE_POTENTIAL is off: absent, not False, so a
                # run without the planner cannot be read as "nothing routed".
                "routable": routable,
                "required_road_m": required_road,
                "travelled_m": round(travelled, 1),
                "steps": steps,
                "reasons": reasons,
                # driving quality
                "speed_median_kmh": round(float(np.median(sp)), 2) if sp.size else 0.0,
                "speed_p90_kmh": round(float(np.percentile(sp, 90)), 2) if sp.size else 0.0,
                "speed_max_kmh": round(float(sp.max()), 2) if sp.size else 0.0,
                "accel_p95": round(float(np.percentile(np.abs(acc), 95)), 2),
                "harsh_frac": round(float((np.abs(acc) > 4.0).mean()), 4),
                "steer_flip_frac": round(float(flips), 4),
                "stopped_frac": round(float((sp < 1.0).mean()), 4) if sp.size else 0.0,
            })
            if recorder is not None:
                recorder.end_episode(bool(arrived or goals_reached > 0))
    finally:
        pass                      # the env belongs to the caller
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--towns", default="Town02,Town03,Town10HD")
    ap.add_argument("--episodes", type=int, default=25, help="per (town, band)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("MAX_EPISODE_STEPS", 2000)))
    ap.add_argument("--out", default="results/policy_eval.json")
    ap.add_argument("--bands", default="")
    ap.add_argument("--video-dir", default="",
                    help="write an mp4 per episode here (experiments.txt s7 "
                         "requires video for every evaluation run)")
    ap.add_argument("--max-videos", type=int, default=25)
    args = ap.parse_args()

    bands = DEFAULT_BANDS
    if args.bands:
        bands = [tuple(float(x) for x in b.split("-"))
                 for b in args.bands.split(",")]

    import train_carla_tqc as T
    from vae_nav.eval_video import EvalVideoRecorder
    from vae_nav.latent_source import LatentSource
    from vae_nav.policy_io import load_agent_state, param_fingerprint

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    if "agent" not in ck:
        raise SystemExit(f"{args.ckpt} holds no weights (keys: {sorted(ck)})")

    ident = ck.get("identity", {})
    print(f"[Eval] checkpoint step={ck.get('step')} "
          f"train_towns={ident.get('train_towns')} "
          f"heldout={ident.get('heldout_towns')}")
    print(f"[Eval] {param_fingerprint(ck['agent'])}")

    latent_source = LatentSource(os.environ.get("VAE_CKPT", ""))
    towns = [t for t in args.towns.split(",") if t]

    # Build the agent once, against a throwaway env, then reuse it everywhere.
    os.environ.setdefault("GOAL_DIST_MIN", "5.0")
    os.environ.setdefault("GOAL_DIST_MAX", "20.0")
    probe = T.make_env(towns[0], args.seed, latent_source, eval_mode=True,
                       port=args.port)
    agent = T.build_agent(probe.observation_space, probe.action_space, args.seed)
    probe.close()
    load_agent_state(agent, ck["agent"])
    print("[Eval] weights loaded\n")

    all_records, t0 = [], time.time()
    for town in towns:
        for (lo, hi) in bands:
            # VIDEO FOR EVERY EVALUATION RUN, beside its results
            # (experiments.txt s7). One recorder per cell so the files carry
            # the town and band they belong to.
            rec = None
            if args.video_dir:
                rec = EvalVideoRecorder(
                    os.path.join(args.video_dir, f"{town}_{int(lo)}-{int(hi)}m"),
                    ck.get("step") or 0, max_videos=args.max_videos,
                    latent_source=latent_source)
            recs = evaluate_cell(agent, town, lo, hi, args.episodes,
                                 latent_source, args.port,
                                 args.seed + int(lo) * 97, args.max_steps,
                                 recorder=rec)
            all_records.extend(recs)
            sr = float(np.mean([r["success"] for r in recs]))
            gp = float(np.mean([r["goals_reached"] for r in recs]))
            # BOTH arrival signals printed, never just the combined one. They
            # disagreed silently once -- goals/ep held at 0.00 while the policy
            # was arriving on 70% of episodes -- and reporting only the merged
            # number is what let that read as "the policy fails" for two whole
            # checkpoints.
            ar = float(np.mean([r["arrived"] for r in recs]))
            top = Counter(k for r in recs for k in r["reasons"]).most_common(2)
            hq = float(np.mean([r["harsh_frac"] for r in recs]))
            fl = float(np.mean([r["steer_flip_frac"] for r in recs]))
            ap = float(np.mean([r["accel_p95"] for r in recs]))
            sm = float(np.mean([r["speed_median_kmh"] for r in recs]))
            # ROUTABLE, SR (ALL) AND SR (ROUTED) SIDE BY SIDE, never merged
            # (experiments.txt s4): a set that refuses hard episodes scores
            # higher on SR-among-routed precisely because it attempted less,
            # so the two can order arms in opposite directions.
            _rt = [r for r in recs if r.get("routable") is not None]
            if _rt:
                routable = float(np.mean([r["routable"] for r in _rt]))
                _ok = [r for r in _rt if r["routable"]]
                sr_routed = (float(np.mean([r["success"] for r in _ok]))
                             if _ok else float("nan"))
                rcol = f"routable={routable:.3f} SR(routed)={sr_routed:.3f} "
            else:
                rcol = "routable=n/a "
            print(f"  {town:9s} {lo:5.0f}-{hi:<5.0f}m  SR(all)={sr:.3f} "
                  f"(n={len(recs)})  {rcol}arrived={ar:.3f} goals/ep={gp:.2f} | "
                  f"harsh={hq:.3f} flips={fl:.3f} accel_p95={ap:6.1f} "
                  f"speed_med={sm:5.1f}km/h  {dict(top)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checkpoint": args.ckpt, "step": ck.get("step"),
                   "identity": ident, "episodes_per_cell": args.episodes,
                   "records": all_records}, f, indent=2)
    print(f"\n[Eval] {len(all_records)} episodes in "
          f"{(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()

"""Multi-seed policy sweep in ONE environment, on ONE simulator.

    python sweep_eval.py --ckpt checkpoints/carla_gated_her_v4/checkpoint_350000.pkl \
        --town Town02 --seeds 1234,1235,1236,1237 --episodes 25 --bands 20-50

WHAT THIS REPLACES, AND WHY. The sweep used to be a shell loop running
eval_policy.py once per seed. Every seed therefore paid for a new PROCESS, a
new CARLA SERVER and a new ENVIRONMENT -- and eval_policy itself builds two
envs (a throwaway probe for the observation/action spaces, then one per cell),
so four seeds meant eight world loads and, because CarlaCore starts each server
on a RANDOM port and kills it on close, up to eight server boots racing each
other's shutdowns. That is what the "CARLA keeps restarting" time-outs were.

Here: kill every CARLA once, start ONE server from Python, build ONE env, load
the agent ONCE, and then just reset. Nothing is rebuilt between seeds or bands.

THE BAND CAN CHANGE ON A LIVE ENV because carla_core.reset_hero reads
GOAL_DIST_MIN/MAX at every reset rather than at construction, and the seed is
passed to env.reset(seed=...) per episode. So "a new seed" needs no new env --
only a different reset seed, which is the whole point of a seed.

PER SEED, NEVER POOLED. Four seeds is the floor, not the target: a result that
holds on three seeds and fails on the fourth has not held. Each seed writes its
own JSON and its own row; the summary reports mean and spread ACROSS seeds and
never a single pooled rate over 4 x n episodes.

ROUTABLE FRACTION IS ITS OWN COLUMN, beside SR(all) and SR(routed). A set that
refuses hard episodes scores higher on SR-among-routed precisely because it
attempted less, so collapsing the two can order arms backwards.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def kill_all_carla(rounds=6):
    """Kill every CARLA on the machine and PROVE none survived.

    Standing rule: other sessions on this box start and stop servers, and a
    survivor holding the port means the sweep silently measures someone else's
    simulator. Three patterns because the launcher, the shell wrapper and the
    shipped binary do not all carry the same string.
    """
    for _ in range(rounds):
        for pat in ("CarlaUE4-Linux-Shipping", "CarlaUE4", "CARLA_0.9.15"):
            subprocess.run(["pkill", "-9", "-f", pat],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        out = subprocess.run(["pgrep", "-f", "CarlaUE4"],
                             capture_output=True, text=True).stdout.strip()
        if not out:
            return True
    print("[Sweep] WARNING: a CARLA process survived every kill round",
          flush=True)
    return False


def summarise(recs):
    """(SR all, routable, SR among routed, failure counter) for one seed."""
    sr = float(np.mean([r["success"] for r in recs])) if recs else float("nan")
    rt = [r for r in recs if r.get("routable") is not None]
    routable = float(np.mean([r["routable"] for r in rt])) if rt else None
    ok = [r for r in rt if r["routable"]]
    sr_routed = float(np.mean([r["success"] for r in ok])) if ok else None
    reasons = Counter(k for r in recs for k in r["reasons"])
    return sr, routable, sr_routed, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--seeds", default="1234,1235,1236,1237")
    ap.add_argument("--episodes", type=int, default=25, help="per seed, per band")
    ap.add_argument("--bands", default="20-50")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("MAX_EPISODE_STEPS", 750)))
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--video-dir", default="")
    ap.add_argument("--max-videos", type=int, default=5)
    ap.add_argument("--video-decode", type=int, default=0,
                    help="1 adds the VAE mask panel to each video. It runs the "
                         "VAE decoder on the GPU per frame, which pushed the "
                         "simulator into its death zone (see eval_video.py); "
                         "off by default for sweeps")
    ap.add_argument("--action-mode", default="sample",
                    choices=("deterministic", "sample"),
                    help="sample = agent.sample_actions (the policy as trained; "
                         "user decision 2026-09-11); deterministic = tanh(mean)")
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds whose JSON for this --tag already exists")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    bands = [tuple(float(v) for v in b.split("-")) for b in args.bands.split(",")]
    stamp = args.tag or time.strftime("%Y%m%d_%H%M")

    import eval_policy as EP
    import train_carla_tqc as T
    from vae_nav.eval_video import EvalVideoRecorder
    from vae_nav.latent_source import LatentSource
    from vae_nav.policy_io import load_agent_state

    print("[Sweep] pre-flight: killing every CARLA on this machine", flush=True)
    kill_all_carla()

    with open(args.ckpt, "rb") as f:
        import pickle
        ck = pickle.load(f)
    ident = ck.get("identity")
    latent_source = T.latent_from_env()
    if ident is not None:
        LatentSource.assert_matches(ident, latent_source)
    print(f"[Sweep] checkpoint step={ck.get('step')} -> {args.ckpt}", flush=True)

    # ONE env for the whole sweep. make_env starts the single Python-owned
    # CARLA (ensure_carla) the first time it is called.
    # ONE env, built once, reset thereafter. NO in-process rebuild: when the
    # simulator dies, the old env's libcarla sensor-stream threads keep
    # pointing at it, and the TimeoutException they throw on a background
    # thread cannot be caught from Python -- measured as `terminate called
    # after throwing carla::client::TimeoutException` and a core dump the
    # moment a replacement env was being built. So a dead simulator ends
    # THIS PROCESS with exit 75; the wrapper reruns it, and --resume skips
    # every seed that already has its JSON. The env is still built exactly
    # once per process, and a completed seed is never redone.
    env = T.make_env(args.town, seeds[0], latent_source, eval_mode=True,
                     port=args.port)
    agent = T.build_agent(env.observation_space, env.action_space, seeds[0])
    load_agent_state(agent, ck["agent"])
    print("[Sweep] weights loaded; env built once and reused", flush=True)

    rows, t0 = [], time.time()
    try:
        for (lo, hi) in bands:
            for seed in seeds:
                out = os.path.join(
                    args.out_dir,
                    f"{args.town.lower()}_seed{seed}_{int(lo)}-{int(hi)}m_{stamp}.json")
                if args.resume and os.path.exists(out):
                    with open(out) as f:
                        prev = json.load(f)
                    recs = prev["records"]
                    sr, routable, sr_routed, reasons = summarise(recs)
                    rows.append({"band": f"{lo:g}-{hi:g}", "seed": seed,
                                 "n": len(recs), "sr": sr, "routable": routable,
                                 "sr_routed": sr_routed,
                                 "reasons": dict(reasons), "out": out,
                                 "resumed": True})
                    print(f"  {args.town} {lo:g}-{hi:g}m seed {seed}: "
                          f"already done (resumed from {out})", flush=True)
                    continue
                rec = None
                if args.video_dir:
                    rec = EvalVideoRecorder(
                        os.path.join(args.video_dir,
                                     f"{args.town}_{int(lo)}-{int(hi)}m_s{seed}"),
                        ck.get("step") or 0, max_videos=args.max_videos,
                        latent_source=latent_source,
                        decode=bool(args.video_decode))
                try:
                    recs = EP.run_episodes(agent, env, args.town, lo, hi,
                                           args.episodes, seed, args.max_steps,
                                           recorder=rec,
                                           action_mode=args.action_mode)
                except RuntimeError as e:
                    # The simulator stopped answering. Nothing is written for
                    # this seed (a partial episode set would silently change n
                    # and the rate with it); the wrapper reruns the process.
                    print(f"[Sweep] seed {seed}: simulator died ({e}). "
                          f"Exiting 75 for the wrapper to rerun.", flush=True)
                    if T._PY_SERVER.get("stop"):
                        T._PY_SERVER["stop"]()
                    os._exit(75)
                sr, routable, sr_routed, reasons = summarise(recs)
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                with open(out, "w") as f:
                    json.dump({"checkpoint": args.ckpt, "step": ck.get("step"),
                               "action_mode": args.action_mode,
                               "town": args.town, "band": [lo, hi],
                               "seed": seed, "episodes": args.episodes,
                               "records": recs}, f, indent=2)
                rows.append({"band": f"{lo:g}-{hi:g}", "seed": seed, "n": len(recs),
                             "sr": sr, "routable": routable,
                             "sr_routed": sr_routed,
                             "reasons": dict(reasons), "out": out})
                print(f"  {args.town} {lo:g}-{hi:g}m seed {seed}: "
                      f"SR(all)={sr:.3f} (n={len(recs)}) "
                      f"routable={'n/a' if routable is None else f'{routable:.3f}'} "
                      f"SR(routed)={'n/a' if sr_routed is None else f'{sr_routed:.3f}'} "
                      f"| {dict(reasons)}", flush=True)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    # ACROSS SEEDS, NOT OVER POOLED EPISODES.
    print(f"\n[Sweep] {args.town}, checkpoint step {ck.get('step')}, actions={args.action_mode}")
    summary = {}
    for (lo, hi) in bands:
        band = f"{lo:g}-{hi:g}"
        srs = [r["sr"] for r in rows if r["band"] == band
               and r["sr"] is not None]
        if not srs:
            continue
        m = float(np.mean(srs))
        sd = float(np.std(srs, ddof=1)) if len(srs) > 1 else float("nan")
        summary[band] = {"per_seed": srs, "mean": m, "sd": sd,
                         "min": min(srs), "max": max(srs), "n_seeds": len(srs)}
        print(f"  {band} m: per-seed {['%.3f' % v for v in srs]}  "
              f"mean={m:.3f} sd={sd:.3f} range={min(srs):.3f}-{max(srs):.3f}")
    path = os.path.join(args.out_dir, f"{args.town.lower()}_sweep_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"checkpoint": args.ckpt, "step": ck.get("step"),
                   "action_mode": args.action_mode,
                   "town": args.town, "seeds": seeds, "bands": bands,
                   "episodes_per_cell": args.episodes,
                   "rows": rows, "summary": summary,
                   "resumed_seeds": [r["seed"] for r in rows
                                     if r.get("resumed")],
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    resumed = [r["seed"] for r in rows if r.get("resumed")]
    if resumed:
        print(f"[Sweep] NOTE: seeds {resumed} were resumed from an earlier "
              f"process of this sweep (the simulator died in between)")
    print(f"[Sweep] {(time.time() - t0) / 60:.1f} min -> {path}")


if __name__ == "__main__":
    main()

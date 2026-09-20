"""Offline MODE-FORCED eval for the 2-mode image-goal arm (plan task 10).

WHY FORCED, NOT SAMPLED. The 2-mode policy trains ONE actor on BOTH modes,
with the mode bit drawn at reset under EXPLORE_PROB. An eval that let the
env draw modes would mix the two tasks into one number and, worse, draw a
DIFFERENT mix per seed -- so each pass here forces the mode through
make_env(eval_mode=True), which reads EVAL_FORCE_MODE at call time and pins
it into TwoModeImageGoalReward for every episode (the "oracle coin was the
mode coin" failure mode: a coin that shares state with the thing measured).
The wrapper publishes info["mode"] on every step and this script CHECKS it:
a pass that ever sees a mode other than the forced one is refused outright,
not averaged away.

TWO PASSES, TWO MEASURES.
  * EVAL_FORCE_MODE=1 (goal): success fraction + episode returns (the sum of
    the wrapper's per-step rewards) over N episodes.
  * EVAL_FORCE_MODE=0 (explore): mean path metres via info["explore_dist_m"]
    (the wrapper's cumulative-path counter, the quantity the reward
    terminates on) + episode length. SUCCESS IS MEANINGLESS IN EXPLORE MODE
    and is neither computed nor reported: the wrapper pins
    info["is_success"]=False on every explore step, and this script strips
    the success columns from the mode-0 records rather than letting a
    meaningless 0.000 sit in the file to be read as a score.

PER-PASS COUNTS. --episodes-goal / --episodes-explore override --episodes for
the GOAL pass / the EXPLORE pass alone (unset: both passes still run
args.episodes, exactly the old default); the JSON header records the
effective counts as "episodes": {"goal": g, "explore": e, "default": n}.

THE SHELL IS THE FIRST LAYER. PERCEPTION, VC1_READOUT, OBS_WITH_*,
IG_SPEED_PREF_K, TARGET_SPEED and TQC_TARGET_MIN/MAX are read at MODULE
IMPORT (vc1_source, image_goal_reward, jaxrl2's critic_updater), so an
os.environ.setdefault inside this script is a SECOND layer, not the first --
by the time it runs, an import-time read may already have consumed the
ambient value. The caller therefore exports the full flag set before python
starts, or sources the launch's provenance snapshot:

    set -a; source results/run_env/carla_image_goal_2mode.env; set +a
    python3 scripts/image_goal_2mode_eval.py --ckpt checkpoints/.../pkl

As the defensive third layer the script itself loads the snapshot file
(EVAL_ENV_FILE, default results/run_env/carla_image_goal_2mode.env) with
setdefault semantics -- shell exports win, snapshot values fill the rest --
then pins the arm's flag set itself, and runs its setdefault block BEFORE
the heavy imports, so repo-side import-time reads are covered too.

ITS OWN SERVER, NEVER A SHARED ONE. This eval starts its own CARLA on
EVAL_PORT (default 2010 -- NOT the training sim's 2000 and NOT one of its
streaming ports). CARLA binds P/P+1/P+2, so the script REFUSES to run if
ANY of the three already answers: ensure_carla's "does the port answer"
poll cannot tell its own server from someone else's, and a client attached
to a port another process is streaming through corrupts that process's
session. The refusal, not the attach, is the behaviour; pick another
EVAL_PORT. Writes results/two_mode_eval/<run>/<ckpt>/eval_mode{0,1}.json.
"""
import argparse
import json
import os
import pickle
import socket
import sys
import time
from collections import Counter

# Repo root and its src/ -- this file lives in scripts/, one level DOWN from
# where image_goal_demo_350k.py sits, so the demo's dirname(abspath(__file__))
# path inserts would point at scripts/ and import nothing.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

# ===========================================================================
# THE ENV BLOCK -- SECOND AND THIRD LAYERS, BEFORE ANY HEAVY IMPORT.
# ===========================================================================
_PORT = int(os.environ.get("EVAL_PORT", "2010"))

# THIRD LAYER: the launch's provenance snapshot, loaded with setdefault
# semantics so the shell's exports always win. An eval must replay the flags
# the checkpoint trained under, and the snapshot is where run_gated_ig.sh
# wrote them. A missing file is reported, not fatal -- the setdefault block
# below still pins every flag the coherence gate demands.
_ENV_FILE = os.environ.get(
    "EVAL_ENV_FILE",
    os.path.join(_ROOT, "results", "run_env", "carla_image_goal_2mode.env"))


def _load_env_file(path):
    """KEY=VALUE lines (optionally `export ` prefixed) into os.environ.

    Only SETS keys that are not already present: the caller's shell exports
    are the first layer and must not be shadowed by a snapshot written at a
    different step of the run.
    """
    if not os.path.isfile(path):
        print(f"[2ModeEval] no env snapshot at {path} -- falling back to the "
              f"script's own setdefault block. Export the full flag set in "
              f"the SHELL for anything this block does not pin.", flush=True)
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if not key or key in os.environ:
                continue
            os.environ[key] = val
            n += 1
    print(f"[2ModeEval] loaded {n} flags from {path}", flush=True)
    return n


_load_env_file(_ENV_FILE)

# Flags the 2-mode pairing's guards REFUSE (run_gated_ig.sh scrubs the same
# list from its inherited environment): the coherence gate exits on them
# before any server starts, and a stale ambient value left over from the
# single-mode arm must not reach it.
for _refused in ("NO_TIMELIMIT", "OBS_WITH_COLLISION",
                 "LANE_FOLLOW", "LANE_EXPLORE", "TIME_TO_GOAL",
                 "TQC_VALUE_FLOOR", "TQC_VALUE_CEILING",
                 "GATED_ALLOW_DISCOUNTED_SORB", "ACTOR_Q_NORM",
                 "EXPLORE_ARRIVAL_BONUS", "EXPLORE_DISCOUNT",
                 "GOAL_BATCH_FRAC", "STEP_COST"):
    os.environ.pop(_refused, None)

# THE ARM'S FLAG SET (run_gated_ig.sh's export block, same values). This is
# the second layer: a caller who sourced the snapshot or exported the flags
# is replaying the run's own record, so this block only fills holes.
# EVAL_FORCE_MODE is deliberately absent -- the script drives it per pass.
os.environ.setdefault("IMAGE_GOAL_TASK", "1")
os.environ.setdefault("GATED", "1")
os.environ.setdefault("GATED_ONE_CRITIC", "1")
os.environ.setdefault("PERCEPTION", "vc1")
os.environ.setdefault("VC1_READOUT", "grid2")
os.environ.setdefault("OBS_WITH_STATE", "1")
os.environ.setdefault("OBS_WITH_MODE", "1")
os.environ.setdefault("OBS_WITH_GOAL", "1")
os.environ.setdefault("OBS_WITH_VECTOR", "0")
os.environ.setdefault("EPISODE_HER", "1")
os.environ.setdefault("HER_MIN_DISP", "5.0")
os.environ.setdefault("IG_GOAL_DISTANCE_THRESHOLD_M", "2.0")
# Raised from 0.1 (user 2026-09-20, both modes): the launcher now pins 0.2,
# so the mirror must not fill the hole with the superseded weight.
os.environ.setdefault("IG_SPEED_PREF_K", "0.2")
# The explore branch's terminal failure penalty (user 2026-09-20): the
# launcher pins 5, so the mirror fills the hole with the same value.
os.environ.setdefault("IG_TERMINAL_PENALTY", "5")
os.environ.setdefault("DISCOUNT", "0.99")
os.environ.setdefault("TOP_QUANTILES_DROP", "2")
os.environ.setdefault("TQC_KAPPA", "1.0")
os.environ.setdefault("TEMP_LR", "3e-4")
# Floor raised 2026-09-20 with the flat penalties (worst goal row -4.0 ->
# discounted floor -400 at gamma=0.99); the launcher now pins -450, so the
# mirror must not fill the hole with the superseded window.
os.environ.setdefault("TQC_TARGET_MIN", "-450")
os.environ.setdefault("TQC_TARGET_MAX", "100")
os.environ.setdefault("EXPLORE_PROB", "0.5")
os.environ.setdefault("EXPLORE_GOAL_DIST_M", "100.0")
os.environ.setdefault("TARGET_SPEED", "20.0")
os.environ.setdefault("ENV_SKIP_IMAGE_FEATURES", "1")
os.environ.setdefault("MAX_EPISODE_STEPS", "1000")
os.environ.setdefault("GATED_GOAL_REWARD", "shaped")
os.environ.setdefault("EXPERT_PROB", "0.0")

# Server settings for THIS process only. CARLA_START_SERVER=1 means the
# python side starts its own server on EVAL_PORT; the envs then attach with
# start_server=False, so no env owns or kills the server mid-run.
os.environ.setdefault("CARLA_START_SERVER", "1")
os.environ.setdefault("CARLA_QUALITY", "Low")
os.environ.setdefault("CARLA_OFFSCREEN", "1")
os.environ["CARLA_PORT"] = str(_PORT)
os.environ["CARLA_PORTS"] = str(_PORT)

# jax memory, same shape as the launch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

# NOW the heavy imports -- after every flag above is in place, so the
# import-time reads inside these modules see the pinned values in the
# no-shell-exports case too.
import train_carla_tqc as T  # noqa: E402
from eval_policy import run_episodes  # noqa: E402
from vae_nav.latent_source import LatentSource, from_env as latent_from_env  # noqa: E402
from vae_nav.policy_io import load_agent_state, param_fingerprint  # noqa: E402

GOAL, EXPLORE = 1, 0


def resolve_episode_counts(args):
    """The per-pass episode counts: each --episodes-* override, else --episodes.

    The user's binding 2026-09-20 decision (6 goal / 4 explore) is applied
    THROUGH these overrides, not by changing --episodes' default, so a bare
    --episodes invocation keeps driving both passes with the count it always
    did. `is not None`, not truthiness: an explicit --episodes-goal 0 is an
    override asking for zero episodes, not an unset flag.
    """
    goal = (args.episodes_goal
            if args.episodes_goal is not None else args.episodes)
    explore = (args.episodes_explore
               if args.episodes_explore is not None else args.episodes)
    return goal, explore


class ModeRecorder:
    """A thin proxy that captures the 2-mode wrapper's per-step info fields.

    run_episodes returns the records eval_policy.py measures, but it
    DISCARDS the step reward and does not surface info["explore_dist_m"],
    info["explore_end"] or info["mode"] -- the fields this eval is for.
    Rather than fork that loop, this proxy wraps the env, records them each
    step, and delegates everything else (including .unwrapped, which
    run_episodes' town check reads, and .close).

    One recorder per pass; each reset() OPENS a new episode list, so
    recorder.episodes lines up with run_episodes' episode i one-to-one.
    """

    def __init__(self, env):
        self._env = env
        self.episodes = []
        self._ep_steps = []

    def reset(self, **kwargs):
        self._ep_steps = []
        self.episodes.append(self._ep_steps)
        return self._env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self._env.step(action)
        entry = {
            "reward": float(reward),
            "mode": info.get("mode"),
            "is_success": bool(info.get("is_success", False)),
            # THE NATIVE ARRIVAL SIGNAL, captured RAW. CoreStateWrapper
            # publishes info["arrived"] = passed_chckp (core_state.py:111) --
            # the env's own arrival, unmerged with this arm's sustained-stop
            # success. run_episodes latches the two TOGETHER into its record
            # (arrived or is_success), so the recorder must take info
            # directly for the unmerged column to exist at all.
            "native_arrived": bool(info.get("arrived", False)),
            "explore_dist_m": (None if info.get("explore_dist_m") is None
                               else float(info["explore_dist_m"])),
            "explore_end": info.get("explore_end"),
            "has_collided": bool(info.get("has_collided", False)),
        }
        self._ep_steps.append(entry)
        return obs, reward, term, trunc, info

    def __getattr__(self, name):
        # Everything else -- unwrapped, close, observation_space, spec --
        # belongs to the real env. Only what __init__ stored stays local.
        return getattr(self._env, name)


def _port_in_use(port):
    """The first of P/P+1/P+2 something already answers on, or None.

    CARLA binds the RPC port and the two streaming ports above it; the
    standing lesson is to NEVER attach a client to a port another process is
    streaming through, and a bare "does the RPC port answer" poll cannot
    tell this script's future server from one that is already there. Any
    answer at all means the port is not ours to take.
    """
    for p in (int(port), int(port) + 1, int(port) + 2):
        with socket.socket() as sk:
            sk.settimeout(0.5)
            if sk.connect_ex(("localhost", p)) == 0:
                return p
    return None


def _per_episode_extras(recorder, forced_mode, records):
    """Join the recorder's per-step log onto run_episodes' records.

    The two are written in the same order (run_episodes resets once per
    episode; the recorder clears once per reset), and a length mismatch
    means the env was rebuilt mid-pass -- which would misalign every join
    below, so it is refused rather than zipped silently.
    """
    if len(records) == 0 or len(recorder.episodes) != len(records):
        raise RuntimeError(
            f"episode log ({len(recorder.episodes)}) does not line up with "
            f"records ({len(records)}) -- the env was rebuilt mid-pass; "
            f"refusing to join mode-forced metrics to the wrong episodes")
    out = []
    for rec, steps in zip(records, recorder.episodes):
        modes = [s["mode"] for s in steps]
        if any(m != forced_mode for m in modes):
            bad = next(i for i, m in enumerate(modes) if m != forced_mode)
            raise RuntimeError(
                f"MODE COIN LEAK: episode {rec['episode']} step {bad} ran "
                f"mode={modes[bad]} under forced mode {forced_mode}. The "
                f"pass is not about the mode it says it is -- refusing to "
                f"report it. (Is something rebuilding the env without "
                f"eval_mode=True?)")
        successes = sum(1 for s in steps if s["is_success"])
        ends = [s["explore_end"] for s in steps if s["explore_end"]]
        extras = {
            "mode": forced_mode,
            "mode_held": True,
            "steps_logged": len(steps),
            "return": round(sum(s["reward"] for s in steps), 4),
            "collided": bool(steps[-1]["has_collided"]),
            "explore_dist_m": (None if steps[-1]["explore_dist_m"] is None
                               else round(steps[-1]["explore_dist_m"], 2)),
            "explore_end": ends[-1] if ends else None,
        }
        if forced_mode == GOAL:
            extras["success_steps"] = successes
            extras["native_arrived"] = any(s["native_arrived"] for s in steps)
            # CROSS-CHECK THE TWO ARRIVAL DEFINITIONS. run_episodes' record
            # success is a MERGED latch (native passed_chckp OR wrapper
            # is_success); success_steps here is the wrapper's sustained
            # stop alone. The two disagree exactly where the arm's
            # definition does the work: a native arrival WITHOUT the
            # sustained stop (record True, 0 steps) or a stop the env's own
            # checkpoint logic never credited (0 native, success_steps>0).
            # That disagreement is warned per episode and both signals are
            # reported side by side -- never merged into one number.
            if bool(rec["success"]) != bool(successes):
                print(f"[2ModeEval] WARNING episode {rec['episode']}: "
                      f"record success={rec['success']} (merged: native "
                      f"passed_chckp OR wrapper is_success) but "
                      f"{successes} sustained-stop is_success steps; "
                      f"native_arrived={extras['native_arrived']}",
                      flush=True)
        else:
            # EXPLORE MODE: success is meaningless by construction (the
            # wrapper pins is_success=False), so the success columns are
            # STRIPPED, not zeroed -- a 0.000 in the file would still be
            # read as a score.
            for dead in ("success", "arrived", "goals_reached"):
                rec.pop(dead, None)
        out.append(extras)
    return out


def _mean_return(extras):
    return sum(e["return"] for e in extras) / len(extras)


def _summary_for(mode, records, extras):
    """The pass's summary, on the mode's own measure."""
    summary = {
        "forced_mode": mode,
        "episodes": len(records),
        "mean_steps": round(float(sum(r["steps"] for r in records))
                            / len(records), 1),
        "mean_return": round(_mean_return(extras), 4),
        "collided_frac": round(float(sum(e["collided"] for e in extras))
                               / len(extras), 4),
        "mode_verified": True,
    }
    if mode == GOAL:
        # SUCCESS FROM THE WRAPPER'S is_success, the arm's own definition
        # (the sustained stop), reported BESIDE -- never merged with -- the
        # env's NATIVE arrival (info["arrived"] = passed_chckp,
        # core_state.py:111), taken RAW from the recorder's per-step log:
        # run_episodes' own record success is a MERGED latch of the two
        # signals, so a "native" column read from it would be max(native,
        # wrapper) under a native label. The two disagree by construction
        # (native = drove to the env's goal point; the arm's = drove there
        # AND HELD the stop window), and the merged number would read as a
        # score of neither.
        summary["success_frac"] = round(
            float(sum(e["success_steps"] > 0 for e in extras))
            / len(extras), 4)
        summary["native_arrived_frac"] = round(
            float(sum(e["native_arrived"] for e in extras)) / len(extras), 4)
        summary["mean_travelled_m"] = round(
            float(sum(r["travelled_m"] for r in records)) / len(records), 1)
        summary["mean_required_m"] = round(
            float(sum(r["required_m"] for r in records)) / len(records), 1)
        summary["success_steps_total"] = sum(e["success_steps"]
                                             for e in extras)
    else:
        # EXPLORE: the path metres the reward terminates on, and the end
        # reasons. NO success field anywhere -- see _per_episode_extras.
        summary["mean_path_m"] = round(
            float(sum(e["explore_dist_m"] for e in extras)) / len(extras), 2)
        summary["explore_end_counts"] = dict(
            Counter(e["explore_end"] or "none" for e in extras))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=os.environ.get(
        "EVAL_CKPT", ""), help="checkpoint .pkl (EVAL_CKPT)")
    ap.add_argument("--town", default=os.environ.get("EVAL_TOWN", "Town01"))
    ap.add_argument("--episodes", type=int,
                    default=int(os.environ.get("EVAL_EPISODES", "25")))
    ap.add_argument("--episodes-goal", type=int, default=None,
                    help="episodes for the GOAL pass alone "
                         "(default: --episodes)")
    ap.add_argument("--episodes-explore", type=int, default=None,
                    help="episodes for the EXPLORE pass alone "
                         "(default: --episodes)")
    ap.add_argument("--seed", type=int,
                    default=int(os.environ.get("EVAL_SEED", "1234")))
    ap.add_argument("--port", type=int, default=_PORT)
    ap.add_argument("--band-lo", type=float,
                    default=float(os.environ.get("EVAL_BAND_LO", "5.0")))
    ap.add_argument("--band-hi", type=float,
                    default=float(os.environ.get("EVAL_BAND_HI", "25.0")),
                    help="the run's curriculum ceiling (GOAL_DIST_TARGET=25)")
    ap.add_argument("--max-steps", type=int, default=int(
        os.environ.get("MAX_EPISODE_STEPS", "1000")))
    ap.add_argument("--action-mode", default=os.environ.get(
        "EVAL_ACTION", "sample"),
        help="'sample' (as trained; the sweep's recorded decision: the mean "
             "under-reads a wide policy) or 'deterministic' (tanh(mean))")
    ap.add_argument("--out-dir", default=os.environ.get("EVAL_OUT_DIR", ""))
    args = ap.parse_args()

    if not args.ckpt:
        ap.error("--ckpt is required (or export EVAL_CKPT)")
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"no checkpoint at {args.ckpt}")
    if args.action_mode not in ("sample", "deterministic"):
        ap.error(f"--action-mode {args.action_mode!r} is not "
                 f"sample|deterministic")
    n_goal, n_explore = resolve_episode_counts(args)

    # THE PORT REFUSAL -- before the lock, before make_env, before
    # ensure_carla can attach to anything that is not ours.
    busy = _port_in_use(args.port)
    if busy is not None:
        raise SystemExit(
            f"REFUSING TO START: port {busy} (eval port {args.port} plus "
            f"its streaming pair) already answers. This eval starts its OWN "
            f"server and never attaches -- a client on a streaming port "
            f"corrupts the process that owns it. Pick another EVAL_PORT and "
            f"check what holds the port before reusing this one.")
    # ONE eval per port, same convention as the launch scripts: a run that
    # WAITED on the lock would hold nothing but risk, so refuse instead.
    import fcntl
    lock_path = f"/tmp/carla_rl_eval_port{args.port}.lock"
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"REFUSING TO START: another eval holds {lock_path} (port "
            f"{args.port}). Two evals on one port race the server start.")

    # CARLA_ROOT must be in place BEFORE ensure_carla runs, or the eval dies
    # mid-launch on a bare KeyError('CARLA_ROOT') -- after the port refusal
    # has passed and with nothing but the traceback to say what was missing.
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root or not os.path.isdir(carla_root):
        raise SystemExit(
            f"REFUSING TO START: CARLA_ROOT={carla_root!r} is not a CARLA "
            f"install directory. ensure_carla launches the simulator from "
            f"it. Export it in the shell (e.g. "
            f"CARLA_ROOT=/home/robotlab/Apps/CARLA_0.9.15) or source the "
            f"run's snapshot: set -a; source "
            f"results/run_env/carla_image_goal_2mode.env; set +a")

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    if "agent" not in ck:
        raise SystemExit(f"{args.ckpt} holds no weights (keys: {sorted(ck)})")
    ident = ck.get("identity", {})
    step = ck.get("step")
    print(f"[2ModeEval] checkpoint step={step} "
          f"{param_fingerprint(ck['agent'])}", flush=True)

    latent_source = latent_from_env()
    try:
        LatentSource.assert_matches(ident, latent_source)
    except Exception as exc:            # noqa: BLE001 -- refuse loudly, early
        raise SystemExit(f"latent backend mismatch: {exc}")

    # OUT DIR: results/two_mode_eval/<run>/<ckpt-file>/, so several
    # checkpoints of one run can be evaluated without overwriting.
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    out_dir = args.out_dir or os.path.join(
        _ROOT, "results", "two_mode_eval", run_dir,
        os.path.splitext(os.path.basename(args.ckpt))[0])

    # Provenance: every arm flag this eval is ABOUT, from the live env, so
    # the JSON says which flags it ran under even if no snapshot was loaded.
    flags = {k: os.environ[k] for k in sorted(os.environ)
             if k.startswith(("IG_", "OBS_WITH_", "TQC_", "VC1", "PERCEPTION",
                              "GATED", "EXPLORE_", "TARGET_", "GOAL_DIST",
                              "IMAGE_GOAL", "EPISODE_HER", "DISCOUNT",
                              "TEMP_LR", "HER_MIN_DISP", "TOP_QUANTILES",
                              "MAX_EPISODE"))}
    # THE BAND ACTUALLY USED, recorded in the flags block. run_episodes pins
    # GOAL_DIST_MIN/MAX from the band at EVERY reset, but it does so on the
    # live env -- so a flags snapshot taken before the first reset would
    # show the ambient band (or nothing) instead of the one these episodes
    # were drawn under.
    flags["GOAL_DIST_MIN"] = str(float(args.band_lo))
    flags["GOAL_DIST_MAX"] = str(float(args.band_hi))
    header = {"checkpoint": args.ckpt, "step": step, "identity": ident,
              "action_mode": args.action_mode, "seed0": args.seed,
              "town": args.town, "band": (args.band_lo, args.band_hi),
              # The EFFECTIVE per-pass counts, not just the --episodes
              # default: with the overrides set the two passes differ, and a
              # single integer here would silently lie about one of them.
              "episodes": {"goal": n_goal, "explore": n_explore,
                           "default": args.episodes},
              "max_steps": args.max_steps,
              "flags": flags}

    # ONE agent, built against the first env's spaces -- the two passes build
    # identical spaces (same flags, same wrapper order), and load_agent_state
    # refuses a structure that does not match the checkpoint's.
    agent = None
    for mode in (GOAL, EXPLORE):
        # THE PASS CONTRACT. make_env reads EVAL_FORCE_MODE from os.environ
        # AT CALL TIME and pins it into the wrapper's force_mode, so it must
        # be in place on the line BEFORE each make_env. A caller's exported
        # value is overridden -- the two passes, not the shell, decide which
        # mode each pass is about.
        prev = os.environ.get("EVAL_FORCE_MODE")
        if prev is not None and prev != str(mode):
            print(f"[2ModeEval] overriding caller's EVAL_FORCE_MODE={prev} "
                  f"with the pass value {mode}", flush=True)
        os.environ["EVAL_FORCE_MODE"] = str(mode)

        env = T.make_env(args.town, args.seed, latent_source, eval_mode=True,
                         port=args.port)
        if agent is None:
            agent = T.build_agent(env.observation_space, env.action_space,
                                  args.seed)
            load_agent_state(agent, ck["agent"])
            print("[2ModeEval] weights loaded", flush=True)
        try:
            t0 = time.time()
            recorder = ModeRecorder(env)
            # THE PASS'S OWN COUNT: the mode comes from EVAL_FORCE_MODE (set
            # above), the episode count from the per-pass override.
            records = run_episodes(agent, recorder, args.town, args.band_lo,
                                   args.band_hi,
                                   n_goal if mode == GOAL else n_explore,
                                   args.seed,
                                   max_steps=args.max_steps,
                                   action_mode=args.action_mode)
            extras = _per_episode_extras(recorder, mode, records)
        finally:
            env.close()      # attached env; our own server goes down with
                             # ensure_carla's atexit, never a shared one
        summary = _summary_for(mode, records, extras)

        doc = dict(header)
        doc["forced_mode"] = mode
        doc["summary"] = summary
        doc["records"] = records
        doc["per_episode"] = extras
        doc["wall_s"] = round(time.time() - t0, 1)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"eval_mode{mode}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"[2ModeEval] mode {mode}: {json.dumps(summary)} -> {path} "
              f"({doc['wall_s']}s)", flush=True)

    # The lock is held to process exit; ensure_carla tears OUR server down at
    # atexit, and nothing here touches any other process.
    print(f"[2ModeEval] done: {n_goal} goal + {n_explore} explore episodes "
          f"(default {args.episodes}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
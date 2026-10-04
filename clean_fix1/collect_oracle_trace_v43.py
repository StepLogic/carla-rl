#!/usr/bin/env python
"""Explore-mode trace driven by the ORACLE, for map coverage (user 2026-10-03).

Same artefact as collect_explore_trace_v43.py -- same pixels, poses, fields and
per-sample uncertainty telemetry -- with two differences, both deliberate:

  THE DRIVER IS THE EXPERT. CARLA's BasicAgent (EXPERT_KIND=basic), wearing
  ExpertOracle's interface and clipped to v43's own action box, drives every
  step. The policy still runs at every kept sample, so the trace records what
  the critics believe about states the POLICY NEVER REACHES, plus the action
  the policy WOULD have taken beside the one the expert took -- a direct
  steering-error signal against a known-good driver.

  DESTINATIONS CHASE UNVISITED GROUND. The expert needs somewhere to go, and
  "drive well" is not "drive somewhere new": v43's explore reward pays forward
  progress with NO novelty term at all, which is why the policy's own trace is
  one ring road driven 80 times. This collector keeps a coarse visited-cell set
  and sends the expert to the NEAREST UNVISITED CELL whenever a route finishes,
  which sweeps the map outward from wherever it starts.

Everything else is held identical to the policy trace so the two compare: the
same checkpoint, the same fault rules (a painted crossing on open road,
off-road and collision each end the segment), the same restart at the nearest
drivable waypoint to the stop, fixed weather, the same seed, the same 2 m
sample spacing, and ENV_RGB_CHANNEL_FIX=1 so the frames are RGB.

The expert's 25/25 benchmark was measured under the OLD action box
(max_throttle 0.4 / max_steer 0.5). v43 removed those caps, so its speed is
governed by EXPERT_TARGET_SPEED rather than by clipping -- survival is
therefore MEASURED per run and printed, never assumed.

usage: collect_oracle_trace_v43.py --ckpt CKPT --town TOWN --out OUT.pkl
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



class CoverageDestinations:
    """Pick where the expert drives next: the NEAREST cell it has not visited.

    "Nearest unvisited" rather than "farthest" or "random" on purpose. Farthest
    routes the car across town through ground it will cover anyway and hands the
    planner long routes that strand it when one ends in a fault; random re-picks
    ground already driven, which is the policy trace's failure. Nearest-unvisited
    fills outward from wherever the car is, so coverage grows as a front and
    every route is short enough to finish.

    Cells are coarse (default 15 m): the aim is streets not visited, not metres
    not touched, and a fine grid would send the expert back down a road it has
    already driven to collect the half of it that fell in another cell.
    """

    def __init__(self, waypoints, cell_m, rng, n_headings=4):
        # DIRECTION IS PART OF THE KEY. Covering a cell while driving north says
        # nothing about driving it south: the car uses the other carriageway,
        # sees a different view, and -- the reason this matters -- a graph built
        # from one-way coverage is one-way. Town02's first sweep declared 100%
        # after driving every street once, and the directed graph it produced
        # had 340 strongly connected components out of 499 nodes, so almost no
        # node could reach another and come back. n_headings=4 buckets a full
        # turn into quadrants, which separates the two carriageways of a road
        # without demanding the car re-drive every gentle curve.
        self.cell_m = float(cell_m)
        self.n_headings = int(n_headings)
        self.rng = rng
        self.xy = np.asarray([[w.transform.location.x, w.transform.location.y]
                              for w in waypoints], np.float64)
        self.wps = list(waypoints)
        # Every (cell, heading) the road graph actually offers: a one-way street
        # contributes one heading, a two-way road two. Coverage is measured
        # against this, so 100% now means "driven both ways where both exist".
        self.cells = {self._cell(np.array([w.transform.location.x,
                                           w.transform.location.y]),
                                 np.radians(w.transform.rotation.yaw))
                      for w in waypoints}
        self.visited = set()
        self.ever = set()          # every cell driven, across all sweeps
        self.sweeps = 0            # times the map was covered and restarted

    def _cell(self, p, yaw=None):
        xy = (int(np.floor(p[0] / self.cell_m)), int(np.floor(p[1] / self.cell_m)))
        if yaw is None:
            return xy
        h = int(np.floor((np.degrees(yaw) % 360.0) / (360.0 / self.n_headings)))
        return xy + (h,)

    def visit(self, pos, yaw=None):
        c = self._cell(np.asarray(pos, np.float64), yaw)
        self.visited.add(c)
        self.ever.add(c)

    def coverage(self):
        """Fraction of reachable cells driven IN THE CURRENT SWEEP."""
        return len(self.visited & self.cells) / max(1, len(self.cells))

    def coverage_total(self):
        """Fraction driven at any point, across every sweep. This is the number
        that describes the artefact; coverage() only drives the next choice."""
        return len(self.ever & self.cells) / max(1, len(self.cells))

    def next_goal(self, pos, min_m):
        """carla.Location of the nearest unvisited waypoint, or None if the map
        is covered. Ties broken at random so two runs do not share one path."""
        pos = np.asarray(pos, np.float64)[:2]
        d = np.linalg.norm(self.xy - pos, axis=1)
        # The cell under the car counts as driven whether or not visit() has
        # been called for it yet: without this the min-distance fallback below
        # routes the expert to where it is already standing, the planner returns
        # a zero-length route, and the segment stalls.
        # A DESTINATION IS A PLACE, not a place-and-heading: the router cannot
        # choose which way the car will arrive. So "fresh" asks whether this
        # waypoint's cell still has ANY heading left undriven.
        driven_xy = {c[:2] for c in self.visited}
        want = {}
        for c in self.cells:
            want.setdefault(c[:2], set()).add(c[2])
        done = {}
        for c in self.visited:
            done.setdefault(c[:2], set()).add(c[2])
        here = self._cell(pos)
        fresh = np.array([
            bool((self._cell(p) != here)
                 and (bool(want.get(self._cell(p), set())
                           - done.get(self._cell(p), set()))
                      or self._cell(p) not in driven_xy))
            for p in self.xy])
        ok = fresh & (d >= float(min_m))
        if not ok.any():
            ok = fresh                       # everything far enough is driven
        if not ok.any():
            # THE MAP IS COVERED. Start a new sweep rather than returning None:
            # returning None ended every episode after one step and span the
            # collector through 5433 junk episodes on Town02 (only 98 cells, so
            # it covers in ~1500 samples). A second pass is still good data --
            # different weather phase, different approach directions -- and
            # coverage_total() remembers that the first pass happened.
            self.sweeps += 1
            self.visited = {c for c in self.visited if c[:2] == here}
            fresh = np.array([self._cell(p) != here for p in self.xy])
            ok = fresh & (d >= float(min_m))
            if not ok.any():
                ok = fresh
            if not ok.any():
                return None                  # a one-cell map: nothing to do
        idx = np.flatnonzero(ok)
        best = idx[d[idx] <= d[idx].min() + self.cell_m]   # near-ties
        return self.wps[int(self.rng.choice(best))].transform.location


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
    ap.add_argument("--driver", choices=("oracle", "policy"), default="oracle",
                    help="who steers. The policy runs at every kept sample "
                         "either way; only the EXECUTED action changes.")
    ap.add_argument("--expert-speed", type=float,
                    default=float(os.environ.get("EXPERT_TARGET_SPEED", 20.0)),
                    help="km/h. Above what the action box can reach, the "
                         "oracle's PID saturates and it stops cornering.")
    ap.add_argument("--coverage-cell-m", type=float, default=15.0,
                    help="visited-cell size for picking destinations. Coarse "
                         "on purpose: the aim is new STREETS, not new metres.")
    ap.add_argument("--dest-min-m", type=float, default=30.0,
                    help="never route to somewhere already under the car.")
    a = ap.parse_args()

    # EVERY FLAG THAT THE CONFIG READS AT IMPORT MUST BE SET FIRST. This is the
    # repo's standing hazard: these modules read os.environ at module scope, so
    # setting them after the import is a silent no-op.
    os.environ["EVAL_FORCE_MODE"] = "0"          # explore half of the gated arm
    # train_carla_tqc.make_expert reads both of these at module scope, so they
    # have to be set before the import, not after it.
    os.environ["EXPERT_KIND"] = "basic"
    os.environ.setdefault("EXPERT_TARGET_SPEED", "20.0")
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

    # The destination chooser needs the road graph, which comes from the LIVE
    # map (the server is already up) rather than the .xodr on disk, so the
    # waypoints are exactly the ones the planner will route over.
    _rng = np.random.default_rng(a.collect_seed)
    _core = env.unwrapped.core
    _wmap = getattr(_core, "map", None) or _core.world.get_map()
    dests = CoverageDestinations(_wmap.generate_waypoints(a.candidate_spacing),
                                 a.coverage_cell_m, _rng)
    print(f"[Oracle] {len(dests.cells)} reachable {a.coverage_cell_m:g} m cells "
          f"on {a.town}", flush=True)
    n_replan, n_expert_none = 0, 0

    rgb_l, obs_d_l, obs_g_l, fields_l = [], [], [], []
    unc_l, pol_act_l = [], []
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
        # Same contract for the policy's own action: pad rather than guess, so
        # row i still means sample i in an artefact written before this field.
        pol_act_l = (list(prev["policy_actions"]) if "policy_actions" in prev
                     else [np.zeros(2, np.float32) for _ in rew_l])
        if len(pol_act_l) != len(rew_l):
            raise SystemExit(f"resume misalignment: {len(pol_act_l)} policy "
                             f"action rows vs {len(rew_l)} samples")
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
            # What the EXPERT executed is "actions"; what the POLICY proposed at
            # the same state is this. The difference is the steering error
            # against a known-good driver, measured on states the policy's own
            # trace never visits.
            "policy_actions": np.asarray(pol_act_l, np.float32),
            "rewards": np.asarray(rew_l, np.float32),
            "uncertainty": unc_l,
            "segment_info": segments,
            "episodes_done": int(eps_done),
            "last_end_pos": (None if last_end_pos is None
                             else np.asarray(last_end_pos, np.float64).tolist()),
            "meta": {
                "purpose": ("oracle-driven explore-mode trace for MAP COVERAGE, "
                            "with pixels for DINOv2 localisation and the "
                            "policy's telemetry on expert states"),
                "driver": a.driver,
                "expert_kind": os.environ.get("EXPERT_KIND"),
                "expert_target_speed_kmh": a.expert_speed,
                "destination_rule": ("nearest unvisited cell, "
                                     f"{a.coverage_cell_m:g} m, "
                                     f"min {a.dest_min_m:g} m away"),
                "coverage_cell_m": a.coverage_cell_m,
                "map_cells_total": len(dests.cells),
                "map_cells_visited": len(dests.ever & dests.cells),
                "map_coverage": dests.coverage_total(),
                "map_sweeps": dests.sweeps,
                "expert_replans": n_replan,
                "expert_gave_up": n_expert_none,
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

        # THE EXPERT IS REBUILT EVERY EPISODE. carla_core.reset_hero destroys
        # and respawns the hero, so an expert held across a reset drives a dead
        # actor -- silently, because BasicAgent keeps returning control for a
        # vehicle that no longer exists.
        expert = None
        if a.driver == "oracle":
            expert = T.make_expert(env, env.action_space)
            if expert is None:
                raise SystemExit("[Oracle] no expert could be built -- refusing "
                                 "to record a policy-driven trace under the "
                                 "oracle's name")

        start_pose = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]

        def _route(pos):
            """Send the expert at the nearest unvisited ground. True if routed."""
            goal = dests.next_goal(pos, a.dest_min_m)
            if goal is None:
                return False
            try:
                expert.reset(goal)
            except Exception as exc:                      # noqa: BLE001
                print(f"[Oracle] ep{ep}: plan failed "
                      f"({type(exc).__name__}: {exc})", flush=True)
                return False
            return getattr(expert, "last_plan_error", None) is None

        if expert is not None:
            dests.visit(start_pose, float(info.get('hero_yaw', 0.0)))
            _route(start_pose)
        last_kept, why, t = None, "cap", 0
        for t in range(a.episode_steps):
            if len(pos_l) >= a.samples:
                why = "budget"
                break
            p = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]
            keep = (last_kept is None
                    or np.linalg.norm(p - last_kept) >= a.candidate_spacing)
            # The POLICY runs at every step whoever is driving: its action is
            # the comparison this trace exists for, and its critics are read at
            # the action actually EXECUTED (Q(s,a) of the realised trajectory).
            policy_action = collect_action(agent, obs, a.action_mode)
            action = policy_action
            if expert is not None:
                dests.visit(p, float(info.get('hero_yaw', 0.0)))
                ex = expert.act()
                if ex is None:                   # route finished, or it failed
                    n_replan += 1
                    ex = expert.act() if _route(p) else None
                if ex is None:
                    # Falling back to the policy would quietly turn this into a
                    # mixed-driver trace, so it ends the segment instead and the
                    # cause is recorded under the oracle's own name.
                    n_expert_none += 1
                    why = "expert_stuck"
                    # Spinning on this is what burned 5433 episodes once. The
                    # re-sweep removed the cause; this refuses to hide a new one.
                    if n_expert_none >= 25:
                        save(ep)
                        raise SystemExit(
                            "[Oracle] 25 unroutable episodes: the expert cannot "
                            "be given a destination, so this is not a coverage "
                            "trace. Refusing to spin.")
                    break
                action = np.clip(np.asarray(ex, np.float32),
                                 env.action_space.low, env.action_space.high)
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
                pol_act_l.append(np.asarray(policy_action, np.float32).copy())
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
              f"cover={dests.coverage_total():.1%}+{dests.sweeps}sw "
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
    # THE TWO NUMBERS THIS RUN EXISTS TO PRODUCE. Coverage is the point of the
    # oracle; the survival rate says whether the expert is actually a good
    # driver under v43's uncapped action box, which its 25/25 benchmark (taken
    # at 0.4 throttle / 0.5 steer) does not establish.
    faults = sum(v for k, v in ends.items()
                 if k not in ("budget", "cap", "expert_stuck"))
    print(f"[Oracle] map coverage {dests.coverage_total():.1%} "
          f"({len(dests.ever & dests.cells)}/{len(dests.cells)} cells of "
          f"{a.coverage_cell_m:g} m)  replans={n_replan}  "
          f"gave_up={n_expert_none}  sweeps={dests.sweeps}", flush=True)
    print(f"[Oracle] expert faults: {faults} over {len(segments)} segments", flush=True)


if __name__ == "__main__":
    main()

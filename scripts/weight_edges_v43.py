#!/usr/bin/env python3
"""Weight map edges with the GOAL CRITIC, in predicted steps (user 2026-10-03).

An edge a->b is weighted by asking the goal critic what it costs to drive from
the state recorded at node a to node b's POSITION: the observation at a is
copied with only its 4-d goal block recomputed for b, through the wrapper's own
transform, and V = Q(s, pi(s)) is read back as steps. No simulator is involved --
`VAEGoalWrapper.goal_block` is a pure function of pose, and the perception latent
is what the camera saw at a and does not depend on which goal the step is
credited to.

SEGLINKS ARE GONE (user 2026-10-03). They spanned respawns, so they were never
drivable; the graph is now one component per segment and this script reports how
many, because that is the honest cost of dropping them. The route back to a
connected graph is critic-certified shortcuts between nodes that are genuinely
close, which this script's weights are the input to -- not a respawn edge.

WHAT TO CHECK BEFORE BELIEVING THE WEIGHTS. A critic that has not learned
distance returns a number regardless, and the graph looks fine. So this prints
the correlation between predicted steps and straight-line metres, and the share
of edges whose predicted cost is NEGATIVE or clipped at the value floor/ceiling.
A critic trained with TQC_VALUE_FLOOR/CEILING saturates, and a saturated edge
weight is not a distance.

usage: weight_edges_v43.py MAPS.json TRACE.pkl --ckpt CKPT.pkl [--placer mad]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)


def spaces_from(trace, env_file):
    """Rebuild the agent's spaces without a simulator.

    The observation space comes from the recorded observations themselves and
    the action box from the RUN'S OWN SNAPSHOT, never a default: v43 removed the
    0.5/0.4 caps, and an agent built against the wrong box loads its weights
    happily and then scales every action.
    """
    from gymnasium.spaces import Box, Dict as DictSpace

    obs = trace["obs_driven"]
    od = {}
    for k, v in obs.items():
        a = np.asarray(v)
        od[k] = Box(low=-np.inf, high=np.inf, shape=a.shape[1:],
                    dtype=a.dtype if a.dtype == np.float32 else np.float32)
    env = {}
    for line in open(env_file):
        if "=" in line and not line.startswith("#"):
            k, _, v = line.strip().partition("=")
            env[k] = v
    need = ("MAX_STEER", "MAX_THROTTLE")
    for k in need:
        if k not in env:
            raise SystemExit(f"{env_file} has no {k}: refusing to guess the "
                             "action box an agent's weights were trained for")
    lo = np.array([-float(env["MAX_STEER"]),
                   float(env.get("MIN_THROTTLE", 0.0))], np.float32)
    hi = np.array([float(env["MAX_STEER"]), float(env["MAX_THROTTLE"])],
                  np.float32)
    print(f"[Spaces] action box from the snapshot: steer +/-{hi[0]:g} "
          f"throttle {lo[1]:g}-{hi[1]:g}")
    return DictSpace(od), Box(low=lo, high=hi, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps")
    ap.add_argument("trace")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-file", default="clean_fix1/results/run_env/"
                                          "carla_gated_v43_segvae_nocaps.env")
    ap.add_argument("--placers", default="mad,temporal,uniform")
    ap.add_argument("--step-cost", type=float, default=1.0,
                    help="THE ARM'S OWN. The gated arms train at 1.0 while "
                         "value_distance defaults to 0.1; the wrong one scales "
                         "every weight by ten without failing.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # Set before the imports that read os.environ at module scope.
    os.environ.setdefault("STEP_COST", str(a.step_cost))
    import train_carla_tqc as T
    from vae_nav.online_maps import build_edges
    from vae_nav.policy_io import load_agent_state
    from vae_nav.vae_goal_wrapper import VAEGoalWrapper
    from vae_nav.value_distance import edge_cost_steps, goal_relabelled_obs

    res = json.load(open(a.maps))
    with open(a.trace, "rb") as fh:
        tr = pickle.load(fh)
    obs_all = tr["obs_driven"]
    fields = tr["fields"]
    pos = np.asarray(tr["positions"], np.float64)[:, :2]
    seg = np.asarray(tr["segments"], int)

    obs_space, act_space = spaces_from(tr, a.env_file)
    agent = T.build_agent(obs_space, act_space, 0)
    with open(a.ckpt, "rb") as fh:
        state = pickle.load(fh)
    load_agent_state(agent, state.get("agent", state))

    # goal_block is a pure function of the pose fields -- bound with no instance
    # so this needs neither a wrapper instance nor a simulator.
    goal_block = lambda f: VAEGoalWrapper.goal_block(None, f)   # noqa: E731

    out = {"trace": res["trace"], "town": res["town"],
           "step_cost": a.step_cost, "ckpt": a.ckpt, "placers": {}}
    for placer in a.placers.split(","):
        m = res["maps"][placer]
        idx = np.asarray(m["indices"], int)
        edges, kinds = build_edges(idx, pos, seg)      # seglinks off
        if not edges:
            print(f"{placer}: no edges")
            continue
        batch = [goal_relabelled_obs({k: np.asarray(v)[idx[i]]
                                      for k, v in obs_all.items()},
                                     fields[idx[i]], pos[idx[j]], goal_block)
                 for i, j in edges]
        stacked = {k: np.stack([b[k] for b in batch]) for k in batch[0]}
        cost, spread = edge_cost_steps(agent, stacked, step_cost=a.step_cost)
        cost = np.asarray(cost, np.float64)
        euclid = np.array([np.linalg.norm(pos[idx[i]] - pos[idx[j]])
                           for i, j in edges])

        # The guard: a critic that never learned distance still returns numbers.
        r = float(np.corrcoef(cost, euclid)[0, 1]) if len(cost) > 2 else np.nan
        neg = float(np.mean(cost < 0))
        out["placers"][placer] = {
            "n_nodes": len(idx), "n_edges": len(edges),
            "n_components": _components(len(idx), edges),
            "weights_steps": cost.tolist(),
            "edges": [[int(i), int(j)] for i, j in edges],
            "euclid_m": euclid.tolist(),
            "spread": np.asarray(spread, np.float64).tolist(),
            "corr_steps_vs_metres": r,
            "frac_negative": neg,
            "median_steps": float(np.median(cost)),
            "median_m_per_step": float(np.median(euclid / np.maximum(cost, 1e-6))),
        }
        print(f"{placer:9s} {len(idx):4d} nodes {len(edges):4d} edges "
              f"{out['placers'][placer]['n_components']:3d} components | "
              f"steps median {np.median(cost):7.2f} "
              f"[{np.percentile(cost, 5):.1f}, {np.percentile(cost, 95):.1f}] "
              f"| corr(steps, metres) {r:+.3f} | negative {neg:.1%}")

    dest = a.out or a.maps.replace(".json", "_weighted.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"-> {dest}")


def _components(n, edges):
    """Connected components of the undirected graph, by union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(i) for i in range(n)})


if __name__ == "__main__":
    main()

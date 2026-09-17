"""Combine a town's short and long episode JSONs into the driving bundle (user 2026-09-15 Town07 repeat).

Same layout as results/episodes/town02_speed10_s7_episodes_short20_long20_seed0.pkl: {meta, short, long}, where
short/long are the sampler JSONs verbatim (plot_trace_episodes.py, sample_long_horizon_episodes.py). Also writes
NoMaD's flat episode list (results/nomad/episodes_short20_long20_seed0.json layout: set, index, env_seed, goal_kind,
route_m, start/goal pose). Short episodes are on-graph by construction (goal = a trace candidate).
"""
import argparse, json, pickle, time


def nomad_episodes(bundle):
    out = []
    for s in ("short", "long"):
        for e in bundle[s]["episodes"]:
            out.append({"set": s, "index": int(e["index"]), "env_seed": int(e["env_seed"]),
                        "goal_kind": e.get("goal_kind", "on_graph"), "route_m": e.get("route_m"),
                        "start_xy": e["start_xy"], "start_yaw_rad": e["start_yaw_rad"],
                        "goal_xy": e["goal_xy"], "goal_yaw_rad": e["goal_yaw_rad"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True)
    ap.add_argument("--long", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--town", required=True)
    ap.add_argument("--out", required=True, help="bundle .pkl")
    ap.add_argument("--nomad-out", required=True, help="NoMaD episodes .json")
    a = ap.parse_args()
    short, long_ = json.load(open(a.short)), json.load(open(a.long))
    bundle = {"meta": {"trace": a.trace, "town": a.town, "created_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                       "sources": {"short": a.short, "long": a.long},
                       "note": "short = harness sampler (straight-line 20-50 m); long = route 150-400 m, 15 on-graph + "
                               "5 off-graph goals (off-graph goal_trace_index is None -> position-only goal)"},
              "short": short, "long": long_}
    pickle.dump(bundle, open(a.out, "wb"))
    eps = nomad_episodes(bundle)
    json.dump({"source": a.out, "episodes": eps}, open(a.nomad_out, "w"), indent=1)
    print(f"[episodes] {a.town}: {len(short['episodes'])} short + {len(long_['episodes'])} long -> {a.out}; "
          f"NoMaD list {len(eps)} -> {a.nomad_out}")


if __name__ == "__main__":
    main()

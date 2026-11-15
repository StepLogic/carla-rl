#!/usr/bin/env python3
"""Regenerate every CARLA table of the image-navigation paper from the bench eval.json files.

Nothing here is transcribed: each number is recomputed from per-episode records, and every
cell is checked (one eval.json per cell, n, town, checkpoint, map build, episode file, paired
seeds, arrival rule) before it is used. Run:

    python make_carla_tables.py --cluster-root <rsync copy of ~/carla-rl/checkpoints>

Blocks
  A  Town07, ALIGNED navigator (EVAL_MAP_NEXT_ADVANCE=1, 2026-11-15), z25 maps in all bands:
       c0 = il_bc_c0/step_15000 (label na25, AR=2), SAC-from-c0 4-town 100k (sacna25, AR=2),
       ViNT dagger3_e1 (vintd3_nx_ar1, AR=1).
  B  ORIGINAL navigator (current node must win), pooled Town02 + Town03 + Town07 -- the cells
     of the 2026-11-14 main table, plus a z25-consistent variant of the easy row.
  K  Key-only z25 maps with ground-truth pose (pose arm; no image navigator), pooled over the
     three towns, with the no-map arm of the same episodes as reference.

Statistics: paired exact McNemar (two-sided binomial on discordant pairs) of each map arm
against the same policy's no-map arm on the same episodes; Holm across the map arms of one
policy within one band. Pooled cells pair by (town, seed).
"""
import argparse
import glob
import json
import os
import re
import sys
from math import comb, sqrt

WT = "/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla/results"
LOCAL = "/home/robotlab/carla-sac-local/checkpoints"
LOGS = "/home/robotlab/carla-sac-local/logs"
VINT_NX = f"{WT}/vint_town07_nx/checkpoints"
VINT_T07 = f"{WT}/vint_town07/checkpoints"
BANDS = ["easy", "medium", "turns"]
BAND_NAME = {"easy": "Easy", "medium": "Medium", "turns": "Hard"}
PRETTY = {"mapless": "None", "uniform": "Uniform", "entropy": "Entropy",
          "sgm_matched": "SGM", "vinttopo": "ViNT map"}


# ----------------------------------------------------------------------------- statistics
def mcnemar_exact(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    m, out, run = len(ps), [0.0] * len(ps), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * ps[i]))
        out[i] = run
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ----------------------------------------------------------------------------- loading
CHECKS = []


def check(cond, msg):
    CHECKS.append(("OK " if cond else "FAIL ") + msg)
    if not cond:
        print("!!! CHECK FAILED: " + msg, file=sys.stderr)


def load_cell(root, run, town):
    hits = glob.glob(os.path.join(root, run, "eval_*", "eval.json"))
    check(len(hits) == 1, f"{run}: exactly one eval.json (found {len(hits)})")
    if not hits:
        raise SystemExit(f"missing cell {root}/{run}")
    d = json.load(open(hits[0]))
    eps = d["episodes_detail"]
    check(d["episodes"] == len(eps) == 100, f"{run}: n=100 (episodes={d['episodes']}, detail={len(eps)})")
    check(list(d["towns"]) == [town], f"{run}: towns == [{town}] (got {d['towns']})")
    seeds = [e["seed"] for e in eps]
    check(len(set(seeds)) == len(seeds), f"{run}: seeds unique")
    sr = sum(bool(e["reached"]) for e in eps) / len(eps)
    spl = sum(float(e["spl"]) for e in eps) / len(eps)
    check(abs(sr - d["success_rate"]) < 1e-9, f"{run}: SR from episodes == summary ({sr} vs {d['success_rate']})")
    check(abs(spl - d["spl"]) < 1e-6, f"{run}: SPL from episodes == summary ({spl:.4f} vs {d['spl']:.4f})")
    fast = sum(1 for e in eps if e["reached"] and (e.get("min_speed_inside_kmh") or 0) > 10.0)
    return dict(run=run, town=town, path=hits[0], ckpt=str(d.get("checkpoint")),
                map_dir=str(d.get("map_dir")), episodes_file=str(d.get("episodes_file")),
                n=len(eps), k=sum(bool(e["reached"]) for e in eps),
                spl_sum=sum(float(e["spl"]) for e in eps),
                by=dict(((town, e["seed"]), bool(e["reached"])) for e in eps),
                xy=dict(((town, e["seed"]), (tuple(e["start_xy"]), tuple(e["goal_xy"]))) for e in eps),
                fast_reached=fast,
                ends=d.get("ends", {}))


def pool(cells):
    out = dict(n=0, k=0, spl_sum=0.0, by={}, xy={}, fast_reached=0, parts=cells)
    for c in cells:
        out["n"] += c["n"]
        out["k"] += c["k"]
        out["spl_sum"] += c["spl_sum"]
        out["fast_reached"] += c["fast_reached"]
        assert not (set(out["by"]) & set(c["by"])), "overlapping (town, seed) keys in a pool"
        out["by"].update(c["by"])
        out["xy"].update(c["xy"])
    out["sr"] = out["k"] / out["n"]
    out["spl"] = out["spl_sum"] / out["n"]
    return out


def compare(base, cells):
    """McNemar of each map arm against base, Holm across the arms given. Returns dict."""
    raw, keys = [], []
    for arm, c in cells.items():
        shared = sorted(set(base["by"]) & set(c["by"]))
        check(len(shared) == base["n"] == c["n"],
              f"paired episodes {arm}: shared {len(shared)} of {base['n']}/{c['n']}")
        om = sum(1 for s in shared if c["by"][s] and not base["by"][s])
        on = sum(1 for s in shared if base["by"][s] and not c["by"][s])
        raw.append(mcnemar_exact(om, on))
        keys.append((arm, om, on))
    out = {}
    for (arm, om, on), p, ph in zip(keys, raw, holm(raw)):
        out[arm] = dict(only_map=om, only_none=on, p=p, p_holm=ph)
    return out


def same_routes(cells, what):
    ref = None
    for c in cells:
        if ref is None:
            ref = c["xy"]
            continue
        same = set(ref) == set(c["xy"]) and all(
            max(abs(a - b) for a, b in zip(ref[s][0] + ref[s][1], c["xy"][s][0] + c["xy"][s][1])) < 0.05
            for s in ref)
        check(same, f"{what}: identical start/goal per seed across cells")


def map_line(log):
    if not os.path.exists(log):
        return "LOG MISSING"
    for line in open(log, errors="replace"):
        if line.startswith("[Map]"):
            return line.strip()
    return "no [Map] line"


def fmt(x):
    return f"{x:.2f}"


# ----------------------------------------------------------------------------- block A
def block_a(report):
    """Town07, aligned navigator, three policies."""
    pol = {
        "c0": dict(root=LOCAL, label="na25", arms=["mapless", "uniform", "entropy", "sgm_matched"],
                   log=lambda b, a: f"{LOGS}/na25_{b}_{a}.log", ar=2, leg="75"),
        "sac": dict(root=LOCAL, label="sacna25", arms=["mapless", "uniform", "entropy", "sgm_matched"],
                    log=lambda b, a: f"{LOGS}/sacna25_{b}_{a}.log", ar=2, leg="75"),
        "vint": dict(root=VINT_NX, label="vintd3_nx_ar1", arms=["mapless", "uniform", "vinttopo"],
                     log=lambda b, a: f"{WT}/vint_town07_nx/logs/bench_vintd3_nx_ar1_{a}_Town07_{b}.log",
                     ar=1, leg="150"),
    }
    res = {}
    for p, cfg in pol.items():
        for band in BANDS:
            cells = {}
            for arm in cfg["arms"]:
                run = f"bench_{cfg['label']}_{arm}_Town07_{band}"
                if not glob.glob(os.path.join(cfg["root"], run, "eval_*", "eval.json")):
                    res[(p, band)] = None
                    report.append(f"[A] {p} {band}: {run} NOT YET AVAILABLE")
                    break
                c = load_cell(cfg["root"], run, "Town07")
                cells[arm] = c
                if arm not in ("mapless", "vinttopo"):   # ViNT's own topomap arm has no image navigator
                    ml = map_line(cfg["log"](band, arm))
                    check("next-node advance ON" in ml, f"{run}: [Map] says next-node advance ON")
                    check(f"leg timeout {cfg['leg']}," in ml, f"{run}: [Map] leg timeout {cfg['leg']}")
                    if True:
                        check(c["map_dir"].rstrip("/").endswith(f"maps_t07_z25/{arm}"),
                              f"{run}: map_dir is maps_t07_z25/{arm} ({c['map_dir']})")
            else:
                ck = {c["ckpt"] for c in cells.values()}
                check(len(ck) == 1, f"[A] {p} {band}: one checkpoint ({ck})")
                ef = {os.path.basename(c["episodes_file"]) for c in cells.values()}
                check(len(ef) == 1, f"[A] {p} {band}: one episode file ({ef})")
                st = compare(cells["mapless"], {a: c for a, c in cells.items() if a != "mapless"})
                res[(p, band)] = dict(cells=cells, stats=st, ckpt=ck.pop(), episodes=ef.pop())
    # cross-policy pairing: same routes for the same seeds
    for band in BANDS:
        avail = [res[(p, band)]["cells"]["mapless"] for p in pol if res.get((p, band))]
        same_routes(avail, f"[A] {band} mapless arms of all policies")
    return pol, res


# ----------------------------------------------------------------------------- block B
def block_b(cl, report):
    """Original navigator, pooled Town02 + Town03 + Town07 (the 2026-11-14 main table)."""
    c0 = {
        "easy": dict(T02T03=lambda a: "bench_fresh_any_mapless" if a == "mapless" else f"bench_fresh_imageany_{a}",
                     T07=lambda a: f"bench_imagez25_{a}"),
        "easy_z25": dict(T02T03=lambda a: "bench_fresh_any_mapless" if a == "mapless" else f"bench_imagez25_{a}",
                         T07=lambda a: f"bench_imagez25_{a}"),
        "medium": dict(T02T03=lambda a: f"bench_medturn_{a}", T07=lambda a: f"bench_imagez25_{a}"),
        "turns": dict(T02T03=lambda a: "bench_turns_mapless" if a == "mapless" else f"bench_turnsz25_{a}",
                      T07=lambda a: f"bench_t07_{a}"),
    }
    vint = {
        "easy": lambda a: f"bench_vintd3_fresh_any_{a}",
        "easy_z25uni": lambda a: "bench_vintd3_z25_uniform" if a == "uniform" else f"bench_vintd3_fresh_any_{a}",
        "medium": lambda a: f"bench_vintd3_medturn_{a}",
        "turns": lambda a: f"bench_vintd3_turns_{a}",
        "turns_z25uni": lambda a: "bench_vintd3_z25_uniform" if a == "uniform" else f"bench_vintd3_turns_{a}",
    }
    res = {}
    for key, fam in c0.items():
        band = "easy" if key.startswith("easy") else key
        arms = ["mapless", "uniform", "entropy", "sgm_matched"]
        pooled, per_town = {}, {}
        for arm in arms:
            parts = []
            for town in ("Town02", "Town03"):
                parts.append(load_cell(cl, f"{fam['T02T03'](arm)}_{town}_{band}", town))
            parts.append(load_cell(LOCAL, f"{fam['T07'](arm)}_Town07_{band}", "Town07"))
            per_town[arm] = {c["town"]: c for c in parts}
            pooled[arm] = pool(parts)
            ck = {c["ckpt"].split("/")[-2] + "/" + c["ckpt"].split("/")[-1] for c in parts}
            check(ck == {"il_bc_c0/step_15000"}, f"[B] c0 {key} {arm}: checkpoint c0 ({ck})")
        same_routes(list(pooled.values()), f"[B] c0 {key} all arms")
        st = compare(pooled["mapless"], {a: pooled[a] for a in arms if a != "mapless"})
        pt = {}
        for town in ("Town02", "Town03", "Town07"):
            pt[town] = compare(per_town["mapless"][town],
                               {a: per_town[a][town] for a in arms if a != "mapless"})
        res[("c0", key)] = dict(pooled=pooled, stats=st, per_town=per_town, per_town_stats=pt)
    for key, fam in vint.items():
        band = key.split("_")[0]
        arms = ["mapless", "uniform", "vinttopo"]
        pooled, per_town = {}, {}
        for arm in arms:
            parts = []
            for town in ("Town02", "Town03"):
                parts.append(load_cell(cl, f"{fam(arm)}_{town}_{band}", town))
            parts.append(load_cell(VINT_T07, f"bench_vintd3_t07_{arm}_Town07_{band}", "Town07"))
            per_town[arm] = {c["town"]: c for c in parts}
            pooled[arm] = pool(parts)
        same_routes(list(pooled.values()), f"[B] vint {key} all arms")
        st = compare(pooled["mapless"], {a: pooled[a] for a in arms if a != "mapless"})
        pt = {}
        for town in ("Town02", "Town03", "Town07"):
            pt[town] = compare(per_town["mapless"][town],
                               {a: per_town[a][town] for a in arms if a != "mapless"})
        res[("vint", key)] = dict(pooled=pooled, stats=st, per_town=per_town, per_town_stats=pt)
    # wrong goal, hard, c0 (no map): Town02/Town03 cluster, Town07 local
    wrong = {}
    for town in ("Town02", "Town03"):
        wrong[town] = load_cell(cl, f"bench_turnswrong_mapless_{town}_turns", town)
    wrong["Town07"] = load_cell(LOCAL, "bench_t07wrong_mapless_Town07_turns", "Town07")
    res["wrong"] = wrong
    return res


# ----------------------------------------------------------------------------- block K
def block_k(cl, res_b, report):
    """Key-only z25 maps, ground-truth pose, pooled over the three towns."""
    fam = {"easy": "bench_oraclez25_{a}", "medium": "bench_medturnoracle_{a}", "turns": "bench_oraclez25_{a}"}
    res = {}
    for band in BANDS:
        cells = {}
        for arm in ("uniform", "entropy", "sgm_matched"):
            parts = []
            for town in ("Town02", "Town03"):
                parts.append(load_cell(cl, f"{fam[band].format(a=arm)}_{town}_{band}", town))
            parts.append(load_cell(LOCAL, f"bench_oraclez25_{arm}_Town07_{band}", "Town07"))
            for c in parts:
                check("keys" in c["map_dir"] or "key" in c["map_dir"].lower() or "oracle" in c["run"],
                      f"[K] {c['run']}: key-only map_dir ({c['map_dir']})")
            cells[arm] = pool(parts)
            cells[arm]["per_town"] = {c["town"]: c for c in parts}
        base = res_b[("c0", "easy" if band == "easy" else band)]["pooled"]["mapless"]
        same_routes([base] + list(cells.values()), f"[K] {band} key-only arms vs no-map reference")
        st = compare(base, cells)
        res[band] = dict(cells=cells, base=base, stats=st)
    return res


# ----------------------------------------------------------------------------- LaTeX
def mark(s):
    """* = map better than None, dagger = worse, both at Holm-adjusted p < 0.05."""
    if not s or s["p_holm"] >= 0.05:
        return ""
    return "$^{*}$" if s["only_map"] > s["only_none"] else "$^{\\dagger}$"


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


JUNCTION_TABLE = r"""% Junction counts carried from the mapping session's t07/tab_junction_nodes.tex (2026-11-13; junction
% membership from logs/junctions_<town>.xodr). Key counts and uniform spacings re-read 2026-11-15 from the
% map metadata: Town02 96 keys / 26.75 m, Town03 323 / 21.6 m, Town07 193 / 30.15 m (t07_build.log:
% "entropy MAD 326 -> 193 image-distinct; uniform 193 (spacing 30.15 m); random 193", 9093 candidates).
\begin{table}[H]
\centering
\caption{Key nodes inside a junction. Uniform, SGM and Random are matched to entropy's key count
(96 in Town02, 323 in Town03, 193 in Town07); the matched uniform spacing is 26.8\,m, 21.6\,m and 30.2\,m.
\emph{Trace}: all candidate frames of the exploration drive.}
\label{tab:junction_nodes}
\begin{tabular}{lrrr}
\toprule
Placement & Town02 & Town03 & Town07 \\
\midrule
Trace   & 894/3664 (24.4\%) & 6454/16242 (39.7\%) & 3004/9093 (33.0\%) \\
\midrule
Entropy & 45/96 (46.9\%) & 175/323 (54.2\%) & 96/193 (49.7\%) \\
Uniform & 27/96 (28.1\%) & 110/323 (34.1\%) & 72/193 (37.3\%) \\
SGM     & 21/96 (21.9\%) & 115/323 (35.6\%) & 46/193 (23.8\%) \\
Random  & 20/96 (20.8\%) & 133/323 (41.2\%) & 67/193 (34.7\%) \\
\bottomrule
\end{tabular}
\end{table}
"""


def emit_tex(pol, A, B, K, path, script_rev):
    L = []
    w = L.append
    nfail = sum(1 for c in CHECKS if c.startswith("FAIL"))
    w("% " + "=" * 98)
    w("% CARLA TABLES of the image-navigation paper. GENERATED -- do not edit numbers by hand.")
    w(f"% Generator: results/paper_tables_2026-11-15/make_carla_tables.py ({script_rev}),")
    w("% branch vint-rerun-2026-11-15 of the vint-carla worktree. Every number is recomputed from the")
    w(f"% per-episode records of the bench eval.json files; {len(CHECKS)} automated checks, {nfail} failed")
    w("% (one eval.json per cell, n=100, town, checkpoint, map build, episode file, paired seeds and")
    w("% identical start/goal across arms and policies, SR/SPL equal to the file's own summary).")
    w("% Cell list with paths: cells_used.json next to this file.")
    w("% " + "=" * 98)
    w("")

    # ---------------------------------------------------------------- T1 junction nodes
    w(JUNCTION_TABLE)

    # ---------------------------------------------------------------- T2 main (block A)
    w("% MAIN TABLE -- Town07, ALIGNED navigator (EVAL_MAP_NEXT_ADVANCE=1), z25 maps in all bands, any")
    w("% speed, LANE_UNKNOWN_MARKS_FATAL unset (=1), CARLA 0.9.15 local, episode sets")
    w("% bench_t07_fresh_episodes.json (easy, medium) and bench_t07_turns_episodes.json (hard).")
    for p in pol:
        r0 = A.get((p, "easy"))
        w(f"%   {p:5s} label {pol[p]['label']:15s} AR={pol[p]['ar']}  image leg {pol[p]['leg']} decisions"
          f" = 150 ticks  ckpt {r0['ckpt'] if r0 else '?'}")
    w("% SAC = frl_c0actor_priv_4towns_lc01_miss_2m (trained on Town05,Town01,Town04,Town10HD per its")
    w("% training log; Town07 held out), checkpoint step_100000: an intermediate snapshot of a 2M-step run.")
    w("% Rates differ by block (c0/SAC AR=2, ViNT AR=1): the claims are within-block map effects only.")
    w("% Runners: carla-sac-local/t07/run_t07_na_ar2.sh (c0), run_t07_sacna.sh (SAC),")
    w("%          vint-carla results/vint_town07_nx/run_vint_t07_nx.sh (ViNT).")
    w(r"\begin{table}[H]")
    w(r"\centering")
    w(r"\footnotesize")
    w(r"\setlength{\tabcolsep}{3.5pt}")
    w(r"\caption{\textbf{Image-goal navigation in CARLA~\cite{dosovitskiy2017carla}}, held-out Town07, "
      r"100 episodes per cell; every policy and map runs the same episodes. Success: reaching within 1\,m "
      r"of the goal, at any speed; a lane invasion reported by the simulator ends the episode. "
      r"Easy: 10--25\,m. Medium: 25--50\,m with at least one junction turn. Hard: 50--149\,m with at least "
      r"two junction turns. Uniform, Entropy and SGM maps hold 193 key nodes each plus 614--653 image gap "
      r"nodes; \emph{ViNT map} is ViNT's own topological graph, built by its own procedure from the Town07 "
      r"expert tour. "
      r"The high-level policy advances to the next path node when that node wins the image match, as in "
      r"Habitat. SAC is initialised from the BC policy and trained for 100k steps in Town01, Town04, "
      r"Town05 and Town10HD. The BC and SAC policies act every second simulator tick and ViNT every tick, "
      r"and every subgoal leg gets 150 ticks, so maps are compared within a policy, not absolute numbers "
      r"across policies. $^{*}$/$^{\dagger}$: better/worse than \emph{None} of the same "
      r"policy, $p<0.05$, paired exact McNemar, Holm-corrected across that policy's maps within each band.}")
    w(r"\label{tab:map_navigation}")
    w(r"\begin{tabular}{lcccccc}")
    w(r"\toprule")
    w(r" & \multicolumn{2}{c}{Easy} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Hard} \\")
    w(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    w(r"Map & SR & SPL & SR & SPL & SR & SPL \\")
    head = {"c0": r"\textit{BC policy}", "sac": r"\textit{SAC fine-tuned from the BC policy}",
            "vint": r"\textit{ViNT}~\cite{shah2023vint}"}
    for p in pol:
        w(r"\midrule")
        w(r"\multicolumn{7}{l}{" + head[p] + r"} \\")
        for arm in pol[p]["arms"]:
            cells = []
            for band in BANDS:
                r = A.get((p, band))
                if r is None:
                    cells += ["--", "--"]
                    continue
                c = r["cells"][arm]
                cells += [fmt(c["k"] / c["n"]) + mark(r["stats"].get(arm)), fmt(c["spl_sum"] / c["n"])]
            w(f"{PRETTY[arm]} & " + " & ".join(cells) + r" \\")
    w(r"\bottomrule")
    w(r"\end{tabular}")
    w(r"\end{table}")
    w("")

    # ---------------------------------------------------------------- T3 long horizon (block A hard)
    w("% LONG-HORIZON TABLE -- the hard band of the main table with the discordant pairs behind each test.")
    w(r"\begin{table}[H]")
    w(r"\centering")
    w(r"\footnotesize")
    w(r"\setlength{\tabcolsep}{4pt}")
    w(r"\caption{\textbf{Long-horizon navigation in CARLA~\cite{dosovitskiy2017carla}}: the hard band of "
      r"Table~\ref{tab:map_navigation} (held-out Town07, routes of 50--149\,m with at least two junction "
      r"turns, 100 episodes per cell, the same episodes in every row). \emph{Map only} / \emph{None only}: "
      r"episodes that succeed with the map and fail without it, and the reverse. $p$: paired exact McNemar "
      r"against \emph{None}, Holm-corrected across that policy's maps.}")
    w(r"\label{tab:long_horizon_carla}")
    w(r"\begin{tabular}{@{}lrrrrr@{}}")
    w(r"\toprule")
    w(r"Map & SR & SPL & Map only & None only & $p$ \\")
    for p in pol:
        r = A.get((p, "turns"))
        w(r"\midrule")
        w(r"\multicolumn{6}{@{}l}{" + head[p] + r"} \\")
        for arm in pol[p]["arms"]:
            if r is None:
                w(f"{PRETTY[arm]} & -- & -- & -- & -- & -- \\\\")
                continue
            c = r["cells"][arm]
            s = r["stats"].get(arm)
            tail = " & -- & -- & --" if s is None else f" & {s['only_map']} & {s['only_none']} & {pfmt(s['p_holm'])}"
            w(f"{PRETTY[arm]} & {fmt(c['k'] / c['n'])} & {fmt(c['spl_sum'] / c['n'])}{tail} \\\\")
    w(r"\bottomrule")
    w(r"\end{tabular}")
    w(r"\end{table}")
    w("")

    # ---------------------------------------------------------------- T4 key-only, true pose (block K)
    w("% KEY-ONLY MAPS, GROUND-TRUTH POSE -- c0 at ACTION_REPEAT=1 (all cells before 2026-11-15), any speed.")
    w("%   key-only maps: cluster maps_expert_jx_keys_z25 (Town02 97 keys, Town03 319), local")
    w("%   t07/maps_t07_keys_z25 (Town07 193). Cells: bench_oraclez25_* (easy, hard; Town07 all bands),")
    w("%   bench_medturnoracle_* (Town02/Town03 medium = >=1 junction turn). Reference: the no-map cells of")
    w("%   the same episodes (bench_fresh_any_mapless / bench_medturn_mapless / bench_turns_mapless,")
    w("%   Town07 bench_imagez25_mapless / bench_t07_mapless). The pose arm uses no image navigator, so")
    w("%   the subgoal-advance rule does not apply. Largest SCC recomputed 2026-11-15 from the map edges.")
    w(r"\begin{table}[H]")
    w(r"\centering")
    w(r"\footnotesize")
    w(r"\setlength{\tabcolsep}{3.5pt}")
    w(r"\caption{\textbf{Key nodes alone, localised by ground-truth pose}, in CARLA, pooled over Town02, "
      r"Town03 and Town07 (100 episodes per town, $n{=}300$ per cell); BC policy acting every simulator "
      r"tick. The maps hold only the key nodes (97, 319 and 193; no gap nodes), and the agent and goal are "
      r"placed on the graph by their true pose. \emph{No map} is the same policy on the same episodes. "
      r"Without gap nodes the graphs fragment: the largest strongly connected component covers 0.93, 0.93 "
      r"and 0.82 of the uniform graph, 0.69, 0.72 and 0.17 of the entropy graph and 0.49, 0.22 and 0.23 of "
      r"the SGM graph (Town02, Town03, Town07), against 1.00 for the full maps. Bands and success as in "
      r"Table~\ref{tab:map_navigation}. $^{*}$/$^{\dagger}$: better/worse than \emph{No map}, $p<0.05$, "
      r"paired exact McNemar, Holm-corrected across the three maps within each band.}")
    w(r"\label{tab:map_navigation_pose}")
    w(r"\begin{tabular}{lcccccc}")
    w(r"\toprule")
    w(r" & \multicolumn{2}{c}{Easy} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Hard} \\")
    w(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    w(r"Map (key nodes only) & SR & SPL & SR & SPL & SR & SPL \\")
    w(r"\midrule")
    w("No map & " + " & ".join(f"{fmt(K[b]['base']['sr'])} & {fmt(K[b]['base']['spl'])}" for b in BANDS) + r" \\")
    w(r"\midrule")
    for arm in ("uniform", "entropy", "sgm_matched"):
        w(f"{PRETTY[arm]} & " + " & ".join(
            f"{fmt(K[b]['cells'][arm]['sr'])}{mark(K[b]['stats'][arm])} & {fmt(K[b]['cells'][arm]['spl'])}"
            for b in BANDS) + r" \\")
    w(r"\bottomrule")
    w(r"\end{tabular}")
    w(r"\end{table}")
    w("")

    # ---------------------------------------------------------------- T5 appendix: original navigator, pooled
    w("% " + "-" * 98)
    w("% APPENDIX / ALTERNATIVE -- the EARLIER protocol (original subgoal rule: the CURRENT node had to win),")
    w("% c0 and ViNT at ACTION_REPEAT=1, pooled Town02 + Town03 + Town07, any speed, crossings fatal.")
    w("% z25-CONSISTENT: every map cell is on the z=2.5 maps the paper describes. This differs from the")
    w("% 2026-11-14 tab_map_navigation_main.tex in two places, both of which used the DENSER")
    w("% maps_expert_jx_img build (255/1104 key nodes) for Town02/Town03:")
    w("%   c0 easy maps: bench_imagez25_* here (z25) instead of bench_fresh_imageany_* (jx_img);")
    w("%   ViNT uniform easy + hard: bench_vintd3_z25_uniform here instead of vintd3_fresh_any_uniform /")
    w("%   vintd3_turns_uniform (jx_img). Medium cells (medturn) were already z25 for both policies.")
    c0b = {"easy": B[("c0", "easy_z25")], "medium": B[("c0", "medium")], "turns": B[("c0", "turns")]}
    vb = {"easy": B[("vint", "easy_z25uni")], "medium": B[("vint", "medium")], "turns": B[("vint", "turns_z25uni")]}
    for name, blk in (("c0", c0b), ("vint", vb)):
        for band in BANDS:
            for arm, c in blk[band]["pooled"].items():
                w(f"%   {name:4s} {band:6s} {PRETTY[arm]:<9} " + ", ".join(x["run"] for x in c["parts"]))
    w(r"\begin{table}[H]")
    w(r"\centering")
    w(r"\footnotesize")
    w(r"\setlength{\tabcolsep}{3.5pt}")
    w(r"\caption{\textbf{Earlier protocol, pooled over three held-out towns} (Town02, Town03, Town07; 100 "
      r"episodes each, $n{=}300$ per cell). Bands, success rule and maps as in Table~\ref{tab:map_navigation}"
      r" (96, 323 and 193 key nodes plus 300--875 image gap nodes), but the high-level policy advanced only "
      r"when the \emph{current} path node won the image match, both policies acted every simulator tick, "
      r"and Town02 and Town03 ran on CARLA 0.9.16 (Town07 on 0.9.15). $^{*}$/$^{\dagger}$ as in "
      r"Table~\ref{tab:map_navigation}.}")
    w(r"\label{tab:map_navigation_orignav}")
    w(r"\begin{tabular}{lcccccc}")
    w(r"\toprule")
    w(r" & \multicolumn{2}{c}{Easy} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Hard} \\")
    w(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    w(r"Map & SR & SPL & SR & SPL & SR & SPL \\")
    for name, blk, arms in (("c0", c0b, ["mapless", "uniform", "entropy", "sgm_matched"]),
                            ("vint", vb, ["mapless", "uniform", "vinttopo"])):
        w(r"\midrule")
        w(r"\multicolumn{7}{l}{" + head[name] + r"} \\")
        for arm in arms:
            w(f"{PRETTY[arm]} & " + " & ".join(
                f"{fmt(blk[b]['pooled'][arm]['sr'])}{mark(blk[b]['stats'].get(arm))} & "
                f"{fmt(blk[b]['pooled'][arm]['spl'])}" for b in BANDS) + r" \\")
    w(r"\bottomrule")
    w(r"\end{tabular}")
    w(r"\end{table}")
    w("")

    # ---------------------------------------------------------------- T6 appendix: original navigator, hard by town
    w("% APPENDIX / ALTERNATIVE -- hard band of the earlier protocol, by town (same cells as above).")
    w("%   wrong goal: bench_turnswrong_mapless_Town02/03_turns (cluster), bench_t07wrong_mapless_Town07_turns.")
    towns = ("Town02", "Town03", "Town07")
    w(r"\begin{table}[H]")
    w(r"\centering")
    w(r"\footnotesize")
    w(r"\setlength{\tabcolsep}{4pt}")
    w(r"\caption{\textbf{Earlier protocol, long-horizon routes by town}: the hard band of "
      r"Table~\ref{tab:map_navigation_orignav} (50--149\,m, at least two junction turns, $n{=}100$ per cell). "
      r"\emph{No Map, wrong goal}: the BC policy on the same episodes given a goal image from elsewhere in "
      r"the town, i.e.\ what lane following alone reaches. $^{*}$/$^{\dagger}$ as in "
      r"Table~\ref{tab:map_navigation}, tested within each town.}")
    w(r"\label{tab:long_horizon_carla_orignav}")
    w(r"\begin{tabular}{@{}lrrrrrr@{}}")
    w(r"\toprule")
    w(r"& \multicolumn{2}{c}{Town02} & \multicolumn{2}{c}{Town03} & \multicolumn{2}{c}{Town07} \\")
    w(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    w(r"Map & SR & SPL & SR & SPL & SR & SPL \\")
    w(r"\midrule")
    w(r"\multicolumn{7}{@{}l}{" + head["c0"] + r"} \\")
    hb = c0b["turns"]
    for arm in ["mapless", "WRONG", "uniform", "entropy", "sgm_matched"]:
        if arm == "WRONG":
            w("No Map, wrong goal & " + " & ".join(
                f"{fmt(B['wrong'][t]['k'] / B['wrong'][t]['n'])} & {fmt(B['wrong'][t]['spl_sum'] / B['wrong'][t]['n'])}"
                for t in towns) + r" \\")
            continue
        w(f"{PRETTY[arm]} & " + " & ".join(
            f"{fmt(hb['per_town'][arm][t]['k'] / 100)}{mark(hb['per_town_stats'][t].get(arm))} & "
            f"{fmt(hb['per_town'][arm][t]['spl_sum'] / 100)}" for t in towns) + r" \\")
    w(r"\midrule")
    w(r"\multicolumn{7}{@{}l}{" + head["vint"] + r"} \\")
    hv = vb["turns"]
    for arm in ["mapless", "uniform", "vinttopo"]:
        w(f"{PRETTY[arm]} & " + " & ".join(
            f"{fmt(hv['per_town'][arm][t]['k'] / 100)}{mark(hv['per_town_stats'][t].get(arm))} & "
            f"{fmt(hv['per_town'][arm][t]['spl_sum'] / 100)}" for t in towns) + r" \\")
    w(r"\bottomrule")
    w(r"\end{tabular}")
    w(r"\end{table}")
    w("")
    open(path, "w").write("\n".join(L))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-root", required=True,
                    help="rsync copy of ~/carla-rl/checkpoints on the cluster (eval.json only)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tex-out", default=None)
    ap.add_argument("--rev", default="uncommitted")
    a = ap.parse_args()
    report = []

    pol, A = block_a(report)
    B = block_b(a.cluster_root, report)
    K = block_k(a.cluster_root, B, report)

    print("=" * 100)
    print("BLOCK A  Town07, aligned navigator (next-node advance), z25 maps, any speed, crossings fatal")
    print("=" * 100)
    for p in pol:
        print(f"\n--- {p}  AR={pol[p]['ar']}  label={pol[p]['label']}")
        for band in BANDS:
            r = A.get((p, band))
            if r is None:
                print(f"  {band}: not available")
                continue
            print(f"  {band}  ckpt={r['ckpt']}  episodes={r['episodes']}")
            for arm, c in r["cells"].items():
                lo, hi = wilson(c["k"], c["n"])
                s = r["stats"].get(arm)
                extra = "" if s is None else (f"  map-only {s['only_map']:2d} none-only {s['only_none']:2d} "
                                              f"p {s['p']:.4f} pHolm {s['p_holm']:.4f}"
                                              + ("  *" if s["p_holm"] < 0.05 else ""))
                print(f"    {PRETTY[arm]:<9} SR {c['k']/c['n']:.3f} [{lo:.2f},{hi:.2f}] "
                      f"SPL {c['spl_sum']/c['n']:.3f}  fast-reached {c['fast_reached']:2d}{extra}")

    print("\n" + "=" * 100)
    print("BLOCK B  original navigator, pooled Town02+Town03+Town07 (n=300)")
    print("=" * 100)
    for key in [k for k in B if isinstance(k, tuple)]:
        r = B[key]
        print(f"\n--- {key}")
        for arm, c in r["pooled"].items():
            s = r["stats"].get(arm)
            maps = sorted({os.path.basename(os.path.dirname(x["map_dir"].rstrip("/"))) for x in c["parts"]})
            extra = "" if s is None else (f"  map-only {s['only_map']:2d} none-only {s['only_none']:2d} "
                                          f"p {s['p']:.4f} pHolm {s['p_holm']:.4f}"
                                          + ("  *" if s["p_holm"] < 0.05 else ""))
            towns = " ".join(f"{x['town'][-2:]}:{x['k']}/{x['n']}" for x in c["parts"])
            print(f"    {PRETTY[arm]:<9} SR {c['sr']:.3f} SPL {c['spl']:.3f}  [{towns}]  fast {c['fast_reached']:2d}"
                  f"  maps {maps}{extra}")
    print("\n    wrong goal (c0, hard, no map): " +
          "  ".join(f"{t}: SR {c['k']/c['n']:.2f} SPL {c['spl_sum']/c['n']:.2f}" for t, c in B["wrong"].items()))

    print("\n" + "=" * 100)
    print("BLOCK K  key-only z25 maps, ground-truth pose, pooled Town02+Town03+Town07")
    print("=" * 100)
    for band in BANDS:
        r = K[band]
        print(f"\n--- {band}  no map (reference) SR {r['base']['sr']:.3f} SPL {r['base']['spl']:.3f}")
        for arm, c in r["cells"].items():
            s = r["stats"][arm]
            towns = " ".join(f"{t[-2:]}:{x['k']}/{x['n']}" for t, x in c["per_town"].items())
            maps = sorted({x["map_dir"].rstrip("/").split("/")[-2] for x in c["per_town"].values()})
            print(f"    {PRETTY[arm]:<9} SR {c['sr']:.3f} SPL {c['spl']:.3f} [{towns}] fast {c['fast_reached']}"
                  f"  maps {maps}  map-only {s['only_map']} none-only {s['only_none']} pHolm {s['p_holm']:.4f}")

    print("\n" + "=" * 100)
    nfail = sum(1 for c in CHECKS if c.startswith("FAIL"))
    print(f"CHECKS: {len(CHECKS)} run, {nfail} failed")
    for c in CHECKS:
        if c.startswith("FAIL"):
            print("  " + c)
    for line in report:
        print(line)

    if a.json_out:
        root = os.path.abspath(a.cluster_root).rstrip("/") + "/"

        def short(p):
            p = str(p)
            for pre, tag in ((root, "cluster:carla-rl/checkpoints/"), (LOCAL + "/", "local:carla-sac-local/checkpoints/"),
                             (WT + "/", "vint-carla:results/")):
                if p.startswith(pre):
                    return tag + p[len(pre):]
            parts = p.rstrip("/").split("/")
            return "/".join(parts[-3:]) if len(parts) > 3 else p

        def strip(c):
            out = {k: v for k, v in c.items() if k in ("run", "town", "n", "k", "spl_sum", "fast_reached")}
            out.update(path=short(c["path"]), ckpt=short(c["ckpt"]), map_dir=short(c["map_dir"]),
                       episodes_file=os.path.basename(str(c["episodes_file"])))
            return out
        dump = {"A": {f"{p}|{b}": {arm: strip(c) for arm, c in r["cells"].items()} | {"stats": r["stats"]}
                      for (p, b), r in A.items() if r},
                "B": {f"{k[0]}|{k[1]}": {arm: [strip(x) for x in c["parts"]] for arm, c in r["pooled"].items()}
                      | {"stats": r["stats"]} for k, r in B.items() if isinstance(k, tuple)},
                "K": {b: {arm: [strip(x) for x in c["parts"]] for arm, c in r["cells"].items()}
                      | {"stats": r["stats"]} for b, r in K.items()},
                "checks": [re.sub(r"/home/[^/]+/carla-rl/", "cluster:carla-rl/", c.replace(root, "cluster:carla-rl/checkpoints/"))
                           for c in CHECKS]}
        json.dump(dump, open(a.json_out, "w"), indent=1, default=str)
    if a.tex_out:
        emit_tex(pol, A, B, K, a.tex_out, a.rev)


if __name__ == "__main__":
    main()

"""Success by GOAL DISTANCE, recovered from an eval run's per-step qtrace.

WHY THIS EXISTS. The eval JSON stores aggregates only -- success, steps_median,
end_reasons -- with no per-episode record, so the near-vs-far split cannot be read
off it. The qtrace sidecar written beside it (EVAL_Q_TRACE=1) IS per step and
carries `ep` and `remaining_m`, so each episode's initial route distance is its
t=0 remaining_m and its outcome is whether `arrived` ever fires. One wide-band
sweep therefore yields the whole breakdown; no band-restricted reruns are needed.

VALIDATED against v29@1M Town01 before use: the per-episode reconstruction gives
27/40 = 0.6750, exactly the `success` the JSON reports. The check is kept as
--verify and runs by default, because a silent mismatch would mean the bins are
measuring something other than the headline number.

The hypothesis this is built to test: at gamma=0.99 a +100 arrival bonus is worth
~0.004 from 1000 steps away, so far goals should degrade while near goals hold.
"""
import argparse, collections, glob, json, os, sys


def episodes_from_qtrace(path):
    """-> {ep: {"d0": initial route distance m, "success": bool, "steps": int,
              "end": 'arrived'|'terminated'|'truncated'|'unknown'}}"""
    rows = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["ep"]].append(r)
    out = {}
    for ep, rs in rows.items():
        rs.sort(key=lambda r: r["t"])
        d0 = rs[0].get("remaining_m")
        arrived = any(bool(r.get("arrived")) for r in rs)
        last = rs[-1]
        end = ("arrived" if arrived else
               "terminated" if last.get("term") else
               "truncated" if last.get("trunc") else "unknown")
        out[ep] = {"d0": d0, "success": arrived, "steps": len(rs), "end": end}
    return out


def verify(json_path, eps):
    """The reconstruction must reproduce the file's own headline success."""
    try:
        agg = json.load(open(json_path))
    except OSError:
        return None
    if "success" not in agg or not eps:
        return None
    got = sum(e["success"] for e in eps.values()) / len(eps)
    return {"reported": agg["success"], "derived": got,
            "match": abs(got - agg["success"]) < 1e-6,
            "episodes_reported": agg.get("episodes"), "episodes_found": len(eps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+",
                    help="qtrace .jsonl files, or eval result .json files (the sidecar is inferred)")
    ap.add_argument("--bins", default="5,20,50,100,200",
                    help="bin edges in metres (default the 5-200 m training band)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()

    edges = [float(x) for x in a.bins.split(",")]
    bands = list(zip(edges[:-1], edges[1:]))

    paths = []
    for p in a.traces:
        for q in sorted(glob.glob(p)):
            paths.append(q if q.endswith(".jsonl") else q + ".qtrace.jsonl")

    results = []
    for qp in paths:
        if not os.path.exists(qp):
            print(f"[skip] no qtrace: {qp}", file=sys.stderr)
            continue
        base = qp[:-len(".qtrace.jsonl")] if qp.endswith(".qtrace.jsonl") else qp
        eps = episodes_from_qtrace(qp)
        missing = [e for e, v in eps.items() if v["d0"] is None]
        v = None if a.no_verify else verify(base, eps)

        name = os.path.basename(base)
        n = len(eps)
        overall = sum(e["success"] for e in eps.values()) / n if n else float("nan")
        print(f"\n=== {name}   {n} episodes, overall success {overall:.3f}")
        if v:
            flag = "OK" if v["match"] else "*** MISMATCH ***"
            print(f"    verify: derived {v['derived']:.4f} vs reported {v['reported']:.4f}  {flag}")
            if not v["match"]:
                print("    REFUSING to trust the bins for this file.", file=sys.stderr)
                continue
        if missing:
            print(f"    WARNING: {len(missing)} episodes have no route distance and are excluded")

        row = {"file": name, "episodes": n, "success": overall, "bands": []}
        # arrived/failed, not arrived/term/trunc: the qtrace's final row often carries
        # neither term nor trunc, so a three-way split silently drops failures into an
        # unreported bucket and the columns stop summing to n. The full end tally is
        # still written to --out-json for anyone who wants it.
        print("     band          n   success   arrived/failed")
        for lo, hi in bands:
            ids = [e for e, x in eps.items() if x["d0"] is not None and lo <= x["d0"] < hi]
            if not ids:
                continue
            s = sum(eps[e]["success"] for e in ids) / len(ids)
            ends = collections.Counter(eps[e]["end"] for e in ids)
            n_arr = sum(eps[e]["success"] for e in ids)
            print("    %4.0f-%4.0f m  %3d    %.3f     %d/%d"
                  % (lo, hi, len(ids), s, n_arr, len(ids) - n_arr))
            row["bands"].append({"lo": lo, "hi": hi, "n": len(ids), "success": s,
                                 "arrived": ends["arrived"], "terminated": ends["terminated"],
                                 "truncated": ends["truncated"]})
        results.append(row)

    if a.out_json and results:
        with open(a.out_json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\n[wrote] {a.out_json}")


if __name__ == "__main__":
    main()

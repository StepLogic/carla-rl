#!/usr/bin/env python3
"""Goal-mode reward audit from an EVAL_Q_TRACE file (user 2026-09-28: "focus on
debugging the goal mode reward").

Recomputes every step's reward from the RAW per-step fields with an independent
implementation of the v19/v25 spec -- not by importing gated_reward -- and
compares it with the reward the env actually paid:

  arrival                        +100, terminal
  otherwise                      -1
    off-road                     -10
    collision                    -10
    lane crossing (lane_any)     solid/double/kerb/grass or no marking data -10;
                                 all marks unpainted (NONE) 0; inside a junction
                                 0; otherwise (broken, open road) -10
    speed bonus, clean step only 0.75 * max(0, 1 - |v - v_target| / v_target)
  truncation                     t >= 1000, or >= 20 off-road / collision /
                                 FINED-crossing steps (counted per episode)

Also reports event freshness: lane_event_lag / collision_event_lag = camera
frame - event frame (0 = the event belongs to this tick).
usage: goal_reward_audit.py TRACE.qtrace.jsonl
"""
import json
import sys
from collections import Counter, defaultdict

ILLEGAL = {"Solid", "SolidSolid", "SolidBroken", "Curb", "Grass"}
UNPAINTED = {"NONE"}


def lane_fine(r):
    if not r.get("lane_any"):
        return 0.0
    marks = r.get("marks") or []
    if not marks or any(m in ILLEGAL for m in marks):
        return 10.0
    if all(m in UNPAINTED for m in marks):
        return 0.0
    if r.get("in_junction"):
        return 0.0
    return 10.0


rows = [json.loads(l) for l in open(sys.argv[1])]
by_ep = defaultdict(list)
for r in rows:
    by_ep[r["ep"]].append(r)

mism, trunc_mism, kinds = [], [], Counter()
lane_lag, coll_lag = Counter(), Counter()
for e, ep in sorted(by_ep.items()):
    ep.sort(key=lambda r: r["t"])
    c_off = c_col = c_cross = 0
    for r in ep:
        fine = lane_fine(r)
        c_off += int(r["off_road"])
        c_col += int(r["collision"])
        c_cross += int(fine > 0)
        if r["arrived"]:
            exp, kind = 100.0, "arrival"
        else:
            exp = -1.0 - 10.0 * r["off_road"] - 10.0 * r["collision"] - fine
            clean = not r["off_road"] and not r["collision"] and fine == 0.0
            vt = r.get("target_speed_kmh") or 0.0
            if clean and vt > 0:
                exp += 0.75 * max(0.0, 1.0 - abs(r["speed_kmh"] - vt) / vt)
            kind = ("off" if r["off_road"] else "") + ("coll" if r["collision"] else "") \
                + (f"cross{int(fine)}" if r.get("lane_any") else "") or "clean"
        kinds[kind] += 1
        if abs(exp - r["r"]) > 1e-3:
            mism.append((e, r["t"], round(r["r"], 3), round(exp, 3), kind, r.get("marks"),
                         r.get("in_junction")))
        want_trunc = (r["t"] + 1 >= 1000 or c_off >= 20 or c_col >= 20 or c_cross >= 20) \
            and not r["arrived"]
        if bool(r["trunc"]) != want_trunc and not (r["t"] + 1 >= 1000):
            trunc_mism.append((e, r["t"], r["trunc"], want_trunc, c_off, c_col, c_cross))
        if r.get("lane_event_lag") is not None:
            lane_lag[r["lane_event_lag"]] += 1
        if r.get("collision_event_lag") is not None:
            coll_lag[r["collision_event_lag"]] += 1

print(f"steps {len(rows)}  episodes {len(by_ep)}")
print(f"step kinds: {dict(kinds)}")
print(f"REWARD mismatches: {len(mism)}")
for m in mism[:20]:
    print("   ep,t,paid,expected,kind,marks,junction:", m)
print(f"TRUNCATION mismatches: {len(trunc_mism)}")
for m in trunc_mism[:10]:
    print("   ep,t,trunc,expected,off,coll,cross:", m)
print(f"lane event lag (frames): {dict(sorted(lane_lag.items()))}")
print(f"collision event lag (frames): {dict(sorted(coll_lag.items()))}")

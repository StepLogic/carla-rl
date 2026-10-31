"""Hindsight relabelling for the DEBiT image-goal SAC arm.

User 2026-10-21: "add HER". The existing vae_nav.image_goal_her cannot be
reused as-is: it targets the old JAX replay, where the goal is a key in a state
vector, and it assumes a GOAL-INDEPENDENT reward. Here the goal reaches the
policy only through DEBiT's binocular reading of (view, goal frame), so every
relabelled row needs a fresh frozen-DEBiT pass, and the goal-mode reward has two
goal-dependent parts: the route-progress potential (info["gated_shaping"]) and
the +GOAL_EXPLORE_ARRIVAL_BONUS on arrival. Its RULES are kept:
  * legs only from the CLEAN prefix -- before the first collision, off-road or
    lane fault (the user's 2026-09-20 dirty cut);
  * a minimum distance to the hindsight goal (min_m), so reaching it took
    travel -- the band is the training band, 5-25 m;
  * the final frame of a time-limited episode is never a goal source.

A RELABELLED STEP'S REWARD is the stored reward with its goal parts removed and
recomputed for the hindsight goal:
    base_t   = r_t - shaping_t - bonus_t        (goal-independent: progress
                                                 along the lane, fines, idle)
    r'_t     = base_t + K * (d_t - d_{t+1})     non-terminal, d = path metres
                                                 left to the hindsight goal
    r'_arr   = base_arr + bonus                  the arrival step, TERMINAL,
                                                 no shaping (as the env)
Arrival is the env's test: within `radius` of the goal position AND at most
`max_speed` km/h -- the hindsight goal is drawn only where the car WAS that
slow, so every leg is a success the env itself would have paid. Two earlier
relabellers in this repo produced rows that never paid arrival or carried the
wrong shaping; the tests pin both.

The path distance stands in for the planner's route_remaining: on a leg the car
actually drove, metres along its own path are the route it took.
"""
import os

import numpy as np


class ImageGoalHER:
    def __init__(self, buf, encoder, pxr, *, n_relabel=2, min_m=5.0,
                 max_m=25.0, radius=2.0, max_speed=10.0, goal_bonus=50.0,
                 explore_bonus=10.0, potential_k=1.0, lane_fault=None,
                 env_id_offset=1000, relabel_explore=True, seed=0,
                 speed_gate_kmh=None):
        self.buf, self.enc, self.pxr = buf, encoder, pxr
        self.n_relabel, self.min_m, self.max_m = int(n_relabel), float(min_m), float(max_m)
        self.radius, self.max_speed = float(radius), float(max_speed)
        self.goal_bonus, self.explore_bonus = float(goal_bonus), float(explore_bonus)
        self.k = float(potential_k)
        # THE SPEED GATE (v27): live goal rows earn no route progress at or
        # above the target speed (gated_reward, PROGRESS_SPEED_GATE under
        # GOAL_ROUTE_ONLY), so hindsight rows must not either. None = off.
        self.speed_gate = None if speed_gate_kmh is None else float(speed_gate_kmh)
        self.lane_fault = lane_fault or (lambda info: False)
        self.offset = int(env_id_offset)
        self.relabel_explore = bool(relabel_explore)
        self.rng = np.random.default_rng(seed)
        self._ep = {}
        # Index of the declaration dimension, or None when STOP_ACTION is off.
        self.stop_dim = 2 if os.environ.get("STOP_ACTION", "0") == "1" else None
        self.counts = {"episodes": 0, "legs": 0, "rows": 0, "no_clean": 0,
                       "no_slow_goal": 0, "skipped_explore": 0}

    # ---- recording ------------------------------------------------------
    def begin(self, env_id, info, mode):
        self._ep[env_id] = {"mode": int(mode), "rows": [], "next_rows": [],
                            "actions": [], "base": [], "frames": [],
                            "next_frames": [], "pos": [], "speed": [],
                            "dirty": [], "lane_tail": []}

    def record(self, env_id, row, action, reward, term, trunc, next_row,
               info_t, info_next):
        e = self._ep.get(env_id)
        if e is None:
            return
        shaping = float(info_next.get("gated_shaping", 0.0) or 0.0)
        bonus = 0.0
        if info_next.get("goal_end") == "reached":
            bonus = self.goal_bonus
        elif info_next.get("explore_end") == "reached":
            bonus = self.explore_bonus
        e["rows"].append(np.asarray(row, np.float16))
        e["next_rows"].append(np.asarray(next_row, np.float16))
        e["actions"].append(np.asarray(action, np.float32))
        # EXPLORE-ONLY PAY (gated_reward: forward progress under GOAL_ROUTE_ONLY,
        # the junction bonus). A relabelled explore episode becomes goal rows,
        # and goal mode pays neither.
        explore_only = float(info_next.get("explore_only", 0.0) or 0.0)
        e["base"].append(float(reward) - shaping - bonus - explore_only)
        e["frames"].append(np.asarray(info_t["debit_cur"]))
        e["next_frames"].append(np.asarray(info_next["debit_cur"]))
        e["pos"].append(np.asarray(info_next.get("hero_pos", (0.0, 0.0)), np.float64))
        e["speed"].append(float(info_next.get("speed_kmh", 0.0) or 0.0))
        e["dirty"].append(bool(info_next.get("collision")) or bool(info_next.get("off_road"))
                          or bool(self.lane_fault(info_next)))
        # LANE-CENTRE SHAPING: the stored reward holds c(s) - gamma c(s'). The
        # env's row was not terminal, but the hindsight arrival row is, where
        # c(s') must be 0 -- so that one row adds gamma c(s') back (_add_leg).
        e["lane_tail"].append(float(info_next.get("lane_shaping_tail", 0.0) or 0.0))

    # ---- flushing -------------------------------------------------------
    def flush(self, env_id, terminated=False, truncated=False):
        e = self._ep.pop(env_id, None)
        if e is None or not e["rows"]:
            return 0
        self.counts["episodes"] += 1
        if e["mode"] != 1 and not self.relabel_explore:
            self.counts["skipped_explore"] += 1
            return 0
        n = len(e["rows"])
        # clean prefix: transitions 0..c-1, all strictly before the first dirty
        # step (the dirty step itself is excluded)
        c = next((t for t, d in enumerate(e["dirty"]) if d), n)
        if truncated and c == n:
            c = n - 1            # a clock artefact is not a destination
        if c < 2:
            self.counts["no_clean"] += 1
            return 0
        pos = np.stack(e["pos"])                       # pos[t] = after step t
        step_m = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
        cum = np.cumsum(step_m)                        # path metres at state t+1
        slow = [g for g in range(c) if e["speed"][g] <= self.max_speed]
        added = 0
        for _ in range(self.n_relabel):
            leg = self._pick_leg(cum, pos, e["speed"], slow)
            if leg is None:
                self.counts["no_slow_goal"] += 1
                continue
            s, j, g = leg
            added += self._add_leg(env_id, e, s, j, g, cum)
        return added

    def _pick_leg(self, cum, pos, speed, slow):
        """(start s, arrival step j, goal state g) or None. The leg's path
        length from s to g must lie in [min_m, max_m]; j is the first step at
        or after s whose next state is within `radius` of g's position and
        slow enough -- at the latest g itself, which is slow by construction."""
        if not slow:
            return None
        for _try in range(20):
            g = int(self.rng.choice(slow))
            # cum[t-1] is the path metres at the START of step t (0 for t=0)
            start_m = np.r_[0.0, cum[:-1]]
            cands = [s for s in range(0, g + 1)
                     if self.min_m <= cum[g] - start_m[s] <= self.max_m]
            if not cands:
                continue
            s = int(self.rng.choice(cands))
            j = g
            for t in range(s, g + 1):
                if (np.linalg.norm(pos[t] - pos[g]) < self.radius
                        and speed[t] <= self.max_speed):
                    j = t
                    break
            return s, j, g
        return None

    def _add_leg(self, env_id, e, s, j, g, cum):
        goal_frame = e["next_frames"][g]
        cur = np.stack(e["frames"][s:j + 1])
        nxt = np.stack(e["next_frames"][s:j + 1])
        gg = np.repeat(goal_frame[None], len(cur), axis=0)
        b_cur = np.asarray(self.enc.encode_batch(cur, gg), np.float32)
        b_nxt = np.asarray(self.enc.encode_batch(nxt, gg), np.float32)
        P = self.pxr.MONO_PIX
        goal_blk = self.pxr.goal_block(goal_frame)
        eid = self.offset + env_id
        for k, t in enumerate(range(s, j + 1)):
            row = e["rows"][t].astype(np.float32).copy()
            nrow = e["next_rows"][t].astype(np.float32).copy()
            row[P], nrow[P] = 1.0, 1.0                    # goal mode
            row[P + 1:], nrow[P + 1:] = b_cur[k], b_nxt[k]
            last = t == j
            if last:
                # STOP_ACTION: the leg's last row is relabelled as an arrival,
                # so its action must DECLARE one -- otherwise HER teaches the
                # critic that arriving silently pays, which the env denies.
                if self.stop_dim is not None:
                    act = np.array(e["actions"][t], np.float32, copy=True)
                    act[self.stop_dim] = 1.0
                    e["actions"][t] = act
                r = e["base"][t] + self.goal_bonus + e["lane_tail"][t]
            else:
                d_prev = cum[g] - (cum[t - 1] if t > 0 else 0.0)
                d_now = cum[g] - cum[t]
                prog = self.k * (d_prev - d_now)
                if (self.speed_gate is not None and prog > 0.0
                        and e["speed"][t] >= self.speed_gate):
                    prog = 0.0
                r = e["base"][t] + prog
            self.buf.add(row, e["actions"][t], float(r), last, False, nrow, 1,
                         env_id=eid, goal=goal_blk)
        self.counts["legs"] += 1
        self.counts["rows"] += j - s + 1
        return j - s + 1

    def describe_counts(self):
        c = self.counts
        return (f"{c['legs']} legs/{c['episodes']} eps, {c['rows']} rows, "
                f"no_slow_goal={c['no_slow_goal']} no_clean={c['no_clean']}")

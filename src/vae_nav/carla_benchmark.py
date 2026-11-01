"""The CARLA replication of the Gibson image-goal benchmark (user 2026-11-01).

THE PROTOCOL BEING REPLICATED, from the peer session that runs it on Gibson
(DEBiT-B in Habitat, measured from the episode files):
  * Mezghani ImageNav val, easy+medium+hard: 14 houses x 3 x 100 = 4,200.
  * Difficulty by GEODESIC distance: easy 1.5-3, medium 3-5, hard 5-10 m.
  * NRNS adds straight vs curved: straight iff geodesic/euclidean <= 1.20.
  * Success = STOP called within 1.0 m; max 1,000 steps.
  * SR and SPL per bin and pooled; arms paired per episode, exact McNemar,
    Holm-corrected across the family.

THE CARLA ANALOGUE:
  * 6 held-out towns (training is Town05+Town01) x 3 bins x 100 = 1,800.
  * Bins by ROUTE distance from the planner: 10-25 / 25-50 / 50-100 m, at
    CARLA's scale (see BINS). Never straight-line: Gibson's bins are
    geodesic, and an unroutable episode is dropped, not re-binned.
  * The FIXED episode set is a list of seeds. Starts and goals are drawn from
    the global RNGs inside rlib's carla_core, and the eval seeds them before
    every reset (train_carla_sac.run_eval), so a seed IS an episode. A probe
    pass resets each seed once and reads its route distance; selection keeps
    the first `per_bin` seeds per town per bin, in seed order.
  * Success has no STOP action (v20/v21's declare action never learned to
    declare), so "stop" means arriving at <= GOAL_ARRIVAL_MAX_SPEED. Both the
    Gibson radius (1.0 m) and our training radius (2.0 m) are reported from
    one run: the eval ends at 1.0 m and also records whether it was within
    2.0 m and slow before the episode ended.
  * Plus a control Gibson does not run: the true goal image against another
    episode's, paired per episode.
"""
from math import comb

# LONGER THAN THE TRAINING BAND, deliberately (user 2026-11-01, "use longer
# distance in carla"). CARLA roads are a different scale from Gibson rooms, and
# the policy was trained on goals up to 25 m: past that, reaching the goal
# needs a MAP built from the exploration trace, which is exactly what the
# peer's map arms test. The easy bin sits mostly inside the training band, so
# the table shows where mapless stops working and maps start to matter.
BINS = (("easy", 10.0, 25.0), ("medium", 25.0, 50.0), ("hard", 50.0, 100.0))
STRAIGHT_RATIO = 1.20      # NRNS


def assign_bin(route_m):
    """Difficulty bin of a route distance; None outside the band or unrouted.
    Half-open [lo, hi) except the top bin, which includes its upper edge."""
    if route_m is None:
        return None
    r = float(route_m)
    for name, lo, hi in BINS:
        if lo <= r < hi or (name == BINS[-1][0] and r == hi):
            return name
    return None


def path_kind(route_m, euclid_m):
    """NRNS's split: straight iff route / euclidean <= 1.20."""
    return "straight" if float(route_m) <= STRAIGHT_RATIO * float(euclid_m) + 1e-9 else "curved"


def select_episodes(probes, per_bin=100, bins=tuple(b[0] for b in BINS)):
    """The fixed episode set from probe records {town, seed, route_m, euclid_m}.

    First `per_bin` seeds per town per bin, in seed order, so the set is a pure
    function of the probe file. A bin that comes up short is an ERROR naming
    the town, bin and count -- padding it, or quietly reporting fewer
    episodes, would make the per-bin numbers incomparable across towns.
    """
    seen = set()
    by_cell = {}
    towns = []
    for p in sorted(probes, key=lambda x: (str(x["town"]), int(x["seed"]))):
        key = (p["town"], int(p["seed"]))
        if key in seen:
            raise ValueError(f"duplicate seed {key[1]} in {key[0]}")
        seen.add(key)
        if p["town"] not in towns:
            towns.append(p["town"])
        band = p.get("band")
        if band is not None and band not in {n for n, _, _ in BINS}:
            raise ValueError(f"unknown band {band!r} for seed {key[1]} in {key[0]}")
        b = assign_bin(p.get("route_m"))
        if b is None or b not in bins:
            continue
        # A probe GENERATED for a band counts only if its route lands in that
        # band's bin: an 'easy' seed that routes to 12 m is neither a clean easy
        # episode nor a medium one (it was drawn from the wrong distribution).
        if band is not None and band != b:
            continue
        by_cell.setdefault((p["town"], b), []).append(p)
    out = []
    for town in towns:
        for b in bins:
            got = by_cell.get((town, b), [])
            if len(got) < per_bin:
                raise ValueError(f"{town} {b}: only {len(got)} probed episodes, "
                                 f"need {per_bin} -- probe more seeds")
            for p in got[:per_bin]:
                e = {"town": town, "seed": int(p["seed"]), "bin": b,
                     "route_m": round(float(p["route_m"]), 2),
                     "euclid_m": round(float(p["euclid_m"]), 2),
                     "kind": path_kind(p["route_m"], p["euclid_m"])}
                if p.get("band") is not None:
                    # The generator band is PART OF THE EPISODE: the eval must
                    # replay the seed under the same band (slurm/bench_<band>.env).
                    e["band"] = p["band"]
                out.append(e)
    return out


# ---- paired statistics -----------------------------------------------------

def mcnemar_p(only_a, only_b):
    """Exact two-sided McNemar on the discordant pairs."""
    n = int(only_a) + int(only_b)
    if n == 0:
        return 1.0
    k = min(int(only_a), int(only_b))
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    """Holm step-down adjusted p-values, in the input's order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def _cell(pairs):
    n = len(pairs)
    if not n:
        return None
    ot = sum(1 for t, w in pairs if t["reached"] and not w["reached"])
    ow = sum(1 for t, w in pairs if w["reached"] and not t["reached"])
    mean = lambda xs: sum(xs) / len(xs)
    return {
        "n": n,
        "sr_true": mean([float(t["reached"]) for t, _ in pairs]),
        "sr_wrong": mean([float(w["reached"]) for _, w in pairs]),
        "spl_true": mean([float(t.get("spl") or 0.0) for t, _ in pairs]),
        "spl_wrong": mean([float(w.get("spl") or 0.0) for _, w in pairs]),
        "sr_true_2m": mean([float(t.get("reached_2m", t["reached"])) for t, _ in pairs]),
        "sr_wrong_2m": mean([float(w.get("reached_2m", w["reached"])) for _, w in pairs]),
        "only_true": ot, "only_wrong": ow, "p": mcnemar_p(ot, ow),
    }


def paired_table(true_rows, wrong_rows, by="bin"):
    """SR/SPL per group plus the pooled row, true vs wrong goal paired per
    episode (town, seed). Holm is applied across the GROUPS -- the family a
    reader compares -- and not to the pooled row."""
    key = lambda r: (r["town"], int(r["seed"]))
    t = {key(r): r for r in true_rows}
    w = {key(r): r for r in wrong_rows}
    if set(t) != set(w):
        raise ValueError(f"the two arms must drive the same episodes: "
                         f"{len(set(t) ^ set(w))} differ")
    pairs = [(t[k], w[k]) for k in sorted(t)]
    groups = {}
    for tr, wr in pairs:
        groups.setdefault(tr[by], []).append((tr, wr))
    table = {g: _cell(ps) for g, ps in groups.items()}
    names = list(table)
    for g, p in zip(names, holm([table[g]["p"] for g in names])):
        table[g]["p_holm"] = p
    table["all"] = _cell(pairs)
    return table


# ---- per-episode scoring at two radii from one run -------------------------

class EpisodeTracker:
    """Bookkeeping for one eval episode, scoring BOTH success radii.

    The episode is run at `radius` (Gibson's 1.0 m): the env ends it there.
    Our training radius (`radius_2m`, 2.0 m) is scored from the same run as
    the FIRST step within radius_2m at <= max_speed -- the step a 2 m eval
    would have ended it as a success. A fault AFTER that step does not undo
    it (the 2 m episode would already be over); a fault ON that step is a
    failure, as the env rules an arrival on a fatal step.

    Distances are 2-D, like the env's own arrival test. SPL uses the shortest
    path to each success REGION (route minus radius) and the metres driven
    when that region was entered.
    """

    def __init__(self, goal_pos, start_pos, max_speed, radius=1.0, radius_2m=2.0):
        import numpy as np
        self._np = np
        self.goal = np.asarray(goal_pos, np.float64)[:2]
        self.prev = np.asarray(start_pos, np.float64)[:2]
        self.max_speed = float(max_speed)
        self.radius, self.radius_2m = float(radius), float(radius_2m)
        self.t = 0
        self.path_m = 0.0
        self.reach2_step = None
        self.path_at_reach2 = None

    def step(self, pos, speed_kmh):
        np = self._np
        p = np.asarray(pos, np.float64)[:2]
        self.path_m += float(np.linalg.norm(p - self.prev))
        self.prev = p
        self.t += 1
        if (self.reach2_step is None
                and float(np.linalg.norm(self.goal - p)) < self.radius_2m
                and float(speed_kmh) <= self.max_speed):
            self.reach2_step = self.t
            self.path_at_reach2 = self.path_m

    def finish(self, reached, t_end, route_m):
        reached = bool(reached)
        reached_2m = reached or (self.reach2_step is not None
                                 and self.reach2_step < int(t_end))

        def spl(ok, radius, path):
            if not ok or route_m is None:
                return 0.0
            ell = max(float(route_m) - radius, 0.1)
            return ell / max(float(path), ell)

        p2 = self.path_at_reach2 if self.path_at_reach2 is not None else self.path_m
        return {"reached": reached, "reached_2m": reached_2m,
                "path_m": round(self.path_m, 3),
                "spl": spl(reached, self.radius, self.path_m),
                "spl_2m": spl(reached_2m, self.radius_2m, p2),
                "reach2_step": self.reach2_step}

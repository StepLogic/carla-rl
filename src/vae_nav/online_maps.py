"""Online MAD, temporal and uniform node placement, ported from the Habitat peer.

Spec relayed 2026-10-03 by the session that built them (rlt-gated/threeview:
mad_online_dinov2_graph.py, sgm_value_graph.py, temporal_global_graph.py). Ported
rule-for-rule; only the units differ, and where they differ it is stated here
rather than silently absorbed.

WHAT IS SHARED BY EVERY PLACER, and must stay shared or comparisons break:
  * dedupe against ANY node already placed, not just the previous one. Their
    last-node-only version gave 58.7 nodes/scene against 16.2, because every
    corridor revisit planted a fresh node.
  * edges = consecutive kept nodes WITHIN a segment, plus a "seglink" from the
    last node of a segment to the first node of the next segment that has one.
  * node counts are MATCHED across placers: temporal and uniform binary-search
    their knob to hit the online-MAD count, so a comparison is about WHERE the
    nodes are and never about how many.

THE ONE PORTING DECISION, STATED LOUDLY. The window is a TRAVEL DISTANCE, not a
frame count -- confirmed by the peer 2026-10-03: their code counts frames only
because their frames are distance-sampled at 0.10 m, and the purpose is LOCAL
adaptivity, because entropy scale drifts between regions and a long window lets
one region donate most of the nodes. So `window_m` is in metres here and the
sample count is derived from the trace's own measured spacing.

Their 15 m was chosen for rooms and corridors. Roads are not rooms, and at our
~2.2 m spacing 15 m is 7 samples -- far too few for a stable median and MAD
(they want >=30 in the window). On their explicit advice the window is set from
OUR geometry instead, once, and not tuned on success rate: 110 m, which is ~50
samples here and about a city block. The faithful-at-their-scale alternative is
to re-collect traces at <=0.5 m spacing and set window_m=15; that costs four
times the samples and storage for the same ground, and is a separate run.

THEIR z, NOT OURS. z=1.0. Their 10-point sweep was a null between 0.5 and 1.0;
z>=1.5 fragments the graph and z>=2.5 yields zero edges. This repo's OFFLINE
`madent` arm uses z=1.5-2 (node_placement.mad_spikes), which sits inside their
fragmenting range -- the two arms are not interchangeable and should not be
compared without saying which z produced which.

THE FAILURE MODE THEY PAID FOR: a TQC actor's entropy can sit at its ceiling, in
which case MAD is 0, every window is degenerate, and the placer emits noise
scaled by candidate density rather than signal. `OnlineMAD.stats()` reports the
degenerate-window share for exactly this reason; check it before believing a map.
(Our own entropy is not pinned -- mean -3.42 nats, sd 1.09 -- but theirs looked
pinned because of an entropy-extraction bug, so the check stays.)
"""
import numpy as np

# Their rules, with the two length scales set for roads (see the module note).
WINDOW_M = 110.0      # rolling-window TRAVEL DISTANCE (theirs: 15 m, for rooms)
WARMUP = 30           # samples required before admission: a STATISTICAL floor,
                      # kept in samples because it is what makes median and MAD
                      # stable, not a property of the geometry
Z = 1.0               # robust-z admission threshold
DEDUPE_M = 0.8        # reject a node within this of ANY node already placed
MAD_SCALE = 0.6745    # MAD -> sigma, i.e. sigma ~ MAD / 0.6745


def _robust(buf):
    """(median, sigma) of a buffer, sigma from the MAD. sigma<=0 means the
    window is degenerate and nothing may be admitted from it."""
    med = float(np.median(buf))
    mad = float(np.median(np.abs(np.asarray(buf) - med)))
    return med, mad / MAD_SCALE


class _Dedupe:
    """Reject anything within `radius` of ANY accepted point."""

    def __init__(self, radius):
        self.radius, self.pts = float(radius), []

    def accept(self, xy):
        xy = np.asarray(xy, np.float64)[:2]
        if any(np.linalg.norm(xy - p) < self.radius for p in self.pts):
            return False
        self.pts.append(xy)
        return True


def online_mad(entropy, positions, segments, window_m=WINDOW_M, warmup=WARMUP,
               z=Z, dedupe_m=DEDUPE_M, carry_buffer=False, guarantee=True,
               window=None):
    """Causal entropy placement. -> (indices, stats).

    Streams the trace in order. For each sample, admit it when its entropy is at
    least median+z*sigma of the ROLLING WINDOW OF PRECEDING SAMPLES, then apply
    the dedupe. No local-maximum test and no refractory distance: those are the
    offline detector's, and the difference is real -- offline keeps the HIGHEST
    entropy sample in a neighbourhood, online keeps the FIRST that qualifies,
    because at decision time it cannot know what is coming.

    The buffer resets at each segment boundary (their carry_buffer=0 default), so
    a segment shorter than `warmup` can never admit anything; `guarantee` then
    keeps that segment's own maximum-entropy sample, subject to the dedupe.
    """
    h = np.asarray(entropy, np.float64)
    pos = np.asarray(positions, np.float64)[:, :2]
    seg = np.asarray(segments, int)
    if not (len(h) == len(pos) == len(seg)):
        raise ValueError(f"ragged inputs: {len(h)} entropies, {len(pos)} "
                         f"positions, {len(seg)} segment ids")

    # The window is a distance; turn it into a sample count with the trace's own
    # measured spacing rather than an assumed one, since thinning varies by run.
    spacing = _median_spacing(pos, seg)
    if window is None:
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("cannot size the window: the trace has no "
                             "within-segment spacing to measure")
        window = max(warmup, int(round(window_m / spacing)))

    dedupe = _Dedupe(dedupe_m)
    picked, buf = [], []
    n_degenerate = n_considered = 0

    for s in _ordered_unique(seg):
        idx = np.flatnonzero(seg == s)
        if not carry_buffer:
            buf = []
        admitted_here = False
        for i in idx:
            if len(buf) >= warmup:
                n_considered += 1
                med, sigma = _robust(buf)
                if sigma <= 1e-9:
                    n_degenerate += 1          # pinned entropy: admit nothing
                elif h[i] >= med + z * sigma and dedupe.accept(pos[i]):
                    picked.append(int(i))
                    admitted_here = True
            buf.append(h[i])
            if len(buf) > window:
                buf.pop(0)
        if guarantee and not admitted_here and len(idx):
            # The segment selected nothing -- usually because it was shorter
            # than the warmup. Keep its best sample so short segments are not
            # silently unrepresented in the map.
            best = int(idx[int(np.argmax(h[idx]))])
            if dedupe.accept(pos[best]):
                picked.append(best)

    picked.sort()
    stats = {
        "n_nodes": len(picked),
        "window_samples": int(window),
        "window_m": spacing * window,
        "sample_spacing_m": spacing,
        "warmup_samples": int(warmup),
        "z": float(z),
        "dedupe_m": float(dedupe_m),
        "degenerate_window_frac": (n_degenerate / n_considered
                                   if n_considered else float("nan")),
        "guarantee": bool(guarantee),
        "carry_buffer": bool(carry_buffer),
    }
    return np.array(picked, dtype=int), stats


def temporal_nodes(positions, segments, n_target, dedupe_m=DEDUPE_M,
                   max_iter=40):
    """THE CONTROL PLACER, not a different kind of map.

    N samples spaced evenly in STEP TIME over the whole trace, then the shared
    dedupe; N is searched so the KEPT count matches `n_target` (the online-MAD
    count). It answers the only question that makes an entropy map interesting:
    does the signal pick better nodes than every Nth step would?

    The search is over N BEFORE dedupe, because dedupe makes the kept count a
    non-decreasing but not one-to-one function of N.
    """
    return _matched(lambda n: _even_in_time(len(positions), n),
                    positions, n_target, dedupe_m, max_iter)


def uniform_nodes(positions, segments, n_target, dedupe_m=DEDUPE_M,
                  max_iter=40):
    """The other control: every node at least s metres along the path, with s
    searched to match `n_target`. Spacing rather than step count, so it differs
    from `temporal_nodes` wherever speed varies."""
    pos = np.asarray(positions, np.float64)[:, :2]
    seg = np.asarray(segments, int)
    s = _arclength(pos, seg)

    def by_spacing(step_m):
        out, last = [], -np.inf
        for i in range(len(s)):
            if s[i] - last >= step_m:
                out.append(i)
                last = s[i]
        return np.array(out, dtype=int)

    lo, hi = 1e-3, max(float(s[-1]), 1.0)
    best = by_spacing(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        kept = _apply_dedupe(by_spacing(mid), pos, dedupe_m)
        best = kept
        if len(kept) > n_target:
            lo = mid                      # too many nodes: space them further
        elif len(kept) < n_target:
            hi = mid
        else:
            break
    return best


def build_edges(indices, positions, segments, seglinks=False):
    """-> (edges, kinds). Consecutive kept nodes WITHIN a segment, and with
    `seglinks=True` also one edge from each segment's last node to the next
    node-bearing segment's first.

    SEGLINKS ARE OFF BY DEFAULT (user 2026-10-03, and the peer's own experience).
    A seglink spans a RESPAWN: the collector restarts a segment at the nearest
    drivable waypoint to where the last one stopped, so the two endpoints are
    not connected by any driveable path the agent can follow. The peer measured
    plans stalling on exactly these. Dropping them leaves the graph with one
    component per segment, which is honest -- the connections that are missing
    were never real -- and the way back to a connected graph is critic-certified
    shortcuts between nodes that are genuinely close, not a respawn edge.
    """
    idx = np.asarray(indices, int)
    seg = np.asarray(segments, int)[idx]
    edges, kinds, last_of_prev = [], [], None
    for s in _ordered_unique(seg):
        here = np.flatnonzero(seg == s)
        for a, b in zip(here[:-1], here[1:]):
            edges.append((int(a), int(b)))
            kinds.append("trace")
        if seglinks and last_of_prev is not None and len(here):
            edges.append((int(last_of_prev), int(here[0])))
            kinds.append("seglink")
        if len(here):
            last_of_prev = here[-1]
    return edges, kinds


# ---------------------------------------------------------------- helpers ---
def _ordered_unique(a):
    """Unique values in FIRST-APPEARANCE order. np.unique would sort, which
    would reorder segments and so reorder the seglinks."""
    seen, out = set(), []
    for v in a:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _even_in_time(n_samples, n):
    if n <= 0 or n_samples == 0:
        return np.empty(0, dtype=int)
    return np.unique(np.linspace(0, n_samples - 1, int(n)).round().astype(int))


def _apply_dedupe(idx, pos, dedupe_m):
    d = _Dedupe(dedupe_m)
    return np.array([int(i) for i in idx if d.accept(pos[i])], dtype=int)


def _matched(candidates_for, positions, n_target, dedupe_m, max_iter):
    pos = np.asarray(positions, np.float64)[:, :2]
    lo, hi = 1, max(2, len(pos))
    best = _apply_dedupe(candidates_for(lo), pos, dedupe_m)
    for _ in range(max_iter):
        mid = (lo + hi) // 2
        kept = _apply_dedupe(candidates_for(mid), pos, dedupe_m)
        if len(kept) < n_target:
            lo = mid + 1
        elif len(kept) > n_target:
            hi = mid - 1
        else:
            return kept
        if abs(len(kept) - n_target) < abs(len(best) - n_target):
            best = kept
        if lo > hi:
            break
    return best


def _arclength(pos, seg):
    """Cumulative distance along the path, with segment jumps not counted: a
    respawn is not travel, and counting it inflates every spacing downstream."""
    d = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    d[1:][seg[1:] != seg[:-1]] = 0.0
    return np.cumsum(d)


def _median_spacing(pos, seg):
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)[seg[1:] == seg[:-1]]
    return float(np.median(d)) if len(d) else float("nan")


# --------------------------------------------------------------- shortcuts ---
# Recipe from the Habitat peer, 2026-10-03, with the ROAD correction they made
# explicit: do NOT use their two-way mean here. Indoors, two nodes 1.5 m apart
# are mutually reachable and V partly encodes orientation anyway; on a road they
# may be opposite carriageways, close in metres and unreachable without a
# U-turn. So edges are DIRECTED, gated on heading BEFORE the critic is asked,
# and certified by the ONE-WAY value.
SHORTCUT_RADIUS_M = 3.0
HEADING_GATE_DEG = 90.0


def shortcut_candidates(xy, yaw, existing_edges=(), radius=SHORTCUT_RADIUS_M,
                        heading_deg=HEADING_GATE_DEG):
    """-> directed (i, j) pairs worth asking the critic about.

    All pairs within `radius` that are not already connected and whose headings
    agree to within `heading_deg`. No k-nearest and no component restriction --
    the peer's procedure, and with a few hundred nodes the all-pairs pass is
    cheap. The heading gate is what the indoor version does not need.
    """
    xy = np.asarray(xy, np.float64)[:, :2]
    yaw = np.asarray(yaw, np.float64)
    n = len(xy)
    have = {(int(a), int(b)) for a, b in existing_edges}
    have |= {(b, a) for a, b in have}
    out = []
    for i in range(n):
        d = np.linalg.norm(xy - xy[i], axis=1)
        for j in np.flatnonzero((d <= radius) & (np.arange(n) != i)):
            j = int(j)
            if (i, j) in have:
                continue
            if abs(_wrap_deg(np.degrees(yaw[j] - yaw[i]))) >= heading_deg:
                continue
            out.append((i, j))
    return out


def _wrap_deg(a):
    """Signed angle difference in (-180, 180]."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def trace_edge_cost_m(indices, positions, segments):
    """Within-segment edge cost as DRIVEN metres along the trace.

    The peer's rule, and the reason for it: trace edges join nodes consecutive
    in the KEPT list, so a long gap becomes a wormhole -- they measured 44% of
    edges too long, up to 45x. Pruning those by length fragmented their graphs,
    which is the failure we already have from dropping seglinks. Costing them
    honestly instead lets a planner avoid a wormhole whenever a certified
    shortcut exists, without removing the only connection when none does.
    """
    idx = np.asarray(indices, int)
    pos = np.asarray(positions, np.float64)[:, :2]
    seg = np.asarray(segments, int)
    s = _arclength(pos, seg)
    out = []
    for a, b in zip(range(len(idx) - 1), range(1, len(idx))):
        if seg[idx[a]] != seg[idx[b]]:
            continue
        out.append(((a, b), float(abs(s[idx[b]] - s[idx[a]]))))
    return out

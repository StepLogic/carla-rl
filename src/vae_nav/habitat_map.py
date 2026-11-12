"""Topological map built the way the peer Habitat repo builds one.

PORTED DELIBERATELY, INCLUDING THE PARTS THAT LOOK WRONG. Where their choice is
load-bearing it is kept; where it is a known defect it is kept but named, and
the one deviation is argued in place rather than made silently.

THE CONSTRUCTION, in their order:

  1. THIN the trace. Greedy walk keeping a point whenever its EUCLIDEAN
     distance from the LAST KEPT point exceeds `spacing`. Not arc length, and
     not rejection against all kept points -- so two passes through the same
     corridor can leave nodes a centimetre apart and nothing rejects them.
     Their measured median nearest-neighbour spacing is 0.18 m against a
     nominal 0.8 m for exactly this reason. Kept nodes are ACTUAL trace poses,
     never resampled.

  2. TRACE EDGES between temporally consecutive kept nodes, WITHIN a segment.
     Nothing spatial. When the trace revisits a corridor the two passes are
     connected by nothing at all. This is why a trace graph is many components
     and routes almost nothing on its own -- theirs is ~27 components routing
     0/60 pairs, and the CARLA teleport map measured 43 components routing 3%.

  3. CERTIFIED SHORTCUTS reconnect the segments, and this is the only thing
     that makes the graph routable. Certification is against the SENSOR
     OBSERVATION held at the node -- never a simulator ray cast, never the road
     network. An edge asserted without sensor evidence is the same class of
     error as the 234 m teleport edges that made the first CARLA map's
     routability fictional.

WHY NOT A RAY CAST, given CARLA offers one. Because the experiment is whether a
robot can build a usable map from what it can SEE. A privileged visibility test
would answer a different question and would flatter the result.

DEVIATION, STATED: their node spacing equals their arrival radius (both 0.8 m),
which makes the waypoint follower a no-op -- a waypoint is already "reached"
when it becomes current, so the follower skips down the chain and hands the
policy a distant node. They confirmed their configuration sits in that regime.
This module defaults `spacing` to 2x the arrival radius instead. Their
denser-is-better result is not evidence against that: they measured it acts
through CERTIFICATION COVERAGE (more node pairs within range -> more shortcuts
-> fewer components), not through subgoal spacing.

MAX SHORTCUT LENGTH IS NOT SCALED FROM THE ARRIVAL RADIUS. Their 6.0 m came
from the usable range of their depth ring, and they have since measured 6.0 to
be far too generous: shortcuts audit at 14.6% traversable in the 3-4 m band and
0 of 22 beyond 4 m. It must come from what the sensor can actually certify, and
`audit_shortcuts` exists to measure that banded by length before the number is
trusted.
"""
import math

import numpy as np

# Their ring geometry, and the values are theirs unless noted.
DEFAULT_HFOV_DEG = 120.0        # forward field the ring covers
DEFAULT_MIN_CONE_DEG = 8.0      # cone half-width floor, widened to beam spacing
DEFAULT_MARGIN_M = 0.3          # absolute clearance required beyond the target
DEFAULT_MARGIN_PER_M = 0.0      # range-proportional slack; 0 = theirs
VALID_RANGE_EPS = 1e-3          # below this a beam is DEAD, not "range zero"


def thin_trace(positions, spacing):
    """Greedy thinning against the LAST KEPT point only. Returns indices.

    Deliberately not a global min-spacing rejection: this reproduces their
    behaviour, including the property that revisits leave near-duplicate nodes.
    Changing it would change the node distribution and make node counts
    non-comparable with theirs.
    """
    pts = np.asarray(positions, np.float64)
    if len(pts) == 0:
        return []
    keep = [0]
    for i in range(1, len(pts)):
        if float(np.linalg.norm(pts[i] - pts[keep[-1]])) >= spacing:
            keep.append(i)
    return keep


def bearing_to(pos_a, yaw_a, pos_b):
    """Bearing of B relative to A's heading, and the distance. CARLA-left-handed.

    The sign convention matches vae_nav.vae_goal_wrapper.goal_block: CARLA's
    basis is forward=(cos yaw, sin yaw), right=(-sin yaw, cos yaw), so LEFT is
    the negative projection onto `right`. Getting this backwards is what made
    the policy steer away from every off-axis goal for the entire project, so it
    is written the same way here rather than re-derived.
    """
    d = np.asarray(pos_b, np.float64)[:2] - np.asarray(pos_a, np.float64)[:2]
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return 0.0, 0.0
    cy, sy = math.cos(yaw_a), math.sin(yaw_a)
    forward = d[0] * cy + d[1] * sy
    left = -(-d[0] * sy + d[1] * cy)
    return math.atan2(left, forward), dist


def visible_from(pos_a, yaw_a, ring_a, pos_b, max_range,
                 hfov_deg=DEFAULT_HFOV_DEG, min_cone_deg=DEFAULT_MIN_CONE_DEG,
                 margin=DEFAULT_MARGIN_M, margin_per_m=DEFAULT_MARGIN_PER_M):
    """Can B be certified visible from A, using only A's beam ring?

    `ring_a` is (n_beams,) of ranges in metres, evenly spaced over hfov_deg and
    centred on A's heading. A range at or below VALID_RANGE_EPS is a DEAD beam.

    Four conditions, all theirs:
      * within `max_range` euclidean
      * within the forward field of view
      * at least one VALID beam inside the cone toward B -- an unmeasured
        direction is never treated as clear
      * every valid beam in that cone reports a range beyond B plus a margin
    """
    ring = np.asarray(ring_a, np.float64)
    n = ring.shape[0]
    if n == 0:
        return False

    bearing, dist = bearing_to(pos_a, yaw_a, pos_b)
    if dist <= 1e-9 or dist > max_range:
        return False

    half_fov = math.radians(hfov_deg) / 2.0
    if abs(bearing) > half_fov:
        return False

    beam_spacing = (2.0 * half_fov) / max(n - 1, 1)
    cone = max(math.radians(min_cone_deg), beam_spacing)

    angles = np.linspace(-half_fov, half_fov, n)
    in_cone = np.abs(angles - bearing) <= cone
    if not np.any(in_cone):
        return False

    # DEAD BEAMS ARE EXCLUDED, not read as zero. One dead pixel read as range 0
    # vetoes an entire arc, which they hit.
    cand = ring[in_cone]
    valid = cand[cand > VALID_RANGE_EPS]
    if valid.size == 0:
        return False                      # unknown is not clear

    return bool(valid.min() > dist + margin + margin_per_m * dist)


def build_graph(positions, yaws, rings, segment_ids, spacing, arrival_radius,
                max_shortcut, min_segment_nodes=0, verbose=True):
    """Trace -> topological graph. Returns (networkx.Graph, node_indices, stats).

    `segment_ids` marks trace breaks: consecutive samples with different ids get
    no trace edge, which is how a severed trace stays severed.
    """
    import networkx as nx

    if spacing <= arrival_radius:
        # Not fatal, because reproducing their configuration exactly is a
        # legitimate thing to want -- but it must be loud.
        print(f"[Map] WARNING: spacing={spacing} <= arrival_radius="
              f"{arrival_radius}. A waypoint is then already 'reached' when it "
              "becomes current, the follower skips down the chain, and the "
              "mapped arm degenerates toward the unmapped one.")

    positions = np.asarray(positions, np.float64)
    yaws = np.asarray(yaws, np.float64)
    segment_ids = np.asarray(segment_ids)

    # Thin WITHIN each segment, exactly as they do -- there is no global pass.
    #
    # MINIMUM SEGMENT LENGTH is a DELIBERATE CHOICE, not an inherited default.
    # Their builder discards any segment yielding fewer than 3 thinned
    # candidates -- under about 2.4 m of travel -- and they measured that this
    # drops roughly 84 of ~114 segments and 53.8% of all explored steps, which
    # never reach the map at all. That is a large silent loss, so here it
    # defaults to 0 (keep everything) and the filtered variant is reported
    # alongside rather than substituted for it.
    keep, dropped_segs, dropped_steps = [], 0, 0
    for seg in np.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == seg)
        local = thin_trace(positions[idx], spacing)
        if min_segment_nodes and len(local) < min_segment_nodes:
            dropped_segs += 1
            dropped_steps += len(idx)
            continue
        keep.extend(int(idx[i]) for i in local)
    keep = sorted(keep)

    G = nx.Graph()
    for n_i, t_i in enumerate(keep):
        G.add_node(n_i, trace_index=int(t_i),
                   position=positions[t_i].tolist(), yaw=float(yaws[t_i]))

    # TRACE EDGES: temporally consecutive kept nodes, same segment only.
    n_trace = 0
    for a in range(len(keep) - 1):
        if segment_ids[keep[a]] != segment_ids[keep[a + 1]]:
            continue
        d = float(np.linalg.norm(positions[keep[a + 1]] - positions[keep[a]]))
        G.add_edge(a, a + 1, weight=d, length=d, type='trace')
        n_trace += 1

    # CERTIFIED SHORTCUTS. Either endpoint certifying is enough: visibility is
    # symmetric in geometry but not in sensor coverage, so requiring both would
    # discard real shortcuts wherever one node happened to be facing away.
    n_short, tested = 0, 0
    pos_kept = positions[keep]
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            if G.has_edge(i, j):
                continue
            d = float(np.linalg.norm(pos_kept[j] - pos_kept[i]))
            if d > max_shortcut:
                continue
            tested += 1
            ok = visible_from(pos_kept[i], yaws[keep[i]], rings[keep[i]],
                              pos_kept[j], max_shortcut)
            if not ok:
                ok = visible_from(pos_kept[j], yaws[keep[j]], rings[keep[j]],
                                  pos_kept[i], max_shortcut)
            if ok:
                G.add_edge(i, j, weight=d, length=d, type='shortcut')
                n_short += 1

    n_comp = nx.number_connected_components(G)
    stats = {
        'trace_samples': int(len(positions)),
        'nodes': G.number_of_nodes(),
        'trace_edges': n_trace,
        'shortcut_edges': n_short,
        'shortcut_pairs_tested': tested,
        'components': n_comp,
        # Their invariant: nodes - trace_edges is the segment count, and they
        # measured it constant at 27.1 across every variant they tried. Worth
        # printing because if it moves with spacing, the trace changed rather
        # than the map.
        'segments_implied': G.number_of_nodes() - n_trace,
        'spacing': spacing,
        'max_shortcut': max_shortcut,
        'min_segment_nodes': int(min_segment_nodes),
        'segments_dropped': int(dropped_segs),
        'steps_dropped': int(dropped_steps),
        'steps_dropped_frac': float(dropped_steps / max(len(positions), 1)),
    }
    if verbose:
        print(f"[Map] {stats['nodes']} nodes  trace={n_trace}  "
              f"shortcuts={n_short}/{tested} tested  components={n_comp}  "
              f"segments_implied={stats['segments_implied']}")
    return G, keep, stats


def audit_shortcuts(G, bands=((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10),
                             (10, 15), (15, 25))):
    """Shortcut count by length band, for the audit they insisted on.

    They set max_shortcut=6.0 m from their ring's range and later measured it
    far too generous -- 14.6% genuinely traversable in the 3-4 m band, 0 of 22
    beyond 4 m. A cap must be justified by an audit, not by a sensor spec sheet,
    and the number of shortcuts in a band is the first half of that; the second
    half is driving them, which needs the simulator.
    """
    out = []
    lens = [(d['length'], u, v) for u, v, d in G.edges(data=True)
            if d.get('type') == 'shortcut']
    for lo, hi in bands:
        n = sum(1 for L, _, _ in lens if lo <= L < hi)
        if n:
            out.append({'band': f"{lo}-{hi}m", 'count': n})
    return out


def routable_fraction(G, n_pairs=300, seed=0):
    """Fraction of random node pairs with a path, and the metric path stats.

    THE DENOMINATOR IS EVERY PAIR DRAWN, not every pair that happened to route.
    Scoring only routable pairs flatters exactly the configurations that
    fragment the graph -- they measured one arm at 54.2%->70.0% that way, and
    53.3%->23.3% once the 39 lost routes were counted back in. Same data,
    opposite conclusion.
    """
    import networkx as nx
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        # Must return the FULL key set: callers format median_hops and
        # median_longest_edge_m unconditionally, and a short dict here crashed
        # the whole sweep with KeyError after the traces were already collected.
        return {'routable': 0.0, 'n_pairs': 0,
                'median_hops': float('nan'),
                'median_longest_edge_m': float('nan')}
    ok, hops, longest = 0, [], []
    for _ in range(n_pairs):
        a, b = (int(rng.integers(0, len(nodes))) for _ in range(2))
        if a == b:
            continue
        try:
            p = nx.shortest_path(G, nodes[a], nodes[b], weight='weight')
        except nx.NetworkXNoPath:
            continue
        ok += 1
        hops.append(len(p) - 1)
        longest.append(max(G[p[k]][p[k + 1]]['length'] for k in range(len(p) - 1))
                       if len(p) > 1 else 0.0)
    return {
        'routable': ok / max(n_pairs, 1),
        'n_pairs': n_pairs,
        'median_hops': float(np.median(hops)) if hops else float('nan'),
        'median_longest_edge_m': float(np.median(longest)) if longest else float('nan'),
    }

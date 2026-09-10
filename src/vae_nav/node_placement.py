"""Four ways to choose graph nodes from one trace, for the placement ablation.

THE POINT OF THE COMPARISON. Every method here consumes the SAME candidate
states and differs only in which it keeps. Node placement is then the only
variable, so a difference in SR/SPL is attributable to it. Anything collected
per method would measure the collection instead.

THERE IS NO NODE BUDGET, AND THAT IS DELIBERATE. The question is the PARETO
FRONTIER of node count against SR/SPL, so nothing is forced onto a shared
number. An earlier version of this file imposed an equal budget -- SGM's own
count, handed to the other three -- which answers a different and narrower
question: "which method is best at exactly n nodes". It also caps the one
method whose sparsity is supposed to be emergent.

THE TWO KINDS OF ARM ARE SWEPT DIFFERENTLY, because their free parameter is not
the same thing:

    uniform, random, policy   have NO intrinsic node count. They must be told
                              how many to place, so node count IS their free
                              parameter and sweeping it traces their curve.
    sgm                       DERIVES its count from its consistency cutoff.
                              That cutoff is swept and the node count is an
                              OUTPUT, never a target.

`n` therefore means "how many this arm was asked for", not "the budget", and it
is meaningless for SGM.

THE FOUR ARMS:

    uniform   equal ARC-LENGTH spacing, at interval midpoints. The honest
              baseline: it needs no policy at all, so a method that cannot beat
              it is not earning its keep.
    random    uniform-random arc-length targets. The floor.
    policy    peaks of the policy's own action entropy. The claim is that the
              policy is most uncertain where the route is DECIDABLE -- at
              junctions -- so entropy peaks are a free junction detector.
    sgm       Sparse Graphical Memory's two-way consistency merge, in
              sgm_placement.py, which reads the critic as a distance.

THE FIRST THREE ARE PORTED FROM node_placement_entropy.py DELIBERATELY, down to
the midpoint and greedy-by-height details. That script already ran this
comparison against junction coverage and found no reliable difference between
peaks and uniform; re-implementing the arms differently here would mean the
SR/SPL result could not be read against that one.

WHY "policy" IS ENTROPY AND NOT ENSEMBLE UNCERTAINTY, which is what the peer's
node-placement work queries. value_distance.edge_reject_mask records the
measurement that rules the latter out for us: critic-ensemble disagreement is a
proxy for LEG LENGTH, not local uncertainty (AUC 0.533 -- chance -- once length
and Qmean are regressed out) and it has the wrong sign. Actor entropy is a
different quantity with its own evidence in this project: measured +0.1232 nats
higher at junctions than on open road, 95% CI [+0.0253, +0.2273].

ARC LENGTH ACROSS A MULTI-EPISODE TRACE accumulates WITHIN a segment and adds
nothing at a segment break, so the jump between one episode's last pose and the
next episode's reset pose is not counted as distance travelled. The consequence,
stated because it is a real approximation: two nodes on either side of a break
can be within min_sep of each other in arc length despite being unrelated. It
costs at most one node per break and is preferred to inventing a teleport
distance that would stretch the uniform arm's spacing.
"""
import numpy as np

from vae_nav.value_distance import H_STEPS

METHODS = ("uniform", "random", "policy", "sgm")


def smooth(x, w):
    """Centred moving average. Odd window; edges shrink rather than pad.

    Padding with the edge value would invent a plateau at both ends, and
    episode starts already carry a large entropy transient -- that plateau
    would become a spurious peak exactly where the policy is only initialising.
    """
    x = np.asarray(x, float)
    if w <= 1 or len(x) < 3:
        return x.copy()
    w = int(w) | 1
    half, out = w // 2, np.empty_like(x)
    for i in range(len(x)):
        a, b = max(0, i - half), min(len(x), i + half + 1)
        out[i] = x[a:b].mean()
    return out


def arclength(xy, segments=None):
    """Cumulative path length in metres, same length as xy.

    A segment break contributes zero, so a reset teleport is not distance.
    """
    xy = np.asarray(xy, np.float64)
    if len(xy) == 0:
        return np.zeros(0)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if segments is not None:
        seg = np.asarray(segments)
        d = np.where(seg[1:] == seg[:-1], d, 0.0)
    return np.concatenate([[0.0], np.cumsum(d)])


def place_uniform(s, n):
    """n nodes at equal arc-length spacing, at the interval MIDPOINTS.

    Midpoints, not endpoints: an endpoint would pin a node at s=0, which is the
    reset pose and not a place anyone would choose.
    """
    s = np.asarray(s, np.float64)
    if n <= 0 or len(s) == 0:
        return np.empty(0, dtype=int)
    edges = np.linspace(s[0], s[-1], int(n) + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    return np.array(sorted({int(np.argmin(np.abs(s - m))) for m in mids}), dtype=int)


def place_random(s, n, seed=0):
    """EXACTLY n distinct states, uniformly at random, in trace order.

    DRAWN OVER STATES, NOT OVER ARC LENGTH, and the difference is the budget.
    Sampling arc-length targets and snapping each to its nearest state -- the
    reference's formulation -- lets two targets land on one state, and the
    duplicates collapse: measured here, 101 targets produced 79 distinct nodes.
    An arm silently 22% sparser than the others is the sparsity confound the
    equal budget exists to prevent, so the budget is honoured exactly instead.

    The candidates are already thinned to near-uniform spacing before any of
    this, so uniform over states and uniform over arc length are nearly the
    same draw; only the duplicate handling differs.
    """
    s = np.asarray(s, np.float64)
    n = int(min(max(n, 0), len(s)))
    if n <= 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(s), size=n, replace=False)).astype(int)


def place_peaks(h, s, n_max, min_sep_m, smooth_w):
    """Local maxima of smoothed entropy, greedily taken HIGHEST-FIRST.

    Greedy-by-height with a separation veto, rather than "every local max then
    thin": thinning left-to-right keeps whichever peak happens to come first,
    an arbitrary tiebreak that would make the arm depend on travel direction.

    MIN-SEPARATION IS NOT A FUDGE. Entropy is autocorrelated over tens of
    steps, so raw maxima arrive in clusters of near-duplicates around one
    event, and nodes a metre apart are not two nodes in any useful sense. The
    uniform arm satisfies the same constraint by construction, so this
    equalises the arms rather than favouring one.
    """
    h, s = np.asarray(h, np.float64), np.asarray(s, np.float64)
    if n_max <= 0 or len(h) == 0:
        return np.empty(0, dtype=int)
    hs = smooth(h, smooth_w)
    picked = []
    for i in np.argsort(-hs):
        if len(picked) >= n_max:
            break
        if all(abs(s[i] - s[j]) >= min_sep_m for j in picked):
            picked.append(int(i))
    return np.array(sorted(picked), dtype=int)


def sgm_cutoff_for_budget(dist, embeddings, target_n, embedding_cutoff,
                          lo=0.1, hi=None, iters=18, order=None):
    """Find the consistency cutoff whose SGM node count is nearest `target_n`.

    WHY SEARCH AT ALL. SGM derives its own sparsity rather than taking a budget,
    which is the property that makes it interesting and also makes it
    incomparable to fixed-budget baselines. Sweeping its one free parameter to
    the budget is the standard way to put a self-selecting method on the same
    axis as arms that are told how many nodes to place.

    MONOTONE, SO BISECTION IS VALID: a larger cutoff admits more merges, so the
    node count falls monotonically in the cutoff. Returns (cutoff, nodes,
    merged_into) for the best cutoff found, so the caller reports what was
    actually used rather than what was asked for.
    """
    from vae_nav.sgm_placement import sgm_select

    hi = float(np.max(dist)) if hi is None else float(hi)
    lo = float(lo)
    best = None
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        nodes, merged = sgm_select(dist, embeddings, embedding_cutoff=embedding_cutoff,
                                   consistency_cutoff=mid, order=order)
        gap = abs(len(nodes) - int(target_n))
        if best is None or gap < best[0]:
            best = (gap, mid, nodes, merged)
        if len(nodes) > target_n:
            lo = mid            # too many nodes -> merge more -> raise cutoff
        else:
            hi = mid
    _gap, cutoff, nodes, merged = best
    return cutoff, nodes, merged


def place(method, s=None, entropy=None, dist=None, embeddings=None, n=None,
          seed=0, min_sep_m=8.0, smooth_w=9, max_steps=H_STEPS, order=None,
          **kw):
    """One entry point, so the driver cannot apply a method inconsistently.

    `n` is HOW MANY NODES THIS ARM WAS ASKED FOR -- the sweep point for uniform,
    random and policy, and meaningless for sgm, whose count comes out of its
    consistency cutoff. It is REQUIRED for the three that take it: defaulting it
    would let an arm quietly place a different number than the point being
    swept, which turns a curve into an unlabelled scatter.
    """
    method = str(method).lower()
    if method in ("uniform", "random", "policy") and (s is None or n is None):
        raise ValueError(f"{method} placement needs arc length s and budget n")
    if method == "uniform":
        return place_uniform(s, n), None
    if method == "random":
        return place_random(s, n, seed=seed), None
    if method == "policy":
        if entropy is None:
            raise ValueError("policy placement needs per-state actor entropy")
        return place_peaks(entropy, s, n, min_sep_m, smooth_w), None
    if method == "sgm":
        if dist is None or embeddings is None:
            raise ValueError("sgm placement needs dist and embeddings")
        from vae_nav.sgm_placement import sgm_select
        return sgm_select(dist, embeddings, order=order, **kw)
    raise ValueError(f"unknown placement method {method!r}; "
                     f"expected one of {METHODS}")


def actor_entropy(agent, observations, seed=0, batch=256):
    """Per-state entropy of the actor, as -log_prob of one sample.

    THE SAME ESTIMATOR THE LEARNER REPORTS (gated_tqc_learner's "entropy" is
    -log_probs.mean() over one sample per state), so a peak here means what a
    training curve means. A closed-form entropy would be a second definition of
    one quantity and the two would drift.
    """
    import jax

    n = len(observations)
    out = np.zeros(n, np.float64)
    key = jax.random.PRNGKey(seed)
    for a in range(0, n, batch):
        chunk = observations[a:a + batch]
        stacked = {k: np.stack([np.asarray(o[k]) for o in chunk])
                   for k in chunk[0]}
        key, sub = jax.random.split(key)
        dist = agent._actor.apply_fn({"params": agent._actor.params}, stacked)
        _act, log_prob = dist.sample_and_log_prob(seed=sub)
        out[a:a + len(chunk)] = -np.asarray(log_prob, np.float64)
    return out

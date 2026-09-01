"""The critic's value, read as a distance the graph can use.

UNITS. Two are in play and they differ by 1/STEP_COST, so they are named
apart everywhere:

    V-units   what the critic predicts, range [-30, 0]
    steps     -V / STEP_COST, non-negative, what Dijkstra receives

Mixing them silently is easy and would put every edge weight out by 10x.
"""
import numpy as np

STEP_COST_DEFAULT = 0.1
# The 10th percentile of the pooled quantiles. Conservative on DISTANCE.
PESSIMISM_PERCENTILE = 10.0


def pooled_quantiles(q):
    """(num_qs, B, n_q) -> (B, num_qs * n_q). All critics' quantiles together."""
    q = np.asarray(q, np.float64)
    if q.ndim != 3:
        raise ValueError(f"expected (num_qs, B, n_q), got shape {q.shape}")
    num_qs, batch, n_q = q.shape
    return q.transpose(1, 0, 2).reshape(batch, num_qs * n_q)


def pessimistic_value(q, percentile=PESSIMISM_PERCENTILE):
    """A LOW percentile of V, i.e. a LONG distance. (num_qs, B, n_q) -> (B,).

    Low rather than mean because distance is -V: an optimistically high V
    understates distance, and graph search actively selects for exactly that
    error. TQC already drops the top (most optimistic) quantiles from its
    target, so this extends the existing bias rather than fighting it.
    """
    return np.percentile(pooled_quantiles(q), percentile, axis=1)


def ensemble_spread(q):
    """Std across critics of their per-critic mean. (num_qs, B, n_q) -> (B,).

    Disagreement between independently-initialised critics is the cheapest
    available signal that a (state, goal) pair is off-distribution.
    """
    q = np.asarray(q, np.float64)
    return q.mean(axis=2).std(axis=0)


def value_to_steps(v, step_cost=STEP_COST_DEFAULT):
    """V-units -> non-negative steps.

    Clipped at zero: the critic may predict a small positive V for a state
    already at the goal, and a negative distance is meaningless downstream.
    """
    if step_cost <= 0:
        raise ValueError(f"step_cost must be positive, got {step_cost}")
    return np.maximum(0.0, -np.asarray(v, np.float64) / step_cost)


# MEASURED over 75 recorded eval episodes: 0.283 m per step (median, max 0.327)
# at ~11 km/h on a 0.1 s timestep.
METRES_PER_STEP = 0.283
H_METRES = 30.0
H_STEPS = H_METRES / METRES_PER_STEP        # 106.0


def goal_relabelled_obs(base_obs, fields, goal_pos, goal_block_fn):
    """Copy `base_obs` with its goal block recomputed for `goal_pos`.

    NO SIMULATOR NEEDED. The VAE latent is what the camera saw at this state
    and is independent of which goal the step is credited to, so only the 12-d
    goal block is rebuilt -- through the wrapper's OWN transform, so relabelled
    observations cannot drift from on-policy ones. Same contract as
    EpisodeRelabelBuffer._relabelled.
    """
    out = dict(base_obs)
    out["goal"] = goal_block_fn({**fields, "goal_pos": goal_pos})
    return out


def edge_reject_mask(cost_steps, spread, max_steps=H_STEPS, tau=None):
    """True where an edge must NOT enter the graph.

    Two independent reasons:
      * beyond H -- V was trained only on goals within the horizon, so a longer
        query is extrapolation and unclipped extrapolation is how the critic
        left its provable range entirely;
      * critics disagree -- the cheapest available off-distribution signal.

    `tau` of None disables the disagreement test, for when no spread
    distribution has been observed yet to set it from.
    """
    cost_steps = np.asarray(cost_steps, np.float64)
    reject = cost_steps > max_steps
    if tau is not None:
        reject = reject | (np.asarray(spread, np.float64) > tau)
    return reject

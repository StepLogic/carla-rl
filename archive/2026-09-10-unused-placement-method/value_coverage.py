"""ARCHIVED, NOT IN USE. See README.md in this directory.

SoRB coverage placement: kept because it is SGM's natural ablation partner,
archived because it was never one of the four baselines and never ran.
"""
import numpy as np

from vae_nav.value_distance import H_STEPS


def value_coverage_select(dist, max_steps=H_STEPS, order=None):
    """SoRB coverage: keep a candidate no existing node can already reach.

    NOT ONE OF THE FOUR ARMS. It is kept because it is SGM's natural ablation
    partner -- SGM's merge test minus the two-way consistency check -- and is
    the thing to reach for if the SGM result needs decomposing. Reachability is
    required in BOTH directions here, because a node the policy can drive to
    but not out of is useless as a waypoint; that is a different question from
    SGM's, which compares how two states behave toward every OTHER node.
    """
    dist = np.asarray(dist, np.float64)
    n = len(dist)
    if n == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    seq = list(range(n)) if order is None else list(order)

    members, covered_by = [], -np.ones(n, dtype=int)
    for i in seq:
        hit = -1
        for node in members:
            if dist[node, i] <= max_steps and dist[i, node] <= max_steps:
                hit = node
                break
        if hit < 0:
            members.append(i)
        else:
            covered_by[i] = hit
    return np.asarray(members, dtype=int), covered_by

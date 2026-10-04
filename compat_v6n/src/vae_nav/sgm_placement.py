"""Sparse Graphical Memory node placement.

Reimplemented for this stack from the reference TensorFlow code at
https://github.com/scottemmons/sgm (Emmons et al., "Sparse Graphical Memory for
Robust Planning", 2020), cloned to /home/robotlab/scratch/sgm. NOT a port: the
reference is TF1-era and built on tf_agents, while our critic is JAX. The
algorithm is small; the reference was read for its exact test, not its code.

WHAT SGM IS. SoRB plus one step at graph construction. Walking a set of
candidate states in order, each one is either MERGED into an existing node or
becomes a new node. It merges when some existing node is close in embedding AND
TWO-WAY CONSISTENT with it under the learned distance. Sparsity is what falls
out; it is not a budget imposed from outside.

THE TWO-WAY TEST, from planner.py's qvalue_consistency. For a candidate s and an
existing node w, over all current nodes k:

    C(s, w) = max(  max_k |d(s,k) - d(w,k)|,      outgoing: where can I get to
                    max_k |d(k,s) - d(k,w)|  )    incoming: who can reach me

Merge when C(s, w) < consistency_cutoff. Both directions are required: two
states can agree on everything they can reach while differing entirely in what
can reach them, and merging those breaks the graph in a way that only shows up
as failed plans much later.

WHY THIS IS A DEMANDING BASELINE HERE, AND WORTH RUNNING. Every quantity above
comes from the critic read as a distance. Measured on this project's two gated
arms, that critic correlates 0.98 with true time-to-goal with hindsight
relabelling and 0.48 without. So SGM should sparsify sensibly on the first and
poorly on the second, which makes it an independent test of that finding rather
than merely another baseline to beat.

THE ALGORITHM TAKES A DISTANCE MATRIX, NOT AN AGENT. Querying the critic is a
separate function below. That split is deliberate: the merge logic is the part
worth testing, and it should be testable without JAX, a checkpoint or a
simulator.
"""
import numpy as np

# Reference defaults, from SGMSearchPolicy.__init__ in the source repo.
EMBEDDING_CUTOFF = 0.05
CONSISTENCY_CUTOFF = 5.0


def embedding_neighbours(embeddings, candidate, cutoff=EMBEDDING_CUTOFF,
                         k=None):
    """Indices of existing nodes within `cutoff` of `candidate`, nearest first.

    A CHEAP PREFILTER, NOT THE TEST. Its only job is to keep the quadratic
    consistency check off obviously unrelated states; the merge decision is
    made entirely by two_way_consistency below. Setting the cutoff too tight
    costs sparsity, never correctness.
    """
    if len(embeddings) == 0:
        return np.empty(0, dtype=int)
    emb = np.asarray(embeddings, np.float64)
    d = np.linalg.norm(emb - np.asarray(candidate, np.float64), axis=1)
    idx = np.argsort(d)
    idx = idx[d[idx] <= cutoff]
    return idx if k is None else idx[:k]


def two_way_consistency(dist, candidate, node, members):
    """C(s, w): the larger of the outgoing and incoming disagreements.

    `dist` is a full square matrix in STEPS over every state under
    consideration; `members` are the indices currently acting as graph nodes.

    Returns np.inf when there is nothing to compare against, so the first
    candidate can never be merged into an empty graph.
    """
    if len(members) == 0:
        return np.inf
    d = np.asarray(dist, np.float64)
    m = np.asarray(list(members), dtype=int)
    outgoing = np.max(np.abs(d[candidate, m] - d[node, m]))
    incoming = np.max(np.abs(d[m, candidate] - d[m, node]))
    return float(max(outgoing, incoming))


def sgm_select(dist, embeddings, embedding_cutoff=EMBEDDING_CUTOFF,
               consistency_cutoff=CONSISTENCY_CUTOFF, order=None):
    """Choose graph nodes. Returns (node_indices, merged_into).

    node_indices  the states kept as nodes, in the order they were added
    merged_into   for every state, the node it merged into, or -1 if it is one

    ORDER MATTERS AND IS EXPOSED. SGM is greedy and sequential: an earlier
    state becomes a node and later ones merge into it. Walking the states in
    trajectory order is the faithful reading of the reference, which consumes a
    replay buffer in collection order. It is exposed so the dependence can be
    measured rather than assumed away.
    """
    n = len(dist)
    if n == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    seq = list(range(n)) if order is None else list(order)

    members, merged_into = [], -np.ones(n, dtype=int)
    for i in seq:
        best = None
        for w in embedding_neighbours(
                [embeddings[m] for m in members], embeddings[i],
                cutoff=embedding_cutoff):
            node = members[int(w)]
            c = two_way_consistency(dist, i, node, members)
            if c < consistency_cutoff:
                best = node
                break          # nearest consistent neighbour wins, as reference
        if best is None:
            members.append(i)
        else:
            merged_into[i] = best
    return np.asarray(members, dtype=int), merged_into


def pairwise_steps(agent, observations, fields, positions, goal_block_fn,
                   step_cost, percentile=None):
    """Full square matrix of predicted steps, dist[i, j] = i -> j.

    Built by relabelling state i's goal block to node j's position and reading
    the critic, which needs NO SIMULATOR: the perception latent is what the
    camera saw and does not depend on which goal the step is credited to.

    STEP_COST MUST BE THE ARM'S OWN. value_distance defaults to 0.1, while the
    gated arms train at 1.0. Passing the wrong one scales every edge weight by
    ten, which does not fail -- it silently produces a graph whose distances are
    all out by an order of magnitude.
    """
    from vae_nav.value_distance import edge_cost_steps, goal_relabelled_obs

    n = len(observations)
    dist = np.zeros((n, n), np.float64)
    kw = {} if percentile is None else {"percentile": percentile}
    for j in range(n):
        batch = [goal_relabelled_obs(observations[i], fields[i], positions[j],
                                     goal_block_fn) for i in range(n)]
        stacked = {k: np.stack([b[k] for b in batch]) for k in batch[0]}
        cost, _spread = edge_cost_steps(agent, stacked, step_cost=step_cost,
                                        **kw)
        dist[:, j] = np.asarray(cost, np.float64)
    return dist


def summarise(members, merged_into, path_length_m):
    """Node count and density, the two numbers a placement method is judged on.

    Density rather than raw count, because the arms are compared across
    trajectories of different lengths and a count alone is not comparable.
    """
    n_nodes = int(len(members))
    return {
        "n_nodes": n_nodes,
        "n_merged": int(np.sum(merged_into >= 0)),
        "path_length_m": float(path_length_m),
        "nodes_per_100m": (float(n_nodes) / max(path_length_m, 1e-9) * 100.0),
    }

# Experimental Setup

This section specifies the policy, its training, and the mapping and evaluation
pipeline used for the node-placement study. All quantities are stated in the
units the implementation uses; where a constant is load-bearing, the reason is
given rather than the value alone.

## A. Simulation and Task

Experiments are run in CARLA 0.9.15 at Epic rendering quality. The perception
front-end is a frozen autoencoder trained on Epic-quality frames, so rendering
quality is a fixed property of the setup and not a tunable: lowering it would
shift the input distribution out from under the encoder.

The agent controls a single ego vehicle in Town02. A task instance is a
point-goal: the agent is given a goal pose sampled at a controlled geodesic
distance from its start and must arrive within an arrival radius. Goal distance
is sampled uniformly within a band `[GOAL_DIST_MIN, GOAL_DIST_MAX]`; training
pins this band to 5-20 m, and evaluation additionally reports 20-40 m and
40-70 m, which are extrapolation with respect to the training distribution.

**Action space.** Two-dimensional continuous control, `a = (steer, throttle)`,
applied as absolute commands rather than increments. Steering is bounded to
±0.5 and throttle to [0, 0.6], with a target speed of 20 km/h. The control
actually applied to the simulator — not the raw action — is what the observation
records (Section B), because under incremental control the two differ.

## B. Observation Space

The policy observes a dictionary of four keys, 145 dimensions in total. The keys
are kept separate rather than concatenated into one flat vector so that each is
projected independently; flattening a heterogeneous vector behind a single
normalisation layer mixes quantities on incomparable scales.

| key | dim | contents |
| --- | --- | --- |
| `latent` | 128 | frozen convolutional autoencoder embedding of the 120×120 RGB frame |
| `goal` | 12 | egocentric goal offset (2), heading as (sin ψ, cos ψ) (2), action history (8) |
| `vector` | 4 | proprioceptive state, bounded to [-5.1, 5.1] |
| `mode` | 1 | commanded task mode, 1 = goal-reaching, 0 = exploration |

**Perception.** The `latent` block is produced by an autoencoder trained offline
on driving frames and held frozen throughout policy training and evaluation. Its
identity is recorded in every checkpoint and asserted before evaluation, since
pairing a policy with a different encoder does not fail loudly — it produces
confident nonsense.

**Goal encoding.** The goal offset is expressed in the vehicle frame as
(forward, left) components of the goal displacement. CARLA's coordinate frame is
left-handed, with basis vectors `forward = (cos ψ, sin ψ)` and
`right = (-sin ψ, cos ψ)`; the lateral component is therefore the negation of the
projection onto `right`. This is stated explicitly because an earlier
implementation applied a textbook right-handed rotation and mirrored the lateral
bearing, which instructed the policy to steer away from the goal at every turn
and manifested only as "drives competently but never arrives."

The action history comprises the last four applied controls (4 × 2), giving the
policy the short-horizon context needed to produce smooth control.

**Mode conditioning.** The commanded mode is an *external task bit*, not a
learned switch. It occupies its own observation key, and in exploration mode the
goal block is **zeroed rather than omitted**, because a single actor cannot
accept two input widths. The permanently-zero channel in exploration mode is the
acknowledged cost of gating.

## C. Reward Functions

The policy is trained on two rewards selected by the mode bit, drawn once per
episode with probability 0.5.

**Goal mode — minimum time.** The goal reward is the SoRB-style minimum-time
objective:

```
r_t = -1        for every step before arrival
r_t =  0        on arrival (terminal)
```

with an episode cap of 1000 steps, so the undiscounted return lies in
[-1000, 0] and the optimal value function is exactly the negated time-to-goal.
No shaping, no arrival bonus, and no coefficients to balance. This choice is not
incidental: the study consumes the critic **as a distance**, and only under this
reward does `Q(s, a)` read as `-(steps to goal)`.

A vehicle that becomes irrecoverably wedged is billed *exactly* the remaining
clock, `-(1 + steps remaining)`, so terminating early is worth precisely
nothing. Without this, any terminal failure would stop the per-step meter and
crashing would become optimal.

**Exploration mode — coverage.** The exploration reward pays displacement
projected onto the road direction, gated on being under target speed, against a
small per-step cost:

```
r_t = -4e-3 - |Δsteer| + progress_t · 1[speed ≤ target]
      - 3.0 · 1[collision] - 5.0 · 1[road departure]
```

**Baseline reward.** The shaped baseline used for comparison receives the
environment's own per-step distance reduction plus projected progress under the
same speed gate, minus terminal penalties for collision, lane departure and
idling, and a per-step steering-smoothness cost. This is the reward the
upstream driving codebase supplies; it is included because a method that cannot
beat the reward practitioners actually use has not demonstrated anything.

**Lane-crossing convention.** Under `LANE_INVASION_NOOP`, crossing a painted
lane marking costs nothing and terminates nothing, while leaving the drivable
surface remains fatal. Crossing markings is unavoidable when turning through
junctions, and treating every crossing as failure terminated 90% of episodes at
44-70 steps in an earlier configuration.

## D. Terminal Conditions

Termination differs by mode, and the difference is material: it determines what
a success rate is comparing.

**Goal mode.** *Only arrival terminates.* The environment's own termination
signal — idle, falling, distance limit, lane invasion, collision, speed,
wrong-direction — is deliberately discarded, and the episode instead truncates
at the 1000-step cap. Under a per-step cost, any terminal failure would stop the
meter and become an escape hatch. A wedge terminates but is billed the full
remaining clock, so it is not an escape either.

**Exploration mode.** Collision or departure from the drivable surface
terminates; the episode truncates at 500 steps. Off-road is retained as fatal
because, with lane crossings free, it is the only remaining term that
discourages leaving the road at all — the progress term pays equally for driving
down a road and for circling.

**Baseline.** The shaped baseline terminates on arrival, collision, idling, lane
invasion or off-road, and additionally honours the environment's own
termination. Because this differs from the gated arms' rule, two relaxations are
provided so the same frozen checkpoint can be evaluated under the gated rule:
one making lane crossings non-terminal, and one in which only arrival
terminates. Both suppress the environment's own termination as well as the
wrapper's; relaxing only the latter leaves the episode ending anyway, silently,
because the environment ORs its raw lane flag into its own done signal.

## E. Policy Learning

The policy is trained with Truncated Quantile Critics (TQC): 5 critics × 25
quantiles, dropping the 2 most optimistic quantiles per network, MLP trunks of
(256, 256), Adam at 3 × 10⁻⁴ for actor and critics and 10⁻⁵ for the temperature,
τ = 0.005, initial temperature 0.2, and a fixed target entropy of −2.0 for a
two-dimensional action space.

**Two critics, two discounts.** Goal and exploration modes have separate critic
networks, because their returns are on different scales and have different
horizons. The goal critic uses **γ = 1.0**, which is mandatory rather than
preferred: at γ = 0.99 the effective horizon is 100 steps, the value saturates
near −10, and a 400-step goal becomes indistinguishable from a 1000-step one
(Q differs by 0.18), destroying exactly the distance reading the study depends
on. Undiscounted return is safe here only because every episode terminates or
truncates within the cap. The exploration critic uses γ = 0.99.

**Batch composition.** Transitions are stored in a mode-partitioned replay
buffer and every batch is drawn 50/50 across modes, so neither critic starves
when episode lengths differ between modes (the goal cap is 1000, exploration
500).

Training runs for 350k environment steps in Town02, with the goal-distance
curriculum pinned to 5-20 m. Two arms are trained identically except for
hindsight relabelling of exploration episodes into goal-mode data; the wrapper
scripts differ only in that flag and the output directory, so the arms cannot
drift apart in any other respect.

## F. Mapping Algorithm

The map is a directed topological graph over states visited by the policy
itself. Construction has three stages and requires no simulator after the first.

**1. Trace collection.** The policy is driven in *exploration* mode for 60
episodes of 400 steps, recording pose, observation, and the fields from which
the goal block is computed. Exploration mode is used because it produces
coverage; the goal half drives direct lines to sampled goals and would map only
those. Candidates are thinned to one per 5 m as they are recorded, since
consecutive simulator steps are ≈0.28 m apart and the next stage is quadratic in
the candidate count. This yields 1,510 candidates over 7,464 m of driving.

**2. All-pairs learned distance.** For every ordered pair (i, j), state i's
observation is relabelled so its goal block points at candidate j's position,
and the goal critic is queried. This requires **no simulator**: the perception
latent records what the camera saw and is independent of which goal the step is
credited to. Relabelling uses the wrapper's own transform, so relabelled
observations cannot drift from on-policy ones. Two implementation constraints
are load-bearing: observations must be forced to goal mode before any query
(the goal block is zeroed at mode 0, and querying the goal critic with
exploration-mode inputs reads the wrong half of a gated policy), and the step
cost must be the arm's own, since a mismatch scales every edge weight by an
order of magnitude without failing.

**3. Node selection and edges.** Four placement rules select nodes from the
shared candidate set (Section H). Edges are then formed between selected nodes,
directed and weighted by the predicted number of steps. An edge is **rejected**
when its predicted cost exceeds the horizon on which the critic was trained
(H = 30 m ≈ 106 steps): beyond that the critic is extrapolating, and graph
search actively selects for optimistic extrapolation error. Nothing is
symmetrised — a one-way street is genuinely asymmetric and the critic represents
that.

Critic-ensemble disagreement is deliberately *not* used as an edge filter.
Measured on this project's agents, disagreement is a proxy for leg length rather
than local uncertainty (AUC 0.533 — chance — once length and Q-magnitude are
regressed out) and has the wrong sign, so rejecting on it would discard the
short, reliable edges.

## G. Localization Algorithm

Localization is by nearest-node lookup in the plane. At episode start, the
current pose and the goal pose are each associated with the nearest graph node
within a radius of 25 m. The radius is bounded by the policy's own competence
rather than tuned: its training band is 5-20 m, so a node within ~25 m is a leg
the policy can be expected to drive; a larger radius would assert entry points
it cannot actually reach.

Localization has three distinct failure modes, and they are reported separately
rather than pooled into the success rate, because they answer different
questions: the start pose has no node within radius, the goal pose has none, or
both localize but the graph contains no path between them. The last is a
statement about graph connectivity, not about localization.

## H. Node Placement Methods

Four rules select nodes from the *same* candidate set, so placement is the only
variable. They differ in what their free parameter is, which determines how each
is swept.

- **uniform** — nodes at equal arc-length spacing, taken at interval midpoints.
  Uses no learned quantity, and is the baseline a method must beat to justify
  its cost.
- **random** — distinct candidates drawn uniformly at random. The floor.
- **policy** — peaks of the policy's own action entropy, smoothed, taken
  greedily by height subject to a minimum separation. Entropy is estimated as
  `-log π(a|s)` for a single sample, the same estimator reported during
  training. The hypothesis is that the policy is least certain where the route
  is *decidable* — at junctions — making entropy a free junction detector.
- **sgm** — Sparse Graphical Memory's two-way consistency merge, in which a
  candidate is merged into an existing node when the two agree, within a
  cutoff, about their distances *to* and *from* every current node.

The first three have no intrinsic node count and are swept over the number of
nodes requested. SGM derives its own sparsity from its consistency cutoff; it is
therefore swept over that cutoff, with node count as an **output**. No shared
node budget is imposed, since the object of study is the Pareto frontier of node
count against navigation performance.

Two implementation notes affect reproducibility. SGM's published cutoffs are
expressed in its reference implementation's value scale and do not transfer: at
the published values nothing merges, and the algorithm silently degenerates to
retaining every candidate. Cutoffs here are therefore derived from measured
quantities on the target critic, and a zero-merge outcome is treated as a hard
error rather than a result. Distances are additionally clipped at the critic's
horizon for the merge test, because the test is a maximum over all nodes and
would otherwise be decided by extrapolated values.

## I. Evaluation Algorithm

**Episode set.** Evaluation uses a fixed episode set drawn from a generator
never touched by training, so every method drives identical start states and
goals. The environment is reset with each episode's own seed; omitting this
makes the fixed set decorative, since the simulator then samples whatever it
likes and "identical episodes across methods" silently fails to hold.

**Navigation.** Per episode: localize start and goal (Section G), compute a
shortest path over the directed graph by Dijkstra with predicted steps as edge
weights, and drive the sequence of nodes by rewriting the policy's goal block to
the current waypoint. A waypoint is advanced when the vehicle is within 4 m; a
leg not completed within 200 steps is abandoned and the next waypoint is
adopted, since a leg the policy cannot drive is an edge the graph should not
have offered. **The final waypoint is always the true goal**, not the last node.
Episodes for which the graph offers no path are still driven, with the true goal
alone; excluding them would score a map on the subset it happened to cover and
would flatter a sparse or poorly connected graph.

**Metrics.** Success (SR) is the environment's own arrival flag, so it is
identical to the metric used elsewhere in this codebase. Success weighted by
Path Length is

```
SPL = (1/N) Σ  S_i · ℓ_i / max(ℓ_i , p_i)
```

where `ℓ_i` is the length of CARLA's own shortest driving route between start
and goal, obtained from the map's route planner, and `p_i` is the distance the
agent actually drove. The geodesic is a road route rather than a Euclidean
distance, since a straight line through a building is not a shortest path and
would credit every method with detours it did not avoid.

Two diagnostics are reported alongside: the number of episodes the graph could
not route, and the number of subgoal legs abandoned on the step budget.

**Metrics are refused where undefined rather than reported as zero.** Success is
not reported for an arm that cannot observe the goal; SPL is reported as *not
applicable*, never 0.000, when no road route is available, since a zero reads as
"took an infinite detour" rather than "was not measured"; and critic calibration
is reported only for arms trained on the minimum-time reward, since a shaped
arm's critic predicts a shaped return and scoring it against elapsed steps
measures nothing. Calibration is computed on arriving episodes only, a timeout
having no true cost-to-go.

## J. Reproducibility Notes

All arms share one trace, one distance matrix, one edge rule, one episode set,
and one policy; placement is the only variable. Results are keyed by band, since
pooling an in-distribution result with an extrapolation one produces a number
describing neither. Every reported configuration is recorded in the result file
alongside the metrics, including the node count actually placed, the consistency
cutoff actually used, and the step cost, so that a row can be checked against
the configuration that produced it rather than against the configuration that
was intended.

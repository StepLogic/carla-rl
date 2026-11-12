# Training a SORB-style agent in CARLA

**Date:** 2026-09-01
**Status:** design approved, not yet implemented

## What this is

A goal-conditioned policy for CARLA whose **value function is a usable distance
metric**. The policy is deliberately short-horizon; long-range navigation comes
from graph search over that metric, as in Search on the Replay Buffer (Eysenbach
et al., 2019).

**The deliverable is V, not success rate.** A policy that reaches 90% of goals
with a value function that cannot rank two edges is useless here; one at 70%
with a well-ordered V is not. Every decision below follows from that.

The graph itself — node selection, trace thinning, the node-selector comparison
against the Habitat peer — is **out of scope** and gets its own spec. This one
produces a trained and validated V; that one consumes it.

## Why the previous configuration was abandoned

Measured, in the runs that preceded this design:

| observation | value |
|---|---|
| policy success, 20–31 m goals | 0.84, lane crossings 0.24 |
| policy success, 100 m goals | 0.28, lane crossings 0.85 |
| eval mean over 10 evals, 20–113 m | 0.440 (in-distribution) |
| scripted PID expert, 100 m goals | **0.000 over 100 episodes** |
| `q` against a −200 return floor | **−279.85**, critic_loss 38.4, td 1.71 |

Two independent failures. The environment rules were unsatisfiable — a
`GlobalRoutePlanner` route under PID control could not complete them, which is
not a hard task but a broken one. And the critic left the provably achievable
range, which for a project whose product is V is disqualifying on its own.

## Decisions

### 1. Short horizon; the graph provides range

SORB's policy is reliable only nearby and the graph composes short hops. The
previous configuration inverted this, training directly at 100–200 m — which
defeats the purpose, since the graph exists precisely so the policy need not
generalise to 200 m. Our own data agrees with SORB: 0.84 success at 20–31 m
against 0.28 at 100 m.

### 2. H = 30 m

H is simultaneously the policy's reliable range and the maximum graph edge; the
graph may only connect nodes the policy can actually traverse.

At the measured **0.283 m/step** (median over 75 recorded eval episodes, ~11
km/h on a 0.1 s timestep), 30 m is ~106 steps. A 200 m route needs ~7 hops.

Per-hop reliability compounds: `0.95^7 = 0.70`, `0.90^7 = 0.48`. Per-hop success
is therefore a first-class metric, not a footnote.

### 3. Arrival and timeout are the only terminals

**No wedge.** An absorbing failure makes V mean "distance to goal *or* death",
and the graph reads V as travel time. A death term routes Dijkstra around
phantom cost.

Collisions and lane crossings carry no penalty and do not end the episode. Their
cost is the time they lose, which is self-calibrating: a crash that costs 8
steps of recovery costs 8; one that ends progress costs the remaining horizon.
No coefficient to defend.

This reverses the `LANE_WEDGE` mechanism used previously. It is also what
finally makes the environment satisfiable — the wedge's rules were failing a PID
expert on 78% of episodes via `off_road` alone.

### 4. Absolute actions

The action is the control directly. Chosen for parity with the Habitat peer,
which keeps the cross-simulator comparison honest.

**Known cost, stated rather than hidden:** nothing penalises steering rate, so
sawing at the wheel is free, and lane crossings never fell below 0.58 at long
range or 0.24 at short range across the whole project. Under this design a
crossing costs only time rather than the episode, which reduces but does not
eliminate the concern. The incremental action space (measured p95 |accel| 0.15
m/s² vs 196.0 absolute under random actions) remains the untested alternative if
lane drift dominates the per-hop failure rate.

### 5. Goal band 2–30 m, uniform

V must be accurate everywhere the graph queries it, and graph edges between
adjacent trace nodes are mostly **short**. A V trained only on 20–30 m goals
would assign extrapolated weights to the majority of its own edges.

This retires the 20 m floor, which was correct when success rate was the
product and is wrong now that V is.

## The MDP

| | value | rationale |
|---|---|---|
| reward | `−0.1` per step, `0` on arrival | pure time |
| terminal | arrival only | no contamination of V |
| truncation | 300 steps | ~2.8× the 106 steps H needs |
| γ | 1.0 | `V = −0.1 × steps` exactly |
| V range | **[−30, 0]** | bounded by construction |
| goal band | 2–30 m uniform | see decision 5 |
| actions | absolute, steer ±0.5, throttle ≤0.6 | peer parity |
| `LANE_WEDGE` | 0 | see decision 3 |
| towns | Town02 train; Town03, Town10HD held out | held-out eval has never run |

### `STEP_COST = 0.1`, not SORB's `−1`

SORB's convention makes `V = −steps` read directly. It is rejected here because
`quantile_huber_loss` computes `huber / kappa`, and mis-scaling that has already
silently killed a run in this project — `kappa=60` against small rewards printed
`critic_loss 0.00` at 22k steps. `STEP_COST=0.1` with `kappa=1.0` is the exact
regime that ran healthily for 500k steps. Steps are recovered as `−V / 0.1`.
Cosmetic fidelity is not worth re-entering a measured failure mode.

### The curriculum is removed

With a fixed 2–30 m band there is nothing to advance. `GOAL_DIST_MIN/MAX` are
pinned and the promote/demote/stall machinery is inert. This also retires the
stall-on-resume bug (the stall counter fired on a timer at every resume because
`curr_max` was restored at target while `last_advance_ep` reset to 0) rather
than depending on its patch.

### The 300-step cap is load-bearing

It is what bounds the undiscounted return under γ=1, and it is 6.7× tighter than
the 2000-step cap under which `q` reached −280. With `TQC_TARGET_MIN=-30`,
values outside the achievable range become unrepresentable rather than merely
unlikely. Clipping targets to `[−30, 0]` adds information rather than bias: every
quantile of a return distribution supported on that interval lies within it, so
a target outside it is unrealisable.

## V as an edge weight

### Querying arbitrary pairs needs no simulator

`Q(s_i, g_j)` is computed from the stored trace. The VAE latent at node `i` is
what the camera saw and is independent of which goal the step is credited to, so
only the 12-d goal block is recomputed via `VAEGoalWrapper.goal_block`. This is
the mechanism `EpisodeRelabelBuffer._relabelled` already uses and has tests
around. `V(s_i) = Q(s_i, π(s_i))` with the actor's action under goal `g_j`.

### Pessimistic aggregation

Distance is `−V`, so *underestimating* distance means V is too high. That is the
error mode that wrecks a graph: Dijkstra preferentially selects edges it believes
are cheap, so optimistic errors get chosen and pessimistic ones ignored. Search
concentrates the error rather than averaging it out.

Units are stated explicitly because the two natural ones differ by 10x and
mixing them is an easy, silent error:

```
q            125 quantiles (5 critics x 25), in V-units, range [-30, 0]
v_pess     = percentile(q, 10)              V-units, conservative (more negative)
cost_steps = -v_pess / STEP_COST            steps; STEP_COST = 0.1

reject if  cost_steps > 106                 beyond H = 30 m; V is extrapolating
reject if  ensemble spread > tau            critics disagree -> untrusted edge
```

Edge weights passed to Dijkstra are `cost_steps` — non-negative, in steps, and
directly comparable to the geodesic ground truth in the validation section.

TQC already drops the **top** quantiles from its target, which in a negative-cost
formulation are the optimistic ones. The default is therefore already biased the
way SORB needs; a low percentile at inference extends that rather than fighting
it. `tau` is to be set from the observed spread distribution, not guessed.

### The graph is directed

This is where CARLA departs from the Habitat peer. Their free-space navmesh is
roughly symmetric, so `d(i,j) ≈ d(j,i)`. A road network is not — one-way
streets, medians, lane direction. Edges are stored directed and V is queried
separately per direction. Assuming symmetry would produce shortcuts through
oncoming traffic.

### Clipping at H is what makes short-horizon training sound

V is trained only on goals ≤30 m, so any query beyond that is extrapolation, and
unclipped extrapolation is how `q` reached −280. Rejecting edges past H means
the graph only asks V questions it was trained to answer.

## Validation

The primary metric is **rank correlation, not absolute error**. Dijkstra depends
only on the relative ordering of edge costs; a V uniformly 30% too long still
routes correctly. Absolute calibration is secondary and affects only whether the
reported cost is meaningful.

Two independent ground truths, which is what makes this a test rather than a
self-consistency check:

1. **Realised steps** — successful eval episodes: `V(s_0, g)` against steps
   actually taken.
2. **Geodesic** — `GlobalRoutePlanner` along-road distance / 0.283 m per step.
   Covers pairs the policy fails, where realised steps do not exist.

### Acceptance gates

Thresholds are estimates until the first measurement exists, and are labelled as
such rather than presented as derived.

| gate | threshold | rationale |
|---|---|---|
| Spearman ρ(−V, true steps) | > 0.8 | what routing needs |
| median absolute relative error | < 25% | reported costs mean something |
| V within [−30, 0] | **100%** | hard gate; this is what failed at q = −280 |
| per-bin ρ (2–5, 5–10, 10–20, 20–30 m) | none < 0.5 | catches "fine at 30 m, garbage at 3 m" |
| per-hop success at 30 m | measured, not gated | `success^hops` sets end-to-end feasibility |

**Per-bin is mandatory.** The 2–30 m band exists because the graph queries V at
short edges. If V is accurate at 25 m and meaningless at 3 m, pooled ρ still
looks acceptable while every short edge is garbage.

**Negative control.** Run the identical evaluation against an untrained critic.
If a random V scores ρ = 0.6 through goal-block geometry alone, then 0.8 is a far
weaker result than it appears.

**Held-out towns are part of this, not a later step.** Every number produced by
this project so far is in-distribution Town02, and that has been true while
in-distribution numbers were reported as results.

### If V fails the gates

In order: re-enable HER for denser short-range distance supervision (currently
off by explicit instruction, and it is the natural fix for sparse coverage);
then more steps; then reconsider a supervised distance head. Named now rather
than improvised later.

## Explicitly out of scope

- Node selection and trace thinning, and the node-selector comparison against
  the peer. Separate spec, consuming this one's V.
- Dijkstra planning and end-to-end 100–200 m routing.
- The long-horizon baseline arm (a policy trained directly at 100–200 m, to show
  what the graph buys). Deferred, not rejected.

## Open questions

- **`tau`**, the ensemble-disagreement rejection threshold, has no defensible
  value until the spread distribution is observed on a trained critic.
- **Whether absolute actions hold up at H = 30 m.** Lane drift never improved
  across 1600 episodes under the previous configuration. It now costs time
  rather than the episode, which should reduce its impact, but that is a
  prediction and not a measurement.

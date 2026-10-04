# Junction-exit novelty for the explore arm

**Date** 2026-09-30 · **Status** design, awaiting review · **Arm** explore (goal mode untouched)

## Intent

The explore policy loops the same places **within a single episode**. The aim is a reward
term that pays for taking a junction exit it has not already taken this episode, so
exploration branches instead of circling.

Success is measured as **distinct `(junction_id, exit_road_id)` pairs per episode**, compared
against the same recipe with the term off. Secondary: explore `dist` and the `reached`
end-reason share should not fall.

## Why it loops today

`explore_reward` pays `PROGRESS_SCALE * progress` where `progress` is displacement along the
**car's own heading**, with no projection onto the road (user decision 2026-09-11, "remove
projection to road"). Driving a tight circuit therefore earns the full progress term on every
step, indefinitely, and a loop around a block crosses no painted line and leaves no road — so
it is both safe and maximally profitable. **Looping is the optimum of the current reward, not
a failure of exploration.** Nothing in the explore reward decreases on revisiting a place.

The existing `K_NOVELTY` view-novelty bonus does not address this and is disabled in the gated
stack. Its own comment records that "a car driving tight circles CAN [earn it] -- every heading
is a different image"; `K_NOVELTY=0.2` exists to bound the bonus *below* the progress term so
circling does not beat driving. It is a safety bound, not an anti-loop mechanism.

## Design

### 1. Publish junction identity

`src/configs/train_env_config_v4_image_goal.py:961-963` already fetches the hero's waypoint and
derives `in_junction`:

```python
_jwp = core.map.get_waypoint(hero.get_transform().location, ...)
_in_junction = bool(_jwp is not None and _jwp.is_junction)
```

Add, on that same waypoint (no additional map query):

- `junction_id` — `_jwp.get_junction().id` when `is_junction`, else `None`
- `road_id` — `_jwp.road_id`, published every step

Both go into `info` beside `in_junction` (emitted at `:1115`).

### 2. Detect a traversal

In the gated wrapper, track the **falling edge** of `in_junction`. On the step the car leaves a
junction, form the pair `(junction_id_entered, road_id_now)`, where `junction_id_entered` is the
id latched on the rising edge. A traversal is recorded once, on exit, not per step inside.

The exit is identified by the **outgoing `road_id`** rather than a discretised left/straight/right.
Road id is exact, unambiguous at junctions whose exits are not 90 degrees apart, and needs no
angle threshold.

### 3. Episode-local visited set

`self._junction_exits = set()`, reset in `reset()` (`gated_reward.py:946`) alongside `_t` and
`_path_m`. It is **episode-local**: cross-episode diversity is explicitly out of scope.

### 4. The reward

On a recorded traversal whose pair is not in the set: add `EXPLORE_JUNCTION_NOVELTY` to that
step's reward and insert the pair. A repeat pays nothing. Default **0.0**, so every existing
run is byte-identical.

Proposed value **10.0**, about four steps of progress (~2.4/step). It cannot be farmed by
circling: a loop re-traverses pairs it already holds and earns zero on every repeat.

**Where it is applied.** `explore_reward(info, disp)` is a pure function with no episode state
and is unit-tested as such; the visited set is per-episode wrapper state. The bonus is therefore
added **in the wrapper's `step()`**, on top of `explore_reward`'s return, exactly as
`EXPLORE_ARRIVAL_BONUS` already is (`gated_reward.py:1125`). `explore_reward` stays pure and its
existing tests stay valid. The pure part — "is this pair new, and what does it pay" — is factored
into a small helper so it can be tested without CARLA.

### 5. Edge cases

- **Episode ends inside a junction.** No falling edge occurs, so no traversal is recorded and
  nothing is paid. Correct: the exit was never taken.
- **Missing ids.** If `junction_id` or `road_id` is `None` (map query failed, or the waypoint is
  off-road), the traversal is discarded rather than keyed on `None` — a `None` key would collapse
  distinct junctions into one pair and silently suppress later bonuses.
- **Entering and leaving on the same road.** A U-turn inside a junction yields an exit road equal
  to the entry road. It is still a legitimate pair and is treated as one; no special case.

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Exit identity | outgoing `road_id` | exact; no angle thresholds; handles 5-way junctions |
| Observation | **unchanged** | keeps checkpoint compatibility and comparability with v29/v36 |
| Scope | episode-local | cross-episode diversity is a separate problem |
| Recurrence | **not in this change** | see Risks |

## Risks

**The reward is not observable, and this is the main risk.** The policy sees one 512-d image
latent. Whether it has already taken an exit this episode is nowhere in that input, so the
critic cannot predict the bonus: at a given junction it sometimes pays 10 and sometimes 0, and
the critic will fit the **average**. The per-state value carries no gradient toward choosing a
*different* exit.

Trajectory-level signal survives — routes covering more distinct pairs do return more, and the
actor is updated through the critic on whole trajectories — but the effect may be weaker than
intended, and the plausible failure mode is that the term encourages **junction traversal in
general** rather than **exit diversity** specifically.

This is accepted deliberately (decision (b)) to avoid changing the observation width, which
would invalidate every existing checkpoint. The closing test below is designed to detect
exactly this failure rather than let it pass as success.

The Markov fix, if the test fails, is a small per-junction "exits already taken" vector appended
to the observation; a GRU is the heavyweight alternative and is out of scope here.

## Testing

Unit (pure, no CARLA), mirroring how `explore_lane_fatal` is tested:

1. A new pair pays `EXPLORE_JUNCTION_NOVELTY` exactly once.
2. The same pair on a later traversal pays 0.
3. Two different exits from the same junction are two distinct pairs.
4. The same exit road from two different junctions are two distinct pairs.
5. The set is empty after `reset()`.
6. Default 0.0 leaves the step reward bit-identical to today.
7. No bonus is paid while inside a junction — only on the falling edge.

Behavioural, on a smoke run:

8. **Distinct `(junction, exit)` pairs per episode** rises against the control.
9. **Junction traversals per episode** — if this rises while distinct pairs per traversal does
   not, the term is rewarding traversal rather than diversity, which is the risk above.

## Flags

| Flag | Default | v37 |
|---|---|---|
| `EXPLORE_JUNCTION_NOVELTY` | 0.0 | 10.0 |

## Out of scope

Cross-episode diversity; a GRU or any recurrence; goal-mode changes; the view-novelty
`K_NOVELTY` term, which stays disabled.

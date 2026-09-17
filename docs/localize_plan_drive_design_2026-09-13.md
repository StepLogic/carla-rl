# Localise → plan → drive on saved maps — design decisions (2026-09-13)

Status: **designed, scripts written and compiled, NOTHING RUN.** The user asked to talk
before starting; the plan-only run awaits an explicit go-ahead. Driving comes after the
plan-only results.

## 1. What the policy observes (v6n, `VAEGoalWrapper`, 401-d)

| key | dim | content | source |
|---|---|---|---|
| `latent` | 384 | MobileNetV3-Small features of the car's CURRENT camera frame | camera only |
| `goal` | 12 | [forward, left] = (goal − hero_pos) rotated by hero_yaw; [sin, cos] of the CAR's yaw; last 4 actions (steer, throttle) | **true world poses** |
| `vector` | 4 | steer/max, throttle/max, speed/target_speed (clip 0–1), goal-visible flag | car state |
| `mode` | 1 | 1 goal, 0 explore (explore zeroes the goal block) | commanded |

**The goal is a relative position from ground truth, not a goal image.** Consequences:
- Image-based localisation can only use `latent` (compare live latent to stored node latents).
- Any critic query (the "fewest predicted steps" ranking, the all-pairs trace matrices) builds the goal
  block from the TRUE car pose and the node position, so it is NOT image-based.
- When driving, every target handed to the policy is a world position.

## 2. Plan-only pipeline — user decisions

| # | question | decision |
|---|---|---|
| 1 | critic step uses true pose | **A** — keep the latent+critic variant, labelled "uses true pose", reported beside image-only |
| 2 | starts are trace points the maps were built from (best case) | fine, no exclusion variant |
| 3 | settle after spawn before reading the latent | **yes, 10 zero-action ticks** |
| 4 | localisation result | **image only**: nearest stored latent (cosine) among the map's nodes; shortlist k=5 for the critic variant |
| 5 | path cost | **every edge weight = 1** (hop count) |
| 6 | maps with disconnected chains ("no path") | fine — **report it** |
| 7 | output | **B**: summary table per map + one figure per map with 4 example episodes |

Earlier answers in the same thread: maps = **all of them**; scope = **plan only**; localisation =
top-k latent + critic (then refined by #4 to image-only as the result); episodes = **all 40, off-graph
goals by image** (spawn at the goal to capture a live latent).

### Inputs
- Trace: `results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl` (v6n@250k, Town02,
  seed 7, 10 km/h collection, deterministic actions, 1,155 candidates).
- Episodes: `results/episodes/town02_speed10_s7_episodes_short20_long20_seed0.pkl`
  (short = 20 pairs, straight-line 20–50 m; long = 20, route 150–400 m, 15 on-graph + 5 off-graph).
- Maps (7): junction (`results/maps/town02_junctions_uniform6_trace_speed10.json`),
  uniform[6 m] and uniform[2 m] trace-edge graphs (`results/figures/uniform_frame_trace_edges/`),
  entropy smooth9/nosmooth × high/low (`results/maps/town02_speed10_s7_*_driving_map.json`).

### Scripts (gated-impl worktree, untracked)
- `capture_episode_views.py` — stage A, CARLA: spawn at each of the 40 starts and 40 goals via
  `set_continue_origin` (origin + destination, lane-snapped, spawn check), hold 10 zero-action ticks,
  store live obs (latent, vector, mode, goal), fields, and ground-truth pose (scoring only) →
  `results/episodes/town02_speed10_s7_episode_views_seed0.pkl`.
- `localize_and_plan.py` — stage B, offline (GPU critic, no CARLA): per map × episode, image-only and
  latent+critic (true pose) localisation of start and goal, GT-nearest node reference, unweighted
  shortest path (hops) from image picks / critic picks / GT nodes → `results/localize_plan/`
  (`localize_plan_results.json`, `localize_plan_summary.csv`, `examples_<map>.png`; examples: short #2
  detour, short #10 direct, long #0 on-graph, long #1 off-graph).

## 3. Driving — options explained, user chose **C**

All options hand the policy world-position targets (the only goal it accepts).

**A — leg by leg on the planned node path** (as the placement harness does). The next path node
becomes the goal; within the 4 m arrival radius, advance; skip on a per-leg timeout; the true final goal
is the last target. + comparable with existing placement results. − long edges become long legs (junction
map up to 350 m; ≳100 m legs already timed out before); relies entirely on the initial localisation.

**B — follow each edge's waypoint polyline.** Targets are the stored trace candidates along each edge
(~2–6 m apart). + short, previously driven legs, far fewer timeouts; suits sparse maps with long edges.
− largely replays the recorded trajectory, so it tests the map's structure less; still depends on the
initial localisation; uniform graphs have no intermediate waypoints (their nodes are already close).

**C — re-localise at every node, then replan (CHOSEN).** On reaching a node (or every N steps), compare
the live latent to the map's node latents, take the nearest node, and replan the shortest path from
there to the goal. + recovers from a wrong initial localisation and from drifting onto a different route;
the real test of image-based localisation for navigation. − a mis-match to a similar-looking road
elsewhere can make the plan jump; needs rules for when to re-localise and how to avoid flip-flopping;
more compute per step; not directly comparable with the placement runs.

### C — user decisions (2026-09-13)

| # | question | decision |
|---|---|---|
| 1 | when to re-localise | **at each node** (on arrival at the current target node) |
| 2 | which re-localisation to accept | **only nodes near the current plan** (a nearest-latent match elsewhere is rejected, so look-alike roads cannot make the plan jump) |
| 3 | what to drive between nodes | **hand the NEXT NODE as the policy's goal** (its world position), A-style legs |
| 4 | termination and metrics | **yes, as proposed**: success radius 4 m; per-leg and total step caps; log re-localisation events, node jumps, replans, final distance, SR and SPL |
| 5 | off-graph goals | **capture an image of the goal, localise it to the map (goal node), plan and drive there; if the goal plan is finished and the goal is not reached, SWITCH TO EXPLORATION MODE** (mode bit → explore) |

### Procedure (C as decided)

1. Spawn at the start; settle 10 ticks; image-localise the start to a map node (nearest latent).
2. Goal: capture the goal view (spawn at the goal beforehand, as in the plan-only capture), image-localise
   it to a goal node.
3. Plan the shortest path (edge weight 1) start node → goal node.
4. Drive: the next path node's world position is the policy's goal (goal mode). On arrival within the
   radius, **re-localise** from the live latent, **accepting the match only if it is near the current
   plan**. If the accepted node differs from the expected one, replan from it.
5. After the last path node, target the true goal position.
6. **If the plan is finished and the goal is not reached, switch the policy to exploration mode.**
7. Log every re-localisation (expected vs matched node, accepted/rejected), node jumps, replans, legs,
   timeouts, final distance to goal, success (≤4 m), SPL.

### Settled for the driving run (user, 2026-09-13 14:15)

- **Maps / edges:** the two-way sequence maps (`results/maps/sequence/*_sequence.json`, edges usable both
  ways), all 7; all 40 episodes (280).
- **Following:** option C. Window for an accepted re-localisation = the expected node plus the next 3 nodes
  along the current plan; a different match replans from it. A leg that does not arrive within 200 steps
  also re-localises (same window); if the match is still the expected node, skip to the next node.
- **Reversed edges** (against the trace's driving direction): driven as planned, counted.
- **Termination:** success = within 4 m of the true goal; ends on collision or the cap
  2 × (planned path m / 0.24) + 300 steps. Lane crossings and off-road are logged, not terminal
  (LANE_WEDGE unset, so a crossing does not freeze the car; `NO_TIMELIMIT=1` removes the env's own cap).
- **After the plan:** one goal-mode leg to the true goal (200-step timeout), then explore mode
  (goal block zeroed) until the cap; applies to every episode, on- or off-graph.
- **Implementation:** `drive_plan_c.py` (CARLA loop), `plan_c.py` (pure rules, `tests/test_plan_c.py`),
  `summarise_drive_plan_c.py`. Output `results/drive_plan_c/sequence_undirected.jsonl`.

### Previously unspecified (proposals as written before the decision)
- **"Near the current plan":** proposed = within the next **3 hops** along the current path (or the
  current node). Other options: within X m of any plan node, or require a latent match margin.
- **Exploration-mode fallback termination:** proposed = end on reaching the goal (true distance ≤4 m, the
  success radius) or on the remaining total step budget. Also: does success in explore mode count as
  success (with SPL on the driven path)?
- **Scope of the fallback:** the user stated it for off-graph goals; proposed to apply it to ANY episode
  whose plan finishes without reaching the goal (including on-graph misses) — confirm.
- **Step caps:** per-leg cap (the harness's `leg_steps`) and total cap (e.g. 3,000 steps for long-horizon
  episodes at ~0.25 m/step) — confirm values.
- **Explore-mode goal handling:** in explore mode the goal block is zeroed, so the policy cannot see the
  goal; arrival is judged by true distance only.

## 4. Next step

Run plan-only (capture a few minutes in CARLA, then offline planning) **only after the user says go**;
then settle the open C questions before any driving run.

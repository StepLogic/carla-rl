# CARLA gated-policy node placement — results and findings, 2026-09-12/13

Policy: `checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl` (MobileNet, option A).
Placement harness: `eval_placement.py` in this worktree. Result JSONs, traces and
critic matrices live under `results/` (gitignored). Run scripts: `scripts/placement_runs_2026-09-13/`.

Protocol common to every placement row below: 60 episodes sampled from the trace
(seed 0, 20–50 m straight-line), arrival radius 4.0 m, localisation radius 15 m,
TARGET_SPEED 20 km/h for navigation. **SR counts routed episodes only**; unroutable
episodes are driven straight at the goal and reported separately as
`success_bypassed` (the "no-map" subset — not a mapless arm over all 60).

## 1. Harness fixes this session
- **Trace continuation was broken.** `CarlaEnv.reset` honours a start waypoint only
  when origin AND destination are set; collection set origin alone, so every v6n
  trace before 2026-09-12 ~22:50 was random-respawn (median restart gap 96–433 m).
  Fixed (`set_continue_origin`, spawn check, 3 misses → exit); tested.
- **Step-budget collection**: `--collect-total-steps`, `NO_TIMELIMIT=1` (TimeLimit and
  explore cap removed), `--collect-end-on` (collision | lane | off_road),
  `--collect-start-xy`, collection-only `COLLECT_TARGET_SPEED`.
- **Directed reachability** rewritten with scipy BFS — identical numbers, ~minutes → seconds.
- `--nodes-per-100m` sets no node count when uniform/random are matched → stage 4 as
  queued was inert and was stopped.

## 2. Exploration traces (all Town02/Town04, seed noted, 10,000 steps, continue from nearest waypoint)
| Trace | Ends on | Target speed | Episodes | Candidates | Revisit ≤2 m | 10 m cells | Folder |
|---|---|---|---|---|---|---|---|
| Town02 s7 | collision | 20 | 9 | 1211 | 74% | 113 | `placement_v6n_town02_budget10k` |
| Town02 s7 | collision, lane, off_road | 20 | 23 | 1196 | 97% | 97 | `placement_v6n_town02_budget10k_laneoff` |
| Town02 s7 | collision, lane, off_road | 10 | 31 | 1155 | 94% | 111 | `placement_v6n_town02_budget10k_laneoff_speed10` |
| Town04 s7 | collision | 20 | 2 | 1200 | 0% | — | `placement_v6n_town04_budget10k` |
| Town04 s8 | collision | 20 | 3 | 1230 | 0% | — | `placement_v6n_town04_budget10k_s8` |
| Town04 s9 (city-grid start) | collision | 20 | 2 | 1160 | 0% | — | `placement_v6n_town04_budget10k_grid` |

- The explore reward has no novelty term (heading progress ×10, 100 m bonus, penalties);
  collection uses the policy's eval action. Coverage is habitual, not exploratory.
- Town04 has ~41.5 km of lanes vs Town02 ~4.2 km; a 10k-step trace (~2.5 km) covers ~6% of
  Town04 and never revisits itself. Lowering TARGET_SPEED to 10 changed driven speed only
  8.9 → 8.6 km/h (it scales the speed observation; the policy picks throttle).
- Figures: `results/figures/exploration_traces_10k_fullmap*/`.

## 3. Placement rows on continued traces
### Town02 s7 collision-only trace — SGM gate 2 m, sgm[123.2/61.6], madent z=2
| Arm | Nodes | Success/routed | No-map success/unrouted |
|---|---|---|---|
| sgm | 622 | 12/60 | — |
| uniform@sgm | 622 | 10/60 | — |
| random@sgm | 622 | 16/60 | — |
| madent | 39 | 0/4 | 23/56 |
| uniform@madent | 39 | 2/36 | 14/24 |
| random@madent | 39 | 7/31 | 18/29 |
Consistency 123.2 is the grid ceiling (user kept the range) → sgm ≈ uniform frame.

### Town04 s9 city-grid trace — SGM gate 4 m (user: "run it whatever it takes"), sgm[16.6/106.5], madent z=1.5
| Arm | Nodes | Success/routed | No-map success/unrouted |
|---|---|---|---|
| sgm | 917 | 18/60 | — |
| uniform@sgm | 917 | 18/60 | — |
| random@sgm | 917 | 18/60 | — |
| madent | 57 | 0/4 | 24/56 |
| uniform@madent | 57 | 5/37 | 13/23 |
| random@madent | 57 | 3/19 | 17/41 |
SGM at 2 m merges nothing on Town04 (candidate spacing 2.07 m > gate). Never pool towns.

### Town04 link-cutoff ablation (madent z=1.5, 57 nodes, matched controls)
Cutoffs 26.6/53.2/79.8/106.5/159.7 steps. Routing saturates from 79.8; madent 0 routed
successes at every cutoff; uniform/random 2–5 routed successes each; every no-map subset
beats its routed episodes. Rows: `results/ablate_linkcutoff_town04_grid_z1.5/`.

## 4. Findings
1. Placement criterion does not matter at the node counts tested: SGM ties or trails
   count-matched uniform/random (Town02 12 vs 10/16; Town04 18/18/18). Paired on Town02
   (three random-respawn traces): sgm-only vs control-only 7 vs 11 (uniform, p=0.48),
   4 vs 12 (random, p=0.077).
2. madent's entropy spikes cluster (Town04: all 57 nodes on one southern road; 42/60 starts
   >100 m away) → ≤4 episodes attach, 0 succeed.
3. Maps hurt at 20–50 m goals: routed legs span ~100–210 m median and time out; the no-map
   subsets succeed 41–78%.
4. No critic orders road distance (route-binned Spearman v4 +0.17, v5 +0.12, v6n +0.03..+0.06)
   — user said not to pursue true distance further.

## 5. Open
- SGM Pareto sweep (Town02, consistency 5.1–123.2 at edge 61.6): paused, only the c123.2 point exists.
- Stage 4 design undecided; nothing committed (user approval pending).

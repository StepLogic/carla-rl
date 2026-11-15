# ViNT Town07 cells: provenance

## The 2026-11-12 cells (label `vintd3_t07`), ANY-SPEED: superseded, do not use for map-benefit claims

- Run 2026-11-12 18:15-19:32 by the vint-carla session on the local workstation, CARLA 0.9.15
  (Town07 is not in the cluster's CARLA image), Epic quality, windowed.
- Policy: the final ViNT, `vint_final/dagger3_e1` (= `vintd3`), local copy at
  `/home/robotlab/vint/final/dagger3_e1`, chosen on training-town validation; run once.
- Harness: a read-only rsync of the cluster's `~/carla-rl-vint` (the code that produced ViNT's
  Town02/Town03 rows), with the settings of `~/carla-rl/slurm/vint_bench_any.sh`:
  - `BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env`, i.e. `GOAL_ARRIVAL_MAX_SPEED=0`:
    success = within 1 m at ANY speed (the rule from 2026-11-09 to 2026-11-13);
  - `GOAL_CAM_MATCH=1 OFF_ROAD_JUNCTION_M=3.5 EVAL_MAP_FALLBACK=mapless EVAL_MAP_RESET_HIDDEN=1
    EVAL_MAP_REPLAN=3`, uniform arm with `EVAL_MAP_LOCALISE=image`;
  - `LANE_UNKNOWN_MARKS_FATAL` NOT set: the code default 1 applied, so a crossing of a lane
    edge with no marking data ended the episode. The 2026-11-15 rule sets it to 0 for evals.
    This is a TABLE-WIDE property of every cell scored before 2026-11-15 (c0, ViNT and RL
    alike), not a handicap on ViNT; such cells are not poolable with cells run under the
    2026-11-15 rule.
- Episodes: `bench_t07_fresh_episodes.json` (easy; medium = the >= 1-turn rule) and
  `bench_t07_turns_episodes.json` (hard), 100 each, from the map benchmark.
- Maps: uniform = `t07/maps_t07_z25/uniform`. ViNT topomap = drive 0 of the Town07 expert tour
  (segments 0-17, 34,536 of 34,681 steps; `trace_expert_Town07_drive0/index.json`).
- Results: `SUMMARY.md`. SR mapless / uniform / ViNT topomap: easy .52/.52/.35,
  medium .15/.33/.18, hard .01/.03/.04.

WARNING. Successes with a map entered the 1 m circle fast: median 10.9-12.8 km/h (uniform) and
11.3-14.1 km/h (ViNT topomap), against 5.8 km/h for mapless. That is the drive-through
signature measured on 2026-11-12 (wrong-goal arrivals entered at ~13 km/h, true-goal ones at
~7.8). So the medium "map benefit" (.15 -> .33) may be largely cars passing through the goal.
These cells are superseded by the live slow-arrival rerun below once it exists. They must not
support a claim about map benefit.

## The rerun (label `vintd3_t07slow`): LIVE slow arrival, current eval rules

`run_vint_t07_slow.sh`, the same harness, episodes, maps and topomap, with two settings changed
to the rules in force since 2026-11-13 / 2026-11-15:
- `BENCH_BASE_OVERLAY=slurm/il_eval_chw.env`: `GOAL_ARRIVAL_MAX_SPEED=10` from v52_reward.env.
  Success = within 1 m at <= 10 km/h, applied LIVE, so a fast pass does not end the episode.
- `LANE_UNKNOWN_MARKS_FATAL=0`: a crossing of an edge with no marking data is not a fault.
Because two settings change, the difference from the 2026-11-12 cells is not the speed rule
alone. Status: prepared, NOT RUN, and ON HOLD. A ViNT-only rerun would leave the main table
mixed (ViNT at slow arrival + non-fatal no-data crossings, c0 and RL at any speed + fatal),
which is worse than a consistent, dated table. It runs only as part of a TABLE-WIDE rerun of
every arm, if the user chooses that over keeping every cell at any speed with a disclosure.

## Every ViNT benchmark cell and the rule it ran under (checked from the logs, 2026-11-15)

All `vintd3` = `vint_final/dagger3_e1`. No ViNT log sets `LANE_UNKNOWN_MARKS_FATAL`, so every
cell below ran with no-data crossings FATAL (the pre-2026-11-15 default).

| Label | Jobs | Towns | Bands | Arrival rule |
|---|---|---|---|---|
| `vintd3_fresh` | 464944-464946 (cluster, 0.9.16) | Town02, Town03 | easy, OLD length-only medium, OLD fresh hard | LIVE <= 10 km/h (old rule, il_eval_chw.env) |
| `vintd3_fresh_any` | 465200-465202 (cancelled midway), 465271-465273 | Town02, Town03 | easy (and partial medium/hard from the cancelled jobs) | any speed |
| `vintd3_medturn` | 466134 (Town02), 466159 (Town03) | Town02, Town03 | medium, >= 1 turn | any speed |
| `vintd3_turns` | 465331-465333 | Town02, Town03 | hard, >= 2 turns | any speed |
| `vintd3_z25` | 465470-465471 | Town02, Town03 | easy, medium, turns; uniform z=2.5 map only | any speed |
| `vintd3_t07` | local, 2026-11-12 | Town07 | easy, medium (>= 1 turn), hard | any speed |

Medium, any speed, mapless / uniform / ViNT topomap: Town02 (466134) .41 / .69 / .50,
Town03 (466159) .04 / .02 / .05.

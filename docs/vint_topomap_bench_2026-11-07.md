# ViNT on the frozen map benchmark, with ViNT's own topological map (2026-11-07)

## Question
How does ViNT do on the frozen fresh map benchmark (`docs/map_benchmark_fresh_2026-11-07.md`)
when it navigates the way it is deployed: a topological map of the exploration
trace (ViNT's `create_topomap`), localised and followed with its own distance head
(ViNT's `navigate.py`)? The comparison inside this block is ViNT with that map against
ViNT without a map, on the same episodes.

## Protocol (fixed before the fresh run; the fresh set was run once)
- **Policy:** ViNT fine-tuned from the released `vint.pth` on c0's expert data, epoch 1,
  chosen on validation (`docs/vint_carla_2026-11-07.md`). Deterministic.
- **Episodes:** the frozen fresh set c0 was scored on: `logs/bench_fresh_episodes.json`
  (md5 b186a08), 600 episodes, Town02 and Town03 × easy / medium / hard × 100.
  Development used only the development set (`logs/bench_long_episodes.json`, seeds
  21000–23110).
- **Success and faults:** identical to the c0 rows.
  - Arrival: within 1 m of the goal, with no speed condition.
  - A crossing, off-road or collision ends the episode. A fault on the arrival step is a failure.
  - Off-road junction rule: `OFF_ROAD_JUNCTION_M=3.5`.
  - SR@2m counts a pass within 2 m only at ≤ 10 km/h (the tracker's default).
  - These settings are frozen in `slurm/vint_bench_map.sh`.
- **Arms (same seeds):**
  - **ViNT mapless:** the real goal image from the start (`ARM=mapless`).
  - **ViNT topomap**, images only (`ARM=vinttopo`, `EVAL_VINT_TOPOMAP`):
    - **Map:** ViNT's topological map of the expert-tour trace that the benchmark's map was built from (one continuous drive).
      - Nodes: a node every 10 frames (1 s, about 3.3 m). Each node keeps its last frame as the node image and its 6-frame ViNT context (stride 3).
      - Drive links: between consecutive nodes in drive order.
      - Revisit links:
        - Candidates are places the tour passes again: within 8 m, heading within 90°, not drive neighbours. They are proposed by the trace's own odometry; this is disclosed, and the exploration's odometry is allowed for map building.
        - A candidate is kept when ViNT's predicted distance is at most τ, the 95th percentile of ViNT's predicted distance between consecutive nodes.
      - Link cost is ViNT's predicted distance.
    - **Per episode:**
      - Start node: the node image ViNT judges nearest ahead of the car's first view (context: that frame six times).
      - Goal node: the node from whose context ViNT judges the goal image nearest.
      - Route: Dijkstra on ViNT link costs.
      - No route, or start node = goal node: driven mapless (counted).
    - **Driving:**
      - `navigate.py`'s ViNT rule along the route: window closest−4 … closest+5, close threshold 3, the closest node advances, reached when the closest node is the goal node.
      - Waypoint 2 goes to the expert's PID. The car cruises at 15 km/h toward nodes.
      - After that, ViNT drives on the real goal image with the approach profile until it arrives.
- **Privileged information:** none while driving. Trace odometry only proposes revisit
  candidates at build time. True pose is used only to score: the `privileged_*`
  columns, which report the start/goal node error and the car's end position against its route.
- **Not the c0 arm.** The map, the localiser (ViNT's distance head, not AnyLoc) and the
  navigator (navigate.py, not subgoal switching with leg timeouts and replanning) all
  differ. ViNT also keeps its frame context across subgoals and approaches only the final goal.
  These rows are therefore a separate block; the mapless rows are the clean policy
  comparison with c0.

## The maps

| Town | Tour | Nodes | Links | τ (ViNT steps) | Revisits kept | Largest strongly connected component |
|---|---|---|---|---|---|---|
| Town02 | 5.0 km | 1,234 | 3,811 | 5.46 | 2,578 / 5,060 | 100% |
| Town03 | 31.4 km | 7,789 | 88,874 | 8.48 | 81,086 / 119,916 | 100% |

ViNT's predicted distance to the next node has a median of 3.59 (Town02) and 3.70 (Town03) steps, against a true 3.33.

## Development (development set only; nothing tuned on it)
- **Topomap arm, 18 episodes (Town02/Town03 × 3 bands × 3):**
  - Outcomes:
    - 1 reached at 1 m and 4 at 2 m.
    - 15 crossings and 2 off-road.
    - 6 failures passed 0.8–1.8 m from the goal. The 0.84 m pass came at 14 km/h on the step of a line crossing.
  - Localisation:
    - Start node within 8 m in 16 of 18 episodes (median error 1.6 m).
    - Goal node within 8 m in 14 of 18 (median 1.5 m); the misses were 15–175 m.
  - Route following:
    - The final leg started in 8 of 18 episodes.
    - The navigator's own route position equalled the true one (median gap 0 nodes).
    - The car ended more than 5 m off its route in 7 of 17 failures.
- **Mapless, Town02 hard, 3 episodes:** 0 of 3, crossings after 10–17 m. On the same
  episodes the topomap arm drove 20–104 m before its fault.
- **Bugs:** none found. One diagnostic was added (route and route position, 145ddbf).
  `navigate.py` was re-read to confirm the closest-node rule.

## Results (fresh set, run once)
The per-episode files and the full report are in `results/vint_map_fresh_2026-11-07/`.
Success is arrival within 1 m. SPL tracks SR in every cell.

**ViNT topomap vs ViNT mapless (same episodes).** The ViNT block below is separate from the c0 rows.

| Cell | ViNT mapless | ViNT topomap | Topomap-only / mapless-only | Exact McNemar p (Holm, 6 cells) | SR@2m mapless / topomap | 2 m: topomap-only / mapless-only (p, Holm) |
|---|---|---|---|---|---|---|
| Town02 easy | 0.29 | 0.18 | 7 / 18 | 0.26 | 0.46 / 0.37 | 14 / 23 (0.56) |
| Town02 medium | 0.07 | 0.15 | 12 / 4 | 0.38 | 0.16 / 0.34 | 27 / 9 (**0.020**) |
| Town02 hard | 0.00 | 0.03 | 3 / 0 | 1.00 | 0.00 / 0.16 | 16 / 0 (**0.0002**) |
| Town03 easy | 0.34 | 0.39 | 15 / 10 | 1.00 | 0.55 / 0.59 | 13 / 9 (0.95) |
| Town03 medium | 0.15 | 0.13 | 10 / 12 | 1.00 | 0.19 / 0.24 | 18 / 13 (0.95) |
| Town03 hard | 0.02 | 0.03 | 2 / 1 | 1.00 | 0.02 / 0.09 | 8 / 1 (0.16) |
| **All 600** | **0.145** | **0.152** | 49 / 45 | 0.76 (raw) | 0.23 / 0.30 | 96 / 55 (0.001, raw) |

- **Fallbacks:** none. No route: 0 of 600; start node = goal node: 0 of 600. Both maps are one strongly connected component.
- **Clock ends:** none in either arm; every failure is a fault.
- **Speed (pooled path over time):** topomap 10.0–12.5 km/h, mapless 8.3–11.5 km/h.
- **2 m score:** a secondary measure that counts a pass within 2 m only at ≤ 10 km/h. The primary score is the frozen 1 m arrival.

**Clean policy comparison (mapless rows, same 600 episodes).** ViNT 0.145 vs c0 0.582.
- c0-only 277, ViNT-only 15, p ≈ 1e-63.
- c0 is ahead in every cell (Holm p ≤ 3e-7).
- c0 scores nearly the same at 1 m and 2 m (0.58 / 0.60); ViNT does not (0.15 / 0.23).

For reference, c0's own map arms (`docs/map_benchmark_fresh_2026-11-07.md`) score image 0.64 and oracle 0.70. Those arms use a different map, localiser and navigator, so they are not compared with ViNT's topomap row.

### What ViNT's own map does for ViNT
- **The images-only localisation works** (scored with true pose, never used to drive).
  - Start node within 8 m of the true start in 94–99% of episodes per cell (median 1.2–2.5 m).
  - Goal node within 8 m in 91–100% (median 0.7–1.3 m).
  - The navigator tracked its route: the median gap between its route position and the true one was 0–1 node.
- **It gets ViNT near the goal more often:**
  - Passed within 2 m (any speed) more often in every cell: Town02 medium 43 vs 16, Town02 hard 19 vs 0, Town03 hard 11 vs 2.
  - SR@2m rises from 0.23 to 0.30.
- **It does not turn those passes into 1 m arrivals.**
  - Near passes that did not arrive missed by a median of 1.35–1.72 m in both arms. ViNT's lateral precision, not the map, is the limit at 1 m.
  - In Town02 easy and medium the topomap arm also reaches the goal faster: median 8.8–9.4 vs 5.8 km/h at the closest approach. It cruises at 15 km/h until the final leg. Town03 is 5.8–6.9 km/h, about the mapless speed.
  - Town02 easy is the one cell where the map lowers SR (0.18 vs 0.29), but the difference is not significant after Holm.
  - On Town02 easy it passed within 2 m more often than mapless (57 vs 47), but only 18 of those passes arrived (mapless 29).
- **Faults dominate both arms.** Per cell: crossings 39–80, off-road 15–33, collisions 0–11.
- **The final leg was reached in 17–81% of episodes,** fewer as the band gets longer. Long routes end at a lane fault before the final leg.

### Why ViNT is so far below c0 here (and below its held-out 0.33 / 0.49)
The held-out evaluation scored arrival at 2 m (the trainer default); this benchmark scores 1 m.
- ViNT gets near the goal but misses by 1.3–1.7 m.
- ViNT keeps losing the lane: crossings are its main failure, as in the held-out evaluation.

**Run notes:**
- Topomap Town02 hard (job 464582) crashed before its first episode finished.
  - The CARLA server segfaulted (signal 11) and the client timed out; the server log is kept as `logs/vfresh_vinttopo_Town02_hard_464582_carla_segfault.log`.
  - It ran on an RTX 2080 Ti (11 GB). There, the topomap arm's Python process holds 3.8 GB after localising over every node, and CARLA reaches about 6.4 GB while loading.
  - The cell was re-run unchanged (job 464592) with the 2080 Ti node excluded. No episode of the crashed job had finished, so nothing was selected.

## Reproduce
- Map:
  `PYTHONPATH=$HOME/vint/pkgs:$HOME/vint/visualnav/train python3 scripts/vint/build_topomap.py --trace checkpoints/trace_expert_jx/trace_expert_<Town> --ckpt checkpoints/vint_carla_expert/epoch_1 --out checkpoints/vint_topomap/epoch_1/<Town>_topomap.npz`
  (Town02 about 3 min, Town03 about 6 min on one GPU).
- One cell:
  `CKPT=$HOME/carla-rl/checkpoints/vint_carla_expert/epoch_1 TOPOMAP=$HOME/carla-rl/checkpoints/vint_topomap/epoch_1 TOWN=Town02 BAND=hard ARM=vinttopo LABEL=vint_fresh PORT=3100 EPISODES=$HOME/carla-rl/logs/bench_fresh_episodes.json sbatch -p gpu --gres=gpu:1 -c 4 --mem=48G -t 10:00:00 --export=ALL slurm/vint_bench_map.sh`
  (`ARM=mapless` for the other arm).
- Report: `scripts/vint/vint_bench_report.py --root checkpoints --label vint_fresh --c0-root results/map_fresh_2026-11-07/checkpoints`
  (or `--root results/vint_map_fresh_2026-11-07/checkpoints` for the committed files).
- Topomap jobs need a GPU larger than 11 GB (`--exclude=node-g104`); see the run notes.

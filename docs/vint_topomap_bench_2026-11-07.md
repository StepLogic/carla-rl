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

## Results
_Pending: jobs 464580–464592._

**Run notes:**
- Topomap Town02 hard (job 464582) crashed before its first episode finished.
  - The CARLA server segfaulted (signal 11) and the client timed out; the server log is kept as `logs/vfresh_vinttopo_Town02_hard_464582_carla_segfault.log`.
  - It ran on an RTX 2080 Ti (11 GB). There, the topomap arm's Python process holds 3.8 GB after localising over every node, and CARLA reaches about 6.4 GB while loading.
  - The cell was re-run unchanged (job 464592) with the 2080 Ti node excluded. No episode of the crashed job had finished, so nothing was selected.

## Reproduce
- Map:
  `PYTHONPATH=$HOME/vint/pkgs:$HOME/vint/visualnav/train python3 scripts/vint/build_topomap.py --trace checkpoints/trace_expert_jx/trace_expert_<Town> --ckpt checkpoints/vint_carla_expert/epoch_1 --out checkpoints/vint_topomap/epoch_1/<Town>_topomap.npz`
  (Town02 about 3 min, Town03 about 15 min on one GPU).
- One cell:
  `CKPT=$HOME/carla-rl/checkpoints/vint_carla_expert/epoch_1 TOPOMAP=$HOME/carla-rl/checkpoints/vint_topomap/epoch_1 TOWN=Town02 BAND=hard ARM=vinttopo LABEL=vint_fresh PORT=3100 EPISODES=$HOME/carla-rl/logs/bench_fresh_episodes.json sbatch -p gpu --gres=gpu:1 -c 4 --mem=48G -t 10:00:00 --export=ALL slurm/vint_bench_map.sh`
  (`ARM=mapless` for the other arm).
- Report: `scripts/vint/vint_bench_report.py --root checkpoints --label vint_fresh`.

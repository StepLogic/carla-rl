# Topological-map image navigation in CARLA: frozen evaluation on fresh seeds (2026-11-07)

The raw per-episode results are in `results/map_fresh_2026-11-07/`: 18 `eval.json` files and `logs/bench_fresh_episodes.json`.

## Question
Does a topological image map built from one exploration trace help an image-goal driving policy reach goals it cannot see, without using pose during navigation?

## Protocol (frozen before the run)
- **Policy:** c0 = `il_bc_c0/step_15000`, a behaviour clone. Deterministic actions, its own eval overlay.
- **Episodes:** a fresh, never-seen set, `logs/bench_fresh_episodes.json`. 600 episodes = Town02 and Town03 × easy / medium / hard, 100 per cell (the Gibson per-bin count).
  - Bands are by road-route distance: easy 10–24 m, medium 25–49 m, hard 50–99 m.
  - Seeds: 31000 / 32000 / 33000 and up. Probed with `slurm/bench_probe_fresh.sh`, selected with `scripts/bench_select.py`.
  - No overlap with the development set (21000–23110), which was used to design the method.
- **Success:** within 1 m of the goal. Collisions, line crossings and off-road end an episode.
  - Off-road uses the junction distance rule (`OFF_ROAD_JUNCTION_M=3.5`): inside a junction it counts only beyond 3.5 m from a lane centre.
  - CARLA's raw check fires on junction asphalt; measured: 74 of 81 junction faults were 1.75–3 m from a lane centre.
  - Results under the original rule are not reported here.
- **Map:** `checkpoints/maps_expert_jx_img/uniform/<Town>.json`, built once with no reference to benchmark episodes.
  - **Trace:** a scripted expert (CARLA BasicAgent) drives one closed coverage tour of every lane in its legal direction AND every junction turn: Town02 5.0 km, Town03 31.4 km, one drive each, with the explorer's own pose (as Habitat's explorer has odometry). c0 rides along (recording only).
  - **Nodes:** uniform placement, key-node count matched to the entropy arm, plus image-coverage gap nodes. No metric gap nodes.
  - **Image rules:** AnyLoc (DINOv2 ViT-B/14 value facet + 32-cluster per-trace VLAD) on our single forward camera.
    - Thresholds come from Habitat's rules scaled by the same-place median (ours 0.72–0.75, theirs 0.61).
    - Arrival threshold = the map's 2% far-false point: 0.515 (Town02), 0.512 (Town03).
    - Nothing was tuned on success rate.
  - **Edges:** transition edges plus the drive's own visits; none between headings more than 90° apart; unit cost.
- **Arms (identical seeds):**
  - **Mapless:** c0 with the true goal image.
  - **Image (main result):** pose-free, as close to the Habitat procedure as our setup allows (`EVAL_MAP_LOCALISE=image`).
    - Start and goal node: top-1 image match, no threshold, no radius.
    - Subgoal reached: it beats the previous and next nodes AND scores at least the threshold, 2 consecutive steps.
    - Leg timeout: 150 steps, then image relocalise and replan, up to 3 times; then the real goal image.
    - GRU reset at every subgoal.
    - No route, or start = goal node: driven mapless.
    - Not matched: Habitat also ends a leg on STOP; c0 has no STOP action.
  - **Oracle (labelled upper bound):** the same map; true pose and goal coordinates for start/goal (joint localisation), arrival within 4 m, and replanning by true pose (100-step stall or more than 15 m off-route); GRU reset per subgoal.

## Results (success rate; SPL tracks SR in every cell)

| | Mapless | **Image** | Oracle (upper bound) | Image-only / mapless-only wins | Exact McNemar p |
|---|---|---|---|---|---|
| Town02 easy | 0.73 | 0.78 | 0.81 | 14 / 9 | 0.40 |
| Town02 medium | 0.68 | 0.73 | 0.81 | 15 / 10 | 0.42 |
| Town02 hard | 0.45 | **0.60** | 0.64 | 19 / 4 | **0.0026** |
| Town03 easy | 0.67 | 0.69 | 0.73 | 6 / 4 | 0.75 |
| Town03 medium | 0.57 | 0.60 | 0.67 | 13 / 10 | 0.68 |
| Town03 hard | 0.39 | 0.46 | 0.55 | 14 / 7 | 0.19 |
| **Hard, both towns (200)** | **0.42** | **0.53** | 0.60 | 33 / 11 | **0.0013** |
| Easy, both towns (200) | 0.70 | 0.74 | 0.77 | 20 / 13 | 0.30 |
| Medium, both towns (200) | 0.63 | 0.67 | 0.74 | 28 / 20 | 0.31 |
| All 600 | 0.58 | 0.64 | 0.70 | 81 / 44 | 0.0012 |

**Image localisation while driving** (true pose used only to score):

| Town | Start node within 8 m | Goal node within 8 m |
|---|---|---|
| Town02 | 95–99 / 100 | 91–98 / 100 |
| Town03 | 92–96 / 100 | 89–92 / 100 |

1 mapless fallback in 600 episodes. The image arm replanned in 11–45 episodes per cell.

## Findings
1. **Long routes:** the pose-free map improves c0 on 50–99 m routes by about +0.11 (0.42 → 0.53, p ≈ 0.001, 200 paired episodes). This matches Habitat's finding that maps help only on long routes.
2. **Short and medium routes:** no significant gain. A large medium-band gain on the development seeds (0.52 → 0.71) did not replicate on fresh seeds.
3. **Cost of image localisation:** about 0.06 below the oracle on average (0.64 vs 0.70).

## Caveats
- **One policy checkpoint**, two towns, 100 episodes per cell; only the Town02 hard cell is significant on its own.
- **The oracle and image arms differ in their replan triggers**, not only in localisation.
- **The map comes from a scripted coverage explorer**, not the evaluated policy.
- **Entropy vs uniform placement** is a separate comparison (next run, same fresh seeds and configuration).

## Development history
How the method reached this form, all on the development seeds:
- maps built from c0's own drive lacked coverage;
- a lanes-only expert tour lacked junction turns;
- CARLA's off-road check misfired in junctions;
- nearest-node localisation snapped to the wrong branch;
- Habitat's thresholds needed scaling to our camera.

The development-seed numbers are not reported as results.

## Entropy vs uniform placement (same frozen configuration, same 600 fresh episodes)
- **Entropy map:** `maps_expert_jx_img/entropy`. Key nodes come from online MAD on c0's explore entropy along the same expert tour, with count matched to uniform, plus image gap nodes. Town02 has 464 nodes, Town03 1,491.
- **Mapless rows** are the same runs as above.

**Image localisation (main):**

| | Mapless | Uniform | Entropy | Entropy-only / uniform-only wins | p |
|---|---|---|---|---|---|
| Town02 easy | 0.73 | 0.78 | 0.78 | 7 / 7 | 1.0 |
| Town02 medium | 0.68 | 0.73 | 0.75 | 6 / 4 | 0.75 |
| Town02 hard | 0.45 | 0.60 | 0.59 | 6 / 7 | 1.0 |
| Town03 easy | 0.67 | 0.69 | 0.66 | 8 / 11 | 0.65 |
| Town03 medium | 0.57 | 0.60 | 0.61 | 9 / 8 | 1.0 |
| Town03 hard | 0.39 | 0.46 | 0.49 | 11 / 8 | 0.65 |
| Hard, pooled (200) | 0.42 | 0.53 | 0.54 | 17 / 15 | 0.86 |

**Pooled, entropy vs mapless:**

| Band | Entropy-only / mapless-only wins | p |
|---|---|---|
| Easy | 19 / 15 | 0.61 |
| Medium | 30 / 19 | 0.15 |
| Hard | 37 / 13 | 0.0009 |

**Oracle (pose) localisation:**

| | Mapless | Uniform | Entropy |
|---|---|---|---|
| Town02 easy | 0.73 | 0.81 | 0.80 |
| Town02 medium | 0.68 | 0.81 | 0.86 |
| Town02 hard | 0.45 | 0.64 | 0.55 |
| Town03 easy | 0.67 | 0.73 | 0.67 |
| Town03 medium | 0.57 | 0.67 | 0.64 |
| Town03 hard | 0.39 | 0.55 | 0.46 |

- **Pooled hard:** entropy 0.505 vs uniform 0.595; entropy-only / uniform-only wins 10 / 28, p 0.005.
- **This is NOT a clean placement effect.** The oracle localises within 8 m by true pose, and the entropy map's sparser coverage (key nodes cluster at junctions; no metric gap nodes) left 27 episodes with no node within 8 m, which fell back to mapless (uniform: 0). Image localisation has no radius and does not suffer this.

**Finding:** with pose-free image localisation, entropy placement equals uniform in every band (pooled p 0.73 / 0.70 / 0.86). Both beat mapless only on long routes. This matches Habitat (uniform is the most consistent placer). Entropy placement adds no measurable benefit for this policy.

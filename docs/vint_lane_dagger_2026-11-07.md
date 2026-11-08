# Fixing ViNT's lane keeping: lane-centre labels, start padding, DAgger (2026-11-07)

Follows `docs/vint_topomap_bench_2026-11-07.md`. There, ViNT scored 0.145 on the fresh map benchmark against c0's 0.582, ending mostly in line crossings. The user asked "can we fix it?", then "use DAGGER".

Nothing here touches the fresh benchmark set. Development and validation sets only.

## Data and selection (verified on disk 2026-11-08)

**Training data is only from the training towns.** Every training dataset comes from Town01, Town04, Town05 and Town10HD only:

| Dataset | Content |
|---|---|
| expert `il_data` | 480 episodes per town |
| DAgger rounds 1 / 2 / 3 | 300 / 300 / 150 episodes per town |
| `carla_expert_lc.tar`, `carla_dagger{1,2,3}.tar` | zero Town02/Town03 frames |

The recorder refuses the evaluation towns in code (`torch_rl/expert_record.HELD_OUT_TOWNS = ("Town02", "Town03")`).

**Selection uses training-town validation only.** Every epoch choice uses c0's validation: Town05 + Town01, seeds 50000+.

**Where the evaluation towns are used.** Town02/Town03 serve only to evaluate:
- The development split (`bench_long_episodes.json`), used for lane checks and the development map set.
- The map under test, built from the evaluation town's exploration drive. It trains no weights.
- The fresh test set: not run again since ViNT v1's one-shot.

**One lapse, disclosed.** The decision that DAgger round 2 beat round 1, which started round 3, also looked at Town02 development lane checks. On validation alone, 49 vs 47/80 is within noise.

**The rule from here (user 2026-11-08):** decisions rest on training-town validation only. Round 3 gets the development map set only if its best epoch reaches >= 54/80, a threshold declared before its results.

## Diagnosis

### Offline: does ViNT steer back toward the lane?
Held-out expert frames, offline `carla.Map`; `scripts/vint/build_lane_labels.py` and the probe scripts in the job folder.

| Metric | Value |
|---|---|
| Recorded expert paths: \|lane offset\| median | 0.08 m |
| Recorded expert paths: steps more than 1 m off | 0.9% |
| v1 labels (the recorded path), offset kept 5.6 m ahead | ~75% |
| ViNT v1 predictions, offset kept at 1.1 / 3.4 / 5.6 m | 94% / 78-85% / 45-58% |
| Expert's clean steering vs executed (noisy) steering, steer per metre of offset | -0.108 vs -0.053 |

ViNT reproduces its labels; the labels themselves are weak.

### Closed loop: lane state while driving
`EVAL_LANE_DEBUG=1`; the overlay `slurm/lanedbg_<band>.env` turns on the env's lane state with a 1e-9 lane-centre weight, which changes only the reward value. Development set, Town02, 30 episodes.

| | c0 | ViNT v1 |
|---|---|---|
| Medium band: \|offset\| median | 0.10 m | 1.02 m |
| Easy band: \|offset\| median | 0.09 m | 1.04 m |
| Easy band: heading error p90 | 5 deg | 178 deg (wrong-way stretches) |
| Easy band: episodes with a crossing | 8 | 20 |
| Easy band: SR (1 m) | 0.63 | 0.20 |

Failure modes seen in the lane lines:
1. **Start-up:** it turns toward the goal from a standstill. Its first context, one frame repeated six times, never appears in training.
2. **Heading:** it holds a steady heading error until a crossing.
3. **Junctions:** turnarounds, then wrong-way driving that no fault catches.

**The goal range matters for this test.** Medium and hard goals (25-99 m) lie beyond ViNT's ~22 m training goal range by design, so mapless medium episodes mix lane keeping with far goals. Lane-keeping checks therefore use the easy band.

### The controller is not the cause
`VINT_ORACLE_AIM=1` (PRIVILEGED, a diagnostic only) drives with ViNT's speed but steers at the true lane-centre aim points.
- Town02 easy: 22/30 arrived, 0 faults (8 ran out of time: straight-ahead aim points never turn toward a goal).
- |offset| median 0.00 m; heading error p90 1 deg.

The failures come entirely from ViNT's predicted waypoints.

## Fix 1: lane-centre aim-point labels and start padding (no DAgger)
The driver aims at one predicted waypoint, so the label is changed to where the expert aims: the lane centre of the car's route, j x 1.1346 m ahead, j = 1..5, with the lane heading.

How the labels are built:
- From an offline `carla.Map`: `src/vae_nav/vint_lane_labels.py` and `scripts/vint/build_lane_labels.py`.
- A label that strays more than 3 m from the car's own path (a wrong branch of the lane chain) keeps the recorded path. 95.6% of frames are relabelled.
- The tar is copied with only `traj_data.pkl` changed (`add_lane_labels_to_tar.py`); the loader reads it via `patch_vint_cf_labels.py`.
- The labels use privileged map information at training time only. The policy still sees only images.

Start padding: `VINT_START_PAD=1` (`patch_vint_start_pad.py`) trains from frame 0, padding the context with the first frame as the driver does. That is 8.2% more samples.

Results (run `vint_carla_lanecentre`, job 464612):
- **Offline:** it aims at the lane centre. Offset kept at 1.1 / 3.4 / 5.6 m: 53% / 29% / 10-19%.
- **Validation** (c0's protocol, Town05 + Town01, 2 m, true goals of 80): epoch 1 **42**, epoch 4 31, epoch 6 33. v1 managed 37, c0 63. The wrong-goal arrival rate is 0 for every epoch.
- **Closed loop, Town02 easy (epoch 1):** |offset| median 1.12 m, SR 0.13. Offline the model aims at the lane centre, but in closed loop it does not hold the lane.

## Fix 2: DAgger
**Collection** (`slurm/vint_dagger.sh`):
- ViNT (lane-centre epoch 1) drives goal episodes in the four training towns.
- The expert labels every frame. It replans when the car is more than 3 m off; the label is its current plan's lane-centre aim points (`EXPERT_RECORD_AIM`: `BasicAgentExpert.plan_xy` + `aim_from_plan`).
- So a wrong turn is labelled back toward the route, and frames without a plan are skipped (`cf_ok`).
- Smoke check: label points lie exactly on lane centres, and every off-centre frame's label points back to the lane.

**Round 1:**
- 1,200 episodes (300 per town, seeds 600000+), 94k steps. ViNT reached 32-51% of these 5-25 m goals.
- Conversion: `carla_dagger1.tar`, with no fault trim (the labels are the expert's), a 16-frame minimum, and the expert's 0.3782 m spacing. 90.7k train frames, 84.2k of them labelled.

**Training** (`vint_carla_dagger1.yaml`, job 464665):
- Data: expert (lane-centre labels) + DAgger, with start padding. DAgger is ~22% of the samples.
- Starting point: from `vint.pth`, as before.

| DAgger round 1 | Validation (true / 80) | Town02 easy SR | \|offset\| median | heading p90 | episodes with a crossing |
|---|---|---|---|---|---|
| epoch 0 | 35 (T05 .250, T01 .625) | 0.33 | 0.72 m | 25 deg | 15 |
| **epoch 1** | **47** (T05 .475, T01 .700) | **0.47** | 0.76 m | 14 deg | 14 |
| epoch 2 | 41 (T05 .350, T01 .675) | 0.43 | 0.58 m | 15 deg | 14 |
| epoch 3 | 44 (T05 .400, T01 .700) | 0.43 | 0.94 m | 10 deg | 14 |
| *c0* | *63* | *0.63* | *0.09 m* | *5 deg* | *8* |
| *ViNT v1* | *37* | *0.20* | *1.04 m* | *178 deg* | *20* |

Wrong-goal arrival rate: 0 for every epoch. Round-1 epoch 1 is the pick, and it drove DAgger round 2.

### Round-1 epoch 1 on the development map set
Settings:
- The 600 development episodes (`logs/bench_long_episodes.json`, never the fresh set), with the frozen benchmark settings and 1 m arrival.
- One job per arm (`slurm/vint_bench_cells.sh`).
- The ViNT topomap is rebuilt with this model. Town02: 1,234 nodes, tau 7.07, 2,770 revisits kept. Town03: 7,789 nodes, tau 10.94. Both fully connected.

| Development set, SR at 1 m (100 per cell) | T02 easy | T02 medium | T02 hard | T03 easy | T03 medium | T03 hard |
|---|---|---|---|---|---|---|
| ViNT v1, mapless | .21 | .09 | .00 | .30 | .06 | .02 |
| **DAgger r1 e1, mapless** | **.65** | **.50** | **.19** | .23 | .08 | .02 |
| **DAgger r1 e1, ViNT topomap** | **.68** | **.59** | **.49** | .33 | .11 | .00 |
| c0 mapless (mapping session, development set) | .71 | .52 | .44 | | | |
| c0 image map arm (same) | .80 | .71 | .65 | | | |

- **Town02:**
  - Mapless ViNT roughly triples, reaching c0 mapless on easy and medium.
  - With its own topomap, ViNT goes from .00 to .49 on hard, above c0 mapless on medium and hard.
  - The near misses are mostly gone: SR at 1 m vs 2 m on easy is .65 / .75, against .21 / .50 before.
- **Town03 does not improve.** Crossings (51-60 per cell) and off-road ends (25-38) still dominate. Town03 is held out, and DAgger collects only in the training towns.

### DAgger round 2 (fast)
- Collection: driven by round-1 epoch 1, 1,200 episodes, 8 jobs. ViNT arrival: Town01 .65-.67 (round 1 .48), Town04 .51-.53.
- Training: warm-started from round-1 epoch 1 on expert + dagger1 + dagger2, stopped after epoch 2.
- Validation (true goal only): epoch 0 = 45 (T05 .40, T01 .725), epoch 1 = 45 (T05 .45, T01 .675). Round 1 epoch 1 had 47.
- Epoch 2's checkpoint was truncated by the early stop (the cancel landed while CephFS was still writing it). It was restored from `latest.pth`, the same epoch, intact, and re-validated.

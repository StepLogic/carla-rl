# Fixing ViNT's lane keeping: lane-centre labels, start padding, DAgger (2026-11-07)

Follows `docs/vint_topomap_bench_2026-11-07.md`. There, ViNT scored 0.145 on the fresh map benchmark against c0's 0.582, ending mostly in line crossings. The user asked "can we fix it?", then "use DAGGER".

Nothing here touches the fresh benchmark set. Development and validation sets only.

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
| epoch 0 | 35 (T05 .250, T01 .625) | **0.33** | 0.72 m | 25 deg | 15 |
| *c0* | *63* | *0.63* | *0.09 m* | *5 deg* | *8* |
| *ViNT v1* | *37* | *0.20* | *1.04 m* | *178 deg* | *20* |

_Epochs 1-3 pending._

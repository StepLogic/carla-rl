# ViNT fine-tuned on c0's expert data, evaluated like c0 (2026-11-07)

User request: "can you also run VINT train on CARLA dataset and evaluate". Choices
(asked): c0's expert data with c0's train/validation split; fine-tune from the
released ViNT weights; evaluate with c0's paired protocol.

## Setup

| | |
|---|---|
| Data | c0's `bc_split.json`: 1,368 train / 72 val expert episodes (Town01/04/05/10HD, 10 Hz). 1,294 / 65 long enough for ViNT (>= 31 frames): 275,938 / 13,986 frames. Fault-ended episodes lose their last 20 frames. |
| Labels | frame k+1 at `pos[k]`; CARLA's y flipped to ViNT's left-positive frame; heading from motion (shards store no yaw). Checked on real shards: with \|steer\| > 0.3 the future path bends the steering's way in 87-90% of frames. |
| Model | released ViNT (`vint.pth`, 29.7 M params; every weight loads, none missing or unexpected), full fine-tune, AdamW 1e-4, cosine, 20 epochs, batch 128, A30. |
| Stride | 3 frames per ViNT step (0.3 s, 1.13 m at 0.3782 m/frame); top distance class 18 steps (~20 m). |
| Waypoints | supervised for every positive goal (ViNT's default only < 10 steps, which assumes near map subgoals; this eval is mapless with goals 5-25 m away). |
| Driving | `torch_rl/vint_driver.py`: ViNT's waypoints executed by the expert's own vendored CARLA PID controller (lookahead 3 m + 0.5 s x speed); speed from ViNT's predicted time-to-goal through the expert's approach profile (15 km/h, 6 km/h within 1.8 s). |
| Eval | `run_eval` with `EVAL_POLICY=vint` -- c0's episodes, faults terminal, 2 m arrival under 10 km/h. Selection like c0: Town05+Town01, 40 paired episodes each from seed 50000, pooled true-goal SR. Held-out: Town02/Town03, 100 paired episodes each from seed 10000. Start and goal poses are identical to c0's runs (40/40, 40/40, 100/100, 98/100). |

## Validation (choosing the epoch, as c0's step was chosen)

| Epoch | Town05 true / wrong | Town01 true / wrong | Pooled true (of 80) | Test waypoint loss |
|---|---|---|---|---|
| 1 | .500 / .000 | .425 / .000 | **37** | .0418 |
| 4 | .300 / .000 | .575 / .000 | 35 | .0424 |
| 6 | .350 / .000 | .500 / .000 | 34 | .0400 |
| 9 | .350 / .000 | .500 / .000 | 34 | .0400 |
| 14 | .375 / .000 | .375 / .000 | 30 | .0399 |
| 19 | .350 / .000 | .350 / .000 | 28 | .0389 |
| c0 step 15000 | .675 / .100 | .900 / .050 | 63 | -- |

Epoch 1 has the most validation successes, so it is the one tested held-out.
The lowest test waypoint loss (epoch 19) drove worst: the offline loss does not
predict driving here.

## Held-out (epoch 1)

| | Town02 true / wrong / gap | Town03 true / wrong / gap |
|---|---|---|
| ViNT epoch 1 | .330 / .000 / +.33 | .490 / .010 / +.48 |
| c0 step 15000 | .690 / .070 / +.62 | .650 / .080 / +.57 |

Paired on the same episodes: Town02 both 29, only c0 40, only ViNT 4 (McNemar
p = 1.7e-8); Town03 both 42, only c0 23, only ViNT 7 (p = 0.005).

## Why ViNT is behind

* **It reads the goal**: it almost never stops at a wrong goal (1 of 400 wrong-goal
  episodes over validation and held-out).
* **It does not keep its lane.** Its failures are crossings and off-road: held-out
  Town02 35 crossings + 25 off-road (c0 2 + 24), Town03 31 + 20 (c0 9 + 24).
  Replays show waypoints that stay straight and nearly constant while the car
  drifts onto a line, on goals dead ahead that c0 reaches.
* **The waypoint head overfits**: training waypoint loss 0.048 -> 0.029 over
  epochs 0-5 while the validation loss stayed at 0.040-0.042; validation SR does
  not rise with training. ViNT learns where the car WENT, and the recording's
  DART steering noise is in those paths; c0 learned the expert's clean labels.
* **Start-up**: 9-14 of 80 validation episodes fail within 5 m (c0: 0). ViNT's
  first context is six identical frames of a car at rest, which training never
  shows it (the first training sample is 1.5 s in), and it turns toward the goal
  at once instead of at the junction.
* Not the speed control: ViNT enters the goal circle at a median 5.8 km/h (c0
  5.7-6.6) and none of its failures came within 2 m too fast.

## Traps found on the way

1. ViNT's dataset bounds goals in frames and then multiplies by the stride again
   (goals past the episode end at stride > 1): `scripts/vint/patch_vint_dataset.py`.
   It also rebuilds distance classes as `range(0, max + 1, stride)` (top 18 at stride 3).
2. The cluster's wandb cannot import (no click): `scripts/vint/stubs/wandb`.
3. CephFS costs ~85 ms per small file: the converter writes one tar (`--tar`).

## Reproduce

`scripts/convert_il_to_vint.py --tar`, `slurm/vint_train.sbatch` (MODE=probe first),
`scripts/vint/export_vint_ckpt.py --link`, `slurm/vint_eval_pair.sh`. Cluster tree
`~/carla-rl-vint`, ViNT code and packages `~/vint`, CARLA ports 3100-3399.

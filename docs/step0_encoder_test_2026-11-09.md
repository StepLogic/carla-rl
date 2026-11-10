# Step 0: is it the encoder? (2026-11-09)

Plan: `docs/superpowers/plans/2026-11-09-image-goal-lane-following.md`, Step 0.

The user asked "is it the encoder?" and chose to run this test before any new training. The privileged critic was approved as an offline diagnostic only.

**Answer.** No. c0's features carry as much lane information as networks trained directly on the same pixels for exactly that job. What fails is the learning signal:
- one 10 Hz steering step leaves no trace in any feature the critic reads;
- a critic given the true lane state, trained the same way on the same demonstrations, still does not learn how much to steer.

**Data.**
- c0 = `il_bc_c0` step_15000 (frozen).
- Recorded demonstrations: Town05, 480 episodes; Town01, 160 episodes (for transfer).
- Lane labels come from the recorded positions plus each town's OpenDRIVE map (`torch_rl/lane_labels.py`). They are diagnostic only and never a policy input.
- Clean-label frames (`clean_label_mask`): moving, off-junction, |offset| ≤ 1 m, |heading error| ≤ 15°, no lane-boundary jump.
  - 53,237 frames from 407 Town05 episodes.
  - Label SD: offset 0.21 m, heading error 3.5°.
  - Heading error is consistent with the next offset change at r = 0.93; geometry check r = 1.000.
- Splits are by episode.

Result files: `results/step0_encoder_2026-11-09/`. Code:
- `scripts/probe_c0_features.py`
- `scripts/probe_pixels_heading.py`
- `scripts/offline_priv_critic.py`

## 1. What the features carry (R² on held-out episodes, linear / MLP)

| Features | Lateral offset | Heading error |
|---|---|---|
| Camera latent (one frame, 512) | 0.70 / 0.81 | 0.05 / 0.05 |
| GRU state (history, 512) | 0.55 / 0.64 | 0.31 / 0.36 |
| DEBiT binocular (frame + goal, 3136) | 0.67 / 0.63 | 0.06 / −0.03 |
| Small CNN trained on the pixels, one frame | 0.72 | 0.04 |
| Small CNN trained on the pixels, two frames | 0.72 | 0.32 |

- Shuffled-label controls: 0.00 everywhere.
- Heading is not in one 128 px frame. It appears only across frames, and c0's GRU recovers as much as a two-frame CNN trained for it.
- Town05 → Town01 transfer (ridge), against within-Town01 values:
  - Camera: offset 0.34 (0.73 within); heading 0.04 (0.20 within).
  - GRU: offset −0.36 (0.44); heading 0.28 (0.17).
  - Binocular: offset 0.06 (0.53); heading −0.04 (0.13).
- So the offset encodings are town-specific, while the GRU's motion-based heading transfers.

## 2. Is one action visible in the next features? (R² of the DART steering noise)

The expert's recordings add iid N(0, σ) noise to the executed steering, with σ = 0.1 or 0.2, independent of the state. The target is that noise averaged over steps t..t+k−1. It is predicted from:
- the features at t and t+k;
- or, as the ceiling, from the TRUE heading and offset change over the window.

Rows: expert-driven, noised steps at ≥ 5 km/h, about 43k per k.

| k | True motion (ceiling) | Camera | GRU (ridge / MLP) | Binocular | Full critic input (MLP) |
|---|---|---|---|---|---|
| 1 | 0.26 / 0.33 MLP | 0.00 | 0.00 / −0.01 | 0.00 | −0.06 |
| 2 | 0.32 / 0.39 | 0.01 | −0.04 / 0.04 | 0.01 | −0.02 |
| 4 | 0.27 / 0.31 | 0.15 (0.18 on clean labels) | 0.03 / 0.22 | 0.10 | 0.12 |

- Features at t alone: 0.00 for every k, as it should be.
- **One step of steering is invisible** in everything the critic reads, although the car's motion shows about a third of it. The critic's target value of the next state therefore cannot depend on the action just taken. That is the Q ≈ V(s) + g(a) pattern seen in every SAC run.
- Over four steps the effect becomes visible: about half to two-thirds of the ceiling. This is the mechanistic case for action repeat (plan C1). A held action pushes in one direction for k ticks, so its effect should be larger than this mean of iid draws.

## 3. Privileged-critic control (offline, diagnostic only)

Setup:
- The same offline critic study as before. Training transitions: 88,175 from 430 episodes.
- Transform: no_arrival (the expert's final approach and arrival reward removed).
- Rewards ×10; 10 LayerNorm critics; REDQ min of 2 random targets; γ 0.99.
- The next action comes from c0's actor at α 0.1. 20k updates.
- The critic's input: true lateral offset, heading error, lane curvature ahead, speed, junction flag, clean-label flag, mode.

The probe compares the best steering on a 21-point grid with the expert's, on held-out episodes.

| Probe | States | Corr. with expert | Best steer at full lock | Mean \|best\| (expert) |
|---|---|---|---|---|
| First 3 Town05 episodes (the earlier probe) | 487 | 0.03 (off-junction 0.05) | 4% | 0.18 (0.03) |
| Every 10th episode | 4,000 | 0.04 (off-junction 0.08, clean labels 0.12) | 9% | 0.47 (0.07) |

- The earlier image critic on c0's features, same study: corr +0.04 with the DrQ head, +0.10 without.
- The matched image-critic rerun inside this script is still running.

Even with the true lane state, TD on these demonstrations does not identify how much to steer. The demonstrations rarely show what over-steering costs, and one step's consequence is small even in the true motion (ceiling R² about 0.3).

## Gate 0 (declared before the results)

| Rule | Condition | Result |
|---|---|---|
| Encoder is the bottleneck | Heading R² < 0.3, OR privileged critic ≥ 0.5 while the image critic < 0.3 | Not met: GRU 0.31 / 0.36; privileged ≤ 0.12 |
| Not the encoder | Heading R² ≥ 0.5 AND privileged critic < 0.3 | Not met: heading 0.36 |
| In between | Otherwise | **Yes.** Both B (imitation) and C (action repeat) go ahead |

The pixel controls (§1) and the visibility test (§2) say more than the gate's two numbers: the encoder is not throwing away information that the frames carry.

## Errors and corrections, recorded

- **The first heading probe was invalid; its log is kept, marked INVALID.**
  - It used the filter |offset| ≤ half a lane width, which kept lane-boundary flips and wrong-way / lane-snapping outliers.
  - Those outliers took the heading label's SD to 15.4° and its relation to the next offset change to r = 0.10.
  - The clean filter was checked on the labels alone before any rerun.
- **The local projection run died at 19:40** (about 387k steps). The cause was a training-loop bug on a simulator hang, fixed in d91affb. The CPU load of these analyses may have triggered the hang.

# v7 Recipe + VC-1 Patch-Token Compressor — Design

**Date:** 2026-09-21. **Status:** approved by user (chat, 2026-09-21).

## Goal

Re-run the v7 gated two-mode arm (point-goal + explore) with two user-specified
changes: (1) the camera pathway is VC-1 ViT-B patch tokens consumed by a
**trainable conv compressor producing a 512-d latent**, and (2) the state block
drops `has_collided`. Everything else is the **exact v7 recipe** (user choice,
AskUserQuestion 2026-09-21).

## What is v7 (the recipe of record)

Snapshot `results/run_env/carla_gated_her_v7.env` + trainer banners
(logs/carla_gated_her_v7.log):

- GOAL leg: minimum-time clock −1/step; per-step fines −5.0 off-road,
  −5.0 lane crossing (NON-terminal); collision TERMINAL with the exact bill
  (−step_cost × remaining clock); arrival is the only ordinary terminal;
  750-step cap. `WEDGE_TRUNCATE=0`, `GATED_GOAL_LANE_TERMINAL=0`.
- EXPLORE leg: progress reward `LANE_PROGRESS_MODE=heading ×10` while
  `speed_factor < 1`, base −4e-3, steer penalty; collision −5.0 non-terminal;
  off-road FATAL (−5.0 on the fatal step); painted lane crossing FATAL (−5.0,
  NONE-marking exempt, junctions forgiven); arrival at 100 m cumulative path
  pays +2.0 and terminates; 500-step cap. `EXPLORE_ARRIVAL_BONUS=2.0` (v7
  value — the 2026-09-18 raise to 10.0 is NOT taken; user chose "exact v7").
- Learner: two-critic `GatedTQCLearner` (mode bit in obs, per-mode critic,
  per-mode temperature), `DISCOUNT=0.99` + `GATED_ALLOW_DISCOUNTED_SORB=1`,
  `ACTOR_Q_NORM=1`, HER on.
- Env: Town01+Town10HD (`configs/town_split_v7.json`, held-out Town03),
  Low quality, windowed render (`CARLA_OFFSCREEN=0`, standing user rule),
  `PERCEPTION=mobilenet` in v7 — **replaced by VC-1 here**.

## The four deltas from v7

### D1 — VC-1 grid2 obs published as a pixels-shaped key

`VcSource` (src/vae_nav/vc1_source.py) with `VC1_READOUT=grid2` already
produces the verified (7, 7, 768) patch-token grid (196 tokens 2×2
average-pooled). NEW: the wrapper publishes it under the obs key **`pixels`**
with shape (7, 7, 768) — not the flat `latent` key — because
`is_image_space` (jaxrl2 utils/misc.py) routes any ndim ≥ 3 key through the
conv encoder. The 37632-d flat `latent` key stays for arms that want it; the
v8 wrapper publishes `pixels` instead of `latent` (one key, same content,
shaped).

Obs layout (v8): `{'goal': (4,), 'pixels': (7,7,768), 'mode': (1,),
'state': (9,)}`. Goal block `[forward, left, sin yaw, cos yaw]` unchanged;
goal block zeroed in explore mode (unchanged); mode bit unchanged.

### D2 — trainable conv compressor (the encoder)

The jaxrl2 `PixelMultiplexer` already applies the conv `D4PGEncoder` to
image-like keys and per-key Dense+LayerNorm to flat keys, then concatenates.
The compressor therefore needs **no new module class** — it is a config:

- `encoder="d4pg"`, `cnn_features=(64, 128, 512)`, `cnn_filters=(3, 3, 1)`,
  `cnn_strides=(2, 2, 1)`, valid padding → conv stack output 1×1×512.
- `latent_dim=512` (the user's number) — PixelMultiplexer's post-conv
  Dense(512)+LayerNorm yields the **512-d compressed latent**; flat keys
  (goal/mode/state) each get their own Dense(512)+LayerNorm+tanh.
- Total trunk input to actor/critic heads: 512 (vision) + 512 (goal) + 512
  (mode) + 512 (state) = 2048 → MLP heads. (If this over-widths the heads,
  `latent_dims` per-key narrowing is the sanctioned knob — see
  pixel_multiplexer.py docstring — but the default equal width ships first.)

Training: compressor params live inside the critic; TQC quantile-critic
gradients train it end-to-end. `_share_encoder` (default ON) copies the
critic's encoder params into the actor each update — correct for a real conv
trunk per the tqc_learner.py comment block. Target critic tracks via the
existing soft updates. VC-1 itself is frozen (torch, `no_grad`,
`requires_grad=False`, existing VcSource code).

Recorded known-harmless quirk: D4PGEncoder's `max(x) > 1 → /255` branch fires
on ViT tokens (their range exceeds 1), but the LayerNorm after the projection
is scale-invariant, so the division is a no-op for learning. Pinned by a unit
test that asserts the conv stack output is finite/learnable under the /255
path.

Recorded decision — **augmentation OFF**: TQCLearner sets
`self.augument = not encoder is None`, so encoder="d4pg" would enable
`batched_random_crop` on the (7,7,768) grid. Cropping a semantic token grid
drops tokens; it is wrong here. The v8 config passes a new `augment_pixels`
override (env `AGENT_AUGMENT_PIXELS=0`, default from the old rule) through to
`augument` so token-grid runs never crop. (The base learner's crop code is
partially commented out; the GatedTQCLearner update path passes
`augument=self.augument` through — the override is set at construction.)

### D3 — state block drops has_collided (user 2026-09-21)

State block becomes `[clip(speed/TARGET_SPEED, 0, 1), *8 prev controls]` =
9-d (was 10-d with the collision flag first). Behavioral meaning, recorded at
user decision time: goal mode terminates on collision anyway; in explore mode
a policy that has crashed once sees no obs difference and learns only from the
−5/step collision penalty. Implemented in `VAEGoalWrapper._state_block` behind
a NEW flag `OBS_STATE_COLLISION` (default 1 = current behaviour; v8 sets 0).
It must NOT be named `OBS_WITH_COLLISION` — that flag already exists and means
the separate image-goal `coll` obs key (vae_goal_wrapper.py:136-143); reusing
the name would silently couple two unrelated switches.

### D4 — exact-v7 recipe, fresh 450k, workstation

Fresh run from step 0 (no warm start: the encoder change makes v7
checkpoints non-transferable, consistent with the user's fresh-450k choice).
450k steps, Town01+Town10HD at Low, windowed, on this workstation. New
launcher `train_v8.sh` deriving from `train_v7.sh` with:
- `RUN_NAME=carla_gated_her_v8_vc1`, `PERCEPTION=vc1`,
  `VC1_READOUT=grid2` (already the vc1_source default), plus the new
  `AGENT_ENCODER=d4pg`, `CNN_FEATURES=64,128,512`, `CNN_FILTERS=3,3,1`,
  `CNN_STRIDES=2,2,1`, `AGENT_LATENT_DIM=512`, `OBS_STATE_COLLISION=0`,
  `AGENT_AUGMENT_PIXELS=0` exported into the snapshot (the v7 lesson: the
  snapshot must carry every flag the trainer reads).
- Everything else byte-identical to the v7 recipe.

The torch/venv story is settled by precedent: the image-goal arm ran
torch+VC-1 inside the training venv (vc1-venv's torch 2.5.1+cu121 works on
driver 535; ig-venv already runs VC-1 on cuda — see the stopped run's
`[Latent] backend 'vc1_vitb_grid2' ... device=cuda` banner).

## Buffer cost (known, accepted, mitigated)

The replay buffer stores the (7,7,768) grid per step: 37632 floats ≈ 150 KB
(float32) → ~68 GB at 450k steps. The image-goal arm held 136k such rows
(~20 GB) comfortably; 450k is 3× that. Mitigation: the v8 buffer stores the
pixels field as **float16** (~34 GB), converting to float32 at sample time.
Token values are post-LayerNorm ViT outputs, well inside fp16 range.
Implementation: a dtype knob on the episode buffer for the pixels key only
(HER relabelled fields — goal block — stay float32; the goal block is what
relabel arithmetic touches).

## Out of scope

- Any change to v7's reward, termination, learner, HER rules, towns, caps,
  discounts, or TQC/gated-critic settings (user: exact v7 recipe).
- The image-goal arm's single-critic route and its reward (frozen as of
  Task 20; its checkpoints remain in checkpoints/).
- Any cluster move (user: workstation).
- Warm-starting, actor freeze, or value-bound recalibration for the goal
  critic beyond what v7 already ran with (`GATED_ALLOW_DISCOUNTED_SORB=1`
  goal-critic semantics unchanged).

## Verification plan

1. Unit tests: state block without collision (9-d, v7 replay flag restores
   10-d); pixels-shaped wrapper output for PERCEPTION=vc1 grid2; the d4pg
   config instantiates PixelMultiplexer with a (7,7,768) obs dict (JAX init,
   no simulator); `augument=False` actually reaches the learner's update
   call; fp16 buffer round-trip.
2. Offline smoke: one CARLA episode stepping the real VC-1 grid → conv
   compressor → action, asserting obs shapes, finite values, and that the
   town-leak check accepts `trained_on_towns=[]`.
3. Training launch (only on user go, standing rules: windowed render,
   prove-alive, kill-by-PID procedure) → first windows show
   `[VAEGoal] obs={'goal': (4,), 'pixels': (7,7,768), 'mode': (1,),
   'state': (9,)} = 37650-d`-class banner and the d4pg encoder banner.
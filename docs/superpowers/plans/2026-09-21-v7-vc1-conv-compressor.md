# v8: v7 Recipe + VC-1 Conv Compressor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the exact v7 gated two-mode recipe with VC-1 grid2 patch tokens compressed by a trainable 512-d conv compressor, and the state block without has_collided.

**Architecture:** frozen torch VC-1 → (7,7,768) grid2 obs under a `pixels` key → jaxrl2 PixelMultiplexer routes it through D4PGEncoder (64,128,512 convs) → Dense(512)+LayerNorm = the 512-d latent; flat keys (goal 4, mode 1, state 9) get per-key Dense+LayerNorm; two-critic GatedTQCLearner unchanged.

**Tech Stack:** JAX/Flax (jaxrl2, imported from /home/robotlab/scratch/robot-learning-tools — NEVER edit that repo), torch VC-1 via existing VcSource, carla-rl trainer `train_carla_tqc.py`.

**Spec:** docs/superpowers/specs/2026-09-21-v7-vc1-conv-compressor-design.md

## Global Constraints

- EXACT v7 recipe otherwise: reward, termination, two-critic GatedTQCLearner, DISCOUNT=0.99 + GATED_ALLOW_DISCOUNTED_SORB=1, ACTOR_Q_NORM=1, HER, Town01+Town10HD (`configs/town_split_v7.json`), Low quality, windowed render.
- VC-1 trunk stays frozen; the ONLY trainable vision params are the conv compressor inside the JAX nets.
- jaxrl2 (robot-learning-tools) is a shared external checkout — DO NOT EDIT IT. All changes live in carla-rl; the augmentation override is a trainer-side post-construction attribute set.
- Never name the new state flag `OBS_WITH_COLLISION` (exists, means the coll key). The new flag is `OBS_STATE_COLLISION` (default 1 = current 10-d block; 0 = 9-d without has_collided).
- Tests always run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q` from the worktree root.
- No CARLA starts, no training runs, no bare `git stash`. Commits in the worktree.

---

### Task 1: State block without has_collided (OBS_STATE_COLLISION)

**Files:**
- Modify: `src/vae_nav/vae_goal_wrapper.py` (`_state_block`, :363 area; flag near the other OBS_WITH_* flags :70-143)
- Test: `tests/test_gated_obs.py`

**Interfaces:**
- Produces: env flag `OBS_STATE_COLLISION` (default "1"); with 0, `_state_block` returns `[clip(speed/TARGET_SPEED,0,1), *history.flatten()]` = 9-d (1 + 2·N_ACTION_HISTORY). Banner `obs={...}` text must show the new state width.
- Consumers: Task 5's launcher exports OBS_STATE_COLLISION=0; nothing else reads it.

- [ ] **Step 1:** Failing test: with OBS_STATE_COLLISION=0 (subprocess or reload pattern used by existing obs tests), state block is 9-d, first element is the speed ratio (a step at speed 0 and history zeros gives all-zeros except ratio 0.0 — pin the layout), and byte-identical for the same step in both modes. With default (unset/1), the block stays exactly the current 10-d `[has_collided, ratio, *hist]` (regression guard).
- [ ] **Step 2:** Implement: module flag `STATE_COLLISION = os.environ.get("OBS_STATE_COLLISION", "1") == "1"` beside WITH_STATE; `_state_block` prepends has_collided only when STATE_COLLISION; STATE_DIM computed from the flag (the existing STATE_DIM = 2 + 2·N_ACTION_HISTORY becomes conditional); banner width follows the space.
- [ ] **Step 3:** Full suite green; commit "feat: OBS_STATE_COLLISION flag — v8 drops has_collided from the state block".

### Task 2: Pixels-shaped VC-1 grid obs key

**Files:**
- Modify: `src/vae_nav/vae_goal_wrapper.py` (`_observation`, observation space, banner)
- Test: `tests/test_gated_obs.py`

**Interfaces:**
- Produces: env flag `OBS_LATENT_AS_PIXELS` (default "0"). When 1: obs key `pixels` with shape (7,7,768) replaces the flat `latent` key (same content, reshaped; assert `source.latent_dim == 37632` at init, SystemExit otherwise); observation space Box entry `pixels` with that shape; banner shows `'pixels': (7, 7, 768)` instead of `'latent'`.
- Consumers: Task 3 (agent kwargs) keys off the same env flags; buffer fp16 (Task 4) stores the `pixels` key.

- [ ] **Step 1:** Failing test: with OBS_LATENT_AS_PIXELS=1 and a stub source (latent_dim=37632, latent() returns a 37632-vector), `_observation` returns key `pixels` shaped (7,7,768) and no `latent` key; values equal the vector reshaped. Default (unset) keeps the flat latent key exactly as today.
- [ ] **Step 2:** Implement: publish `np.asarray(latent, np.float32).reshape(7, 7, 768)` under `pixels`; space + banner follow.
- [ ] **Step 3:** Suite green; commit "feat: publish VC-1 grid2 tokens as a (7,7,768) pixels obs key".

### Task 3: Conv-compressor agent kwargs + augmentation override

**Files:**
- Modify: `train_carla_tqc.py` (`agent_kwargs`, the make-agent function :270-296)
- Test: new `tests/test_v8_agent_kwargs.py` (pure config assertions; a JAX init smoke if the env allows — guard with `pytest.importorskip("jax")`)

**Interfaces:**
- Produces: env flags `AGENT_ENCODER` (values "none"|"d4pg"; default "none" = today's behaviour), `CNN_FEATURES`/`CNN_FILTERS`/`CNN_STRIDES` (comma lists; default to today's TQCLearner defaults), `AGENT_LATENT_DIM` (default 50 = today), `AGENT_AUGMENT_PIXELS` (default "1" = today).
- Consumers: Task 5's launcher exports AGENT_ENCODER=d4pg, CNN_FEATURES=64,128,512, CNN_FILTERS=3,3,1, CNN_STRIDES=2,2,1, AGENT_LATENT_DIM=512, AGENT_AUGMENT_PIXELS=0.

- [ ] **Step 1:** Failing tests: agent_kwargs returns encoder="d4pg", cnn_features=(64,128,512), cnn_filters=(3,3,1), cnn_strides=(2,2,1), latent_dim=512 under the v8 env; defaults reproduce today's dict exactly (no new keys when unset); AGENT_AUGMENT_PIXELS=0 → a freshly constructed TQCLearner (small obs, jax, importorskip) has `.augument is False` after construction; unset leaves True for encoder="d4pg" (pin the old hardcode).
- [ ] **Step 2:** Implement in `agent_kwargs` (comma-split int lists; strip spaces) and the construction sites: after building ANY learner with encoder != "none", if AGENT_AUGMENT_PIXELS=0 set `learner.augument = False` (trainer-side attribute override — the GatedTQCLearner update path forwards `self.augument`; NO edits to the jaxrl2 repo).
- [ ] **Step 3:** Suite green; commit "feat: d4pg conv-compressor kwargs + augmentation override for v8".

### Task 4: fp16 buffer storage for the pixels field

**Files:**
- Modify: the episode buffer the gated trainer uses (`src/vae_nav/episode_her.py` record/sample path — find where obs dicts are stored and returned)
- Test: `tests/test_image_goal_her.py` (or a new focused test file)

**Interfaces:**
- Produces: env flag `BUFFER_PIXELS_FP16` (default "0"); when 1, the `pixels` array is stored as float16 and restored to float32 on read; goal/mode/state stay float32.
- Consumers: Task 5's launcher sets BUFFER_PIXELS_FP16=1.

- [ ] **Step 1:** Failing test: record a row with obs {"goal": f32, "pixels": f32(7,7,768), "mode": f32, "state": f32} under the flag; the stored field is fp16 (assert dtype or max-abs-err ≤ 1e-2 after round-trip) and sampled observations return float32 with allclose to the original within fp16 tolerance; without the flag, byte-identical float32 storage as today.
- [ ] **Step 2:** Implement the dtype conversion at record and sample paths, keyed on the obs key name "pixels" only.
- [ ] **Step 3:** Suite green; commit "feat: fp16 buffer storage for pixels (v8, ~34 GB at 450k)".

### Task 5: Launcher train_v8.sh

**Files:**
- Create: `train_v8.sh` (copy `train_v7.sh` — read it first; it is the launcher of record)
- Test: `bash -n train_v8.sh`

**Interfaces:**
- Consumes: every flag from Tasks 1-4.
- Produces: RUN_NAME=carla_gated_her_v8_vc1; new exports in the snapshot's required-vars list: PERCEPTION=vc1, VC1_READOUT=grid2, AGENT_ENCODER=d4pg, CNN_FEATURES=64,128,512, CNN_FILTERS=3,3,1, CNN_STRIDES=2,2,1, AGENT_LATENT_DIM=512, OBS_STATE_COLLISION=0, OBS_LATENT_AS_PIXELS=1, AGENT_AUGMENT_PIXELS=0, BUFFER_PIXELS_FP16=1. MAX_STEPS=450000. Everything else byte-identical to v7's recipe (TOWN_LIST=configs/town_split_v7.json, Low, CARLA_OFFSCREEN=0, GATED=1, two-critic route, HER on).

- [ ] **Step 1:** Copy train_v7.sh → train_v8.sh; change RUN_NAME/output paths; add the new exports to BOTH the export block and the snapshot required-vars regex (the v7 lesson: the snapshot must carry every flag the trainer reads); banner line naming the compressor config (d4pg 64,128,512 → 512-d, aug off, fp16 buffer).
- [ ] **Step 2:** `bash -n train_v8.sh` passes; diff against train_v7.sh shows ONLY the intended changes.
- [ ] **Step 3:** Commit "feat: train_v8.sh launcher — v7 recipe + VC-1 conv compressor".

### Task 6: Integration tests (no simulator)

**Files:**
- Create: `tests/test_v8_pixel_obs_integration.py`
- Test: itself

**Interfaces:**
- Consumes: Tasks 1-3's flags; the real jaxrl2 PixelMultiplexer/D4PGEncoder via train_carla_tqc's agent construction (importorskip jax).

- [ ] **Step 1:** Test A (JAX init): build the v8 obs space `{'goal': Box(4,), 'pixels': Box((7,7,768)), 'mode': Box((1,)), 'state': Box((9,))}` and init TQCLearner AND GatedTQCLearner under the v8 env flags; assert encoder_*_pixels params exist in both actor and critic, the /255 path leaves finite outputs, and one fake forward produces the documented trunk width (vision 512 + per-key 512s).
- [ ] **Step 2:** Step a one-update smoke on random batches (goal rows + explore rows mixed) asserting finite losses and that `learner.augument is False` on the path actually used.
- [ ] **Step 3:** Suite green; commit "test: v8 pixel-obs agent integration".

---

## Verification

1. Full suite: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q` green after every task.
2. Offline smoke (controller-run, before launch, on user go): one CARLA episode with the real VC-1 grid → conv compressor → action; assert obs banner `{'goal': (4,), 'pixels': (7,7,768), 'mode': (1,), 'state': (9,)}`, finite values, town-leak check passes with trained_on_towns=[].
3. Training launch ONLY on explicit user go, following every standing rule (windowed render verified in prove-alive, kill-by-PID stop procedure).
# Learner validation and a FastRLAP-matched learner (2026-11-10)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Find out whether our SAC implementation learns correctly on problems with known answers,
and provide a fresh PyTorch learner that matches FastRLAP's (kylestach/fastrlap-release) exactly.

**Why (user, 2026-11-10):** "have you considered that our implementation might be bad". The CARLA
critic shows Q flat in the action, best action at the edge and no fault anticipation. A bug in
`GatedRecurrentSAC` would produce the same symptoms, and the offline critic study and the
privileged-critic control build that same class. It has never been run on a task with a known answer.

**Architecture:** three learners on benchmark tasks, judged by thresholds declared below before
any run. A = CleanRL's single-file SAC (the control). B = our `GatedRecurrentSAC` as the C1 runs
configured it. C = a fresh port of jaxrl5's `SACLearner` + `DrQLearner` (`torch_rl/frl/`).

**Tech stack:** PyTorch 2.5.1, gymnasium 1.4.0, mujoco 3.17 + dm_control 1.0.50 (T2), CPU-only
Slurm jobs on the cluster in `~/bench-venv` (inherits `carla310`'s torch; run with
`PYTHONNOUSERSITE=1`, because `~/.local` holds an older torch 2.3.1 / gymnasium 0.29.1 that
otherwise shadows it). `stable-baselines3` 2.6.0 installed `--no-deps` for CleanRL's replay
buffer; it declares gymnasium < 1.2 (we have 1.4). The control's result is what validates that.

**Spec:** the reference code itself, pinned to `main` of kylestach/fastrlap-release as fetched
2026-11-10: `jaxrl5/jaxrl5/agents/sac/sac_learner.py`, `agents/drq/{drq_learner,augmentations}.py`,
`networks/{mlp,ensemble,state_action_value,pixel_multiplexer}.py`, `networks/encoders/d4pg_encoder.py`,
`distributions/tanh_normal.py`, `agents/sac/temperature.py`; configs
`fastrlap-ros/offroad_learning/config/{drq,redq}_config.py`; loops
`fastrlap-mujoco-env/train_online_pixels.py`, `fastrlap-ros/offroad_learning/src/offroad_learning/training.py`.

## Global constraints

- User rules: all experiments on the cluster; at most one GPU job, only the L40 (these are CPU
  jobs); one seed per arm ("no seeds"); thresholds declared before results; no quantile critics.
- The L40 queue (B2 Town01 → B1 Town04 → B1 Town10HD → c1 → a3val) is not touched.
- Learner C is a separate module with its own entry point; `train_carla_sac.py` and the plain-SAC
  refusal are untouched by this plan.
- Every new module is written test-first (`PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).

## What learner C must match (from the reference)

| Part | Reference behaviour |
|---|---|
| Critic | Ensemble of `num_qs` heads: MLP(256,256) → Dense(1); every hidden layer Dense → (Dropout) → LayerNorm → ReLU (`activate_final=True`); input `[obs, action]` |
| Target | sample `num_min_qs` of the TARGET heads without replacement, once per critic update; `y = r + γ·mask·min(Q')`; minus `γ·mask·α·log π(a'|s')` only if `backup_entropy` |
| Critic loss | `mean((Q_i − y)²)` over heads and batch |
| Target update | Polyak `τ` after every critic update |
| Actor | MLP(256,256, ReLU, activate_final) → Dense mean, Dense log_std (clip [−20, 2]) → tanh-squashed diagonal Gaussian |
| Actor loss | `mean(α·log π(a|s) − mean over ALL heads Q(s,a))` |
| Temperature | `log α` parameter, init `log(init_temperature)`; loss `α·(H − H_target)`, H = −mean log π from the actor step |
| Update | batch split into `utd_ratio` minibatches: critic updated on each, then actor and temperature once on the last |
| Inits | Xavier-uniform kernels, zero biases (flax defaults) |
| Default target entropy | `−action_dim / 2` when not given |
| Pixel mode (DrQ) | d4pg encoder (Conv-ReLU ×4, VALID) → Dense(latent) → LayerNorm → tanh, per network; pixels /255; encoder owned by the critic and copied into the actor before every update; the actor's encoder output is stop-gradient; random crop (pad 4, edge) of obs and next_obs with independent draws per sample |
| Embeddings mode | precomputed features → Dense(50) → LayerNorm → tanh per network (`stop_gradient=True`); states get their own Dense(50) → LN → tanh |
| Loop (sim) | random actions before `start_training`; one `update(batch, utd)` per env step; mask = 1 unless terminated (time-limit truncation keeps 1); target entropy annealed `−25e−7` per step |

## Declared pass thresholds (written before any run)

One seed per arm. "Eval" = mean return of 10 deterministic episodes (mean action), seeds 1000–1009.

| Task | Setting | Control A must reach (else setup broken, stop) | A learner passes if |
|---|---|---|---|
| **T1** Pendulum-v1, states | 50k env steps | eval ≥ −250 | eval ≥ −250 |
| **T2-env** DMC cartpole swing-up, states | action repeat 1, 200k env steps | eval ≥ 700 | (control only: checks the env wrapper) |
| **T2** DMC cartpole swing-up, pixels | 84 px RGB, 3-frame stack, action repeat 8, 100k env steps | (no pixel control: bar is DrQ's published 759 ± 92) | eval ≥ 650 |

Reading: A fails → setup problem, nothing else read. A passes, B fails, C passes → our learner has a
bug; every SAC claim is withdrawn until found (bisect B vs C on T1). B passes, C fails → the port is
wrong; fix before CARLA. All pass → both implementations sound; T3 (toy lane keeping, next plan)
then tests the CARLA mechanism itself.

## Tasks

### Task 1: Benchmark env and the control on T1
- [ ] Vendor CleanRL `cleanrl/sac_continuous_action.py` at a pinned commit into `scripts/bench/cleanrl_sac_continuous.py`, unmodified except an end-of-run deterministic eval (10 episodes, seeds 1000–1009) and a JSON result file.
- [ ] Slurm wrapper `slurm/bench_cpu.sh` (CPU partition, `~/bench-venv`, `PYTHONNOUSERSITE=1`).
- [ ] Launch A on T1 (Pendulum-v1, 50k steps). Expected: eval ≥ −250.

### Task 2: T2 env and its control
- [ ] Check offscreen MuJoCo rendering on a CPU node (`MUJOCO_GL=egl`, then `osmesa`); if neither works, T2's pixel arms wait for an L40 slot (user's call).
- [ ] `scripts/bench/dmc_env.py`: dm_control suite → gymnasium env (`states` and `pixels` modes, action repeat, frame stack), registered ids `dmc/cartpole-swingup-states-v0`, `dmc/cartpole-swingup-pixels-v0`. Tests: shapes, action repeat sums rewards, frame stack order, terminal/truncation flags.
- [ ] Save 4 rendered frames for a visual check.
- [ ] Launch A on T2-env (states, 200k steps). Expected: eval ≥ 700.

### Task 3: Learner C (`torch_rl/frl/`)
- [ ] `networks.py`: MLP, StateActionValue ensemble (batched heads), TanhNormal actor, D4PG encoder, Multiplexer (pixel / embeddings / states). Tests pin shapes, init, LayerNorm placement, stop-gradient.
- [ ] `learner.py`: `SACLearner.update(batch, utd_ratio, update_temperature=True)`, `DrQLearner` (encoder sharing + random crop). Tests: target with and without `backup_entropy`; subset min without replacement; actor uses the mean of all heads; temperature loss sign; critic updated `utd` times and actor once; Polyak after each critic step; crop pads 4 with edge values and keeps shape.
- [ ] `scripts/bench/frl_train.py`: the sim loop (random start, mask rule, entropy annealing, eval, JSON).
- [ ] Launch C on T1 and T2 (pixels). Configs: FastRLAP `redq_config` (T1) and `drq_config` (T2) values, with target entropy from the learner's own default (`−action_dim/2`), since the configs' −3.0 / −2.0 are tuned for the 2-D car.

### Task 4: Learner B on the benchmarks
- [ ] Adapter `scripts/bench/gated_sac_adapter.py`: benchmark observations → the rows `GatedRecurrentSAC` expects (one mode, `ONE_CRITIC=1`, the C1 config). Tests: row layout, mode column, masks.
- [ ] Launch B on T1 and T2. Same thresholds.

### Task 5: Report
- [ ] Outcome table against the thresholds, ledger lines, memory; message the user.

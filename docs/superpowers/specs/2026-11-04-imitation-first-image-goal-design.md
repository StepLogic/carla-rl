# Imitation first, then reinforcement: image-goal navigation in CARLA

Date: 2026-11-04. Branch `worktree-sac-image-goal`. Cluster repo `~/carla-rl` on
`login1.acg.maine.edu`.

## 1. Intent and why

The user wants an agent that **follows lanes and navigates to a goal given only
a goal image** in CARLA, and that can **keep going goal after goal for 200 m**.

Ten reinforcement-learning runs (v30-v39) established what blocks this, by
measurement:

- **Goal-reading works.** v30 (`slurm/v30_overlay.env`, checkpoint
  `checkpoints/sac_debitmono_t05t01_v30/step_300000`) scores 0.56 on held-out
  Town02 with the true goal image against 0.03 with another episode's image
  (n=100, paired McNemar p<0.0001). Today's code still reproduces it (0.525,
  n=40).
- **Driving is the bottleneck.** In towns it never trained on, v30 crosses a
  lane line or leaves the road every 12-32 m on average (median unbroken run
  6-11 m; Town02 31.7 m, Town04 22.2 m, Town10HD 19.8 m, Town03 12.2 m), and
  75-82% of failures are lane crossings. 18 of its 19 Town02 eval failures
  crashed before getting within 2 m of the goal. Overshooting the goal is rare
  (1 of 19).
- **Reward tuning has not fixed it.** v31-v39 changed the goal side of the
  reward (retreat charge, junction zones, chaining, arrival rules), the action
  floor, and the goal-resample rule. All lost to v30. v37-v39 learned their
  training towns but scored ~0 on Town02.

Diagnosis: reinforcement learning is learning lane keeping from pixels, without
the car's speed or steering in its observation, from a −1 at the moment of a
crossing, on two towns. CARLA's own autopilot (`BasicAgent`) drove 25/25 test
routes with zero crossings. So the plan is to **teach the driving by
imitation**, then **fine-tune with v30's reward**.

## 2. Decisions

The user approved the imitation route ("proceed", 2026-11-04). The two items
below were proposed together with it and not separately objected to, so they
stand **as my calls**, open to correction:

- **Training towns: Town01, Town04, Town05, Town10HD. Held out: Town02, Town03.**
  Nothing is ever recorded, trained or tuned on the held-out towns. Cost:
  Town04 and Town10HD stop being held-out cells of the long-distance benchmark
  (`logs/bench_long_episodes.json`). Their numbers must be reported as
  training-town numbers from now on.
- **Success bar** (section 6).

Fixed by the user earlier, and unchanged here:
- Encoder: frozen DEBiT-tiny binocular plus the trained monocular ResNet.
- Actor observation: frame, mode bit, binocular vector. No speed, no previous
  action.
- Action space: steer [−1, 1], throttle [0, 1], no brake, no reverse.
- Evals end on any fault (crossing, off-road, collision).

## 3. Goals, non-goals, anti-goals

**Goals**
1. A policy that keeps its lane on held-out towns for ≥200 m between failures.
2. On held-out towns it reaches image goals 5-25 m away, and does so because of
   the image (the paired wrong-goal control).
3. It chains goals to 200 m.

**Non-goals**
- No maps, no localisation.
- No change to the encoder, the actor observation or the action space. If M0
  shows the expert cannot drive without a brake, STOP and bring it to the user.
- No reward design beyond v30's in the RL stage.

**Anti-goals** (failure even if numbers look good)
- A policy that succeeds by lane following while ignoring the image, i.e.
  wrong-goal success ≈ true-goal success.
- Any data, training or tuning on Town02 or Town03.
- A single-seed result presented as the answer.
- Changing the eval rules to lift numbers. Faults always end eval episodes;
  arrival stays 2 m at ≤10 km/h.

## 4. The pipeline

### M0. Make the expert run on the cluster, and benchmark it (gate)
- `src/vae_nav/expert_oracle.py` (`BasicAgentExpert`) imports CARLA's
  `agents` package from a hard-coded local 0.9.15 path. The cluster client env
  (`~/carla-cluster/envs/carla310`) has no `agents`.
- The 0.9.16 container `~/carla-cluster/carla0916nv.sif` ships it at
  `/workspace/PythonAPI/carla/agents`. Copy it out to
  `dependencies/carla_agents/agents` and import from there (networkx 3.4.2 and
  shapely 2.0.4 are already in the env).
  - `apptainer exec` does not work on the login node; use a compute node.
  - `slurm/sync_to_cluster.sh` uses `--delete`, so the copy must either live in
    the local tree and be synced like the other dependencies, or sit under an
    excluded path.
- **The expert has never been checked under today's controls.** It was
  benchmarked with the old box (throttle ≤0.4, steer ≤0.5, with brake).
  `_to_action` folds brake into negative throttle, which today's env clips to
  0, so the expert can only coast.
- Benchmark it, recording nothing, on the training towns: 5-25 m goals and
  100-200 m drives.
- **Gate:** ≥95% arrivals on 5-25 m goals, and ≥200 m mean between failures.
  If it fails, stop and report; a brake decision is the user's.

### M1. Record expert demonstrations
- A new recording mode in `train_carla_sac.py` (alongside `EVAL_TRACE` /
  `run_trace`): the expert drives, the env and wrappers run exactly as in
  training, and every step is stored as the replay row the learner would
  store.
  - Row: `pxr.mono_row(frame, mode, binoc)`.
  - Goal block: `pxr.goal_block`.
  - The env reward, terminal and mode.
  - Action in actor space via `to_actor`.
  - Speed and pose, for analysis.
  - Use `build_env(..., encoder=enc)` so `binoc` is computed as in training.
- **Both modes, about 50/50, as v30 trains:**
  - Goal mode (mode bit 1): the env's own 5-25 m goal, junction fraction
    0.85, the latched goal image.
  - Explore mode (mode bit 0): the expert drives 100-200 m along lanes to a
    destination unrelated to the latched goal image. This teaches long lane
    keeping, and it makes the lane-keeping metric measurable with the same
    deterministic explore traces used for v30.
- **Recovery data (my call, builder may tune):** per episode, a steering noise
  level σ ∈ {0, 0.1, 0.2} is added to the EXECUTED action, while the stored
  label is always the expert's clean action at that state. This puts "how to
  come back from drifting" into the data.
- About 300k steps across the four training towns (my estimate).
  - At 120×120 uint8 frames plus fp16 binoc this is ~16 GB.
  - Store under `checkpoints/` (the sync deletes other untracked paths).
  - Shards must load straight into `torch_rl.episode_buffer.DeviceEpisodeBuffer`.

### M2. Behaviour cloning
- Instantiate `GatedRecurrentSAC` exactly as `train_carla_sac.main()` does, so
  the result loads with `agent.load()`. Then `run_eval`, `run_trace` and
  `INIT_FROM` work unchanged.
- Train the visual encoder, the GRU and the actor together on a cloning loss.
  - Use chunks from `DeviceEpisodeBuffer.sample_chunks` with the learner's
    burn-in (`_roll` gives core states).
  - Then `cat([s, _state(obs)])` and `_actor_forward` give mean and log_std.
  - Unlike SAC's update, where the actor sees detached features, the encoder
    must learn from this loss.
- Loss: builder's choice between squashed-Gaussian NLL and MSE on tanh(mean).
  Requirement: the deterministic action must imitate well, and log_std must
  not be degenerate for the SAC stage. Sync the target copies after cloning.

### M3. Measure (decision point)
- Lane keeping: deterministic explore traces (`EVAL_TRACE=1`,
  `EVAL_FORCE_MODE=0`, `TRACE_NOISE_SCALE=0`, `TRACE_END_ON_CROSSING=1`) on
  Town02 and Town03. Mean metres between failures, against v30's 31.7 and
  12.2 m.
- Navigation: `run_eval` on Town02 and Town03, n=100, true goal and
  `WRONG_GOAL_MODE=1`, paired McNemar (`~/paired_wrong.py`). Also the approach
  fields now in `eval.json`.
- Goes to M4 if lane keeping misses the bar, otherwise to M5.

### M4. Correct the drift (only if M3 misses lane keeping)
- One or two DAgger rounds. The cloned policy drives; at every visited state
  `BasicAgentExpert.act()` supplies the label (it replans from the car's
  actual pose).
- Add the new data, retrain, re-measure.

### M5. Reinforcement fine-tune
- SAC with v30's reward and launch settings, `INIT_FROM` the clone.
- The expert data stays as a second replay, with a cloning loss on the actor
  so the policy does not forget how to drive. This is new learner code. Weight
  and decay are the builder's call. Warm starts without this have destroyed
  goal steering before.
- **Goal chaining to 200 m:** use `GOAL_CHAIN_M=200` with chained goals drawn
  **5-25 m ahead**, inside the image's range. Today's 10-30 m past the next
  junction gives 40-90 m goals, which I suspect, but have not shown, taught
  v37-v39 that the image is uninformative.
- Chaining is training-only today, so a chain eval mode is needed for the
  200 m criterion.

## 5. Constraints and house rules
- All training and evals run on the ACG cluster (L40s on node-g102). Local
  work is analysis only.
- Every run sources `slurm/v52_reward.env`, plus a named overlay via
  `REWARD_OVERLAY`. Launch settings go in the overlay, never on the command
  line: v35's first launch silently lost HER that way.
- After launch, prove the run is alive (process, fresh log, GPU). Lead training
  reports with `policy=`.
- Never edit a running bash script.

## 6. Success bar (my proposal)
| Measure | Today (v30) | Target |
|---|---|---|
| Mean metres between failures, Town02 / Town03 | 31.7 / 12.2 | ≥200 / ≥200 |
| Town02 success, 5-25 m, n=100 | 0.56 | ≥0.80 |
| Town02 wrong-goal success | 0.03 | ≤0.10 (p<0.001) |
| Town03 success, 5-25 m | 0.35 | ≥0.60 |
| 200 m chains completed, Town02 | ~0 | ≥50% |
| Seeds for the final recipe | 1 | 2 |

## 7. Risks
- **The expert without a brake** (M0 gate).
- **Compounding error.** The clone drifts into states the expert never
  visited. The recovery noise in M1 and DAgger in M4 exist for this.
- **Far-goal branch choice.** Beyond ~30 m the goal image does not overlap the
  view, so which branch to take at a junction cannot be inferred from it.
  That is why goal-mode data and chained legs stay within 5-25 m.
- **Fine-tune drift.** RL fine-tuning can erode what cloning taught. The
  cloning loss in M5 guards against this.

## 8. Left to the builder
- Shard format and recording parallelism.
- The cloning loss form.
- Noise levels, within the intent of section 4.
- The M5 cloning weight and schedule.
- Exact data volume.
- How the chain eval is wired.

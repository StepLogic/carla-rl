# Plan: image-goal navigation with lane following in CARLA (2026-11-09)

## Context

You asked for a full plan for a car that reaches a goal shown only as an image while staying in its lane. This is the paper's task: the navigator uses images only, and any privileged signal is confined to an arm labelled as an oracle.

Where things stand. All numbers are deterministic, and a fault ends the episode.
- **Driver.** The imitation clone c0 (`il_bc_c0` step_15000) is the only learner that works.
  - Held-out Town02: .69 success (wrong goal .07). Town03: .65 (.08). Goals 5–25 m.
  - Metres between failures: 464 in Town02, 210 in Town03.
- **Short routes.** Fresh map benchmark, success = within 1 m at any speed:
  - Without a map: easy .77, medium .71.
  - Maps add a few points at most (best: SGM .79/.75).
- **Long routes fail.** Multi-turn set: 200 episodes, 50–150 m, at least 2 junction turns.
  - No map .06, uniform .11, entropy (z=2.5) .13, SGM .15.
  - True-pose localisation on key-only maps: .08–.12. ViNT: .09 or lower.
  - Town03: .07 or lower for every arm.
  - How the 200 uniform-map episodes end: 109 lane faults (64 off-road, 45 crossings), 66 time-outs, 22 reached.
  - True pose does no better, so the bottleneck is the driver, not localisation. The driver must keep its lane through junction turns and reach each subgoal.
  - Sources: `results/map_anyspeed_2026-11-09/hard_set_z25_summary.txt`, `docs/sections/map_navigation.tex`.
- **RL.**
  - Every SAC fine-tune of c0 collapsed within a few hundred to about 1,000 actor updates.
  - SAC from scratch learned to use the goal by 25–50k steps (P2: 31–36 of 80), then lost it by 100k (8 of 80).
  - In every variant the critic behaves like V(s) + g(a). The variants: DrQ critic head, imitation-pretrained frozen head, twin critics, no actor freeze.
  - Its best steering is at full lock in most states and barely tracks the expert's steering (correlation −0.10 to +0.42).
  - The paper's RL rows are empty.
- **Spec bar** (`docs/superpowers/specs/2026-11-04-imitation-first-image-goal-design.md`):
  - Town02 success ≥ .80 with wrong goal ≤ .10.
  - Town03 success ≥ .60.
  - ≥ 300 m between failures in both towns.
  - ≥ 50% of 200 m chains completed.
  - Met: Town03 success and Town02 distance between failures. Still missing: Town02 success, Town03 distance between failures and chains (never run).
  - The spec's two-seed line is dropped. You, 2026-11-09: "no seeds - one policy that works".

What this plan delivers:
1. **One policy that works.** It meets the spec bar and does better on the multi-turn set. It is whichever candidate passes its gates and scores best on training-town validation: the improved clone, or an RL policy if one beats it.
2. An RL row: either a SAC that works, or an honest report that it fails, with the cause.
3. Final numbers run once, on episodes never used to choose anything.

## Decisions for you (my recommendation in bold)

1. **Reopen data collection for the clone.**
   - You closed DAgger on 2026-11-05.
   - I recommend **one DAgger round under deployment conditions**: c0 drives chained goals past junctions in the training towns, and the expert relabels every frame (B1). Junction turns are where c0 fails.
   - Size: **300 chains of 200 m per training town**, about 600k frames in total.
   - Evidence it can work: ViNT's lane-centre labels plus 3 DAgger rounds took it from .32/.11/.01 to .51/.40/.14 on the fresh set.
   - Risk: the earlier clone's DAgger round 1 (s1) raised its wrong-goal rate to .22/.15. Gate B checks for this.
2. **The RL arm.** I recommend **a three-run action-repeat test** (k = 1, 2, 4) before any long run (C1).
   - It stays inside the plain-SAC rule: action repeat is standard in DrQ and DrQ-v2.
   - The cause it tests: at 10 Hz one action barely changes the next frame, so Q(s,a) collapses toward V(s) (Tallec et al., ICML 2019).
   - If it fails its gate, you choose one of two:
     - a recurrent PPO fine-tune. Its critic is V(s) only. The code is in robot-learning-tools `torch_rl/recurrent_ppo.py`. It is a new algorithm, so it needs your approval.
     - **report the best validated SAC checkpoint, with the critic analysis as the finding.**
3. **Control rate.** Action repeat lowers the RL arm's control rate from 10 Hz to 5 or 2.5 Hz. The clone stays at 10 Hz.
4. **Final episodes.** The multi-turn episode set (41000/42000) was used to choose z=2.5. I recommend **a new, never-used multi-turn episode set for the final table**, run once when you say so.
5. **Stop** two things:
   - further projection-head variants;
   - the pending cluster run `sac_c0_plain_rs10` (task 36). It is the same recipe as runs that collapsed.

   The local projection run keeps going until you stop it.

## Rules every step follows

- **Evaluation towns.** No training data from Town02/Town03, and no decision based on them.
  - Choices rest on Town05 + Town01 validation.
  - Every threshold is written down before its result arrives.
- **Navigator inputs.** Images only. True pose appears only in a labelled oracle arm.
- **SAC runs.**
  - Source `slurm/plain_sac.env` last.
  - LayerNorm critic.
  - v52 reward plus the named overlay.
  - Evals run inside the job (`slurm/train_with_evals.sbatch`), one at a time.
  - Evals are deterministic and faults end the episode.
- **Every eval reports:**
  - true-goal success with the paired wrong-goal success;
  - crossings per episode;
  - metres between failures.

  Success = within 1 m at any speed (your 2026-11-09 rule). Comparisons use exact McNemar tests, with Holm correction across the declared set.
- **Cluster practice.**
  - All runs on the cluster. Deploy by tmp + mv.
  - Warn the mapping session before editing `train_carla_sac.py`.
  - RL ports: 4000–5999 and 7800–7999.
  - Kill CARLA by PID or port.
- **Not changing:** the DEBiT encoder and the observation (yours, fixed), the reward, and the set of comparison arms.

## Step 0 — Is it the encoder? (offline, about a day, before any new training)

You chose on 2026-11-09:
- the encoder test runs first, before any new training;
- the privileged critic runs offline only.

What we know:
- c0 drives with these features: 68/80 on validation, 464 m between failures in Town02.
- Run S2 kept c0's encoder and GRU frozen and still went from 68/80 to 0/80 in about 1,200 SAC actor updates.
- So the features are good enough to drive; the critic can't use them to judge how much to steer.

Not yet tested:
- **Heading in c0's features.** Whether they carry heading error, which is what one steering step changes. The October seg-VAE probe found offset R² .73–.79 but heading ≤ .20, with no transfer between towns.
- **Encoder trained by the critic.** Whether a critic that trains the encoder (as DrQ does) would do better than one on the frozen copy-the-expert encoder.

The tests:
- **0a. Feature probe** `scripts/probe_c0_features.py`.
  - Data: c0 step_15000 on recorded Town05 + Town01 demo episodes, read the way `bc_loss` reads them.
  - Inputs, probed separately: the camera latent, the GRU state and the DEBiT binocular features.
  - Targets: lateral offset, heading error, and the one-step heading change.
  - Labels: the shards store `pos` only, so labels come from positions plus each town's OpenDRIVE map. That is the approach of vint-carla's `scripts/vint/build_lane_labels.py`.
  - Probes: ridge and a small MLP.
  - Splits by episode, a shuffled-label control, and a Town05 → Town01 transfer test.
  - Test file: `tests/test_lane_labels.py`, label builder on a synthetic straight lane and a synthetic curve.
- **0b. Privileged-critic control.** A diagnostic only, never in a result. Offline only, as you approved; no CARLA run.
  - Re-run the offline critic study (`scripts/offline_critic_slope.py` / `torch_rl/slope_diag.py`) with critic input = true offset, heading error, curvature ahead, speed and the goal's relative position. Same demonstrations, 20k updates.
  - Measure the steering correlation and the full-lock share with the 21-point probe.
- **Gate 0 (declared now).**
  - **Encoder is the bottleneck** if either holds:
    - heading-error R² below 0.3 on held-out episodes;
    - the privileged critic reaches steering correlation ≥ 0.5 while the image critic stays below 0.3 (it was +0.04 and +0.10).

    The encoder options come to you, since the encoder setup is fixed by your 2026-10-14 rule: encoder trained through the critic (DrQ), higher resolution, a patch-grid readout.
  - **Not the encoder** if heading R² ≥ 0.5 and the privileged critic also stays below 0.3. The problem is then the training signal, and C1 (action repeat) goes ahead.
  - **In between:** reported as such, and both B and C go ahead.

## A — Find where long routes fail (training towns, evals only)

- **A1. Training-town maps.**
  - Run expert coverage tours of Town05 and Town01 with the tour planner (`EXPERT_TRACE`, `TRACE_UNBROKEN`), using the same settings as `checkpoints/trace_expert_jx`.
  - Build uniform image maps with `scripts/build_image_sim_maps.py`, output to `checkpoints/maps_train_jx_img/uniform/<Town>.json`.
- **A2. Multi-turn validation set.** Built the way `slurm/turns_chain.sh` built the Town02/03 set:
  - probes with `EVAL_PROBE=1 EVAL_PROBE_TURNS=1` and `slurm/bench_turns.env`;
  - selection with `scripts/select_turn_episodes.py`;
  - 100 episodes per town, 50–150 m, at least 2 turns;
  - episode IDs from 51000, checked against the standard validation episodes and every recorded shard.
- **A3. c0 baselines.** One job, evals in sequence, all with `EVAL_TRAJ=1`. That option already exists (mapping session, feb0e83).
  - **Standard validation pair** (80 true + 80 wrong goal) at 1 m any speed.
    - New overlay: `slurm/il_eval_val_chw_any1m.env` = jr35nb + `GOAL_RADIUS_M=1.0`, as on vint-carla 0f889b1.
    - This re-measures c0's baseline under the current rule.
  - **Chain eval**, 200 m, 100 per town (`slurm/il_eval_chain_chw.env`), run twice:
    - `GOAL_CHAIN_STYLE=ahead`, the spec metric;
    - `past_junction`, one junction per goal.
  - **The A2 set** with `slurm/bench_map_debug.sh`, three arms: no map, uniform image, true-pose oracle.
- **A4. `scripts/route_failure_taxonomy.py`.**
  - Reads `eval.json` and `traj/ep*.json` and assigns every failure to one class:
    - lane fault inside a junction or up to 15 m after one;
    - lane fault elsewhere;
    - wrong branch: leaves the junction more than 5 m off the planner's road route;
    - stall: below 2 km/h for at least 50 steps;
    - passed the subgoal without switching (map arms only): within 3 m of the node, no switch within 20 steps;
    - collision;
    - other time-out.
  - Tests on synthetic trajectories: `tests/test_route_failure_taxonomy.py`.
  - It reuses ideas from the mapping session's untracked `scripts/entropy_hard_*.py`. Those are theirs; ask before copying.
- **Gate A (declared now).**
  - If chain completion and true-pose multi-turn success are low, and lane faults plus wrong branches are at least 50% of failures: the driver is the problem, so do B1.
  - If chains complete but the map arm mostly fails by passing subgoals or stalling: it is a switching problem. Hand it to the mapping session; B1 waits.
  - If wrong branches outnumber lane faults in the map arm: also do B2.

## B — A better driver by imitation (main line; needs decision 1)

- **B1. DAgger in chains.**
  - c0 drives chained goals (`GOAL_CHAIN_STYLE=past_junction`, 5–25 m, up to 200 m) in Town01/04/05/10HD.
  - The expert labels every frame, using `EXPERT_RECORD=1 EXPERT_DRIVER=policy`, `torch_rl/expert_record.py` and `slurm/il_dagger.sh`.
  - After a fault, the recovery gate (`torch_rl/recovery.py`) takes over so the chain continues.
  - New code: the recorder keeps one goal per episode (`il_record.env`: `GOAL_KEEP_ON_PASS=1`). Add chaining to it, with tests.
  - Episode IDs 900000+. `run_record` already refuses Town02/03.
- **B2. Map-subgoal recording** (only if Gate A calls for it).
  - The expert drives multi-turn routes in the training towns.
  - The goal input is the next node image of that town's map. It switches when the expert passes the node.
  - Needs A1-style maps for Town04 and Town10HD too.
  - New recorder hook, with tests.
- **B3. Clone c1.**
  - c0's recipe (`slurm/il_bc_chw.env`: MONO_CHW, speed, BC_AUG, 15k updates). Only the data changes.
  - Data: M1 plus B1 (plus B2), selected by an explicit shard list, not by `BC_MAX_SEED`.
  - One training run, validation inside the job, checkpoint chosen by the Task 7 rule.
- **Gate B (declared now).** c1 against c0 on training-town validation. c1 must meet all four:
  - multi-turn uniform-image success at least +0.10 pooled, McNemar p < 0.05;
  - standard validation true-goal success no more than 0.03 below c0's;
  - wrong-goal gap no more than 0.05 below c0's;
  - chain completion (ahead style) not below c0's.
- **Gate B outcome.**
  - Pass: c1 replaces c0. One more round may follow under the same rule, stopping at the first round that fails.
  - Fail: c0 stays, and the round is reported.

## C — The RL row (needs decision 2)

- **C1. Test the cause: one action barely changes the next frame.**
  - **Action repeat.** Add `ACTION_REPEAT=k` as a wrapper right after `GatedReward` in `torch_rl/carla_env_build.build_env`.
    - It sums rewards, combines fault and arrival flags across the k ticks, and stops early when the episode ends.
    - The DEBiT frame is encoded once per decision. `MAX_EPISODE_STEPS` counts decisions.
    - Tests with a fake env: `tests/test_action_repeat.py`.
    - Recorded as allowed (DrQ) next to `PLAIN_SAC_RULES`.
  - **Steering probe** `scripts/critic_steer_probe.py`, promoted from the job-tmp `diag_q_extremes.py`.
    - For each checkpoint: best steering on a 21-point grid vs the expert's, on recorded Town05 states.
    - Reports the correlation and the share of states where the best steering is full lock.
  - **c0 at k = 2 and k = 4** on standard validation, no training. Does the clone survive the slower rate?
  - **Three SAC-from-scratch runs**, k = 1, 2, 4.
    - Town05 + Town01, 100k decisions each.
    - P2's task overlay (`slurm/sac_scratch_2t_u5_orterm_idle_nb.env`) plus `plain_sac.env`.
    - Evals inside the job at 25k, 50k, 75k and 100k, with the probe at each checkpoint.
    - k = 1 is the control.
- **Gate C (declared now).**
  - A value of k is GO if, at 50k, its probe correlation is at least 0.5 with full lock in no more than 20% of states, and its validation success at 100k is not below its 50k value.
  - If k = 1 fails the same test, that confirms the cause.
- **C2 (GO).** Either continue that k to 1M, or fine-tune c0/c1 at that k if C1 showed the clone holds up there. One run. Select by validation, then send to the benchmark.
- **C3 (NO-GO).**
  - The RL row is the best validated SAC checkpoint so far: highest pooled validation true-goal success, ties broken by the gap.
  - The paper reports the critic analysis.
  - PPO comes back to you as an option, with its cost.

## D — The map layer (the mapping session's)

- They own placement, subgoal switching, the benchmark harness and the fresh sets.
- I hand over each final driver as real files with their provenance (task 35).
- I also send the Gate A failure breakdown. It shows them switching failures without anyone touching Town02/03.

## E — Final evaluation (once, when you say so)

Runs for the one final policy, with c0 for reference and the RL row:
- **Held-out pairs.** Town02 + Town03, 100 true + 100 wrong goals each, via `slurm/il_eval_pair.sh`. Move `~/paired_wrong.py` into `scripts/`.
- **Metres between failures** on the Town02/03 explore traces.
- **Chain eval.** 200 m, 100 per town.
- **Map benchmark.**
  - Easy/medium on the fresh 600, plus the new multi-turn set.
  - Arms: no map, uniform, entropy and SGM (all image-based), plus the true-pose oracle.

Each line of the spec bar is reported for the final policy, with Wilson intervals and paired tests with Holm correction.

## F — Harness fixes (mine)

- **F1. Orphaned CARLA servers.**
  - Bug: a crashed eval leaves its CARLA server running. Orphans on 4600/4650/4700 held 11.8 GB and caused an out-of-memory failure.
  - Fix: stop the server on exit in `torch_rl/carla_server.CarlaServer`, and make `slurm/eval_sidecar.sh` kill whatever still holds the eval's port after each eval.
  - Test: when the eval raises, the server is stopped.
- **F2.** Move `~/paired_wrong.py` into `scripts/` with a test on fixed sample data.

## Paper items (for you or the mapping session to edit)

- The Protocol paragraph says hard = 50–100 m, but the tables use the multi-turn set (50–150 m, at least 2 turns).
- The Results paragraph still quotes 0.42 → 0.53.
- Table 3's "Too fast" column predates the any-speed rule.
- The RL rows are empty.

## Order

1. **Now:**
   - commit this plan as `docs/superpowers/plans/2026-11-09-image-goal-lane-following.md`;
   - save your rule to memory: "no seeds - one policy that works" (single-run final policy; it overrides the spec's two-seed line);
   - Step 0 (0a and 0b, offline). Report Gate 0 with its tables;
   - alongside, with no training: F1, F2, A1 tours, and C1 code (the wrapper and the probe).
2. **After Gate 0:**
   - **Encoder is the bottleneck:** the encoder options come to you, and nothing trains until you choose.
   - **Otherwise:** A2–A4, and C1's evals and three runs, in parallel. Report Gates A and C with their tables.
3. **Next:** B1 → B3 (one run) → Gate B, then C2 or C3. The final policy is the best candidate on training-town validation.
4. **Last:** E when you say so, then paper updates with the mapping session.

## Risks

- **DAgger can weaken goal use** (the earlier clone's round, s1, did). Gate B's gap test catches it. Junction chains make the goal decide the branch.
- **Action repeat coarsens steering.** Crossings per episode show the cost. k = 2 is the middle ground.
- **Cluster faults.** Covered by F1, one eval per job, and atomic deploys.

## Verification

- **Code.** Every new test fails first, then passes. The full suite runs with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and keeps the known baseline (44 failed, 3 errors) with no new failures.
- **Runs.**
  - Alive means: the process exists, the log has a fresh line, and the GPU is in use.
  - Done means the `eval.json` files exist, not just a marker.
- **Eval-town data.** Before claiming no eval-town data, list the towns in every data root and tar used for training.
- **Gate records.** Each gate decision quotes n, episode sets and rules from the `eval.json` files on disk. It goes into `.superpowers/sdd/2026-11-04-imitation-first-image-goal/progress.md`.

# Gated Policy Design

Date: 2026-09-09
Branch: `add/baselines-policy-approach` (design work on `worktree-gated-policy-design`)
Status: **awaiting review — no implementation written**

## 1. Context, and a corrected premise

The task began as "take over the gated policy the other session trained." That
premise was wrong, and the correction shapes everything below.

**No gated policy has been trained in CARLA.** The peer session that was asked
for a handover never trained one. The trained gated policies live in Habitat,
under `/home/robotlab/scratch/robot-learning-tools`, and belong to a third
session. They were trained in the Cantwell scene only, and their Maricopa
evaluation is invalid because that scene has two storeys and routes wormhole
between floors.

What exists in carla-rl is two thirds of a design, written but never run:

| File | Lines | State |
| --- | --- | --- |
| `src/vae_nav/gated_reward.py` | 157 | Untracked, never run against CARLA |
| `src/vae_nav/mode_split_buffer.py` | 87 | Untracked, **untested** |

The evidence manifest at
`evidence/2026-09-09_entropy_at_junctions/MANIFEST.md` describes the replay
buffer as tested. That claim is false. The pytest cache holds 235 collected
tests across fourteen files and none of them touch either file. Both must be
treated as unverified sketches.

The missing third piece is the two-critic learner.

## 2. Why gate at all

The honest answer is **not** performance, and the design should not claim
otherwise.

The usual argument for a gated multi-task policy is sample sharing: both modes
learn a common skill, so training them together beats training them apart. That
argument is weak here. Our perception latent comes from a frozen autoencoder
outside the policy, so the only shared trainable parameters are small projection
and head layers. The peer ran the pre-check that would have supported the
argument and it came back negative: collision rates over the first 35k steps
were 0.0154 for the goal-conditioned arm and 0.0103 for the goal-free arm, both
already at the floor. There is no obstacle-avoidance skill left for a shared
actor to transfer.

Two reasons remain, and both are good:

**The three-way ablation.** Gated against goal-only against goal-free is a
comparison neither codebase has. The paper's own ablation pillar asks whether
the goal input earns its place. The two arms in this repo currently disagree in
their source comments about the answer: `gated_reward.py` zeroes the goal block
in explore mode and calls the dead inputs a cost worth paying, while
`explore_obs_wrapper.py` omits goal and mode entirely on the grounds that "a
constant input is not free, it is a channel the encoder must learn to ignore."
A gated arm is the experiment that settles that disagreement, because it is the
only configuration in which the same network sees both conditions.

**Runtime mode switching.** A single checkpoint that can be commanded to explore
or to drive to a waypoint is operationally simpler than two, and the map builder
needs both behaviours from the same agent.

**Assumption stated explicitly:** this design is built for the ablation. If the
goal is instead a performance win, the design does not change but the success
criteria in section 9 do, and the expected outcome is that gating loses.

## 3. Which two arms

There are three arms in the repo, not two, and picking the wrong pair makes the
work impossible rather than merely hard.

| Arm | Observation keys | Perception latent | Dim |
| --- | --- | --- | --- |
| `carla_lanefollow` | goal, latent, vector | 128-d autoencoder | 144 |
| `carla_lanefollow_nogoal` | latent, vector | 128-d autoencoder | 132 |
| `carla_lane_explorer` | explore, latent | 384-d MobileNet | — |

These two layouts are confirmed independently of the source: the evidence
manifest records the preserved goal-free checkpoint as 132-d and the preserved
goal-conditioned checkpoint as 144-d, matching the table above exactly.

**Gate the first two.** They already share the frozen 128-d autoencoder at
`checkpoints/lane_vae_120/best.pkl`, share the goal wrapper, and differ only by
the twelve-value goal block. Their action spaces are already identical:
`Box([-0.5, 0.0], [0.5, 0.6])`, reached through the same config function, so no
work is needed on the action side at all.

**Do not gate against the explorer arm.** Its perception backend differs, and
the backend is chosen by an environment variable inside the latent source rather
than by any wrapper. Unifying would force one backend on both, and whichever is
dropped makes that arm's existing checkpoints unloadable. The checkpoint loader
refuses mismatched pytrees loudly and by design, so this would not fail quietly,
but it would still cost every checkpoint on one side.

The explorer arm stays a separate baseline. That is the right outcome anyway,
because it varies three mechanisms at once and the ablation wants one variable.

## 4. Observation design

The gated observation is the goal arm's observation plus one scalar:

```
{ goal: (12,), latent: (128,), vector: (4,), mode: (1,) }
```

**The goal block is zeroed in explore mode, not omitted.** One actor cannot
accept two input shapes. This is the dead-input cost of gating, and it is
precisely the cost the goal-free arm exists to measure against.

**The mode bit gets its own key rather than being appended to `vector`.** The
network gives every non-image key its own dense projection followed by layer
normalisation before concatenation. A separate key therefore gives the mode bit
its own projection instead of forcing it through a layer norm shared with steer,
throttle and speed, where a single bit among four continuous values would be
partly normalised away. It also keeps the existing four-value vector layout
untouched, so goal-arm checkpoints stay loadable for warm starts.

**The mask is read from the observation, not carried alongside the batch.** The
replay buffer strips the mode from the transition dict on insert, so the learner
recovers it from `observations["mode"]`. This is exactly what the Habitat
implementation does, and it means no change to the batch schema.

## 5. The learner

Port, do not rewrite. The Habitat `GatedTQCLearner` is 228 lines, subclasses the
same jaxrl2 `TQCLearner` this repo vendors, expects the same attribute names
(`_actor`, `_critic`, `_target_critic_params`, `_temp`), builds on the same
`PixelMultiplexer` and `StateActionQuantileEnsemble`, and contains no Habitat
imports whatsoever. It is a genuine port, not an aspiration.

Three pieces move:

1. **`quantile_huber_loss_per_sample`**, about 19 lines, into
   `dependencies/jaxrl2/jaxrl2/agents/tqc/quantiles.py`. This repo's
   `quantile_huber_loss` reduces over the batch, which averages away the very
   per-sample structure the mask needs.
2. **`GatedTQCLearner`**, a new module beside `tqc_learner.py`. Second critic
   with fresh parameters, its own optimiser and its own target parameters;
   second temperature; masked critic losses; per-row selection in the actor.
3. **Checkpoint keys** for the second critic, its target, and the second
   temperature, plus a warm-start path that loads a plain checkpoint into the
   gated learner.

The vendored jaxrl2 is a nested git repository with its own history, so these
changes are version-controlled there rather than left untracked. That is a
deliberate choice to record, because it means the learner work does not appear
in this repo's log.

**Masked losses, never a per-sample discount.** Each critic computes its target
at its own scalar discount over the whole slice, then zeroes the off-mode rows
out of its loss. Off-mode rows contribute exactly zero gradient. This is worth
stating because the tempting alternative, a per-sample gamma vector inside one
shared target, destroys the goal critic's value semantics: with a single critic
the goal Q can no longer be read as a distance, and the map builder reads it as
exactly that.

**The actor selects per row** on both the Q value and the temperature, using the
just-updated critics. Both critics are evaluated on the whole batch and half of
each result is discarded. That waste is accepted in exchange for a shape the JIT
compiler can handle without dynamic control flow.

### A correction that matters

The Habitat rationale for two critics is a discount mismatch: the goal arm runs
minus one per step at discount 1.0 while the explore arm runs a coverage bonus
at 0.99. **That rationale does not transfer to CARLA.** Both lane arms here run
at 0.99. The goal arm's run script does not set the discount at all and inherits
the trainer default of 0.99; the goal-free script sets 0.99 explicitly. The
discount 1.0 case belongs to the time-to-goal reward, which is a different arm.

The case for two critics in CARLA rests instead on **return scale**. The goal
arm's shaped reward produces returns of order 100; the goal-free arm's produces
order 10. A single critic fitted to both is dominated by the larger, and the
Huber parameter is derived from the reward scale, so one value cannot serve
both. That is a sufficient reason, but it is a different reason, and the code
comments should say so rather than inheriting Habitat's.

## 6. Replay

`ModeSplitBuffer` swaps in at the single insert site in the trainer, routing on
the mode published by the reward wrapper.

The failure it prevents was measured, not hypothesised. In Habitat, explore
episodes ran long enough that roughly 90% of environment steps were exploration;
the goal critic then received between 10 and 37 percent of each batch and
goal-mode evaluation collapsed from a 40 to 65 percent band down to 15 percent.
The goal arm was starved, not broken.

**CARLA inverts the imbalance rather than removing it.** Here the goal arm's
episode cap is 2000 steps and the goal-free cap is 500, so goal rows will
dominate instead. The fixed split handles either direction, which is the point
of fixing composition on sample rather than on insert.

**The ratio is fixed per batch, not per sample.** A stochastic mixture would
give the same ratio in expectation but a variable one per update, and since each
critic trains on a masked subset, a batch that happened to hold three goal rows
would give the goal critic a three-sample gradient.

One detail to carry over: because the buffer concatenates goal rows first, a
unit-to-data ratio above one makes the last slice all-explore, and per-slice
statistics must be weighted by mask count or the goal critic appears to report
zero.

## 7. Environment side

`GatedReward` draws the mode once per episode from its own generator, or accepts
a forced mode from an evaluator. Once per episode and never mid-episode, because
each critic must see a whole return from a single mode.

Two things in the existing sketch need attention before it runs:

- It re-derives termination in explore mode with the junction exemption, because
  the environment's own termination fires on any lane crossing and would end an
  episode every time the car turned through a junction. That logic is shared
  with the wedge filter and is already pinned by tests. The gated path must use
  the same helper rather than a copy.
- Its explore branch reads a displacement from `info["hero_pos"]`. That coupling
  to the info dictionary is untested and is the most likely place for a silent
  zero.

Do not port the Habitat wrapper's geometry. Its coordinate convention is y-up
with the ground plane in x and z, and its measured forward vector is Habitat's.
CARLA is z-up and left-handed. The repo already contains a cautionary tale here:
the goal wrapper carried a mirrored lateral bearing for an unknown period
because a textbook right-handed rotation was applied to a left-handed simulator,
and a goal 20 metres to the right was reported as 20 metres to the left. Forward
was always right, so it presented as "drives but never arrives." Re-derive and
measure; do not assume.

## 8. Testing

The house style here is strong and should be followed rather than invented. The
repo has seventeen test files, and the explorer arm in particular is covered for
reward, observation wrapper, visit ring, novelty, dead reckoning, and a
config-coherence test whose only job is to pin environment-variable
combinations.

Tests to write **before** the corresponding implementation:

1. `tests/test_mode_split_buffer.py` — fixed 50/50 composition under a 90/10
   insert imbalance and under the inverted 10/90 imbalance CARLA will actually
   produce; correct behaviour when one side is empty; and the nested observation
   mapping is joined leaf-wise rather than collapsing to an object array. That
   last case is the subtle one: the flax mapping type implements the mapping
   protocol but does not subclass dict, so a naive dict check fails silently and
   produces an unindexable batch much later.
2. `tests/test_gated_reward.py` — mode is drawn once and held for the episode;
   a forced mode overrides the draw; each mode returns its own reward and its
   own termination rule; the junction exemption applies in explore mode only.
3. `tests/test_gated_config.py` — environment-variable coherence, matching the
   existing config-coherence test for the explorer. Reject a gated run that
   leaves a stale goal reward enabled, that mismatches the two episode caps
   without saying so, or that sets a perception backend inconsistent with the
   warm-start checkpoint.
4. `tests/test_gated_learner.py` — the load-bearing one. After a pure-goal
   batch, the explore critic's parameters are bit-identical; after a pure-explore
   batch, the goal critic's are. The Habitat implementation has exactly this
   pair of tests and they are the reason its masking can be trusted.

## 9. Success criteria

For the ablation reading, the run succeeds if it produces a clean three-way
comparison, **not** if the gated arm wins.

Report per arm, under identical evaluation:

- Goal-mode success and path efficiency, with the evaluator forcing goal mode.
- Explore-mode return, episode length, and the termination breakdown.

**Do not report the trainer's success metric for explore mode.** It measures
arrival at a goal the policy cannot see and is not paid for. The finished
goal-free run reports 0.000 success over 60 episodes for exactly this reason,
and every one of its evaluation videos is labelled a failure. The same caveat
applies to the gated arm's explore half.

The comparison is only meaningful if the arms share seeds, town split, and
evaluation trajectories.

## 10. Operational constraints

- One RTX 4070 with 12.3 GB, shared with another user's Habitat jobs. Our floor
  is about 5.1 GB, since the CARLA server takes 2.8 GB before JAX starts and JAX
  adds roughly 2.3 GB.
- Gate launches on about 6000 MiB free, polled. Six run scripts already
  implement this pattern; reuse it rather than writing a seventh.
- Take the CARLA port lock. Seventeen run scripts hold
  `/tmp/carla_rl_port2000.lock` for their lifetime through a non-blocking lock
  and refuse to start if it is held, so any two CARLA jobs on this machine are
  mutually exclusive. Three sessions share the one card and only one can hold it
  at a time, so a long run is something to schedule rather than launch.
- A free-memory check is necessary but not sufficient. JAX can still fail while
  instantiating a CUDA command buffer, which no byte count predicts. The
  goal-free run script already disables that allocation through an XLA flag and
  the gated script should inherit it.
- Never gate on process names. The CARLA binary's name is 23 characters and the
  kernel truncates the comm field to 15, so an exact-name process match can
  never succeed. Every cleanup path in the repo used to have this bug and
  reported success while servers kept running and held 2.8 GB. Use a full
  command-line match with a bracketed first character, and never place that
  pattern unbracketed in a long-running command, where a concurrent cleanup
  would match and kill the shell.
- The goal arm's usable checkpoint for a warm start is the 50k step one, not the
  latest. The peer reports that later checkpoints regressed badly at long range,
  because the curriculum start and target distances are pinned equal so the
  curriculum never advances. That specific measurement has not been independently
  reproduced here and should be re-checked before it is relied on; what is
  confirmed is that the 50k checkpoint is the one preserved as evidence for both
  arms. The goal-free arm's best is its final 350k checkpoint, though it
  plateaued early and the last quarter million steps bought little.

## 11. Risks

- **Reward hacking in explore mode.** If explore episodes truncate on a wedge, a
  free respawn becomes a reward channel.
- **A stale explore critic after a warm start** will drag the actor. The Habitat
  implementation damps the explore-mode actor gradient for a warm-up period and
  the port should keep that.
- **Changing which observation slots are masked costs roughly 150k steps of
  relearning.** Settle the layout before the first long run.
- **The two uncalibrated numbers in the explorer arm fail silently.** They do not
  affect this design directly, because we are not gating against the explorer,
  but they are a live reminder that a well-formed reward can be inert.

## 12. Out of scope, but found on the way

Three defects surfaced during this design and are recorded so they are not lost.
None should be fixed inside this work.

1. **The explorer arm's reward is speed-gated but its observation carries no
   speed channel.** The goal wrapper's own comments state that a reward paying
   only below target speed is unlearnable when the policy cannot observe its own
   speed. Both lane arms enable the speed channel; the explorer does not, and no
   comment anywhere justifies the omission.

   Two refinements, confirmed with the peer after this spec was first written.
   Setting the speed flag would **not** fix it, because the explorer path uses
   the explore observation wrapper rather than the goal wrapper, and that flag
   belongs to the goal wrapper. The fix has to add a speed channel to the 21-value
   explore block, which changes the observation shape and therefore invalidates
   the explorer's checkpoints. Separately, the dead reckoner is **not** affected:
   the core state wrapper does publish speed into the info dictionary, so the
   reckoner receives a real value. Only the observation lacks it. Anyone widening
   this bug report should not claim the reckoner is starved.

   **This is a decision, not a patch, and it belongs to the user rather than to
   any agent.** Adding a speed channel widens the 21-value explore block, which
   changes the observation shape, which makes the explorer's existing checkpoint
   unloadable. The loader refuses mismatched pytrees loudly and by design. So
   the real cost is retraining the explorer arm from scratch, not adding a slot.
   Nobody should apply this fix without that trade being put to the user first.
2. **`src/configs/goal_conditioned_env_config.py` is dead code.** It is 1831
   lines, modified in the working tree, and imported by nothing.
3. **Most of the working stack is untracked**, including the goal wrapper and
   the lane-follow reward, not only the two gate files. Committing the gate work
   requires a deliberate decision about what else comes with it.

## 13. Open question for review

Section 2 assumes the gate is being built for the three-way ablation. If the
intent is a performance win instead, the implementation is unchanged but the
success criteria in section 9 are wrong, and the expected result is that the
gated arm loses to both specialists.

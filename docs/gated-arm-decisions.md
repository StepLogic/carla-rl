# Gated arm: decisions, corrections and open questions

Running record for the gated goal/explore arm, 2026-09-09. Everything here is
either a decision that changes results, a correction to something this project
believed, or a question still open. Implementation detail lives in the commits.

## 1. What the arm is

One policy serving both the goal-reaching and the goal-free task, switched by a
bit drawn once per episode. It is the **third leg of a three-way ablation**
against `carla_lanefollow` (goal-only) and `carla_lanefollow_nogoal`
(goal-free). It is **not** expected to beat either specialist: perception is a
frozen autoencoder outside the policy, so the only shared trainable parameters
are small heads, and a pre-check found collision rates already at the floor in
both arms. Gating losing to both is a result, not a failure.

Observation, identical for the actor and both critics:

| Key | Width |
| --- | --- |
| goal | 12 |
| latent | 128 |
| vector | 4 |
| mode | 1 |

145 total, matching neither existing arm (144 and 132), so **a gated run is
from scratch, never a fine-tune**.

## 2. The goal reward is SORB minimum time

`-1` per step until arrival, `0` on arrival. The critic is then the cost-to-go,
`Q = -(steps to goal)`, which is what a topological map builder needs and no
shaped reward provides.

Three consequences, all load-bearing:

- **The goal critic must run undiscounted.** At 0.99 the horizon is 100 steps,
  so Q saturates near -100 and cannot separate a 400-step goal from a 1000-step
  one, measured 0.18 apart here. `gated_config` refuses the combination.
- **The goal critic needs value bounds, applied per critic.** Undiscounted, it
  has no bounded fixed point on a state the agent never escapes. The bounds must
  NOT come from the global `TQC_TARGET_MIN/MAX`, which are baked into every
  trace at import and would impose the goal arm's window on the explore critic.
- **Collisions must not end a goal episode.** With a per-step cost, a terminal
  failure stops the meter and crashing immediately becomes optimal. A wedge does
  end the episode but is billed exactly for the time it would have burned, so
  ending early is worth precisely nothing.

## 3. Lane invasion is a no-op

Crossing a marking costs nothing and ends nothing.

- Goal mode already ignored lane state; this was verified through the real
  wrapper stack, with the lane-follow reward underneath signalling terminal, not
  merely read off the reward function.
- Explore mode changed: a crossing previously cost 5.0 and terminated.
- **Off-road is deliberately still fatal.** The explore progress term pays the
  same for circling as for driving down the road, so off-road is the only
  remaining reason to stay on it.

## 4. Corrections to things this project believed

Each of these was stated confidently somewhere before being measured.

**The Habitat two-critic rationale does / does not transfer — it depends on the
reward.** Under the shaped reward both arms run at 0.99 and only return scale
separates them. Under SORB the discounts genuinely differ and it does transfer.
Asserting either unconditionally is wrong.

**The explore critic's targets are not positive.** The justification for leaving
it unclipped originally rested on that. Measured, they are negative early
(around -63 over ~180 steps), dominated by the steer-change penalty. The
decision stands; the reason is provenance, not sign.

**CARLA's memory footprint is not ~2.8 GB.** Measured across runs: 2611, 2775,
2808 and 5297 MiB, with 9221 MiB for CARLA and JAX together. The repo-wide
6000 MiB gate derives from the low end of that range recorded as typical, and it
passes on a card that cannot hold the run. Use 9500.

**Two out-of-memory failures look alike and need different fixes.** The phantom
CUDA command-buffer allocation is cured by
`XLA_FLAGS=--xla_gpu_enable_command_buffer=`. A genuine shortfall is not, and
needs a higher gate.

## 5. Things that were silently unreadable

None of these were bugs in behaviour; all of them would have wasted time.

- The observation banner listed three keys for a four-key gated run.
- Episode statistics averaged goal and explore together, whose returns differ in
  sign and by two orders of magnitude.
- Actor entropy printed `nan`, which reads as a diverged policy, because the
  masked critic update omitted the key the trainer prints.
- The explore critic appeared in no log line at all, so a starved second critic
  would have left no trace.
- The lane-invasion setting had no banner, so confirming it meant reading
  `/proc/<pid>/environ`.

## 6. Open questions

1. **Should off-road also be free in explore mode?** Currently fatal. Making it
   free removes the last incentive for that arm to stay on the road.
2. **The goal-free baseline was trained with the lane penalty.** Its explore half
   and the gated one now differ by more than the gate, which is the one thing
   the comparison is supposed to isolate. Either retrain that baseline with
   `LANE_INVASION_NOOP=1`, or state in the write-up that the explore arms are
   not directly comparable.
3. **GPU scheduling.** At ~9.2 GB for a CARLA run and ~4.8 GB for a Habitat
   stage on one 12.3 GB card, the two cannot run together. This is a queue.

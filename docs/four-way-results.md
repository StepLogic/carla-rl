# Four-way ablation results

Twenty evaluations, 60 episodes each, every arm driving the **same** episode
set from a fixed seed. Raw per-arm JSON lives in `results/four_way`, which is
gitignored; regenerate with `./run_four_way_eval.sh`.

The two `goal_only_*` rows are the SAME frozen checkpoint as `goal_only`, run
under a relaxed termination rule. They exist because the extrapolation numbers
below were confounded, and they are what resolves it. See "The termination
confound, resolved".

```
band 5-20m  (in-distribution)
arm                     success  steps(med)  n_arr   ep_len  timeout     q_bias  q_corr
goal_only                 0.983         409     59    408.7    0.000        n/a     n/a
goal_only_lanenoop        0.983         406     59    400.3    0.000        n/a     n/a
goal_only_freerun         1.000         407     60    407.2    0.000        n/a     n/a
goal_free                   n/a         n/a      0     58.4    0.000        n/a     n/a
gated                     1.000          50     60     47.2    0.000       -150    0.48
gated_her                 1.000          48     60     45.1    0.000        -38    0.98

band 20-40m  (EXTRAPOLATION -- training is pinned to 5-20 m)
arm                     success  steps(med)  n_arr   ep_len  timeout     q_bias  q_corr
goal_only                 0.317         432     19    287.6    0.000        n/a     n/a
goal_only_lanenoop        0.567         450     34    361.1    0.000        n/a     n/a
goal_only_freerun         0.850         449     51    538.8    0.150        n/a     n/a
goal_free                   n/a         n/a      0     91.7    0.000        n/a     n/a
gated                     0.900          78     54    173.8    0.100       -172    0.35
gated_her                 0.967          82     58    111.2    0.033        -84    0.61

band 40-70m  (EXTRAPOLATION -- training is pinned to 5-20 m)
arm                     success  steps(med)  n_arr   ep_len  timeout     q_bias  q_corr
goal_only                 0.283         433     17    249.5    0.000        n/a     n/a
goal_only_lanenoop        0.250         436     15    265.6    0.000        n/a     n/a
goal_only_freerun         0.517         455     31    738.6    0.483        n/a     n/a
goal_free                   n/a         n/a      0    146.6    0.000        n/a     n/a
gated                     0.817         133     49    292.1    0.183       -119    0.40
gated_her                 0.800         127     48    314.4    0.200       -109    0.82

EXPLORE half (gated arms only; no goal, so no bands and no success)
arm                                            ep_len   how episodes ended
gated                                           339.5   off_road 32, trunc 27, collision 1
gated_her                                       299.5   off_road 39, trunc 14, collision 7
```

The termination rules, since three rows differ only by them:

| arm | what ends an episode |
| --- | --- |
| `goal_only` | arrival, collision, idle, off-road, **or crossing a lane marking** |
| `goal_only_lanenoop` | arrival, collision, idle, off-road. A crossing is free. |
| `goal_only_freerun` | **arrival only**, or the 1000-step cap. The gated arms' own rule. |
| `gated`, `gated_her` | arrival only, or the cap (SORB discards the env's `done`) |

## What the data supports

**Minimum time transforms speed, and this is the finding that survives
everything.** 48-50 steps to goal against the shaped baseline's 409 in
distribution, 78-82 against 449 at 20-40 m, 127-133 against 455 at 40-70 m.
Identical episodes, and a 5.5x margin that no termination rule can produce --
the deconfounded baseline arrives at a comparable rate over the medium band and
still takes five times as long to do it.

**Hindsight relabelling does NOT change the policy and DOES fix the critic.**
Success and time-to-goal are indistinguishable between the two gated arms at
every band. Correlation between the critic's value and true time-to-goal:

| band | gated | gated + hindsight |
| --- | --- | --- |
| 5-20 m | 0.48 | **0.98** |
| 20-40 m | 0.35 | **0.61** |
| 40-70 m | 0.40 | **0.82** |

Better at every band, decisively in-distribution. This is the result for the
map-distance claim: relabelling is what makes the value function readable as a
metric. **A correlation of 0.98 means a single linear rescale would make the
critic accurate**; at 0.48 no rescale helps, because the ordering itself is
unreliable. The plain gated arm never exceeds 0.48 anywhere.

**A long-range advantage survives the deconfound.** At 40-70 m the gated arms
reach 0.817/0.800 against the identically-terminated baseline's 0.517. That gap
is 18 episodes and is not a rule artefact -- but see below for what it IS.

## The termination confound, resolved

The previous version of this document recorded "the gated arms extrapolate; the
baseline does not" (0.90/0.82 against 0.32/0.28) and flagged it as confounded: a
lane crossing ended the baseline's episode and was a no-op for the gated arms,
which killed 35 and 43 of its 60 episodes while the gated arms only ever timed
out. Re-running the same frozen checkpoint under the gated arms' own rule
settles how much of the gap was which.

| band | as recorded | crossing free | gated's rule | gated | share of the gap that was the RULE |
| --- | --- | --- | --- | --- | --- |
| 20-40 m | 0.317 | 0.567 | 0.850 | 0.900 | **91%** |
| 40-70 m | 0.283 | 0.250 | 0.517 | 0.817 | **44%** |

**At 20-40 m the claim does not survive.** Nearly all of that gap was the
termination rule. 0.850 against 0.900 is three episodes at n=60, well inside
noise. The baseline was being stopped, not out-navigated.

**At 40-70 m roughly half the gap is real** -- and the honest reading is that
this remainder is the SPEED result wearing a different hat, not an independent
one. The baseline needs a median 455 steps to arrive; the cap is 1000; so at
long range **48.3% of its episodes hit the cap**, against 18-20% for the gated
arms. Its mean episode length is 738.6 steps. The long-range success gap is what
a 5.5x slower policy looks like once a finite step budget starts binding. Under
an unbounded budget much of it would likely close.

So the defensible statement is one claim, not two: **minimum-time training makes
the policy roughly five times faster to goal, and at long range that speed is
what converts into success within a fixed budget.** "The gated arms generalise
beyond their training range and the baseline does not" overstates it.

## What the data does NOT support

**Hindsight's explore half is worse**: 39 off-road terminations against 32,
seven collisions against one, shorter episodes. It buys a usable critic and
costs road-keeping.

**Success saturates in-distribution.** All goal-conditioned arms sit at or near
1.000 at 5-20 m, so success alone separates nothing there; the discrimination
comes from time-to-goal and from the extrapolation bands.

**The deconfounded arms are out of distribution after a failure.** `lanenoop`
and `freerun` run a policy that was TRAINED with those events terminal, so its
behaviour in the steps after a crossing is not behaviour it was ever optimised
for. This is a fact about the baseline rather than a flaw in the measurement --
the gated arms were trained under their rule and evaluated under it -- but it
means the deconfounded rows are a LOWER bound on a baseline retrained under the
same rule. The 40-70 m `lanenoop` row (0.250, below the original 0.283) is
consistent with that: two episodes, but in the direction the caveat predicts.

**One seed, one town.** Trained and evaluated in Town02 throughout. No error
bars. Treat four-episode margins as noise and the correlation gaps as real.

## Reading notes

- `success` is refused for `goal_free` by construction: it measures arrival at a
  goal that policy cannot see and is not paid for. Reported as n/a rather than
  0.000, which would read as "worse" instead of "not applicable".
- `q_bias`/`q_corr` are refused for arms not trained on minimum time; the shaped
  arm's critic predicts a shaped return, so scoring it against steps measures
  nothing. That is why the three `goal_only_*` rows carry no calibration.
- Calibration is measured on **arriving episodes only**: a timeout has no true
  cost-to-go, so including it would score the critic against a censored label.

## Next

1. Node-placement comparison (uniform / random / entropy peaks / SGM) by SR and
   SPL, on the gated arms. Harness: `./run_placement_eval.sh`.
2. A second town. `TOWN=` now selects it for both drivers. NOTE that
   `configs/town_split.json` designates Town03 and Town10HD as the frozen
   held-out set; Town01 is a train-split town this policy merely never saw.
3. More seeds before any of this goes in a paper. Every margin above is n=60,
   one seed, one town.

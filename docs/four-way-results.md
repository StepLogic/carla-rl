# Four-way ablation results

Fourteen evaluations, 60 episodes each, every arm driving the **same** episode
set from a fixed seed. Raw per-arm JSON lives in `results/four_way`, which is
gitignored; regenerate with `./run_four_way_eval.sh`.

```
band 5-20m  (in-distribution)
arm             success  steps(med)  n_arr   ep_len  timeout     q_bias  q_corr
goal_only         0.983         409     59    408.7    0.000        n/a     n/a
goal_free           n/a         n/a      0     58.4    0.000        n/a     n/a
gated             1.000          50     60     47.2    0.000       -150    0.48
gated_her         1.000          48     60     45.1    0.000        -38    0.98

band 20-40m  (EXTRAPOLATION -- training is pinned to 5-20 m)
arm             success  steps(med)  n_arr   ep_len  timeout     q_bias  q_corr
goal_only         0.317         432     19    287.6    0.000        n/a     n/a
goal_free           n/a         n/a      0     91.7    0.000        n/a     n/a
gated             0.900          78     54    173.8    0.100       -172    0.35
gated_her         0.967          82     58    111.2    0.033        -84    0.61

band 40-70m  (EXTRAPOLATION -- training is pinned to 5-20 m)
arm             success  steps(med)  n_arr   ep_len  timeout     q_bias  q_corr
goal_only         0.283         433     17    249.5    0.000        n/a     n/a
goal_free           n/a         n/a      0    146.6    0.000        n/a     n/a
gated             0.817         133     49    292.1    0.183       -119    0.40
gated_her         0.800         127     48    314.4    0.200       -109    0.82

EXPLORE half (gated arms only; no goal, so no bands and no success)
arm                                            ep_len   how episodes ended
gated                                           339.5   off_road 32, trunc 27, collision 1
gated_her                                       299.5   off_road 39, trunc 14, collision 7
```

## What the data supports

**Minimum time transforms speed.** 48-50 steps to goal against the shaped
baseline's 409, on identical episodes, with equal or better success.

**Hindsight does not change the policy and transforms the critic.** Success and
time-to-goal are indistinguishable between the two gated arms at every band.
Correlation between the critic's value and true time-to-goal:

| band | gated | gated + hindsight |
| --- | --- | --- |
| 5-20 m | 0.48 | **0.98** |
| 20-40 m | 0.35 | **0.61** |
| 40-70 m | 0.40 | **0.82** |

Better at every band, decisively in-distribution. This is the result for the
map-distance claim: relabelling is what makes the value function readable as a
metric. The plain gated arm never exceeds 0.48 anywhere.

**A correlation of 0.98 means a single linear rescale would make the critic
accurate**; at 0.48 no rescale helps, because the ordering itself is unreliable.
That is the difference between a usable metric with a calibration constant and
one that is not usable at all.

**The gated arms extrapolate; the baseline does not.** 0.90 and 0.82 against
0.32 and 0.28 beyond the training range.

## What the data does NOT support

**The extrapolation gap is confounded.** Lane invasion terminates episodes for
the baseline and is a no-op for the gated arms, so part of that difference is a
termination rule rather than the gate. The termination breakdown shows it
plainly: the baseline fails almost entirely by lane invasion (35 and 43 of 60),
while the gated arms only ever time out. **Removing the confound costs one flag
and one re-run of the baseline, with no retraining.**

**Hindsight's explore half is worse**: 39 off-road terminations against 32,
seven collisions against one, shorter episodes. It buys a usable critic and
costs road-keeping.

**Success saturates in-distribution.** All goal-conditioned arms sit at or near
1.000 at 5-20 m, so success alone separates nothing there; the discrimination
comes from time-to-goal and from the extrapolation bands.

**One seed, one town.** Trained and evaluated in Town02 throughout. No error
bars. Treat the four-episode margins (0.967 vs 0.900) as noise and the
correlation gaps as real.

## Reading notes

- `success` is refused for `goal_free` by construction: it measures arrival at a
  goal that policy cannot see and is not paid for. Reported as n/a rather than
  0.000, which would read as "worse" instead of "not applicable".
- `q_bias`/`q_corr` are refused for arms not trained on minimum time; the shaped
  arm's critic predicts a shaped return, so scoring it against steps measures
  nothing.
- Calibration is measured on **arriving episodes only**: a timeout has no true
  cost-to-go, so including it would score the critic against a censored label.

## Next

1. Re-run `goal_only` with lane invasion non-terminal, to remove the confound.
2. More seeds and a held-out town before any of this goes in a paper.
3. Node-placement comparison (entropy / uniform / random / SGM) by SR and SPL.

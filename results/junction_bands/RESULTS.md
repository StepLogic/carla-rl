# Junction-turn bands: results (analysis as pre-specified)

c0 il_bc_c0 step_15000, mapless, local CARLA 0.9.15, the map benchmark's harness. Success = within 1 m of the goal at any speed (headline); slow arrival = success with entry into the 1 m circle at <= 10 km/h (secondary). Checks passed: every wrong-goal cell ran the same seeds in the same order as its true-goal partner, and the seeds are the episode file's.

## Every cell

| Town | Band | Goal | n | Successes | SR | SPL | Slow arrivals | Median entry km/h | Median route m | Contains a left | Ends |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Town02 | 1 turn | true | 100 | 21 | 0.21 | 0.21 | 9 | 10.5 | 92 | 57/100 | off_road 40, reached 21, clock 18, crossing 18, collision 3 |
| Town02 | 1 turn | wrong | 100 | 21 | 0.21 | 0.21 | 0 | 17.2 | 92 | 57/100 | off_road 42, reached 21, clock 19, crossing 11, collision 7 |
| Town02 | 2 turns | true | 100 | 6 | 0.06 | 0.05 | 2 | 11.9 | 118 | 68/100 | off_road 51, clock 21, crossing 18, reached 6, collision 4 |
| Town02 | 2 turns | wrong | 100 | 6 | 0.06 | 0.05 | 0 | 17.9 | 118 | 68/100 | off_road 60, clock 19, crossing 9, reached 6, collision 6 |
| Town02 | 3+ turns | true | 100 | 11 | 0.11 | 0.10 | 4 | 11.6 | 133 | 80/100 | clock 34, off_road 32, crossing 19, reached 11, collision 4 |
| Town02 | 3+ turns | wrong | 100 | 5 | 0.05 | 0.04 | 0 | 19.6 | 133 | 80/100 | off_road 33, clock 33, crossing 22, collision 7, reached 5 |
| Town07 | 1 turn | true | 100 | 18 | 0.18 | 0.17 | 10 | 9.6 | 97 | 46/100 | crossing 37, off_road 26, reached 18, clock 11, collision 8 |
| Town07 | 1 turn | wrong | 100 | 10 | 0.10 | 0.10 | 0 | 14.9 | 97 | 46/100 | crossing 47, off_road 26, clock 13, reached 10, collision 4 |
| Town07 | 2 turns | true | 100 | 3 | 0.03 | 0.03 | 2 | 9.6 | 122 | 73/100 | off_road 45, crossing 39, collision 11, reached 3, clock 2 |
| Town07 | 2 turns | wrong | 100 | 0 | 0.00 | 0.00 | 0 | - | 122 | 73/100 | crossing 46, off_road 38, collision 11, clock 5 |
| Town07 | 3+ turns | true | 100 | 7 | 0.07 | 0.07 | 3 | 11.4 | 126 | 68/100 | off_road 38, crossing 37, collision 13, reached 7, clock 5 |
| Town07 | 3+ turns | wrong | 100 | 1 | 0.01 | 0.01 | 0 | 18.8 | 126 | 68/100 | off_road 41, crossing 40, collision 16, clock 2, reached 1 |

## Test 1, headline: goal use at any speed (true vs wrong, McNemar)

| Town | Band | True only | Wrong only | p | p (Holm, 6) |
|---|---|---|---|---|---|
| Town02 | 1 turn | 14 | 14 | 1 | 1 |
| Town02 | 2 turns | 4 | 4 | 1 | 1 |
| Town02 | 3+ turns | 9 | 3 | 0.146 | 0.584 |
| Town07 | 1 turn | 14 | 6 | 0.115 | 0.577 |
| Town07 | 2 turns | 3 | 0 | 0.25 | 0.75 |
| Town07 | 3+ turns | 7 | 1 | 0.0703 | 0.422 |

## Test 1, secondary: goal use with slow arrival (true vs wrong, McNemar)

| Town | Band | True only | Wrong only | p | p (Holm, 6) |
|---|---|---|---|---|---|
| Town02 | 1 turn | 9 | 0 | 0.00391 | 0.0195 |
| Town02 | 2 turns | 2 | 0 | 0.5 | 1 |
| Town02 | 3+ turns | 4 | 0 | 0.125 | 0.5 |
| Town07 | 1 turn | 10 | 0 | 0.00195 | 0.0117 |
| Town07 | 2 turns | 2 | 0 | 0.5 | 1 |
| Town07 | 3+ turns | 3 | 0 | 0.25 | 0.75 |

## Test 2, H-left: true-goal success with vs without a left turn (Fisher)

1-turn band: left vs right. Hypothesis: lower with a left turn.

| Town | Band | With a left: success / n | Without: success / n | p | p (Holm, 6) |
|---|---|---|---|---|---|
| Town02 | 1 turn | 6/57 (0.11) | 15/43 (0.35) | 0.00551 | 0.022 |
| Town02 | 2 turns | 2/68 (0.03) | 4/32 (0.12) | 0.081 | 0.162 |
| Town02 | 3+ turns | 5/80 (0.06) | 6/20 (0.30) | 0.00752 | 0.0226 |
| Town07 | 1 turn | 2/46 (0.04) | 16/54 (0.30) | 0.00125 | 0.0075 |
| Town07 | 2 turns | 2/73 (0.03) | 1/27 (0.04) | 1 | 1 |
| Town07 | 3+ turns | 1/68 (0.01) | 6/32 (0.19) | 0.00406 | 0.0203 |

## Descriptive only (no test)

No-left stratum across bands, true goal (n printed; too few for a test):

- Town02: 1 turn 15/43, 2 turns 4/32, 3+ turns 6/20
- Town07: 1 turn 16/54, 2 turns 1/27, 3+ turns 6/32

True-goal success by turn-direction sequence (successes / routes):

- Town02 1 turn: L 6/57, R 15/43
- Town02 2 turns: LL 1/21, LR 1/26, RL 0/21, RR 4/32
- Town02 3+ turns: LLL 0/12, LLR 0/10, LRL 2/10, LRR 0/2, RLL 1/10, RLR 2/18, RRL 0/18, RRR 6/20
- Town07 1 turn: L 2/46, R 16/54
- Town07 2 turns: LL 1/25, LR 0/28, RL 1/20, RR 1/27
- Town07 3+ turns: LLL 0/5, LLR 0/7, LRL 1/6, LRR 0/20, LRRR 0/1, RLL 0/6, RLR 0/9, RLRR 0/2, RRL 0/11, RRR 4/28, RRRL 0/1, RRRR 2/4

Pilot (Town07, 20 true-goal episodes per band, seeds after the evaluated 100; harness check only): see logs/eval_jbpilot_*.log.


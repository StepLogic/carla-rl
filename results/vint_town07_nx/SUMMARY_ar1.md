| Band | Arm | n | SR | SPL | SR@2m | Ends | Median entry km/h (successes) | vs mapless: b / c, p |
|---|---|---|---|---|---|---|---|---|
| easy | mapless | 100 | 0.50 | 0.48 | 0.55 | reached 50, crossing 31, off_road 17, collision 2 | 5.8 | - |
| easy | uniform | 100 | 0.49 | 0.47 | 0.55 | reached 49, crossing 30, off_road 16, collision 5 | 13.7 | 12 / 13, p = 1 |
| easy | vinttopo | 100 | 0.35 | 0.33 | 0.38 | reached 35, crossing 34, off_road 24, collision 7 | 12.0 | 9 / 24, p = 0.0135 |
<!-- easy: episodes file bench_t07_fresh_episodes.json -->
| medium (>= 1 turn) | mapless | 100 | 0.19 | 0.19 | 0.19 | crossing 50, off_road 21, reached 19, collision 10 | 5.8 | - |
| medium (>= 1 turn) | uniform | 100 | 0.20 | 0.19 | 0.23 | crossing 50, off_road 27, reached 20, collision 3 | 10.9 | 14 / 13, p = 1 |
| medium (>= 1 turn) | vinttopo | 100 | 0.18 | 0.17 | 0.21 | crossing 55, off_road 22, reached 18, collision 5 | 13.8 | 16 / 17, p = 1 |
<!-- medium: episodes file bench_t07_fresh_episodes.json -->
| hard (>= 2 turns) | mapless | 100 | 0.01 | 0.01 | 0.01 | crossing 45, off_road 45, collision 9, reached 1 | 5.8 | - |
| hard (>= 2 turns) | uniform | 100 | 0.05 | 0.05 | 0.05 | crossing 42, off_road 41, collision 12, reached 5 | 5.8 | 5 / 1, p = 0.219 |
| hard (>= 2 turns) | vinttopo | 100 | 0.00 | 0.00 | 0.01 | crossing 52, off_road 35, collision 13 | - | 0 / 1, p = 1 |
<!-- turns: episodes file bench_t07_turns_episodes.json -->

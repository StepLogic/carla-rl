| Band | Arm | n | SR | SPL | SR@2m | Ends | Median entry km/h (successes) | vs mapless: b / c, p |
|---|---|---|---|---|---|---|---|---|
| easy | mapless | 100 | 0.52 | 0.49 | 0.57 | reached 52, crossing 30, off_road 16, collision 2 | 5.8 | - |
| easy | uniform | 100 | 0.52 | 0.50 | 0.55 | reached 52, crossing 28, off_road 18, collision 2 | 12.8 | 13 / 13, p = 1 |
| easy | vinttopo | 100 | 0.35 | 0.34 | 0.39 | reached 35, crossing 31, off_road 25, collision 9 | 11.3 | 5 / 22, p = 0.00151 |
<!-- easy: episodes file bench_t07_fresh_episodes.json -->
| medium (>= 1 turn) | mapless | 100 | 0.15 | 0.15 | 0.16 | crossing 49, off_road 28, reached 15, collision 8 | 5.8 | - |
| medium (>= 1 turn) | uniform | 100 | 0.33 | 0.32 | 0.35 | crossing 39, reached 33, off_road 24, collision 4 | 10.9 | 25 / 7, p = 0.0021 |
| medium (>= 1 turn) | vinttopo | 100 | 0.18 | 0.18 | 0.23 | crossing 46, off_road 32, reached 18, collision 4 | 14.1 | 16 / 13, p = 0.711 |
<!-- medium: episodes file bench_t07_fresh_episodes.json -->
| hard (>= 2 turns) | mapless | 100 | 0.01 | 0.01 | 0.01 | off_road 50, crossing 41, collision 8, reached 1 | 5.8 | - |
| hard (>= 2 turns) | uniform | 100 | 0.03 | 0.03 | 0.03 | crossing 43, off_road 42, collision 12, reached 3 | 5.9 | 3 / 1, p = 0.625 |
| hard (>= 2 turns) | vinttopo | 100 | 0.04 | 0.04 | 0.04 | off_road 40, crossing 40, collision 16, reached 4 | 12.1 | 4 / 1, p = 0.375 |
<!-- turns: episodes file bench_t07_turns_episodes.json -->

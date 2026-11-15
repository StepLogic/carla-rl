# ViNT Town07 rerun with the aligned navigator (label `vintd3_nx_ar1`), 2026-11-15

Run 14:11-15:27 on the local workstation, CARLA 0.9.15, by `run_vint_t07_nx.sh` with
`VINT_AR=1`. Comparable with the map benchmark's c0 rerun of the same day (label `na25`).

- Policy: final ViNT `dagger3_e1`, run once, at its native rate (`ACTION_REPEAT=1`, user
  decision 2026-11-15). The c0 block runs at AR=2, so the two blocks differ in rate: state the
  rate per block, and never compare absolute numbers between them.
- Code: `code_27b7fb6` (branch vint-rerun-2026-11-15 = ViNT + worktree-sac-image-goal @7a4ee1d).
  `image_nav.py`, `carla_benchmark.py`, `train_env_config_v4_image_goal.py` and `image_map.py`
  are byte-identical to the c0 rerun's tree.
- Rules: any speed (`il_eval_chw_anyspeed.env`); `LANE_UNKNOWN_MARKS_FATAL` unset (default 1);
  `EVAL_MAP_NEXT_ADVANCE=1` (logged per uniform cell: "next-node advance ON"); leg timeout 150
  and leg steps 100 (150 ticks); fallback mapless, reset_hidden 1, replan 3, GOAL_CAM_MATCH=1.
  The flag is inert for the mapless and ViNT-topomap arms: only the uniform arm tests it.
- Episodes: `bench_t07_fresh_episodes.json` (easy; medium = >= 1 turn) and
  `bench_t07_turns_episodes.json` (hard). Maps: `t07/maps_t07_z25` for all bands. ViNT topomap:
  the 2026-11-12 build from drive 0 of the expert tour (an input, not eval code).
- Results: `SUMMARY_ar1.md` (per cell, with paired McNemar against ViNT's own no-map arm).
  Route-by-route comparison with the 2026-11-12 cells: `OLD_VS_NEW.txt` (descriptive only).

Reading:
- Within the block, the uniform map does not change ViNT's success on any band: easy .49 vs .50,
  medium .20 vs .19, hard .05 vs .01, all n.s. ViNT's own topomap is worse than no map on easy
  (.35 vs .50, 9 / 24 discordant, p .0135; Holm over 2, p .027).
- The 2026-11-12 medium gain of the uniform map (.15 -> .33, p .002) does not survive: 19 of
  its 33 successes are lost and 6 gained, more than the run-to-run churn of the arms the flag
  cannot touch (mapless 5 lost / 9 gained, topomap 12 / 12). Those lost successes had entered
  at a median 11.4 km/h, the drive-through signature.
- Per-route outcomes churn a lot between runs even with identical settings (the topomap arm on
  medium keeps only 6 of its 18 successes), so comparisons belong within one run's block.

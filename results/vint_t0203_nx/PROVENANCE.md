# ViNT Town02 + Town03 with the aligned navigator (label `vintd3_nx_ar1`), queued 2026-11-15

The cluster half of `results/vint_town07_nx`, so ViNT's aligned-navigator block pools over the
same three towns as the c0 block. Requested by the mapping session after the user's "run 1" there
(c0 Town02/Town03 next-advance cells, jobs 466595-466606 Town02 and 466607-466618 Town03).

- **Job:** 466619 (`vint_nx_t0203`), ONE job for all 18 cells (2 towns x 3 bands x 3 arms), serial,
  `--dependency=afterany:466606:466618`, so it starts only after both c0 chains end (one GPU at a
  time from this session). `-p gpu --gres=gpu:1 -c 4 --mem=48G -t 14:00:00
  --exclude=node-g104,node-g106`. Log: cluster `~/carla-rl-vint-nx/logs/vint_nx_t0203_466619.log`,
  one log per cell in the same folder.
- **Code:** cluster `~/carla-rl-vint-nx` = this branch's snapshot `results/vint_town07_nx/code_27b7fb6`
  (the exact Town07 code), deployed by rsync to a staging folder and moved into place, refusing to
  overwrite. Only change: `dependencies/rlib_integration` points to the cluster's copy. Checked on the
  cluster: 14/14 navigator tests pass (plain-Python runner, no pytest in the env), ViNT and navigator
  modules import from the tree, `ImageNavigator(next_advance=...)` present. `image_nav.py`,
  `carla_benchmark.py`, `train_env_config_v4_image_goal.py` are md5-identical to the c0 cluster tree
  `~/carla-rl-na`; `image_map.py` differs only by the frame-cache fix (output bit-identical).
- **Runner:** `vint_nx_cells.sh` (this folder; deployed as `slurm/vint_nx_cells.sh`), the cluster
  version of `run_vint_t07_nx.sh`. Same settings: any speed (`il_eval_chw_anyspeed.env`),
  `LANE_UNKNOWN_MARKS_FATAL` unset, `EVAL_MAP_NEXT_ADVANCE=1`, `VINT_AR=1` -> leg timeout 150 / leg
  steps 100 (150 ticks), GOAL_CAM_MATCH=1, fallback mapless, reset_hidden 1, replan 3,
  OFF_ROAD_JUNCTION_M=3.5, image localisation for the uniform arm.
- **Inputs (cluster):** policy `~/carla-rl/checkpoints/vint_final/dagger3_e1`; maps
  `maps_expert_jx_img_z25` for ALL bands (the z25 build, also for easy); ViNT topomaps
  `vint_topomap/dagger3_e1` (the ones every earlier ViNT Town02/Town03 cell used); episodes
  `bench_fresh_episodes.json` (easy), `bench_<Town>_medturn_episodes.json` (medium, >= 1 turn),
  `bench_turns_episodes.json` (hard) -- the same files as the c0 `na25` cells.
- **Output:** `~/carla-rl-vint-nx/checkpoints/bench_vintd3_nx_ar1_{mapless,uniform,vinttopo}_Town0{2,3}_{easy,medium,turns}`.
  Ports 3110-3280 (this session's range 3100-3399).
- **Not covered:** the SAC block's Town02/Town03 cells (24) are not queued by anyone as of 18:05.

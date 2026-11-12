# Same seeds, two CARLA builds: Town02 episode geometry (2026-11-12)

Question: does a benchmark seed give the same episode on the cluster's CARLA 0.9.16 and on
the local CARLA 0.9.15? It matters because Town07 is not in the cluster image, so its rows
are probed and evaluated locally while the Town02 and Town03 rows come from the cluster.

- Cluster: job 465250, the 1000-seed turns probe of Town02 (seeds 41000-41999), CARLA 0.9.16.
- Local: the first 20 of those seeds re-probed on CARLA 0.9.15 with the identical settings:
  `GOAL_CAM_MATCH=1 EVAL_PROBE=1 EVAL_PROBE_TURNS=1 BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env`,
  `REWARD_OVERLAY=slurm/bench_turns.env`, `ENV_TOWNS=Town02`, `EVAL_SEED=41000`, `EVAL_EPISODES=20`
  (reset only, no policy driving).

Result (`RESULT.txt`, from `compare_probes.py` on the two JSON files here):
start pose, goal pose and planner route length identical to 0.000 m on all 20 seeds; turns
and junctions equal on 20/20. The episode geometry does not depend on the build; only the
rendering differs.

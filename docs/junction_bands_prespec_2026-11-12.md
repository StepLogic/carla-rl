# Junction-turn bands: pre-specified design (2026-11-12, before any band result)

Purpose (user, 2026-11-12): "we want to show goal directedness". Difficulty bands by the
number of junction turns, in Town02 and Town07, built jointly with the map benchmark.

Written before any band episode is evaluated. Changes after results arrive are listed at the
end with the reason.

## Episodes
- A junction TURN is a run of RoadOption LEFT or RIGHT on the 2 m planner route from start to
  goal (the probe's `turns` field). A junction crossed straight scores 0.
- Bands: exactly 1 turn, exactly 2 turns, 3 or more turns.
- All bands are drawn from the same route window, 50-149 m (`slurm/bench_turns.env`), so route
  length is roughly matched. The median route length per band is reported.
- Selection is REPRESENTATIVE: the first 100 qualifying seeds per band and town in seed order
  (`scripts/select_turn_episodes.py --min-turns/--max-turns/--bin`). It is not balanced by
  direction.
- Probes:
  - Town02: the cluster's 1000-seed probe (job 465250, seeds 41000-41999), plus more seed
    blocks if the 3+ band has fewer than 100.
  - Town07: the map benchmark's local probe (seeds 41000-41999).
  - Seeds give identical episodes on CARLA 0.9.16 and 0.9.15 (results/junction_bands/crossversion).
- The turn-direction sequence of every episode is reconstructed offline from its start and
  goal (carla.Map + GlobalRoutePlanner at 2 m) and checked against the probe's turn count.

## Runs
- c0 (il_bc_c0 step_15000), mapless, local CARLA 0.9.15. Every episode is run twice on the
  same seed: once with the TRUE goal image, once with a WRONG goal image (WRONG_GOAL_MODE=1).
- Map arms on these episodes are the map benchmark's.
- A pilot of 20 episodes per band in Town07, true goal only, checks the harness and shows any
  floor. The pilot uses qualifying seeds after the first 100 of each band, so the evaluated
  sets are untouched by it. All bands are then run in full, whatever the pilot shows: a band
  at the floor is itself a result.

## Scoring
- Headline: success = within 1 m of the goal at any speed (the user's current rule).
- Secondary, from the same runs: success with entry into the 1 m circle at <= 10 km/h
  (separates aimed arrivals from drive-throughs).

## Tests
1. Goal use, per band and town: true vs wrong goal on the same seeds, exact McNemar, two-sided.
2. Direction (hypothesis H-left, pre-specified here; it comes from a post-hoc Town02
   observation and c0's Town07 drive): for mapless c0, true-goal success is LOWER on routes
   that contain a left turn than on routes that contain none. Test: Fisher's exact,
   two-sided, per band and town; the 1-turn band compares left vs right.
3. Counts by the full direction sequence (LL, LR, RL, RR, ...) are reported descriptively,
   with no test.
- 6 tests per family (3 bands x 2 towns); p-values are reported with a Holm correction
  within each family.

## Changes after results
(none yet)

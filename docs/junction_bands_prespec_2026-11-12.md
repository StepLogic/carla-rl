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
  Result of that check: the rebuilt count equals the probe's on all 740 selected rows
  (`scripts/annotate_turn_directions.py`; files in `results/junction_bands/episodes/`).

## Runs
- c0 (il_bc_c0 step_15000), mapless, local CARLA 0.9.15. Every episode is run twice on the
  same seed: once with the TRUE goal image, once with a WRONG goal image (WRONG_GOAL_MODE=1).
- These bands run c0 mapless ONLY. The map benchmark's existing hard band (the first 100
  qualifying seeds with >= 2 turns in 50-149 m) uses a different selection rule, straddles
  the 2-turn and 3+ bands, and is reported separately; its rows are not results on these bands.
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
- The 1-turn band's left-vs-right comparison is reported in every case, even if the other
  bands sit at the floor: it is the simplest goal-directedness statement and needs no map.

## Comparisons across bands (added 2026-11-12 ~12:20, before any junction-band result)
Written after the map benchmark's partial Town07 hard-band mapless result was known (SR 0.029
after 68 of 100 episodes; a different episode set). The addition only removes claims.
- The bands measure routes with more turning decisions. Turn count, route length and the
  share of routes with a left turn rise together because that is how the towns are built
  (episode sets: contains a left 57/68/80% in Town02 and 46/73% in Town07 for 1/2/3+ turns;
  median route 92/118/133 m and 97/122 m). A difference between band means is therefore
  not attributed to turn count alone, and no test is run across bands.
- Across-band comparison within the no-left stratum (the only one not confounded by left
  turns) is descriptive only, with n printed: about 43/32/20 routes in Town02 and 54/27
  in Town07, too few for a test.
- The isolated comparison is H-left within each band (test 2 above).

## Checks before any test
- Each wrong-goal cell's episodes_detail lists the same seeds, in the same order, as its
  true-goal partner; the analysis asserts this before any McNemar test.
- entry_speed_kmh is read per episode from episodes_detail (slow-arrival scoring).

## Reported regardless, per cell
n, successes, success rate, SPL, the ends breakdown, median route metres, the fraction of
routes that contain a left turn, and the median entry speed of the successes. A cell at the
floor reports all of it and is never dropped.

## Optional extension (b): map arms on these exact episode files
Not part of the plan above; it needs GPU time (6 band-town cells x 3 map arms) and is the
user's decision. If it runs, these tests apply (written by the map benchmark, before results):
- PRIMARY. Each map arm vs mapless, paired McNemar (exact two-sided on discordant pairs; the
  arms share seeds), per band and town, any-speed 1 m arrival. Holm within each (band, town)
  family of 3 arm comparisons.
- SECONDARY. The same comparisons scored under slow arrival (entry <= 10 km/h), reported
  alongside, never substituted for the primary. Any speed is the user's rule.
- H-LEFT FOR THE MAP ARMS. Among the pairs where exactly one of {map arm, mapless} succeeded,
  test whether which one succeeded is associated with the stratum (contains-left vs no-left):
  Fisher exact on the 2x2 of stratum x winner. This is the paired form of "the map's
  advantage is concentrated on left-turn routes", the hypothesis of the post-hoc Town02 split
  (2/72 -> 17-23/72 on contains-left, p <= 0.0007; nothing significant on no-left).
- WRONG-GOAL CONTROL ON THE MAP ARMS too: same seeds, McNemar true vs wrong within each arm.
  Without it, "the map helps" cannot be separated from the map arm being handed an easier goal,
  or from the subgoal sequence acting as a route prior whatever the final goal image shows.
  For a map arm the wrong goal image also moves the localised GOAL NODE, so the arm plans to the
  wrong place: the control measures the whole pipeline's dependence on the goal image, not the
  policy's alone. It is the right control, but not the same quantity as the mapless gap, and
  the two gaps are not differenced against each other.
- PLACEMENT VS PLACEMENT IS DESCRIPTIVE ONLY. Entropy vs uniform vs SGM gets no significance
  test at these sample sizes (no pairwise placement difference reached significance in the
  pooled hard band, n = 200).
- The "reported regardless" list above applies to every map-arm cell too.

## Changes after results
None. The analysis ran exactly as specified above: `results/junction_bands/analyse_bands.py`,
output `results/junction_bands/RESULTS.md` (runs finished 2026-11-12 22:00).

# CARLA tables of the image-navigation paper, regenerated 2026-11-15

The user asked for every CARLA table in the experiments section to be updated. Nothing here is
typed by hand: `make_carla_tables.py` recomputes every number from the per-episode records of the
bench `eval.json` files and checks each cell before using it (one eval.json per cell, n=100, town,
checkpoint, map build, episode file, paired seeds, identical start/goal across arms and policies,
SR/SPL equal to the file's own summary).

| file | what |
|---|---|
| `make_carla_tables.py` | generator; prints the full numbers (Wilson CIs, discordant pairs, fast arrivals) and writes the .tex |
| `carla_tables.tex` | the generated tables, each with its provenance as LaTeX comments |
| `experiments_section_updated.tex` | the user's section with the four CARLA tables replaced, three `% NOTE` comments on affected prose, and the appendix tables at the end |
| `experiments_section_original.tex` | the section as pasted by the user, for diffing |
| `build_section.py` | original + generated tables -> updated section |
| `cells_used.json` | every cell used, with path, checkpoint, map build, episode file, n, successes |
| `generator_output.txt` | the generator's printed output for this version (1218 checks, 0 failed) |
| `carla_tables_preview.pdf` | `carla_tables.tex` compiled on its own, to look at the layout |

Regenerate (needs an rsync copy of the cluster's `~/carla-rl/checkpoints`, eval.json files only):

    python make_carla_tables.py --cluster-root <copy> --json-out cells_used.json --tex-out carla_tables.tex
    python build_section.py experiments_section_original.tex carla_tables.tex experiments_section_updated.tex

## What each table now is

* `tab:junction_nodes`: Town07 column added (96/193 entropy keys in a junction). Junction counts are carried
  from the mapping session's `t07/tab_junction_nodes.tex` (2026-11-13). Key counts and spacing were re-read
  from the map metadata.
* `tab:map_navigation` (main): **held-out Town07, aligned navigator** (next-node advance, as in Habitat), z25
  maps in every band. Three policies: BC c0 (`na25`, AR=2), SAC fine-tuned from c0 (`sacna25`, AR=2,
  100k-step snapshot, trained on Town01/04/05/10HD) and ViNT (`vintd3_nx_ar1`, AR=1). This is the only
  protocol on which all three policies ran the same navigator. Rates differ by block, so only
  within-policy map effects are claims.
* `tab:long_horizon_carla`: the hard band of the main table, with the discordant pairs behind each test.
* `tab:map_navigation_pose`: key nodes only, localised by **true pose**, pooled over Town02/03/07 (n=300).
  Current medium band, z25 key-only maps, a no-map reference row and the largest SCC per town. The empty RL
  block is removed because no RL key-only cells exist.
* Appendix `tab:map_navigation_orignav` / `tab:long_horizon_carla_orignav`: the **earlier protocol**
  (original subgoal rule, AR=1, pooled over three towns), **z25-consistent**. This reproduces the draft's
  pooled table exactly except ViNT uniform easy (0.56 -> 0.54).

## Corrections settled today (agreed with the mapping session)

The 2026-11-14 regeneration had put two denser-map cell families (`maps_expert_jx_img`, 255/1104 keys) into
rows whose other towns were on z25:
c0 easy Town02/03 (`fresh_imageany_*`) and ViNT uniform easy/hard (`vintd3_fresh_any_uniform`,
`vintd3_turns_uniform`). The z25 cells (`imagez25_*_Town0{2,3}_easy`, `vintd3_z25_uniform_*`) pair with the
no-map arms and are the ones used here. With them the draft's easy row (0.79 / 0.78 / 0.77) and ViNT hard
uniform 0.08 are correct. The 0.81 and 0.07 values were the error.

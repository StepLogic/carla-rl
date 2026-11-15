#!/usr/bin/env bash
# Town07 3+ turn band, after the extra probe blocks (run_bands.sh step 1): the first 100
# qualifying seeds in seed order, and a 20-seed pilot set after them (skipped, with a
# note, if fewer than 120 qualify). Turn directions annotated and validated.
set -eu
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla
J=$W/results/junction_bands
PY=/home/robotlab/carla-sac-local/venv/bin/python
XODR=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive
export PYTHONPATH=/home/robotlab/carla-sac-local/rlib_integration${PYTHONPATH:+:$PYTHONPATH}
P="$J/probe_turns_Town07_mapping.json"
for s in 42000 43000 44000; do P="$P $J/checkpoints/jb_probe_turns_Town07_b$s/eval_step_15000/probe.json"; done
sel() {
  local out=$1; shift
  $PY "$W/scripts/select_turn_episodes.py" "$J/episodes/$out.raw.json" $P "$@"
  $PY "$W/scripts/annotate_turn_directions.py" "$J/episodes/$out.raw.json" "$J/episodes/$out.json" \
      --probe $P --xodr-dir "$XODR"
  rm "$J/episodes/$out.raw.json"
}
sel Town07_turns3plus --min-turns 3 --bin turns3plus
sel Town07_turns3plus_pilot --min-turns 3 --bin turns3plus_pilot --skip 100 --per-town 20 \
  || echo "fewer than 120 qualifying: no 3+ pilot set"

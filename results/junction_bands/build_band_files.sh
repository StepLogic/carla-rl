#!/usr/bin/env bash
# Junction-band episode files (pre-spec: docs/junction_bands_prespec_2026-11-12.md).
# Representative: the first 100 qualifying seeds per band and town in seed order, 50-149 m.
# Town07 pilots: the next 20 qualifying seeds after the first 100 (no overlap).
# Every file is then annotated with turn directions, validated against the probe counts.
set -eu
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla
J=$W/results/junction_bands
PY=/home/robotlab/carla-sac-local/venv/bin/python
# the planner imports rlib_integration: the local copy, identical to the cluster's
export PYTHONPATH=/home/robotlab/carla-sac-local/rlib_integration${PYTHONPATH:+:$PYTHONPATH}
XODR=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive
T02="$J/probe_turns_Town02_cluster.json $J/probe_turns_Town02_b42000_cluster.json"
T07="$J/probe_turns_Town07_mapping.json"
mkdir -p "$J/episodes"
sel() {   # town-probes out args...
  local probes=$1 out=$2; shift 2
  $PY "$W/scripts/select_turn_episodes.py" "$J/episodes/$out.raw.json" $probes "$@"
  $PY "$W/scripts/annotate_turn_directions.py" "$J/episodes/$out.raw.json" "$J/episodes/$out.json" \
      --probe $probes --xodr-dir "$XODR"
  rm "$J/episodes/$out.raw.json"
}
sel "$T02" Town02_turns1     --min-turns 1 --max-turns 1 --bin turns1
sel "$T02" Town02_turns2     --min-turns 2 --max-turns 2 --bin turns2
sel "$T02" Town02_turns3plus --min-turns 3 --bin turns3plus
sel "$T07" Town07_turns1     --min-turns 1 --max-turns 1 --bin turns1
sel "$T07" Town07_turns2     --min-turns 2 --max-turns 2 --bin turns2
sel "$T07" Town07_turns1_pilot --min-turns 1 --max-turns 1 --bin turns1_pilot --skip 100 --per-town 20
sel "$T07" Town07_turns2_pilot --min-turns 2 --max-turns 2 --bin turns2_pilot --skip 100 --per-town 20
echo "Town07 3+ needs more probe blocks (46 qualifying per 1000 seeds)."

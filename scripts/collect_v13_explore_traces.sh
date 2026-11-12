#!/usr/bin/env bash
# EXPLORE-MODE TRACES WITH PIXELS, v13, Town02 then Town03.
# User 2026-09-26: "collect trace in exploration mode with v13 in town02 and
# town03 and store all observations, store position of observations, store all
# relevant data also images to enable dinov2 localisation."
#
# Decisions taken with the user the same day: 5000 kept samples per town, the
# full record (RGB + pose + both observation framings + VC-1 features via the
# latent block + action/reward), no 100 m explore cut with a nearest-waypoint
# restart, and 120x120 frames only -- the resolution the policy sees and the
# one the earlier DINOv2-vs-MobileNet result was measured at, so the numbers
# stay comparable.
#
# PINNED TO clean_fix1, where v13 trained. The entry script there is a REAL
# FILE, not a symlink: CPython resolves a symlinked script when it sets
# __file__, which silently re-pins sys.path to the main tree and would run this
# against another session's uncommitted reward. Measured 2026-09-25.
#
# ONE TOWN PER SERVER: the env attaches to whatever answers on the port, so a
# reused server would drive the previous town's map. The kill is port-scoped so
# a peer's server on another port lives.
#
# RESUMABLE: the collector writes its pickle atomically after every episode and
# continues from it, so a supervisor kill costs one episode, not the town.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1

say(){ echo "=== [v13trace $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my_carla(){ for _ in 1 2 3 4 5; do pkill -9 -f 'carla-port=2000' 2>/dev/null; sleep 2;
  pgrep -f 'carla-port=2000' >/dev/null || return 0; done; return 1; }

SAMPLES="${SAMPLES:-5000}"
rc_all=0
for TOWN in Town02 Town03; do
  OUT="results/v13_traces/$(echo "$TOWN" | tr 'A-Z' 'a-z')_explore_trace.pkl"
  # DONE means the artefact holds the full budget, not merely that it exists:
  # a resumable collector leaves a valid partial pickle behind after a kill.
  if [ -s "$OUT" ] && python3 - "$OUT" "$SAMPLES" <<'PY'
import pickle, sys
d = pickle.load(open(sys.argv[1], "rb"))
sys.exit(0 if len(d["positions"]) >= int(sys.argv[2]) else 1)
PY
  then say "SKIP $TOWN: $OUT already holds $SAMPLES samples"; continue; fi

  say "$TOWN starting (fresh server, target $SAMPLES samples)"
  kill_my_carla || { say "FAILED: port-2000 CARLA survived the kill"; rc_all=1; continue; }
  sleep 10
  SAMPLES="$SAMPLES" bash collect_explore_v13.sh "" "$TOWN"
  if [ -s "$OUT" ]; then
    say "$TOWN wrote $OUT"
  else
    say "FAILED: $TOWN produced no $OUT"; rc_all=1
  fi
done
kill_my_carla
if [ "$rc_all" = 0 ]; then say "V13 EXPLORE TRACES DONE"; else say "V13 EXPLORE TRACES FAILED"; fi
exit "$rc_all"

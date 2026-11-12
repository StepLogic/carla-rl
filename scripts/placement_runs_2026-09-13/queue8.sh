#!/bin/bash
# QUEUE 8, 2026-09-12 16:25. ONE queue, sequential, so its stages cannot race.
#  1-2. rerun the two speed-8/radius-4 sweeps that FAILED (simulator died) but
#       were marked done: queue5's check was "any sweep json newer than 70 min",
#       which matched an OLDER sweep. This check requires a json created AFTER
#       the stage started, for the right TOWN and the right CHECKPOINT.
#  3.   retry value-vs-distance for v6c (crashed: bad_weak_ptr in a sensor
#       callback -- simulator-side, not the harness).
#  4.   stage 4 of experiments.txt, node count vs SR, on the v6n Town02 traces.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
Q="$W/results/queue_markers"; mkdir -p "$Q" logs
say(){ echo "=== [Q8 $(date +%H:%M:%S)] $* ==="; }
# queue4 counts as busy for its WHOLE life: it has 45 s settle gaps between
# stages (and Town04 placement still to run), which an idle check can slip into.
busy(){ pgrep -f "tmp/[q]ueue4\.sh" >/dev/null || pgrep -f "[t]rain_carla_tqc.py" >/dev/null || pgrep -f "[s]weep_eval.py" >/dev/null || pgrep -f "[e]val_placement.py" >/dev/null; }
wait_idle(){ for _ in $(seq 1 2880); do busy || { sleep 20; busy || return 0; }; sleep 30; done; return 1; }
settle(){ for _ in 1 2 3 4 5 6; do pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null; sleep 2; ps -eo args | grep -qi '[C]arlaUE4' || break; done; sleep 45; }
V6N=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl
V6M=checkpoints/carla_gated_her_v6m/checkpoint_450000.pkl

sweep_stage(){  # name town ckpt envfile
  local m=$1 town=$2 ck=$3 env=$4
  [ -f "$Q/$m" ] && { say "skip $m"; return; }
  for attempt in 1 2 3; do
    wait_idle; settle
    local t0; t0=$(date +%s)
    say "START $m attempt $attempt -> logs/q8_$m.log"
    RUN_ENV=$env ACTION_MODE=deterministic CKPT=$ck TOWN=$town VIDEO=1 MAX_VIDEOS=5 ./run_sweep.sh > "logs/q8_$m.log" 2>&1
    local lc; lc=$(echo "$town" | tr 'A-Z' 'a-z')
    local ok
    ok=$(find results -maxdepth 1 -name "${lc}_sweep_*.json" -newermt "@$t0" -print0 2>/dev/null | xargs -0 -r grep -l "$(basename $(dirname $ck))" 2>/dev/null | head -1)
    grep -aE "per-seed|simulator died|Traceback" "logs/q8_$m.log" | tail -3 | cut -c1-140
    if [ -n "$ok" ]; then date +%H:%M > "$Q/$m"; say "DONE $m -> $ok"; return; fi
    say "no fresh $town result for $(basename $(dirname $ck)) -- retrying"
  done
  say "FAILED $m after 3 attempts"
}
sweep_stage sweep8r4b_v6n_town04 Town04 $V6N results/run_env/carla_gated_her_v6n_town04.env
sweep_stage sweep8r4b_v6m_town02 Town02 $V6M results/run_env/carla_gated_her_v6m_town02.env

# 3. value-vs-distance, v6c, same fix as the v4/v5 runs that succeeded
if [ ! -f "$Q/vvd_v6c" ]; then
  for attempt in 1 2 3; do
    wait_idle; settle
    d=results/vvd/v6c; mkdir -p $d; rm -f $d/trace_s7.pkl.dist.npy
    say "START vvd_v6c attempt $attempt"
    ( set +u; eval "$(grep -E '^export ' run_gated.sh)"; set -u
      export PERCEPTION=vae PYTHONPATH=src CARLA_START_SERVER=1 CARLA_QUALITY=Epic EXPERT_PROB=0 ENV_SKIP_IMAGE_FEATURES=1
      python -u eval_placement.py --ckpt checkpoints/carla_gated_her_v6c/checkpoint_75000.pkl --arm v6c --town Town02 \
        --trace-cache $d/trace_s7.pkl --collect-seed 7 --stop-after matrix --out-dir $d \
      && python -u calibrate_critic.py --traces $d/trace_s7.pkl --horizon-m 50 --gate-m 2.0 --out $d/calibration.json
    ) > logs/q8_vvd_v6c.log 2>&1
    [ -f $d/calibration.json ] && { date +%H:%M > "$Q/vvd_v6c"; say "DONE vvd_v6c"; break; }
    say "vvd_v6c attempt $attempt failed: $(grep -aE 'Error|bad_weak_ptr' logs/q8_vvd_v6c.log | tail -1 | cut -c1-90)"
  done
fi

# 4. stage 4 -- needs the Town02 v6n calibration, which now exists
wait_idle
[ -f results/placement_v6n_town02/calibration.json ] || { say "no v6n Town02 calibration; stage 4 skipped"; }
if [ -f results/placement_v6n_town02/calibration.json ]; then
  exec /home/robotlab/.claude/jobs/af872d1b/tmp/queue6_stage4.sh
fi
say "QUEUE 8 COMPLETE"

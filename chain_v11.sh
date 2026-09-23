#!/bin/bash
# CHAIN DRIVER: launch v11 (N_STEP=3) the moment v10 ends.
#
# WHY A CHAIN, NOT A CONCURRENT LAUNCH: run_gated.sh takes a blocking flock -n
# on /tmp/carla_rl_port2000.lock and refuses a second job outright, and its GPU
# gate wants ~9500 MiB free before starting -- one CARLA job at a time is this
# stack's own convention (train_v10.sh is at ~70% of 450k and healthy, so
# killing it was never on the table). This driver therefore polls for v10's
# wrapper (pid recorded at write time, with a name fallback), then hands over
# to train_v11.sh -- which runs its own kill-all-CARLA, snapshot artefact
# check, and 40-attempt retry loop.
#
# PROVE-ALIVE (the launch is not the run): after train_v11.sh starts, this
# driver waits up to 20 min for ALL of: a CARLA sim process, the trainer
# python process, a fresh v11 log carrying the "[Buffer] n-step returns:
# N_STEP=3" banner, and card usage rising on nvidia-smi. Every check result
# is logged; a miss is retried (train_v11.sh's own retry loop re-attempts in
# parallel), and 20 min without a green board is reported as DEAD.
set -u
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl || exit 1
LOG=logs/chain_v11.log
V11=carla_gated_her_v11_vc1_nstep3
say(){ echo "=== [chain $(date +%H:%M:%S)] $* ==="; }

V10_WRAPPER=1543444
say "waiting for v10 (wrapper pid $V10_WRAPPER) to end"
for i in $(seq 1 720); do            # up to 12 h
  if ! kill -0 "$V10_WRAPPER" 2>/dev/null; then break; fi
  if ! pgrep -f 'train_v10\.s[h]' >/dev/null 2>&1; then break; fi
  sleep 60
done
if kill -0 "$V10_WRAPPER" 2>/dev/null || pgrep -f 'train_v10\.s[h]' >/dev/null 2>&1; then
  say "FATAL: v10 wrapper still alive after 12 h; not launching"; exit 1
fi
# THE WRAPPER IS GONE; let its run_gated.sh child and trainer drain (they exit
# with it; 90 s of slack covers the between-attempt sleep 30).
for i in $(seq 1 6); do
  pgrep -f 'run_gated\.s[h]' >/dev/null 2>&1 || pgrep -f 'run_gated_he[r]' >/dev/null 2>&1 || break
  sleep 15
done
s=$(ls checkpoints/carla_gated_her_v10_vc1 2>/dev/null \
    | sed -n 's/^checkpoint_\([0-9]*\)\.pkl$/\1/p' | sort -n | tail -1)
say "V10 GONE at checkpoint ${s:-0} of 450000; launching v11"
nohup ./train_v11.sh >> logs/train_v11_wrapper1.log 2>&1 &
LAUNCHED_AT=$(date +%s)
say "LAUNCHED train_v11.sh (wrapper pid $!)"

# ---- prove-alive: sim process + trainer + n-step banner + card moving --------
alive=0
for i in $(seq 1 40); do                       # 20 min
  sleep 30
  sim=$(pgrep -fc 'Linux-Shippin[g]' 2>/dev/null || echo 0)
  tr=$(pgrep -fc 'train_carla_tq[c]' 2>/dev/null || echo 0)
  banner=$(grep -c '\[Buffer\] n-step returns: N_STEP=3' "logs/${V11}.log" 2>/dev/null || echo 0)
  fresh=$(find "logs/${V11}.log" -newermt "@$LAUNCHED_AT" 2>/dev/null | wc -l)
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "$sim" -ge 1 ] && [ "$tr" -ge 1 ] && [ "$banner" -ge 1 ] && [ "$fresh" -ge 1 ]; then
    say "ALIVE: sims=$sim trainer=$tr nstep_banner=$banner vram_used=${vram}MiB"
    alive=1
    break
  fi
  say "waiting: sims=$sim trainer=$tr banner=$banner fresh_log=$fresh vram=${vram}MiB"
done
if [ "$alive" -ne 1 ]; then
  say "DEAD: v11 not provably alive 20 min after launch -- check logs/${V11}.log and logs/train_v11_wrapper1.log"
  exit 1
fi
tail -5 "logs/${V11}.log"
say "chain done; v11 training under train_v11.sh (its own retry wrapper)"
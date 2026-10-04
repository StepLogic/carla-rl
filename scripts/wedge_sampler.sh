#!/bin/bash
# Wedge diagnostics sampler (2026-09-21): Town10HD wedges the CARLA server's
# RPC with ~5 GB VRAM free, so this watches for "sim alive but port 2000 not
# answering" and, after 60 s of deadness, auto-captures a gdb stack of the
# server into logs/wedge_stack_*.txt before the retry loop kills it.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
DEAD=0
while true; do
  SIMS=$(pgrep -fc "Linux-Shippin[g]" 2>/dev/null); SIMS=${SIMS:-0}
  if [ "$SIMS" -gt 0 ]; then
    if python3 -c "import socket;s=socket.socket();s.settimeout(2.0);r=s.connect_ex(('localhost',2000));s.close();exit(0 if r==0 else 1)"; then
      DEAD=0
    else
      DEAD=$((DEAD+1))
      P=$(pgrep -f "Linux-Shippin[g]" | head -1)
      echo "$(date +%H:%M:%S) RPC DEAD (streak=$DEAD) vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) pid=$P"
      if [ -n "$P" ] && [ "$DEAD" -ge 6 ]; then
        gdb -p "$P" -batch -ex "thread apply all bt 8" > "logs/wedge_stack_$(date +%H%M%S).txt" 2>&1 &
        echo "$(date +%H:%M:%S) stack capture launched for pid $P"
        DEAD=0
      fi
    fi
    echo "$(date +%H:%M:%S) vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB sims=$SIMS"
  else
    echo "$(date +%H:%M:%S) sims=0"; DEAD=0
  fi
  sleep 10
done

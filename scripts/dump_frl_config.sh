#!/usr/bin/env bash
# Resolve a learner-C run's configuration WITHOUT starting anything: the run script's own exports
# (slurm/train_sac.sbatch, in order), then slurm/v52_reward.env, then $REWARD_OVERLAY -- exactly the
# order the run script uses -- and print every variable this adds or changes, sorted, followed by
# learner C's resolved loop settings and network config from the tree's own code.
# usage: REPO=<tree> REWARD_OVERLAY=slurm/<overlay>.env [FRL_*=...] PY=<python> bash scripts/dump_frl_config.sh
# Two machines run the same learner when their outputs differ only in paths, towns and ports.
set -u
: "${REPO:?}" "${REWARD_OVERLAY:?}" "${PY:?}"
TMPD=$(mktemp -d)
env | sort > "$TMPD/before"
# the run script's exports between the image/python choice and the reward overlays
eval "$(sed -n '/^export ENV_TOWNS=/,/^\. "\$REPO\/slurm\/v52_reward.env"/p' "$REPO/slurm/train_sac.sbatch" \
        | grep -E '^export ' | grep -vE 'PYTHONPATH|CARLA_SIF|DEBIT_SRC')"
. "$REPO/slurm/v52_reward.env"
. "$REPO/$REWARD_OVERLAY"
env | sort > "$TMPD/after"
echo "### run configuration (variables set by the run script + overlays)"
comm -13 "$TMPD/before" "$TMPD/after" | grep -vE '^(PWD|OLDPWD|SHLVL|_)='
echo "### learner C, resolved from the tree's code"
cd "$REPO" && PYTHONPATH="$REPO:$REPO/src" CUDA_VISIBLE_DEVICES= "$PY" -I - <<'EOF'
import dataclasses, os, sys
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "src")]
from torch_rl.frl import carla as fc
s = fc.LoopSettings.from_env()
print("LoopSettings:", {k: v for k, v in dataclasses.asdict(s).items()
                        if k not in ("expert", "device", "replay_device")})
print("LoopSettings paths/devices:", {k: getattr(s, k) for k in ("expert", "device", "replay_device")})
print("REDQ:", fc.REDQ)
print("LATENT_DIM:", fc.LATENT_DIM)
EOF
rm -rf "$TMPD"

#!/bin/bash
# Build dependencies/ig-venv: system site-packages + torch cu121 shadow.
#
# WHY A SHADOW VENV. System torch is 2.13.0+cu130; the driver (535) exposes
# CUDA 12.2, so torch.cuda is permanently False. Downgrading the SYSTEM torch
# would touch every other consumer on this box. A --system-site-packages venv
# keeps all trainer deps from the system and installs torch cu121 into the
# venv's site-packages, which python resolves FIRST -- so this interpreter
# alone sees a working CUDA torch.
#
# WHY THE vc_models PATH IS RESOLVED, NOT HARDCODED. dependencies/eai-vc is
# UNTRACKED and exists only in the main checkout, not in this worktree, so
# "$_ROOT/dependencies/eai-vc/vc_models" fails as written here. Resolution
# order: (a) EAI_VC_DIR env override; (b) the worktree-local path, for checkouts
# that do carry it; (c) the sibling main checkout's path derived from git.
set -euo pipefail
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_VENV="$_ROOT/dependencies/ig-venv"
[ -d "$_VENV" ] && { echo "[ig-venv] already exists at $_VENV"; exit 0; }
python3 -m venv --system-site-packages "$_VENV"
"$_VENV/bin/pip" install -q torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# WHY PIN cudnn 9.24 ON TOP OF torch. torch cu121 depends on
# nvidia-cudnn-cu12==9.1.0.70, which lands in the venv and SHADOWS the user
# site's 9.24.0.43 that the system jaxlib (0.6.2, compiled against cudnn 9.8)
# resolves to -- jax's conv kernels then die with "DNN library initialization
# failed". Installing the version jaxlib already requires into the venv (same
# major as torch's, forward-compatible) fixes jax without touching JAX itself;
# torch conv verified still working against 9.24. Do NOT "fix" this by
# installing jax/jaxlib into the venv -- that forks the learner's jax silently.
"$_VENV/bin/pip" install -q "nvidia-cudnn-cu12==9.24.0.43"

# WHY PIN protobuf 6.31.1. --system-site-packages exposes the system's
# protobuf to the venv, but the resolver picked 5.29.6 for the venv while
# tensorboard 2.21.0's protos are gencode 6.31.1 -- torch.utils.tensorboard
# then dies at import ("Runtime version cannot be older than the linked
# gencode version"), which the trainer hits the moment it opens its summary
# writer. Install the exact gencode version; jaxlib accepts 6.x (< 7).
"$_VENV/bin/pip" install -q "protobuf==6.31.1"

_VC_MODELS=""
if [ -n "${EAI_VC_DIR:-}" ] && [ -d "$EAI_VC_DIR" ]; then
    _VC_MODELS="$EAI_VC_DIR"
elif [ -d "$_ROOT/dependencies/eai-vc/vc_models" ]; then
    _VC_MODELS="$_ROOT/dependencies/eai-vc/vc_models"
else
    # git-common-dir of a worktree is <main-checkout>/.git, whose dirname is
    # the main checkout's root.
    _MAIN="$(git -C "$_ROOT" rev-parse --git-common-dir | xargs dirname 2>/dev/null)"
    if [ -n "$_MAIN" ] && [ -d "$_MAIN/dependencies/eai-vc/vc_models" ]; then
        _VC_MODELS="$_MAIN/dependencies/eai-vc/vc_models"
    fi
fi
if [ -z "$_VC_MODELS" ]; then
    echo "[ig-venv] FATAL: vc_models source dir not found; set EAI_VC_DIR" >&2
    exit 1
fi
echo "[ig-venv] installing vc_models from $_VC_MODELS"
"$_VENV/bin/pip" install -e "$_VC_MODELS"
echo "[ig-venv] built at $_VENV"
#!/bin/bash
# Training Script for Image Goal Navigation with Single Frame
# ============================================================

set -e  # Exit on error

# Configuration
# =============
SAVE_DIR="./checkpoints/image_goal_single"
TB_DIR="./logs/image_goal_single"
IMAGE_SIZE=84
NUM_STACK=1
BATCH_SIZE=128
MAX_STEPS=5000000
START_TRAINING=5000
EVAL_INTERVAL=10000
LOG_INTERVAL=1000
SEED=42

# Memory optimization for PyTorch
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Kill any existing CARLA processes
echo "Cleaning up existing CARLA processes..."
pkill -9 -f CarlaUE4 2>/dev/null || true
sleep 2

echo "=============================================="
echo "Image Goal Navigation Training"
echo "=============================================="
echo "Configuration:"
echo "  - Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  - Frame stack: ${NUM_STACK} (single image)"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Max steps: ${MAX_STEPS}"
echo "  - Save dir: ${SAVE_DIR}"
echo "  - Log dir: ${TB_DIR}"
echo "=============================================="
echo ""

# Run training
echo "Starting training..."
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online_pixels_image_goal.py \
    --save_dir="${SAVE_DIR}" \
    --log_dir="${TB_DIR}" \
    --seed=${SEED} \
    --image_size=${IMAGE_SIZE} \
    --num_stack=${NUM_STACK} \
    --batch_size=${BATCH_SIZE} \
    --max_steps=${MAX_STEPS} \
    --start_training=${START_TRAINING} \
    --eval_interval=${EVAL_INTERVAL} \
    --log_interval=${LOG_INTERVAL} \
    --eval_episodes=5 \
    --tqdm=True \
    --save_video=False

echo ""
echo "=============================================="
echo "Training completed!"
echo "Checkpoint saved to: ${SAVE_DIR}"
echo "Logs saved to: ${TB_DIR}"
echo "=============================================="

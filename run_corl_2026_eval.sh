#!/bin/bash
# CoRL 2026 Image-Goal Evaluation Pipeline
# ==========================================
# Step 1: Map Town02 (skip if already exists)
# Step 2: Create trajectories (skip if already exists)
# Step 3: Evaluate all models with camera-ready videos

set -e

MAP_DIR="corl-2026-results/town02_map"
TRAJ_DIR="corl-2026-results/trajectories"
OUTPUT_DIR="corl-2026-results"

# Step 1: Map Town02
if [ -f "${MAP_DIR}/map_metadata.json" ]; then
    echo "Map already exists at ${MAP_DIR}, skipping map creation."
else
    echo "Creating Town02 map..."
    python corl_2026_map_town02.py --town Town02 --output-dir "${MAP_DIR}" --skip-existing
fi

# Step 2: Create trajectories
if [ -f "${TRAJ_DIR}/benchmark_info.json" ]; then
    echo "Trajectories already exist at ${TRAJ_DIR}, skipping trajectory creation."
else
    echo "Creating benchmark trajectories..."
    python corl_2026_create_trajectories.py --town Town02 --map-dir "${MAP_DIR}" --output-dir "${TRAJ_DIR}"
fi

# Step 3: Evaluate all models with videos
echo ""
echo "=============================================="
echo "Running evaluation: Image-Goal DrQ vs baselines"
echo "=============================================="
echo ""

XLA_PYTHON_CLIENT_PREALLOCATE=false python corl_2026_evaluate_image_goal.py \
    --model all \
    --town Town02 \
    --benchmark-dir "${TRAJ_DIR}" \
    --map-dir "${MAP_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-videos \
    --video-dir "${OUTPUT_DIR}/videos"

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Videos saved to: ${OUTPUT_DIR}/videos"
echo "=============================================="
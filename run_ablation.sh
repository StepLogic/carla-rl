#!/bin/bash
# Ablation Study: Map Sparsity vs Navigation Success
# ====================================================
# Evaluates all models across different map sparsity levels on Town02.
#
# Three sparsity modes:
#   spatial     - subsample by minimum inter-node distance (meters)
#   similarity  - subsample by VLAD feature cosine similarity threshold
#   both        - spatial first, then similarity
#
# Prerequisites:
#   - CARLA server running on port 2000 (for evaluation)
#   - Dense-image map at corl-2026-results-dense-image/town02_map/
#   - Benchmark trajectories at corl-2026-results-dense-image/trajectories/
#
# Usage:
#   ./run_ablation.sh                                       # Full spatial ablation
#   ./run_ablation.sh --sparsity-mode similarity             # Feature-similarity ablation
#   ./run_ablation.sh --sparsity-mode both                  # Combined spatial + similarity
#   ./run_ablation.sh --maps-only                           # Build maps only (no CARLA)
#   ./run_ablation.sh --model drq_ig                        # Run only DrQ-IG
#   ./run_ablation.sh --model nomad --min-distances 5 10 20 # Specific models/distances
#   ./run_ablation.sh --plot-only                           # Generate plots from results
#   ./run_ablation.sh --localization-mode image             # Image-based localization
#   ./run_ablation.sh --sparsity-mode similarity --similarity-thresholds 0.3 0.5

set -e

# Configuration
MAP_DIR="corl-2026-results-dense-image/town02_map"
BENCHMARK_DIR="corl-2026-results-dense-image/trajectories"
RESULTS_DIR="ablation-results"
CARLA_PORT=2000

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "${MAP_DIR}/map_data.pkl" ]; then
    echo "ERROR: Map data not found at ${MAP_DIR}/map_data.pkl"
    echo "Run map collection first."
    exit 1
fi

if [ ! -f "${BENCHMARK_DIR}/benchmark_info.json" ]; then
    echo "ERROR: Benchmark trajectories not found at ${BENCHMARK_DIR}/benchmark_info.json"
    echo "Run trajectory creation first."
    exit 1
fi

# Check CARLA server
if ! pgrep -f "CarlaUE4" > /dev/null 2>&1; then
    echo "WARNING: CARLA server not detected. Evaluation will fail."
    echo "Start CARLA first: ./CarlaUE4.sh -carla-port=${CARLA_PORT}"
    echo ""
    echo "Continuing with --maps-only to build sparse maps..."
    EXTRA_ARGS="--maps-only"
fi

# Environment setup
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

SPARSITY_MODE="${1:-spatial}"

echo ""
echo "=============================================="
echo "ABLATION: Map Sparsity vs Navigation Success"
echo "=============================================="
echo "Map:          ${MAP_DIR}"
echo "Benchmark:    ${BENCHMARK_DIR}"
echo "Results:      ${RESULTS_DIR}"
echo "Sparsity mode: ${SPARSITY_MODE}"
echo ""

# Step 1: Build sparse maps (fast, no CARLA needed)
echo "Step 1: Building sparse maps..."
python ablation.py \
    --map-dir "${MAP_DIR}" \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --sparsity-mode "${SPARSITY_MODE}" \
    --maps-only \
    $@

echo ""
echo "Sparse maps built at ${RESULTS_DIR}/sparse_*/"

# If --maps-only was explicitly passed, stop here
if echo "$@" | grep -q -- "--maps-only"; then
    echo "Maps-only mode requested. Done."
    exit 0
fi

# If --plot-only was passed, skip evaluation
if echo "$@" | grep -q -- "--plot-only"; then
    echo "Generating plots from existing results..."
    python ablation.py \
        --plot-only \
        --results-dir "${RESULTS_DIR}"
    echo "Plots saved to ${RESULTS_DIR}/"
    exit 0
fi

# Step 2: Plot similarity distribution (informative for choosing thresholds)
echo ""
echo "Step 1b: Generating similarity distribution plot..."
python plot_similarity_distribution.py \
    --map-dir "${MAP_DIR}" \
    --output-dir "${RESULTS_DIR}" || true

# Step 3: Run evaluation (requires CARLA)
echo ""
echo "Step 2: Running evaluation (this takes several hours)..."
echo ""

# Pass through any CLI args
python ablation.py \
    --map-dir "${MAP_DIR}" \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --sparsity-mode "${SPARSITY_MODE}" \
    $@

echo ""
echo "=============================================="
echo "Ablation complete!"
echo "Results:  ${RESULTS_DIR}/ablation_results.pkl"
echo "Plots:    ${RESULTS_DIR}/ablation_results.png"
echo "          ${RESULTS_DIR}/ablation_per_model.png"
echo "          ${RESULTS_DIR}/similarity_distribution.png"
echo "=============================================="
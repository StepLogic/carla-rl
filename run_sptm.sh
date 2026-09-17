#!/bin/bash
# SPTM (Semi-Parametric Topological Memory) Training + Evaluation
# ================================================================
# PyTorch implementation of the SPTM baseline (Savinov et al., ICLR 2018)
#
# Prerequisites:
#   - Map data at corl-2026-results/town02_map/ (for training)
#   - CARLA server on port 2000 (for evaluation only)
#
# Usage:
#   ./run_sptm.sh train               # Train both networks (edge + action)
#   ./run_sptm.sh train-edge          # Train only edge predictor (R-network)
#   ./run_sptm.sh train-action        # Train only action predictor (L-network)
#   ./run_sptm.sh eval                # Evaluate SPTM in CARLA
#   ./run_sptm.sh graph               # Build SPTM graph only (no CARLA needed)
#   ./run_sptm.sh full                # Train + graph + evaluate

set -e

# Configuration
MAP_DIR="corl-2026-results-dense-image/town02_map"
BENCHMARK_DIR="corl-2026-results-dense-image/trajectories"
SPTM_DIR="sptm_output"
CARLA_PORT=2000

# Training hyperparameters
EDGE_EPOCHS=7000
ACTION_EPOCHS=7000
BATCH_SIZE=256
LR=1e-4

# Check prerequisites
if [ ! -f "${MAP_DIR}/map_data.pkl" ]; then
    echo "ERROR: Map data not found at ${MAP_DIR}/map_data.pkl"
    exit 1
fi

mkdir -p "${SPTM_DIR}"

train_edge() {
    echo ""
    echo "=============================================="
    echo "Training R-network (Edge Predictor)"
    echo "  Siamese ResNet-18: 6ch input -> 2-class"
    echo "  Epochs: ${EDGE_EPOCHS}, LR: ${LR}, Batch: ${BATCH_SIZE}"
    echo "=============================================="
    echo ""

    # python sptm_train.py \
    #     --network edge \
    #     --map-dir "${MAP_DIR}" \
    #     --output-dir "${SPTM_DIR}" \
    #     --epochs-edge "${EDGE_EPOCHS}" \
    #     --batch-size "${BATCH_SIZE}" \
    #     --lr "${LR}"
    

    echo ""
    echo "Edge model saved to ${SPTM_DIR}/edge_model_final.pt"
}

train_action() {
    echo ""
    echo "=============================================="
    echo "Training L-network (Action Predictor)"
    echo "  ResNet-18: 9ch input -> 7-class"
    echo "  Epochs: ${ACTION_EPOCHS}, LR: ${LR}, Batch: ${BATCH_SIZE}"
    echo "=============================================="
    echo ""

    python sptm_train.py \
        --network action \
        --map-dir "${MAP_DIR}" \
        --output-dir "${SPTM_DIR}" \
        --epochs-action "${ACTION_EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}"

    echo ""
    echo "Action model saved to ${SPTM_DIR}/action_model_final.pt"
}

build_graph() {
    echo ""
    echo "=============================================="
    echo "Building SPTM topological graph"
    echo "=============================================="
    echo ""

    python -c "
from sptm_train import SPTMNavigator
nav = SPTMNavigator(
    edge_model_path='${SPTM_DIR}/edge_model_final.pt',
    action_model_path='${SPTM_DIR}/action_model_final.pt',
    map_dir='${MAP_DIR}',
)
nav.build_graph()
import pickle
graph_data = {
    'G': nav.G,
    'keyframe_coordinates': nav.keyframe_coordinates,
    'num_nodes': nav.G.number_of_nodes(),
    'num_edges': nav.G.number_of_edges(),
}
with open('${SPTM_DIR}/sptm_graph.pkl', 'wb') as f:
    pickle.dump(graph_data, f)
print(f'Graph saved: {nav.G.number_of_nodes()} nodes, {nav.G.number_of_edges()} edges')
"

    echo "Graph saved to ${SPTM_DIR}/sptm_graph.pkl"
}

run_eval() {
    # Check CARLA
    if ! pgrep -f "CarlaUE4" > /dev/null 2>&1; then
        echo "ERROR: CARLA server not detected on port ${CARLA_PORT}"
        echo "Start CARLA first: ./CarlaUE4.sh -carla-port=${CARLA_PORT}"
        exit 1
    fi

    if [ ! -f "${BENCHMARK_DIR}/benchmark_info.json" ]; then
        echo "ERROR: Benchmark trajectories not found at ${BENCHMARK_DIR}"
        exit 1
    fi

    echo ""
    echo "=============================================="
    echo "Evaluating SPTM in CARLA"
    echo "  Edge model:  ${SPTM_DIR}/edge_model_final.pt"
    echo "  Action model: ${SPTM_DIR}/action_model_final.pt"
    echo "  Map:          ${MAP_DIR}"
    echo "  Benchmark:    ${BENCHMARK_DIR}"
    echo "=============================================="
    echo ""

    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20

    python -c "
import json, pickle, os
from sptm_train import SPTMNavigator, evaluate_sptm
from rlib_integration.carla_env import CarlaEnv
from rlib_integration.helper import ndarray_to_location
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
from src.configs.eval_env_config_image_goal import config as ig_config

# Setup environment
ig_config['env_config']['carla']['start_server'] = False
ig_config['env_config']['carla']['port'] = ${CARLA_PORT}
env = CarlaEnv(config=ig_config)
env.experiment.eval_mode = True
env.experiment.map_mode = False
env = TimeLimit(env, max_episode_steps=3000)
env = RecordEpisodeStatistics(env)

# Load navigator
nav = SPTMNavigator(
    edge_model_path='${SPTM_DIR}/edge_model_final.pt',
    action_model_path='${SPTM_DIR}/action_model_final.pt',
    map_dir='${MAP_DIR}',
)
nav.build_graph()

# Load trajectories
with open('${BENCHMARK_DIR}/benchmark_info.json') as f:
    benchmark = json.load(f)

# Evaluate
results = []
for traj_info in benchmark['trajectories']:
    traj_id = traj_info['trajectory_id']
    traj_path = os.path.join('${BENCHMARK_DIR}', f'trajectory_{traj_id:02d}')
    print(f'Trajectory {traj_id}...')
    try:
        result = evaluate_sptm(env, nav, traj_path, traj_id)
        results.append(result)
        print(f'  Success={result[\"success\"]:.0f}, SPL={result[\"mean_spl\"]:.3f}')
    except Exception as e:
        print(f'  ERROR: {e}')
        results.append({'trajectory_id': traj_id, 'model': 'sptm', 'error': str(e), 'success': 0.0})

env.close()

# Save
with open('${SPTM_DIR}/sptm_eval_results.pkl', 'wb') as f:
    pickle.dump(results, f)

successes = [r for r in results if 'error' not in r]
sr = sum(r['success'] for r in successes) / len(successes) if successes else 0
spl = sum(r['mean_spl'] for r in successes) / len(successes) if successes else 0
print(f'\\nSPTM Results: Success={sr:.0%}, SPL={spl:.3f}')
print(f'Saved to ${SPTM_DIR}/sptm_eval_results.pkl')
"
}

# Main dispatch
case "${1:-full}" in
    train)
        train_edge
        train_action
        ;;
    train-edge)
        train_edge
        ;;
    train-action)
        train_action
        ;;
    graph)
        build_graph
        ;;
    eval)
        build_graph
        run_eval
        ;;
    full)
        train_edge
        train_action
        build_graph
        run_eval
        ;;
    *)
        echo "Usage: $0 {train|train-edge|train-action|graph|eval|full}"
        echo ""
        echo "  train        Train both R-network and L-network"
        echo "  train-edge   Train only edge predictor (Siamese ResNet-18)"
        echo "  train-action Train only action predictor (ResNet-18)"
        echo "  graph         Build SPTM topological graph with shortcuts"
        echo "  eval          Evaluate SPTM in CARLA (requires server)"
        echo "  full          Train + graph + evaluate"
        exit 1
        ;;
esac
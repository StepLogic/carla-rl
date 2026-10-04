#!/bin/bash

# Define arrays for difficulty levels and their corresponding towns
declare -A town_mapping
town_mapping["easy"]="Town02"
town_mapping["medium"]="Town02"
town_mapping["hard"]="Town02"
town_mapping["trajectories"]="Town02"
# Models to evaluate
# Models to evaluate
models=("PixelBCLearner" "DrQLearner") #"DrQLearner"
declare -A checkpoints
checkpoints["PixelBCLearner"]="/home/robotlab/scratch/carla-rl/best_models/final_bc/checkpoint_49"
checkpoints["DrQLearner"]="/home/robotlab/scratch/carla-rl/best_models/final_drq/checkpoint_1"

# Number of trajectories per difficulty (0-4)
num_trajectories=5

# Function to run evaluation for a specific configuration
run_evaluation() {
    local model=$1

    local map_dir="baseline_maps/${model}"
    local checkpoint_path=${checkpoints[$model]}
    echo "Running evaluation for:"
    echo "- Model: ${model}"
    echo "- Map Directory: ${map_dir}"
    echo "-----------------------------------"

    XLA_PYTHON_CLIENT_PREALLOCATE=false python src/benchmark_rl_policy_random_trajectory.py \
        --model=${model} \
        --map_dir=${map_dir} \
        --checkpoint_path=${checkpoint_path}

    echo "Build complete for ${model}"
    echo "==================================="
}

# Main execution loop
for model in "${models[@]}"; do
    run_evaluation "$model"
done

echo "All evaluations completed successfully!"
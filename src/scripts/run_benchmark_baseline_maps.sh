#!/bin/bash

# Define arrays for difficulty levels and their corresponding towns
declare -A town_mapping
town_mapping["easy"]="Town02"
town_mapping["medium"]="Town02"
town_mapping["hard"]="Town02"
town_mapping["trajectories"]="Town02"
# Models to evaluate
models=("vint" "nomad" "gnm")

# Number of trajectories per difficulty (0-4)
num_trajectories=5

# Function to run evaluation for a specific configuration
run_evaluation() {
    local model=$1

    local map_dir="baseline_maps/${model}"

    echo "Running evaluation for:"
    echo "- Model: ${model}"
    echo "- Map Directory: ${map_dir}"
    echo "-----------------------------------"

    python src/benchmark_policy_baseline_map.py \
        --model=${model} \
        --map_dir=${map_dir} 

    echo "Build complete for ${model}"
    echo "==================================="
}

# Main execution loop
for model in "${models[@]}"; do
    run_evaluation "$model"
done

echo "All evaluations completed successfully!"
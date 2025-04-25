#!/bin/bash

# Define arrays for difficulty levels and their corresponding towns
declare -A town_mapping
town_mapping["easy"]="Town02"
town_mapping["medium"]="Town03"
town_mapping["hard"]="Town05"

# Models to evaluate
models=("PixelBCLearner" "DrQLearner") #"DrQLearner"
declare -A checkpoints
checkpoints["PixelBCLearner"]="/home/robotlab/scratch/carla-rl/best_models/final_bc/checkpoint_49"
checkpoints["DrQLearner"]="/home/robotlab/scratch/carla-rl/best_models/model-sac-33/checkpoint_800000"


# Number of trajectories per difficulty (0-4)
num_trajectories=5

# Function to run evaluation for a specific configuration
run_evaluation() {
    local difficulty=$1
    local trajectory=$2
    local model=$3
    local town=${town_mapping[$difficulty]}
    local map_dir="evaluation_trajectory/${difficulty}/${trajectory}"
    local checkpoint_path=${checkpoints[$model]}

    echo "Running evaluation for:"
    echo "- Difficulty: ${difficulty}"
    echo "- Town: ${town}"
    echo "- Trajectory: ${trajectory}"
    echo "- Model: ${model}"
    echo "- Map Directory: ${map_dir}"
    echo "-----------------------------------"

   XLA_PYTHON_CLIENT_PREALLOCATE=false  python src/build_map.py \
        --model=${model} \
        --n_eval_episodes=10 \
        --deterministic=true \
        --map_dir=${map_dir} \
        --town=${town} \
        --checkpoint_path=${checkpoint_path}

    echo "Evaluation complete for ${difficulty}/${trajectory}/${model}"
    echo "==================================="
}

# Main execution loop
for model in "${models[@]}"; do
    for difficulty in "easy" "medium" "hard"; do
        for ((trajectory=0; trajectory<num_trajectories; trajectory++)); do
            run_evaluation "$difficulty" "$trajectory" "$model"
        done
    done
done

echo "All evaluations completed successfully!"
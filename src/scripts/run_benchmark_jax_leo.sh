# #!/bin/bash

# # Define arrays for difficulty levels and their corresponding towns
# declare -A town_mapping
# town_mapping["easy"]="Town02"
# town_mapping["medium"]="Town02"
# town_mapping["hard"]="Town02"

# # Models to evaluate
# models=("PixelResNetBCLearner") #"DrQLearner"
# declare -A checkpoints
# checkpoints["PixelResNetBCLearner"]="/home/kojogyaase/Projects/Research/carla-rl/checkpoints/leo_bc/checkpoint_70"
# # checkpoints["DrQLearner"]="/home/robotlab/scratch/carla-rl/best_models/model-sac-16/checkpoint_1650000"


# # Number of trajectories per difficulty (0-4)
# num_trajectories=5

# # Function to run evaluation for a specific configuration
# run_evaluation() {
#     local difficulty=$1
#     local trajectory=$2
#     local model=$3
#     local town=${town_mapping[$difficulty]}
#     local map_dir="evaluation_trajectory/${difficulty}/${trajectory}"
#     local checkpoint_path=${checkpoints[$model]}

#     echo "Running evaluation for:"
#     echo "- Difficulty: ${difficulty}"
#     echo "- Town: ${town}"
#     echo "- Trajectory: ${trajectory}"
#     echo "- Model: ${model}"
#     echo "- Map Directory: ${map_dir}"
#     echo "-----------------------------------"

#    XLA_PYTHON_CLIENT_PREALLOCATE=false  python3 src/benchmark_rl_policy_leo.py \
#         --model=${model} \
#         --n_eval_episodes=5 \
#         --deterministic=true \
#         --map_dir=${map_dir} \
#         --town=${town} \
#         --checkpoint_path=${checkpoint_path}

#     echo "Evaluation complete for ${difficulty}/${trajectory}/${model}"
#     echo "==================================="
# }

# # Main execution loop
# for model in "${models[@]}"; do
#     for difficulty in "easy" "medium" "hard"; do
#         for ((trajectory=0; trajectory<num_trajectories; trajectory++)); do
#             run_evaluation "$difficulty" "$trajectory" "$model"
#         done
#     done
# done

# echo "All evaluations completed successfully!"
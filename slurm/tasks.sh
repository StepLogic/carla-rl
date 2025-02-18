#!/bin/bash
#SBATCH --job-name=rl_training       # Job name
#SBATCH --partition=gpu              # Partition/Queue name
#SBATCH --mail-type=END,FAIL        # Mail events
#SBATCH --mail-user=egyaase@maine.edu # Where to send mail
#SBATCH --ntasks=1                   # Run on single node
#SBATCH --cpus-per-task=8           # Run with 8 threads
#SBATCH --mem=150gb                 # Job memory request
#SBATCH --time=96:00:00             # Time limit hrs:min:sec
#SBATCH --output=rl_error_%j.log    # Standard output and error log
#SBATCH --gres=gpu:l40:1            # Request 1 L40 GPU

# Change directory
cd ~/carla-rl

# Load required module
module load apptainer

# Create logs directory
mkdir -p "$HOME/carla_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to start training for a town
train_town() {
    local town=$1
    local port=$2
    local log_dir="$HOME/carla_logs/${TIMESTAMP}_${town}"
    mkdir -p "$log_dir"
    
    echo "Starting CARLA server for ${town} on port ${port}"
    
    # Start CARLA server with error logging
    nohup singularity run --nv -e "$HOME/containers/carla-0.9.15.sif" \
    /home/carla/CarlaUE4.sh \
    -RenderOffScreen \
    -nosound \
    -benchmark \
    -fps=60 \
    --carla-rpc-port="${port}" \
    -prefernvidia > "$log_dir/carla_server.log" 2>&1 &
    
    local carla_pid=$!
    
    # Check if CARLA server started
    if ! ps -p $carla_pid > /dev/null; then
        echo "ERROR: Failed to start CARLA server for ${town}. Check logs at $log_dir/carla_server.log" >&2
        return 1
    fi
    
    echo "CARLA server started for ${town} with PID ${carla_pid}"
    
    # Wait for CARLA initialization
    sleep 30
    
    # Start training with error logging
    nohup singularity run --nv "$HOME/containers/acg.simg" \
    python "$HOME/carla-rl/src/ppo_lane_following.py" \
    "${town}" \
    "${port}" > "$log_dir/training.log" 2>&1 &
    
    local training_pid=$!
    
    # Store PIDs for cleanup
    echo "${carla_pid}" >> /tmp/carla_pids_$$
    echo "${training_pid}" >> /tmp/training_pids_$$
    
    echo "Training started for ${town} with PID ${training_pid}"
}

# Create temporary files for PIDs
touch /tmp/carla_pids_$$
touch /tmp/training_pids_$$

# Error handling for the whole script
set -e

echo "Starting training processes at ${TIMESTAMP}"

# Start training for each town in parallel
for town in "Town01" "Town02" "Town03"; do
    port=$((2000 + ${#town}))
    train_town "$town" "$port" || {
        echo "ERROR: Failed to start training for ${town}" >&2
        exit 1
    }
done

# Cleanup function
cleanup() {
    echo "Cleaning up processes..."
    
    # Kill CARLA servers
    if [ -f /tmp/carla_pids_$$ ]; then
        while read pid; do
            echo "Killing CARLA server with PID ${pid}"
            kill $pid 2>/dev/null || echo "Failed to kill CARLA server PID ${pid}"
        done < /tmp/carla_pids_$$
        rm /tmp/carla_pids_$$
    fi
    
    # Kill training processes
    if [ -f /tmp/training_pids_$$ ]; then
        while read pid; do
            echo "Killing training process with PID ${pid}"
            kill $pid 2>/dev/null || echo "Failed to kill training PID ${pid}"
        done < /tmp/training_pids_$$
        rm /tmp/training_pids_$$
    fi
    
    echo "Cleanup completed"
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Wait for all processes to finish
wait

echo "All training processes completed at $(date)"
exit 0
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

# Load required module
module load apptainer

# Function to start training for a town
train_town() {
    local town=$1
    local port=$2
    
    echo "Starting CARLA server for ${town} on port ${port}"
    
    # Start CARLA server
    nohup singularity run --nv -e "$HOME/containers/carla-0.9.15.sif" \
    /home/carla/CarlaUE4.sh \
    -RenderOffScreen \
    -nosound \
    -benchmark \
    -fps=60 \
    --carla-rpc-port="${port}" \
    -prefernvidia &
    
    local carla_pid=$!
    
    # Check if CARLA server started
    if ! ps -p $carla_pid > /dev/null; then
        echo "Failed to start CARLA server for ${town}"
        return 1
    fi
    
    # Wait for CARLA initialization
    sleep 30
    
    # Start training
    singularity run --nv "$HOME/containers/acg.simg" \
    python "$HOME/carla-rl/src/ppo_lane_following.py" \
    "${town}" \
    "${port}" &
    
    # Store PIDs for cleanup
    echo "${carla_pid}" >> /tmp/carla_pids_$$
}

# Create temporary file for PIDs
touch /tmp/carla_pids_$$

# Start training for each town in parallel
train_town "Town01" 2000 &
train_town "Town02" 2001 &
train_town "Town03" 2002 &

# Wait for all background processes to complete
wait

# Cleanup function
cleanup() {
    echo "Cleaning up processes..."
    if [ -f /tmp/carla_pids_$$ ]; then
        while read pid; do
            kill $pid 2>/dev/null
        done < /tmp/carla_pids_$$
        rm /tmp/carla_pids_$$
    fi
}

# Set up trap for cleanup
trap cleanup EXIT

# Wait for all processes to finish
wait

exit 0
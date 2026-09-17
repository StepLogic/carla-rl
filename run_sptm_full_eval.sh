#!/bin/bash
# ============================================================================
# run_sptm_full_eval.sh
# Full SPTM evaluation for both towns + failure mode analysis
# ============================================================================

set -e

PROJECT_DIR="/home/robotlab/scratch/carla-rl"
RESULTS_DIR="$PROJECT_DIR/corl-2026-experiments/results"
FAILURE_DIR="$PROJECT_DIR/corl-2026-experiments/results_failure_modes"
OUTPUT_DIR="$PROJECT_DIR/sptm_output_carla"
CARLA_PORT=2000

mkdir -p "$RESULTS_DIR"
mkdir -p "$FAILURE_DIR"

echo "========================================================================"
echo "SPTM Full Evaluation Script"
echo "========================================================================"
echo ""

# Clean old SPTM dense results to force re-evaluation
for town in town02 town05; do
    RESULT_FILE="$RESULTS_DIR/${town}_dense_sptm_results.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "Removing old result: $RESULT_FILE"
        rm -f "$RESULT_FILE" "${RESULT_FILE%.json}.pkl"
    fi
done

echo ""
echo "Running SPTM evaluation on dense maps for both towns..."
echo "  Edge model: $OUTPUT_DIR/edge_model_best.pt"
echo "  Action model: $OUTPUT_DIR/action_model_best.pt"
echo "  CARLA port: $CARLA_PORT"
echo ""

python "$PROJECT_DIR/corl-2026-experiments/evaluate_sptm.py" \
    --town all \
    --map-type dense \
    --edge-model "$OUTPUT_DIR/edge_model_best.pt" \
    --action-model "$OUTPUT_DIR/action_model_best.pt" \
    --carla-port "$CARLA_PORT" \
    2>&1 | tee "$RESULTS_DIR/sptm_eval_log.txt"

echo ""
echo "========================================================================"
echo "Evaluation complete. Analyzing failure modes..."
echo "========================================================================"
echo ""

python3 << 'PYEOF'
import json
import os
import pickle
import sys

PROJECT_DIR = "/home/robotlab/scratch/carla-rl"
RESULTS_DIR = os.path.join(PROJECT_DIR, "corl-2026-experiments", "results")
FAILURE_DIR = os.path.join(PROJECT_DIR, "corl-2026-experiments", "results_failure_modes")

def analyze_failure_modes(town):
    result_file = os.path.join(RESULTS_DIR, f"{town}_dense_sptm_results.pkl")
    if not os.path.exists(result_file):
        print(f"  {town}: result file not found, skipping")
        return None

    with open(result_file, 'rb') as f:
        data = pickle.load(f)

    per_traj = data.get('per_trajectory_results', [])
    failures = []
    successes = []

    for traj in per_traj:
        traj_id = traj.get('trajectory_id', '?')
        success = traj.get('success', 0.0)
        spl = traj.get('mean_spl', 0.0)
        dist = traj.get('mean_distance', 0.0)
        shortest = traj.get('shortest_distance', 0.0)

        if success > 0.5:
            successes.append({
                'trajectory_id': traj_id,
                'spl': spl,
                'distance': dist,
                'shortest': shortest,
            })
        else:
            failures.append({
                'trajectory_id': traj_id,
                'spl': spl,
                'distance': dist,
                'shortest': shortest,
            })

    # Categorize failures
    stuck = [f for f in failures if f['distance'] < 10.0]
    drove_but_timeout = [f for f in failures if f['distance'] >= 10.0]

    summary = {
        'town': town,
        'total_trajectories': len(per_traj),
        'successes': len(successes),
        'failures': len(failures),
        'success_rate': data.get('overall_success_rate', 0.0),
        'mean_spl': data.get('overall_mean_spl', 0.0),
        'mean_distance': data.get('overall_mean_distance', 0.0),
        'failure_modes': {
            'stuck_at_start': {
                'count': len(stuck),
                'avg_distance': sum(f['distance'] for f in stuck) / max(len(stuck), 1),
                'trajectory_ids': [f['trajectory_id'] for f in stuck],
            },
            'drove_but_timeout': {
                'count': len(drove_but_timeout),
                'avg_distance': sum(f['distance'] for f in drove_but_timeout) / max(len(drove_but_timeout), 1),
                'trajectory_ids': [f['trajectory_id'] for f in drove_but_timeout],
            },
        },
        'success_details': [
            {'trajectory_id': s['trajectory_id'], 'spl': s['spl'], 'distance': s['distance']}
            for s in successes
        ],
    }

    output_file = os.path.join(FAILURE_DIR, f"{town}_dense_sptm_failure_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  {town}: {len(successes)}/{len(per_traj)} successes")
    print(f"    Stuck at start (<10m): {len(stuck)}")
    print(f"    Drove but timed out:   {len(drove_but_timeout)}")
    print(f"    Saved: {output_file}")
    return summary

print("Analyzing SPTM failure modes for both towns...")
print("")

for town in ['town02', 'town05']:
    analyze_failure_modes(town)
    print("")

print("Failure mode analysis complete!")
PYEOF

echo ""
echo "========================================================================"
echo "All done! Results saved to:"
echo "  - $RESULTS_DIR/"
echo "  - $FAILURE_DIR/"
echo "========================================================================"

#!/usr/bin/env python
"""
Benchmark Evaluation Script
Evaluates NoMaD, GNM, and ViNT on all trajectories in benchmark data.
Computes success rate, SPL, and other metrics for comparison.
"""

from collections import defaultdict
import os
import pickle
import json
import signal
import time
import numpy as np
from datetime import datetime

from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
import psutil
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.helper import ndarray_to_location
from rlib_integration.carla_env import CarlaEnv
from navigation_policies.baseline_policies.nomad_policy import NoMaD
from navigation_policies.baseline_policies.gnm_policy import GNM_Policy
from navigation_policies.baseline_policies.vint_policy import ViNT_Policy
from src.agent_wrapper import SetPointAgent
from src.configs.eval_config_v2 import config

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK_DIR = "./corl-2026-experiments/trajectories"
RESULTS_DIR = "./corl-2026-baselines/dist_func_pruning"
MAP_DIR="./corl-2026-experiments/maps"
TOWN_NAME = "Town02"
MAX_EPISODE_STEPS = 2000
TERMINATION_BUDGET = 2  # Max retries per trajectory
SKIP_INDEX = 1  # Frame skip for topological map
TEST_HEURISTIC_SPARSIFICATION=False
# Model checkpoints
MODEL_CHECKPOINTS = {
    "gnm": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/gnm.pth",
    "nomad": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/nomad.pth",
    "vint": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/vint.pth"
}

MODEL_CLASSES = {
    "gnm": GNM_Policy,
    "nomad": NoMaD,
    "vint": ViNT_Policy
}

# Models to evaluate
MODELS_TO_EVALUATE = ["nomad","vint","gnm"]

os.environ['XLA_FLAGS'] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# ============================================================================


def load_benchmark_info(benchmark_dir):
    """Load benchmark information."""
    benchmark_path = os.path.join(benchmark_dir, "benchmark_info.json")
    with open(benchmark_path, "r") as f:
        return json.load(f)



def evaluate_single_trajectory(env, model_type, trajectory_path, traj_id,skip_index=SKIP_INDEX,town="Town02"):
    """Evaluate a single model on a single trajectory."""
    
    # Load trajectory data
    is_map_empty = False
    shortest_path_oracle = []
    path_of_agent = []
    results={}
    threshold = 0.0
    episode_counts=0
    aux_path = os.path.join(trajectory_path, "aux.pkl")
    with open(aux_path, "rb") as f:
        aux_data = pickle.load(f)
        # print(aux_data)
    
    start_location = ndarray_to_location(aux_data["start"])
    goal_location = ndarray_to_location(aux_data["goal"])
    
    # Set start and goal
    env.unwrapped.set_start_transform(start_location)
    env.unwrapped.set_destination_transform(goal_location)
    unwrapped_env=env.unwrapped
    # Compute shortest path distance
    route_planner = GlobalRoutePlanner(env.unwrapped.core.map, 2.0)
    trace = route_planner.trace_route(start_location, goal_location)
    
    shortest_distance = 0
    prev_waypoint = None
    for wp, _ in trace:
        if prev_waypoint is not None:
            shortest_distance += prev_waypoint.transform.location.distance(wp.transform.location)
        loc_obj = wp.transform.location 
        shortest_path_oracle.append((loc_obj.x, loc_obj.y, loc_obj.z))
        prev_waypoint = wp
    
    if shortest_distance <= 2.0:
        shortest_distance = start_location.distance(goal_location)

    while not is_map_empty and episode_counts<20:
        # Initialize model
        
        MODEL = MODEL_CLASSES[model_type]
        checkpoint_path = MODEL_CHECKPOINTS[model_type]
        similarity_matrix=None
        # if TEST_HEURISTIC_SPARSIFICATION:
        similarity_matrix= f"cache/{model_type}/distance_matrix_{traj_id}.pth"
            
            
        map_dir=f"{MAP_DIR}\{town.lower()}"
        agent = MODEL(
            ckpt_path=checkpoint_path,
            mode="navigate",
            skip_index=skip_index,
            map_dir=map_dir,
            similarity_matrix=similarity_matrix,
            use_similarity=True,
            traj_id=traj_id,
            threshold=trajectory_data[model_type][traj_id]["threshold"],
            # explore=False
        )
        

        

        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        success_rate = []
        SPL = []
        distance_completed = []
        termination_budget = TERMINATION_BUDGET
        sp_agent = SetPointAgent(env.unwrapped.core.hero)
        nav_time=[]
        while termination_budget > 0:
            observation, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            
            while not done:
                t0 = time.perf_counter()
                waypoints = np.array(agent.eval_action(env.unwrapped.experiment.obs_img[...,None]))
                t1=time.perf_counter()
                nav_time.append(t1-t0)
                actions = sp_agent.run_step(waypoints,False)
                # print(actions)
                loc_obj = unwrapped_env.core.hero.get_transform().location 
                current_location = (loc_obj.x, loc_obj.y, loc_obj.z)
                path_of_agent .append(current_location)
                _, reward, done, truncated, info = env.step(actions)
                
                episode_reward += reward
                episode_length += 1
                done = done or truncated
            
            # Record metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            

            # # Check termination
            if info.get("is_success", False):
                termination_budget = 0
            else:
                termination_budget -= 1
                env.unwrapped.set_start_transform(env.unwrapped.experiment.wp.transform.location)
            if termination_budget <=0:
                if "is_success" in info:
                    success_rate.append(float(info["is_success"]))
                else:
                    success_rate.append(0.0)
                
                if "distance_travelled" in info:
                    agent_distance = float(info["distance_travelled"])
                    distance_completed.append(agent_distance)
                    
                    # Compute SPL
                    S = int(info.get("is_success", 0))
                    spl = S * (shortest_distance) / max(shortest_distance, agent_distance)
                    SPL.append(spl)
            
            # is_map_empty = len(set(agent.path)) == 2
            # is_map_empty
            # breakpoint()
            print("AGENT",print(),len(aux_data["locations"]))
            # Compute statistics
            demo_results = {
                "success": np.max(success_rate) if success_rate else 0.0,
                "success_rate": np.mean(success_rate) if success_rate else 0.0,
                "mean_spl": np.mean(SPL) if SPL else 0.0,
                "max_spl": np.max(SPL) if SPL else 0.0,
                "mean_reward": np.mean(episode_rewards),
                "mean_length": np.mean(episode_lengths),
                "mean_distance": np.mean(distance_completed) if distance_completed else 0.0,
                "shortest_distance": shortest_distance,
                "num_nodes": len(set(agent.utilized_nodes)),
                "filtered_nodes": len(set(agent.filtered_nodes)),
                "shortest_path":len(agent.path),
                "num_attempts": TERMINATION_BUDGET - termination_budget,
                "path_of_agent":path_of_agent,
                "map_utilized":agent.path,
                "dense_node_location":aux_data["locations"],
                "sparse_node_location":[aux_data["locations"][i] for i in agent.path],
                "shortest_path_agent":shortest_path_oracle
            }
            # breakpoint()
            print(f"Using Threshold {threshold}  - Nodes utilized {len(set(agent.utilized_nodes))} - Filtered Nodes {len(set(agent.filtered_nodes))}")
            if not TEST_HEURISTIC_SPARSIFICATION:
                is_map_empty = True
                results = demo_results
            else:
                results[threshold]=demo_results
            # print()
            threshold += 1.0
            episode_counts+=1
    print("Perf compt time",np.mean(nav_time))
    return results


def evaluate_model_on_benchmark(env, model_type, benchmark_info,town="Town02"):
    """Evaluate a single model on all benchmark trajectories."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()}")
    print(f"{'='*60}")
    
    all_results = []
    
    for traj_info in benchmark_info["trajectories"]:
        traj_id = traj_info["trajectory_id"]
        traj_path = traj_info["path"]
        
        print(f"\n  Trajectory {traj_id}: {traj_info['route_length']:.1f}m, {traj_info['num_images']} images")
        # breakpoint()
        # nodes_to_utilize=benchmark_info["nodes_to_utilize"][str(traj_id)]
        # try:
        results = evaluate_single_trajectory(env, model_type, traj_path,traj_id,town)
        
        # Handle different result structures
        if TEST_HEURISTIC_SPARSIFICATION:
            # When testing heuristic sparsification, results is a dict of {threshold: metrics}
            # We'll store all thresholds but also pick the best one for aggregate stats
            best_threshold = max(results.keys(), key=lambda t: results[t].get("success", 0))
            best_results = results[best_threshold].copy()
            best_results["trajectory_id"] = traj_id
            best_results["route_length"] = traj_info["route_length"]
            best_results["all_thresholds"] = results  # Keep all threshold results
            best_results["best_threshold"] = best_threshold
            all_results.append(best_results)
            
            # Print results for all thresholds
            print(f"\n  Results by threshold:")
            for thresh in sorted(results.keys()):
                print(f"    Threshold {thresh:.1f}: Success={results[thresh]['success']:.2f}, "
                      f"SPL={results[thresh]['mean_spl']:.3f}, "
                      f"Nodes={results[thresh]['num_nodes']}, "
                      f"Filtered={results[thresh]['filtered_nodes']}")
            print(f"  Best threshold: {best_threshold:.1f}")
        else:
            results["trajectory_id"] = traj_id
            results["route_length"] = traj_info["route_length"]
            all_results.append(results)
            print(f"  Success={results['success']:.2f}, SPL={results['mean_spl']:.3f}")
        
        # except Exception as e:
        #     print(f"    ERROR: {e}")
        #     all_results.append({
        #         "trajectory_id": traj_id,
        #         "success": 0.0,
        #         "error": str(e)
        #     })
    
    # Aggregate statistics
    successful_results = [r for r in all_results if "error" not in r]
    
    aggregate = {
        "model": model_type,
        "num_trajectories": len(benchmark_info["trajectories"]),
        "num_successful_evals": len(successful_results),
        "overall_success_rate": np.mean([r["success"] for r in successful_results]) if successful_results else 0.0,
        "overall_mean_spl": np.mean([r["mean_spl"] for r in successful_results]) if successful_results else 0.0,
        "overall_mean_distance": np.mean([r["mean_distance"] for r in successful_results]) if successful_results else 0.0,
        "per_trajectory_results": all_results
    }
    
    return aggregate


def run_full_benchmark(benchmark_dir, models_to_evaluate, results_dir,town):
    """Run full benchmark evaluation on all models."""
      
    # Load benchmark info
    benchmark_info = load_benchmark_info(benchmark_dir)
    
    print(f"\n{'#'*60}")
    print(f"BENCHMARK EVALUATION")
    print(f"{'#'*60}")
    print(f"Benchmark: {benchmark_dir}")
    print(f"Town: {benchmark_info['town']}")
    print(f"Trajectories: {benchmark_info['num_trajectories']}")
    print(f"Models: {models_to_evaluate}")
    
    # Setup environment
    config["env_config"]["carla"]["town"] = town
    config["env_config"]["carla"]["start_server"] = True
    env = CarlaEnv(config=config)
    env.experiment.eval_mode = True
    env.experiment.map_mode = False
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = RecordEpisodeStatistics(env)
    
    # Evaluate each model
    all_model_results = {}
    
    for model_type in models_to_evaluate:
        # try:
            results = evaluate_model_on_benchmark(env, model_type, benchmark_info,town)
            all_model_results[model_type] = results
            # Save results
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save full results
            results_path = os.path.join(results_dir, f"benchmark_results_{timestamp}.pkl")
            with open(results_path, "wb") as f:
                pickle.dump(all_model_results, f)
            
        # # except Exception as e:
        #     print(f"\nERROR evaluating {model_type}: {e}")
        #     all_model_results[model_type] = {"error": str(e)}
    
    # Close environment
    env.close()
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    results_path = os.path.join(results_dir, f"benchmark_results_{timestamp}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(all_model_results, f)
    
    # Save summary JSON
    summary = {
        "timestamp": timestamp,
        "benchmark_dir": benchmark_dir,
        "town": benchmark_info["town"],
        "num_trajectories": benchmark_info["num_trajectories"],
        "models_evaluated": models_to_evaluate,
        "results": {}
    }
    
    for model, results in all_model_results.items():
        if "error" not in results:
            summary["results"][model] = {
                "success_rate": results["overall_success_rate"],
                "mean_spl": results["overall_mean_spl"],
                "mean_distance": results["overall_mean_distance"]
            }
        else:
            summary["results"][model] = {"error": results["error"]}
    
    summary_path = os.path.join(results_dir, f"benchmark_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print_benchmark_summary(all_model_results)
    
    print(f"\nResults saved to: {results_dir}")
    
    return all_model_results


def print_benchmark_summary(all_model_results):
    """Print formatted benchmark summary."""
    
    print(f"\n{'#'*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'#'*60}")
    
    # Table header
    print(f"\n{'Model':<10} {'Success Rate':<15} {'Mean SPL':<12} {'Mean Distance':<15}")
    print("-" * 55)
    
    for model, results in all_model_results.items():
        if "error" in results:
            print(f"{model:<10} ERROR: {results['error']}")
        else:
            print(f"{model:<10} {results['overall_success_rate']*100:>10.1f}% "
                  f"{results['overall_mean_spl']:>12.3f} "
                  f"{results['overall_mean_distance']:>12.1f}m")
    
    print("-" * 55)
    
    # Per-trajectory breakdown
    print(f"\nPer-Trajectory Results:")
    print(f"{'Traj':<6}", end="")
    for model in all_model_results.keys():
        print(f"{model:<12}", end="")
    print()
    print("-" * (6 + 12 * len(all_model_results)))
    
    # Get trajectory count from first valid model
    num_trajectories = 0
    for results in all_model_results.values():
        if "per_trajectory_results" in results:
            num_trajectories = len(results["per_trajectory_results"])
            break
    
    for traj_idx in range(num_trajectories):
        print(f"{traj_idx:<6}", end="")
        for model, results in all_model_results.items():
            if "per_trajectory_results" in results and traj_idx < len(results["per_trajectory_results"]):
                traj_result = results["per_trajectory_results"][traj_idx]
                if "error" in traj_result:
                    print(f"{'ERR':<12}", end="")
                else:
                    success = "✓" if traj_result["success"] > 0 else "✗"
                    spl = traj_result["mean_spl"]
                    print(f"{success} {spl:.2f}      ", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()


def main():
    """Main entry point."""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Run full benchmark
    for town in ["Town02","Town05"]:
        results = run_full_benchmark(
            benchmark_dir=BENCHMARK_DIR,
            models_to_evaluate=MODELS_TO_EVALUATE,
            results_dir=RESULTS_DIR,
            town=town
        )
        
    return results


if __name__ == "__main__":
    main()

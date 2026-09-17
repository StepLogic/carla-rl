#!/usr/bin/env python
"""
Trajectory Video Generator with Policy Distributions

This script loads evaluation results and creates an animated video showing:
- Agent's trajectory through the environment
- Policy distributions (lane, goal, combined) at each timestep
- Junction highlighting
- Real-time statistics
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from scipy.stats import multivariate_normal
import argparse
from pathlib import Path
from collections import defaultdict


class TrajectoryVideoGenerator:
    """Generate animated video of agent trajectory with policy distributions."""
    
    def __init__(self, pickle_path, output_dir="./trajectory_videos"):
        """
        Initialize the generator.
        
        Args:
            pickle_path: Path to evaluation results pickle file
            output_dir: Directory to save output videos
        """
        self.pickle_path = pickle_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"Loading data from: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        print(f"Loaded {self.stats['total_episodes']} episodes")
        print(f"Success rate: {self.stats['success_rate']:.1f}%")
    
    def get_episode_data(self, episode_id):
        """Extract episode data from stats."""
        # Search in successful and failed episodes
        all_episodes = (self.stats['successful_episode_list'] + 
                       self.stats['failed_episode_list'])
        
        for ep in all_episodes:
            if ep['episode'] == episode_id:
                # breakpoint()
                return ep
        
        raise ValueError(f"Episode {episode_id} not found in results")
    
    def extract_trajectory_positions(self, trajectory):
        """
        Extract x, y positions from location strings.
        
        Args:
            trajectory: Episode trajectory dict
            
        Returns:
            positions: List of (x, y) tuples
        """
        positions = []
        for loc_str in trajectory['locations']:
            # Parse location string like "(123.4, 567.8)"
            try:
                loc_str = loc_str.strip('()')
                x, y = map(float, loc_str.split(','))
                positions.append((x, y))
            except (ValueError, AttributeError):
                # If parsing fails, use (0, 0) or previous position
                if positions:
                    positions.append(positions[-1])
                else:
                    positions.append((0, 0))
        
        return np.array(positions)
    
    def compute_policy_pdf(self, samples, grid_size=100):
        """
        Compute 2D PDF for policy distribution.
        
        Args:
            samples: Array of actions (N, 2) - [steer, throttle]
            grid_size: Resolution of PDF grid
            
        Returns:
            X, Y, Z: Meshgrid coordinates and PDF values
        """
        if len(samples) < 2:
            return None, None, None
        
        # Estimate distribution parameters
        mu = np.mean(samples, axis=0)
        cov = np.cov(samples.T) + np.eye(2) * 1e-6
        
        # Create grid
        steer = np.linspace(-1, 1, grid_size)
        throttle = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(steer, throttle)
        pos = np.dstack((X, Y))
        
        # Compute PDF
        pdf = multivariate_normal(mu, cov).pdf(pos)
        
        return X, Y, pdf
    
    def create_episode_video(self, episode_id, fps=10, dpi=100, 
                            skip_frames=1, show_all_distributions=True):
        """
        Create animated video for a single episode.
        
        Args:
            episode_id: Episode number to visualize
            fps: Frames per second for output video
            dpi: Resolution of output video
            skip_frames: Only render every Nth frame (for faster processing)
            show_all_distributions: Show lane/goal/combined or just combined
        """
        print(f"\nGenerating video for Episode {episode_id}...")
        
        # Get episode data
        ep_data = self.get_episode_data(episode_id)
        trajectory = ep_data['trajectory']
        is_success = ep_data['success']
        # breakpoint()
        # goal_location= ep_data['goal_location'] 
        # Extract trajectory positions
        positions = self.extract_trajectory_positions(trajectory)
        
        # Extract goal location (relative to start position)
        goal_location_relative = trajectory.get('goal_location', None)
        if goal_location_relative is not None:
            # Convert relative goal to absolute position
            # Goal is relative to first position
            # start_pos = positions[0]
            goal_pos = np.array(goal_location_relative[:2])  # Use x, y only
            # breakpoint()
        else:
            goal_pos = None
        
        # Extract actions
        lane_actions = np.array(trajectory['lane_actions'])
        goal_actions = np.array(trajectory['goal_actions'])
        combined_actions = np.array(trajectory['combined_actions'])
        
        # Extract other data
        rewards = np.array(trajectory['rewards'])
        is_junction = trajectory['is_junction']
        
        # Setup figure
        if show_all_distributions:
            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            # Trajectory plot (spans 2 rows, 2 columns)
            ax_traj = fig.add_subplot(gs[:, :2])
            
            # Policy distribution plots
            ax_lane = fig.add_subplot(gs[0, 2])
            ax_goal = fig.add_subplot(gs[0, 3])
            ax_combined = fig.add_subplot(gs[1, 2])
            
            # Statistics
            ax_stats = fig.add_subplot(gs[1, 3])
            ax_stats.axis('off')
            
            policy_axes = [ax_lane, ax_goal, ax_combined]
            policy_titles = ['Lane Policy', 'Goal Policy', 'Combined Policy']
            policy_actions = [lane_actions, goal_actions, combined_actions]
        else:
            fig = plt.figure(figsize=(16, 8))
            gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
            
            # Trajectory plot
            ax_traj = fig.add_subplot(gs[0, :2])
            
            # Combined policy distribution
            ax_combined = fig.add_subplot(gs[0, 2])
            
            policy_axes = [ax_combined]
            policy_titles = ['Combined Policy']
            policy_actions = [combined_actions]
            
            # Add stats as text
            ax_stats = None
        
        # Status color
        status = "SUCCESS" if is_success else "FAILURE"
        status_color = 'green' if is_success else 'red'
        
        fig.suptitle(f'Episode {episode_id} - {status}', 
                    fontsize=16, fontweight='bold', color=status_color)
        
        # Pre-compute all PDFs for efficiency
        print("  Pre-computing policy distributions...")
        all_pdfs = []
        for i in range(0, len(combined_actions), skip_frames):
            step_pdfs = []
            for actions in policy_actions:
                # Use a window of actions around current timestep
                window_start = max(0, i - 10)
                window_end = min(len(actions), i + 10)
                action_window = actions[window_start:window_end]
                
                X, Y, Z = self.compute_policy_pdf(action_window)
                step_pdfs.append((X, Y, Z))
            all_pdfs.append(step_pdfs)
        
        def init():
            """Initialize animation."""
            ax_traj.clear()
            for ax in policy_axes:
                ax.clear()
            return []
        
        def update(frame_idx):
            """Update function for animation."""
            actual_step = frame_idx * skip_frames
            if actual_step >= len(positions):
                actual_step = len(positions) - 1
            
            # Clear axes
            ax_traj.clear()
            for ax in policy_axes:
                ax.clear()
            
            # =============================================
            # TRAJECTORY PLOT
            # =============================================
            
            # Plot full trajectory (faded)
            ax_traj.plot(positions[:, 0], positions[:, 1], 
                        'b-', alpha=0.3, linewidth=1, label='Full Trajectory')
            
            # Plot trajectory up to current point
            ax_traj.plot(positions[:actual_step+1, 0], 
                        positions[:actual_step+1, 1],
                        'b-', linewidth=2, label='Completed')
            
            # Current position
            ax_traj.plot(positions[actual_step, 0], 
                        positions[actual_step, 1],
                        'ro', markersize=15, label='Current Position', zorder=5)
            
            # Highlight junctions
            for i in range(min(actual_step + 1, len(is_junction))):
                if is_junction[i]:
                    circle = plt.Circle((positions[i, 0], positions[i, 1]), 
                                       radius=5, color='orange', alpha=0.3, zorder=3)
                    ax_traj.add_patch(circle)
            
            # Start and end markers
            ax_traj.plot(positions[0, 0], positions[0, 1], 
                        'gs', markersize=12, label='Start', zorder=4)
            if is_success:
                ax_traj.plot(positions[-1, 0], positions[-1, 1],
                           'g^', markersize=12, label='Goal', zorder=4)
            ax_traj.plot(goal_pos[0], goal_pos[1], 
                        'rs', markersize=12, label='Real Goal', zorder=4)
            ax_traj.set_xlabel('X Position (m)', fontsize=11)
            ax_traj.set_ylabel('Y Position (m)', fontsize=11)
            ax_traj.set_title(f'Agent Trajectory - Step {actual_step}/{len(positions)-1}',
                            fontsize=12, fontweight='bold')
            ax_traj.legend(loc='upper right', fontsize=9)
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_aspect('equal', adjustable='datalim')
            
            # =============================================
            # POLICY DISTRIBUTION PLOTS
            # =============================================
            
            pdf_data = all_pdfs[frame_idx]
            
            for ax, title, (X, Y, Z), actions in zip(
                policy_axes, policy_titles, pdf_data, policy_actions):
                
                if X is None:
                    ax.text(0.5, 0.5, 'Insufficient Data', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot PDF contour
                contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
                ax.contour(X, Y, Z, levels=8, colors='white', 
                          alpha=0.3, linewidths=0.5)
                
                # Plot current action
                current_action = actions[actual_step]
                ax.plot(current_action[0], current_action[1], 
                       'r*', markersize=20, label='Current Action', zorder=5)
                
                # Plot mean
                window_start = max(0, actual_step - 10)
                window_end = min(len(actions), actual_step + 10)
                window_mean = np.mean(actions[window_start:window_end], axis=0)
                ax.plot(window_mean[0], window_mean[1],
                       'wo', markersize=10, label='Window Mean', zorder=4)
                
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Steer', fontsize=10)
                ax.set_ylabel('Throttle', fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)
            
            # =============================================
            # STATISTICS TEXT
            # =============================================
            
            if show_all_distributions and ax_stats is not None:
                ax_stats.clear()
                ax_stats.axis('off')
                
                # Compute cumulative reward
                cumulative_reward = np.sum(rewards[:actual_step+1])
                
                # Current action values
                current_lane = lane_actions[actual_step]
                current_goal = goal_actions[actual_step]
                current_combined = combined_actions[actual_step]
                
                # Junction status
                at_junction = "YES" if is_junction[actual_step] else "NO"
                junction_color = 'orange' if is_junction[actual_step] else 'green'
                
                stats_text = f"""
EPISODE STATISTICS

Episode: {episode_id}
Status: {status}
Step: {actual_step} / {len(positions)-1}

Position:
  X: {positions[actual_step, 0]:.1f} m
  Y: {positions[actual_step, 1]:.1f} m

At Junction: {at_junction}

Current Actions:
  Lane:     [{current_lane[0]:6.3f}, {current_lane[1]:6.3f}]
  Goal:     [{current_goal[0]:6.3f}, {current_goal[1]:6.3f}]
  Combined: [{current_combined[0]:6.3f}, {current_combined[1]:6.3f}]

Rewards:
  Current:    {rewards[actual_step]:7.3f}
  Cumulative: {cumulative_reward:7.3f}
  
Distance: {ep_data['distance']:.2f} m
                """
                
                ax_stats.text(0.05, 0.95, stats_text, 
                            transform=ax_stats.transAxes,
                            fontsize=10, verticalalignment='top',
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            return []
        
        # Create animation
        print(f"  Creating animation...")
        n_frames = len(all_pdfs)
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=n_frames, interval=1000/fps,
                           blit=False, repeat=True)
        
        # Save video
        video_filename = f"episode_{episode_id}_{'success' if is_success else 'fail'}.mp4"
        video_path = self.output_dir / video_filename
        
        print(f"  Saving video to: {video_path}")
        writer = FFMpegWriter(fps=fps, bitrate=5000)
        anim.save(str(video_path), writer=writer, dpi=dpi)
        
        plt.close(fig)
        print(f"  ✓ Video saved successfully!")
        
        return video_path
    
    def create_summary_comparison(self, success_episode_id=None, 
                                  fail_episode_id=None, output_name="comparison.png"):
        """
        Create a static comparison plot between successful and failed episodes.
        
        Args:
            success_episode_id: ID of successful episode (or None for first successful)
            fail_episode_id: ID of failed episode (or None for first failed)
            output_name: Output filename
        """
        print("\nCreating summary comparison plot...")
        
        # Get episodes
        if success_episode_id is None and self.stats['successful_episode_list']:
            success_ep = self.stats['successful_episode_list'][0]
        else:
            success_ep = self.get_episode_data(success_episode_id)
        
        if fail_episode_id is None and self.stats['failed_episode_list']:
            fail_ep = self.stats['failed_episode_list'][0]
        else:
            fail_ep = self.get_episode_data(fail_episode_id)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Episode Comparison: Success vs Failure', 
                    fontsize=16, fontweight='bold')
        
        for idx, (ep, title_prefix, color) in enumerate([
            (success_ep, 'SUCCESS', 'green'),
            (fail_ep, 'FAILURE', 'red')
        ]):
            row = idx
            
            # Trajectory
            ax_traj = axes[row, 0]
            positions = self.extract_trajectory_positions(ep['trajectory'])
            
            ax_traj.plot(positions[:, 0], positions[:, 1], 
                        color=color, linewidth=2, alpha=0.7)
            ax_traj.plot(positions[0, 0], positions[0, 1],
                        'go', markersize=12, label='Start')
            if ep['success']:
                ax_traj.plot(positions[-1, 0], positions[-1, 1],
                           'g^', markersize=12, label='Goal')
            else:
                ax_traj.plot(positions[-1, 0], positions[-1, 1],
                           'rx', markersize=12, label='Stopped')
            
            # Highlight junctions
            for i, is_junc in enumerate(ep['trajectory']['is_junction']):
                if is_junc:
                    circle = plt.Circle((positions[i, 0], positions[i, 1]),
                                       radius=5, color='orange', alpha=0.2)
                    ax_traj.add_patch(circle)
            
            ax_traj.set_title(f'{title_prefix} - Episode {ep["episode"]} Trajectory',
                            fontweight='bold')
            ax_traj.set_xlabel('X Position (m)')
            ax_traj.set_ylabel('Y Position (m)')
            ax_traj.legend()
            ax_traj.grid(True, alpha=0.3)
            ax_traj.set_aspect('equal', adjustable='datalim')
            
            # Combined policy distribution (overall)
            ax_dist = axes[row, 1]
            combined_actions = np.array(ep['trajectory']['combined_actions'])
            
            X, Y, Z = self.compute_policy_pdf(combined_actions, grid_size=150)
            if X is not None:
                contour = ax_dist.contourf(X, Y, Z, levels=20, cmap='viridis')
                ax_dist.contour(X, Y, Z, levels=10, colors='white',
                              alpha=0.3, linewidths=0.5)
                plt.colorbar(contour, ax=ax_dist, label='Probability Density')
                
                # Plot mean
                mean_action = np.mean(combined_actions, axis=0)
                ax_dist.plot(mean_action[0], mean_action[1],
                           'r*', markersize=20, label='Mean')
            
            ax_dist.set_xlim(-1, 1)
            ax_dist.set_ylim(0, 1)
            ax_dist.set_title(f'{title_prefix} - Combined Policy Distribution',
                            fontweight='bold')
            ax_dist.set_xlabel('Steer')
            ax_dist.set_ylabel('Throttle')
            ax_dist.legend()
            ax_dist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved comparison plot: {output_path}")
        plt.close()
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate trajectory videos from evaluation results')
    parser.add_argument('--pickle_file', type=str,default="/home/robotlab/scratch/carla-rl/eval_results_combined/evaluation_results_20251117_022825.pkl",
                       help='Path to evaluation results pickle file')
    parser.add_argument('--episodes', type=int, nargs='+', default=None,
                       help='Specific episode IDs to visualize (default: first success and first failure)')
    parser.add_argument('--output_dir', type=str, default='./trajectory_videos',
                       help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for output video')
    parser.add_argument('--dpi', type=int, default=100,
                       help='Resolution (DPI) of output video')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Only render every Nth frame (for faster processing)')
    parser.add_argument('--simple', action='store_true',
                       help='Show only combined policy (faster rendering)')
    parser.add_argument('--comparison', action='store_true',
                       help='Generate summary comparison plot')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TrajectoryVideoGenerator(args.pickle_file, args.output_dir)
    
    # Determine episodes to visualize
    if args.episodes:
        episodes_to_viz = args.episodes
    else:
        # Default: first successful and first failed episode
        episodes_to_viz = []
        if generator.stats['successful_episode_list']:
            episodes_to_viz.append(
                generator.stats['successful_episode_list'][4]['episode'])
        if generator.stats['failed_episode_list']:
            episodes_to_viz.append(
                generator.stats['failed_episode_list'][0]['episode'])
    
    print(f"\nGenerating videos for episodes: {episodes_to_viz}")
    
    # Generate videos
    for ep_id in episodes_to_viz:
        try:
            generator.create_episode_video(
                ep_id,
                fps=args.fps,
                dpi=args.dpi,
                skip_frames=args.skip_frames,
                show_all_distributions=not args.simple
            )
        except Exception as e:
            print(f"  ✗ Error generating video for episode {ep_id}: {e}")
    
    # Generate comparison plot
    if args.comparison:
        try:
            generator.create_summary_comparison()
        except Exception as e:
            print(f"  ✗ Error generating comparison plot: {e}")
    
    print(f"\n{'='*80}")
    print(f"All videos saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
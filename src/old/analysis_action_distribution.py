import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def load_datasets(dataset_folder="datasets", town_filter=None):
    """
    Load all pickle files from the dataset folder, with optional town filtering.
    
    Args:
        dataset_folder (str): Path to the folder containing dataset files
        town_filter (str, optional): Only load datasets from this town
        
    Returns:
        list: List of loaded replay buffers
    """
    pattern = os.path.join(dataset_folder, "*.pkl")
    if town_filter:
        pattern = os.path.join(dataset_folder, f"goal_condition_{town_filter}_data_*.pkl")
    
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No dataset files found matching pattern: {pattern}")
    
    print(f"Found {len(files)} dataset files")
    
    datasets = []
    for file_path in tqdm(files, desc="Loading datasets"):
        try:
            with open(file_path, "rb") as f:
                replay_buffer = pickle.load(f)
                datasets.append(replay_buffer)
                print(f"Loaded {file_path} with {replay_buffer._size} samples")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return datasets

def extract_steering_angles(datasets):
    """
    Extract steering angles from a list of replay buffers.
    
    Args:
        datasets (list): List of replay buffer objects
        
    Returns:
        numpy.ndarray: Array of steering angle values
    """
    all_steering_angles = []
    
    for replay_buffer in datasets:
        # Handle different types of replay buffers
        if hasattr(replay_buffer, 'dataset_dict'):
            # For ReplayBuffer implementation with dataset_dict
            if isinstance(replay_buffer.dataset_dict['actions'], np.ndarray):
                # Standard ReplayBuffer
                actions = replay_buffer.dataset_dict['actions'][:replay_buffer._size]
            elif isinstance(replay_buffer.dataset_dict['actions'], list):
                # VariableCapacityBuffer
                actions = np.array(replay_buffer.dataset_dict['actions'])
            else:
                raise ValueError(f"Unknown actions type: {type(replay_buffer.dataset_dict['actions'])}")
        elif hasattr(replay_buffer, '_storage'):
            # For older ReplayBuffer implementation with _storage
            actions = replay_buffer._storage["actions"][:replay_buffer._size]
        else:
            raise ValueError(f"Unknown replay buffer type: {type(replay_buffer)}")
            
        steering_angles = actions[:, 0]  # First column contains steering angles
        all_steering_angles.append(steering_angles)
    
    return np.concatenate(all_steering_angles)

def plot_steering_distribution(steering_angles, bins=50, save_path=None):
    """
    Create and display a histogram of steering angle distribution.
    
    Args:
        steering_angles (numpy.ndarray): Array of steering angle values
        bins (int): Number of histogram bins
        save_path (str, optional): Path to save the plot image
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate statistics
    mean = np.mean(steering_angles)
    median = np.median(steering_angles)
    std_dev = np.std(steering_angles)
    min_val = np.min(steering_angles)
    max_val = np.max(steering_angles)
    
    # Plot histogram
    n, bins, patches = plt.hist(steering_angles, bins=bins, alpha=0.7, color='steelblue')
    
    # Add a line for the mean
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.4f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.4f}')
    
    # Add titles and labels
    plt.title('Distribution of Steering Angles', fontsize=16)
    plt.xlabel('Steering Angle', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # Add text with statistics
    stats_text = f"Statistics:\n"
    stats_text += f"Mean: {mean:.4f}\n"
    stats_text += f"Median: {median:.4f}\n"
    stats_text += f"Std Dev: {std_dev:.4f}\n"
    stats_text += f"Min: {min_val:.4f}\n"
    stats_text += f"Max: {max_val:.4f}\n"
    stats_text += f"Total Samples: {len(steering_angles)}"
    
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='top', fontsize=10)
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def analyze_steering_angles_by_scenario(datasets):
    """
    Analyze steering angles in different driving scenarios.
    
    Args:
        datasets (list): List of replay buffer objects
    """
    # This would require additional information or labeling in the dataset
    # For now, let's create a placeholder for future implementation
    print("Scenario-based analysis would require additional labeling in the dataset.")
    print("This could be implemented by adding scenario labels during data collection.")

def main():
    parser = argparse.ArgumentParser(description='Analyze steering angle distribution from collected datasets')
    parser.add_argument('--dataset_folder', default='datasets', help='Folder containing dataset files')
    parser.add_argument('--town', help='Filter datasets by town name')
    parser.add_argument('--bins', type=int, default=50, help='Number of histogram bins')
    parser.add_argument('--save_path', help='Path to save the histogram image')
    
    args = parser.parse_args()
    
    try:
        # Load datasets
        datasets = load_datasets(args.dataset_folder, args.town)
        
        # Extract steering angles
        steering_angles = extract_steering_angles(datasets)
        
        # Plot distribution
        plot_steering_distribution(steering_angles, bins=args.bins, save_path=args.save_path)
        
        # Additional analysis could be performed here
        print(f"\nAnalyzed {len(steering_angles)} steering angle samples")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
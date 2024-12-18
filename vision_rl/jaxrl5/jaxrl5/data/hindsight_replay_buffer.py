from collections import deque
import copy
import numpy as np
from typing import Optional, Iterable, Callable, Dict, Any
from enum import Enum
from flax.core import frozen_dict
import gymnasium as gym

from jaxrl5.data.dataset import DatasetDict
from jaxrl5.data.replay_buffer import ReplayBuffer, _sample

class GoalSelectionStrategy(Enum):
    """Goal selection strategies for HER."""
    FINAL = "final"
    FUTURE = "future"
    EPISODE = "episode"

class HindsightReplayBuffer(ReplayBuffer):
    """
    Hindsight Experience Replay buffer with support for mixed scalar and array info values.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        goal_selection_strategy: str = "final",
        n_sampled_goals: int = 4,
        success_threshold: float = 0.0,
        relabel_fn: Optional[Callable[[DatasetDict], DatasetDict]] = None,
        info_keys: Optional[list[str]] = None,
        info_shapes: Optional[Dict[str, tuple]] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            relabel_fn=relabel_fn,
        )
        
        self.goal_selection_strategy = goal_selection_strategy
        self.n_sampled_goals = n_sampled_goals
        self.success_threshold = success_threshold
        self.her_ratio = 1 - (1.0 / (n_sampled_goals + 1))
        
        # Initialize episode tracking arrays
        self._current_episode_steps = 0
        self._episode_lengths = np.zeros(capacity, dtype=np.int32)
        
        # Initialize info storage with proper shapes
        self._info_keys = info_keys or []
        self._info_shapes = info_shapes or {}
        if self._info_keys:
            self.dataset_dict['infos'] = {
                key: deque(maxlen=capacity)
                for key in self._info_keys
            }

    def insert(self, data_dict: DatasetDict):
        """Insert a transition with proper handling of scalar and array info."""
        # Handle info values
        if 'infos' in data_dict and self._info_keys:
            info_dict = data_dict['infos']
            if isinstance(info_dict, dict):
                for key in self._info_keys:
                    if key in info_dict:
                        value = info_dict[key]
                        # Convert scalar to numpy array if needed
                        if np.isscalar(value):
                            value = np.array([value], dtype=np.float32)
                        # Ensure correct shape for array values
                        elif isinstance(value, (list, np.ndarray)):
                            value = np.asarray(value, dtype=np.float32)
                            expected_shape = self._info_shapes.get(key, (1,))
                            if value.shape != expected_shape:
                                value = value.reshape(expected_shape)
                        self.dataset_dict['infos'][key].append(value)

        # Add episode tracking info to observations
        if isinstance(data_dict['observations'], dict):
            data_dict['observations']['index'] = np.array([self._current_episode_steps])
            data_dict['observations']['ep_len'] = np.array([0])  # Will be updated on episode end
            
            data_dict['next_observations']['index'] = np.array([self._current_episode_steps + 1])
            data_dict['next_observations']['ep_len'] = np.array([0])
        else:
            # If observations is not a dict, create one
            obs_dict = {
                'obs': data_dict['observations'],
                'index': np.array([self._current_episode_steps]),
                'ep_len': np.array([0])
            }
            next_obs_dict = {
                'obs': data_dict['next_observations'],
                'index': np.array([self._current_episode_steps + 1]),
                'ep_len': np.array([0])
            }
            data_dict['observations'] = obs_dict
            data_dict['next_observations'] = next_obs_dict
        
        # Insert the transition using parent class method
        super().insert(data_dict)
        
        # Update episode tracking
        self._current_episode_steps += 1
        
        # If episode ended, update episode lengths and reset counter
        if data_dict.get('dones', False):
            # Update ep_len for all steps in the episode
            ep_start = self._insert_index - self._current_episode_steps
            if ep_start < 0:  # Handle wraparound
                ep_start += self._capacity
            
            # Update all steps in this episode with the final length
            for i in range(self._current_episode_steps):
                idx = (ep_start + i) % self._capacity
                if isinstance(self.dataset_dict['observations'], dict):
                    self.dataset_dict['observations']['ep_len'][idx] = self._current_episode_steps
                    self.dataset_dict['next_observations']['ep_len'][idx] = self._current_episode_steps
                
            self._episode_lengths[ep_start:ep_start + self._current_episode_steps] = self._current_episode_steps
            self._current_episode_steps = 0

    def _sample_goals(self, indices: np.ndarray, strategy: str = "episode") -> np.ndarray:
        """Sample future observations as goals using the specified strategy."""
        if strategy == "future":
            return self.sample_future_observation(indices, "uniform")
        elif strategy == "final":
            ep_begin = indices - _sample(self.dataset_dict['observations']['index'], indices)
            ep_len = _sample(self.dataset_dict['observations']['ep_len'], indices)
            final_indices = (ep_begin + ep_len - 1) % self._capacity
            return _sample(self.dataset_dict['observations'], final_indices)
        elif strategy == "episode":
            ep_begin = indices - _sample(self.dataset_dict['observations']['index'], indices)
            ep_len = _sample(self.dataset_dict['observations']['ep_len'], indices)
            random_offset = np.random.randint(0, ep_len, size=indices.shape)
            episode_indices = (ep_begin + random_offset) % self._capacity
            return _sample(self.dataset_dict['observations'], episode_indices)
        else:
            raise ValueError(f"Unknown goal selection strategy: {strategy}")

    def _relabel_batch(self, samples: Dict) -> Dict:
        """Relabel a batch of samples with new goals."""
        samples = frozen_dict.unfreeze(samples)
        batch_size = samples['observations']['obs'].shape[0]
        n_relabel = int(batch_size * self.her_ratio)
        
        if n_relabel == 0:
            return frozen_dict.freeze(samples)
            
        relabel_indices = np.random.choice(batch_size, size=n_relabel, replace=False)
        new_goals = self._sample_goals(relabel_indices, self.goal_selection_strategy)
        # breakpoint()
        resampled=copy.deepcopy(samples)
        resampled['observations']['goal'][relabel_indices] = new_goals["goal"]
        resampled['next_observations']['goal'][relabel_indices] = new_goals["goal"]
        breakpoint()
        resampled["infos"]["goal"]=new_goals["infos"]["goal"]
        
        if self._relabel_fn is not None:
            samples = self._relabel_fn(samples,resampled)
        
        return frozen_dict.freeze(samples)

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
        sample_futures=None,
        sample_futures_key=None,
        relabel: bool = True
    ) -> frozen_dict.FrozenDict:
        """Sample a batch of transitions and apply HER relabeling."""
        samples = super().sample(
            batch_size=batch_size,
            keys=keys,
            indx=indx,
            sample_futures=sample_futures,
            sample_futures_key=sample_futures_key,
            relabel=False
        )
        
        if relabel:
            samples = self._relabel_batch(samples)
        return samples

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        """Get an iterator over batches of samples."""
        if "relabel" not in sample_args:
            sample_args["relabel"] = True
        return super().get_iterator(queue_size, sample_args)
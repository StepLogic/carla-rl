from collections import deque
import copy
import numpy as np
from typing import Optional, Iterable, Callable, Dict, Any
from enum import Enum
from flax.core import frozen_dict
import gymnasium as gym

from jaxrl5.data.dataset import DatasetDict
from jaxrl5.data.replay_buffer import ReplayBuffer, _sample
class HindsightReplayBuffer(ReplayBuffer):
    def __init__(
        self, 
        observation_space: gymnasium.Space, 
        action_space: gymnasium.Space, 
        capacity: int,
        relabel_obs_fn=None
    ):
        super().__init__(observation_space, action_space, capacity)
        self._episodes = defaultdict(list)  # Removed lambda as it's not needed
        self._episode_idx = 0
        self.n_sampled_goals = 2
        self.relabel_obs_fn = relabel_obs_fn

    def insert(self, data_dict: DatasetDict):
        if data_dict["dones"]:
            self._episode_idx += 1
            
            
            # Reset episode if buffer is full
            if (self._insert_index + 1) % self._capacity == 0:
                if len(self._episodes[self._episode_idx]) > 0:  # Check if episode has data
                    # virtual_indices = np.random.choice(
                    #     self._episodes[self._episode_idx], 
                    #     size=min(2, len(self._episodes[self._episode_idx])), 
                    #     replace=False
                    # )
                    
                    for idx in range(min(len(self._episodes[self._episode_idx]),self.n_sampled_goals):
                        v_idx=min(random.choice(self._episodes[self._episode_idx]),len(self._episodes[self._episode_idx])-2)
                        o_idx=random.choice(self._episodes[self._episode_idx][self._episodes[self._episode_idx].index(v_idx):])
                        original_dict={}
                        virtual_dict = {}
                        for k, v in self.dataset_dict.items():
                            if isinstance(v, dict):
                                virtual_dict[k] = {}
                                for sub_k, sub_v in v.items():
                                    virtual_dict[k][sub_k] = sub_v[v_idx].copy()
                            else:
                                virtual_dict[k] = v[v_idx].copy()
                        for k, v in self.dataset_dict.items():
                            if isinstance(v, dict):
                                original_dict[k] = {}
                                for sub_k, sub_v in v.items():
                                    original_dict[k][sub_k] = sub_v[o_idx].copy()
                            else:
                                original_dict[k] = v[o_idx].copy()
                        
                        if callable(self.relabel_obs_fn):
                            max_len = max(i for i in self._episodes[self._episode_idx])
                            delta=o_idx-v_idx
                            is_near_goal=delta>0 and delta<2
                            virtual_dict = self.relabel_obs_fn(original_dict,virtual_dict,is_near_goal, max_len=max_len)
                            super().insert(virtual_dict)
                            #goal prositve samples
                            virtual_dict = self.relabel_obs_fn(original_dict,original_dict,True, max_len=max_len)  
                            super().insert(virtual_dict)          
                self._episode_idx = 0
                self._episodes.clear()  # Clear all episodes
                self._episodes[self._episode_idx] = []

        # Insert the original data
        super().insert(data_dict)
        self._episodes[self._episode_idx].append(self._insert_index)
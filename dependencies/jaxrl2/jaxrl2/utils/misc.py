from typing import Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init
import numpy as np

def is_image_space(observation: jnp.ndarray) -> bool:
    """Check if an observation is image-like (has 3+ dimensions)."""
    try:
        return len(observation.shape) >= 3
    except:
        breakpoint()

def process_observation(observation: Union[jnp.ndarray, Dict, FrozenDict]) -> Dict:
    """Convert observation to consistent dict format."""
    if isinstance(observation, (dict, FrozenDict)):
        return observation
    else:
        # If single array passed, treat as primary observation
        return {'obs': observation}
    
def augment_observations(
    rng: jnp.ndarray,
    observations: Union[np.ndarray, jnp.ndarray, Dict],
    aug_func: None
) -> Tuple[jnp.ndarray, Union[jnp.ndarray, Dict]]:

    # Handle direct array input
    if isinstance(observations, (np.ndarray, jnp.ndarray)):
        if is_image_space(observations):
            rng, split_rng = jax.random.split(rng)
            return rng, aug_func(split_rng, observations)
        return rng, observations
    
    # Process dictionary observations
    new_observations = observations.copy()

    # Iterate through observations and augment image-like ones
    for key, value in observations.items():
        if is_image_space(value):
            rng, split_rng = jax.random.split(rng)
            aug_value = aug_func(split_rng, value)
            new_observations = new_observations.copy(add_or_replace={key: aug_value})
    
    return rng, new_observations



def augment_batch(
    rng: jnp.ndarray,
    batch: Dict,
    aug_func: None,
) -> Tuple[jnp.ndarray, Dict]:
    # Get observations and next_observations
    observations = batch["observations"]
    next_observations = batch["next_observations"]
    
    # Handle observations
    rng, aug_observations = augment_observations(rng, observations, aug_func)
    new_batch = batch.copy(add_or_replace={"observations": aug_observations})
    
    # Handle next_observations
    rng, aug_next_observations = augment_observations(rng, next_observations, aug_func)
    new_batch = new_batch.copy(add_or_replace={"next_observations": aug_next_observations})
    
    return rng, new_batch




    # th.autograd.set_detect_anomaly(True)
# import torch as th
import ssl
import os

from PIL import Image
# import torch
# from torch.utils.data import Dataset

import numpy as np
from jax.tree_util import tree_map
# from torch.utils import data
import os
from flax.core.frozen_dict import unfreeze
from flax.training import checkpoints
import numpy as np

def load_pretrained(checkpoint_dir,agent):
    loaded_params = checkpoints.restore_checkpoint(checkpoint_dir, target=None)
    actor_params = unfreeze(agent.actor.params)
    for k, v in loaded_params["critic"]["params"].items():
        if 'encoder' in k: #load only encoder
            actor_params[k] = v
    critic_params = unfreeze(agent.critic.params)

    return agent.replace(
        actor=agent.actor.replace(
            params=unfreeze(actor_params),
        ),
        critic=agent.critic.replace(
            params=unfreeze(critic_params),
        ),
        target_critic=agent.target_critic.replace(
            params=unfreeze(critic_params),
        ),
    )

def load_checkpoints(checkpoint_dir,agent):
    loaded_params = checkpoints.restore_checkpoint(checkpoint_dir, target=agent)
    return loaded_params

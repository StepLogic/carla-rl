from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.types import PRNGKey


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray, **kwargs) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations, **kwargs)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations,)
    # print(np.array(dist.stddev()))
    return dist.mode()


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions__with_stds(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode(),dist.stddev()

class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions_with_std(self, observations: np.ndarray, **kwargs) -> np.ndarray:
        actions,stds = _eval_actions__with_stds(self.actor.apply_fn, self.actor.params, observations, **kwargs)
        return np.asarray(actions),np.asarray(stds)


    def eval_actions(self, observations: np.ndarray, **kwargs) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations, **kwargs)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray, **kwargs) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, **kwargs
        )
        return np.asarray(actions), self.replace(rng=new_rng)

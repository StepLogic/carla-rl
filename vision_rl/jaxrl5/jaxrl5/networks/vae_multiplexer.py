from typing import Dict, Optional, Tuple, Type, Union

import cv2
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl5.networks import default_init

def prepare_batches(x):
    # Check the number of dimensions
    if x.ndim == 4:  # Shape (32, 32, 3, 6)
        # Transpose to make 6 the batch dimension, resulting in (6, 32, 32, 3)
        x = jnp.transpose(x, (3, 0, 1, 2))
    elif x.ndim == 5:  # Shape (batch, 32, 32, 3, 6)
        # Transpose to move 6 to the batch dimension
        # batch_size = x.shape[0]
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # Resulting in (batch * 6, 32, 32, 3)
        x = x.reshape(-1,*x.shape[2:])  # Flatten batch and new batch
    else:
        raise ValueError("Input tensor must be either 4D (32, 32, 3, 6) or 5D (batch, 32, 32, 3, 6).")
    
    return x
# def decode_images_for_debugging(encoder,x):


class VAEMultiplexer(nn.Module):
    encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]
    latent_dim: int
    stop_gradient: bool = False
    pixel_keys: Tuple[str, ...] = ("pixels",)
    depth_keys: Tuple[str, ...] = ()

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
        output_range: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        # observations = FrozenDict(observations)
        if len(self.depth_keys) == 0:
            depth_keys = [None] * len(self.pixel_keys)
        else:
            depth_keys = self.depth_keys

        xs = []
        for i, (pixel_key, depth_key) in enumerate(zip(self.pixel_keys, depth_keys)):
      
            x = observations[pixel_key]
            if self.encoder_cls is not None:

                x = x.astype(jnp.float32) / 255.0 #normalize image
                if depth_key is not None:
                    x = jnp.concatenate([x, observations[depth_key]], axis=-2)
                reshape_after=x.ndim == 5
                batch=x.shape[0]
                x=prepare_batches(x)
                x=self.encoder_cls(x)
                if not reshape_after:
                    x = x.flatten()
                else:                
                    x = x.reshape(batch, -1)
            
                # x = jnp.concatenate([self.encoder_cls(x[..., j])[0] for j in range(x.shape[-1])], axis=-1)
            if self.stop_gradient:
                # We do not update conv layers with policy gradients.
                x = jax.lax.stop_gradient(x)

            x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm(epsilon=1e-5)(x)
            # jax.debug.print("🤯 {x} 🤯", x=x)
            x=jax.numpy.nan_to_num(x, copy=True, nan=1e-5)
            x = nn.tanh(x)
            xs.append(x)
   
        x = jnp.concatenate(xs, axis=-1)
        # jax.debug.print("dsitribution  {x}",x=x)
        # if "states" in observations:
        #     y = nn.Dense(self.latent_dim, kernel_init=default_init())(
        #         observations["states"]
        #     )
        #     y = nn.LayerNorm()(y)
        #     y = nn.tanh(y)

        #     x = jnp.concatenate([x, y], axis=-1)
        # breakpoint()
        if actions is None:
            if output_range is None:
                return self.network_cls()(x, training)
            else:
                return self.network_cls()(x, training, output_range=output_range)
        else:
            if output_range is None:
                return self.network_cls()(x, actions, training)
            else:
                return self.network_cls()(x, actions, training, output_range=output_range)
from typing import Dict, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init
from jaxrl2.utils.misc import is_image_space, process_observation

# from typing import Dict, Optional, Union
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# from flax.core.frozen_dict import FrozenDict
# from jaxrl2.networks.constants import default_init

# class PixelMultiplexer(nn.Module):
#     encoder: nn.Module
#     network: nn.Module
#     latent_dim: int
#     stop_gradient: bool = False

#     @nn.compact
#     def __call__(
#         self,
#         observations: Union[FrozenDict, Dict],
#         actions: Optional[jnp.ndarray] = None,
#         training: bool = False,
#     ) -> jnp.ndarray:
#         observations = FrozenDict(observations)
#         assert (
#             len(observations.keys()) <= 2
#         ), "Can include only pixels and states fields."

#         x = self.encoder(observations["pixels"])

#         if self.stop_gradient:
#             # We do not update conv layers with policy gradients.
#             x = jax.lax.stop_gradient(x)

#         x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
#         x = nn.LayerNorm()(x)
#         x = nn.tanh(x)

#         if "states" in observations:
#             y = nn.Dense(self.latent_dim, kernel_init=default_init())(
#                 observations["states"]
#             )
#             y = nn.LayerNorm()(y)
#             y = nn.tanh(y)

#             x = jnp.concatenate([x, y], axis=-1)

#         if actions is None:
#             return self.network(x, training=training)
#         else:
#             return self.network(x, actions, training=training)


class PixelMultiplexer(nn.Module):
    encoder: nn.Module
    network: nn.Module
    latent_dim: int
    stop_gradient: bool = False
    # siamese:bool=False
    # def setup(self):
    #     # Submodule names are derived by the attributes you assign to. In this
    #     # case, "dense1" and "dense2". This follows the logic in PyTorch.
    #     # self.encoder_dict={
    #     #     "encoder_1":self.encoder
    #     # }
    #     if self.siamese:

    #     self.dense1 = nn.Dense(32)
    #     self.dense2 = nn.Dense(32)
    @nn.compact
    def __call__(
        self,
        observations: Union[jnp.ndarray, Dict, FrozenDict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        # Handle both array and dict inputs
        observations = process_observation(observations)
        # Convert to FrozenDict if needed
        if not isinstance(observations, FrozenDict):
            observations = FrozenDict(observations)
        processed_features = []
        # Process each observation
        for key, value in observations.items():
            if is_image_space(value):
                # if jnp.max(value)>1.0:#normalize
                #     value=(value/255).astype(jnp.float32)
                value = jax.lax.cond(
                    jnp.max(value) > 1.0,
                    lambda x: (x / 255).astype(jnp.float32),  # true_fn: normalize
                    lambda x: x.astype(jnp.float32),               # false_fn: keep as is
                    value
                )
                x = self.encoder(name=f"encoder_{key}")(value)
                if self.stop_gradient:
                    x = jax.lax.stop_gradient(x)
                x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
            else:
                # Handle continuous observations
                x = nn.Dense(self.latent_dim, kernel_init=default_init())(value)
            
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            processed_features.append(x)
        
        # Combine all processed features
        if len(processed_features) > 1:
            x = jnp.concatenate(processed_features, axis=-1)
        else:
            x = processed_features[0]
        
        # Pass through the network
        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)
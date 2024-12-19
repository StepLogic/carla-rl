from typing import Any, Optional

from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
tfb = tfp.bijectors

import jax
import jax.numpy as jnp

# Inspired by
# https://github.com/deepmind/acme/blob/300c780ffeb88661a41540b99d3e25714e2efd20/acme/jax/networks/distributional.py#L163
# but modified to only compute a mode.


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False, low=-1.0, high=1.0):
        self.shift = (high + low) / 2.0
        self.scale = (high - low) / 2.0
        bijector = tfb.Chain([
            tfb.Shift(self.shift),
            tfb.Scale(self.scale),
            tfb.Tanh(),
            tfb.Scale(1.0 / self.scale),
            tfb.Shift(-self.shift),
        ], name="ranged_tanh")

        super().__init__(
            distribution=distribution, bijector=bijector, validate_args=validate_args
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())
    def stddev(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.stdev())
    
    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
# from typing import Any, Optional

# import jax.numpy as jnp
# import tensorflow_probability.substrates.jax as tfp

# tfd = tfp.distributions


# class TanhTransformedDistribution(tfd.TransformedDistribution):  # type: ignore[name-defined]
#     """
#     From https://github.com/ikostrikov/walk_in_the_park
#     otherwise mode is not defined for Squashed Gaussian
#     """

#     def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):  # type: ignore[name-defined]
#         super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args)

#     def mode(self) -> jnp.ndarray:
#         return self.bijector.forward(self.distribution.mode())
    
#     def stddev(self) -> jnp.ndarray:
#         return self.bijector.forward(self.distribution.stddev())

#     @classmethod
#     def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
#         td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
#         del td_properties["bijector"]
#         return td_properties

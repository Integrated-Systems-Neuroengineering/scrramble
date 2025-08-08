import jax
import math
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
from functools import partial



class PartialSumsLayer(nnx.Module):
    """
    Module for single partial sums layer.
    Takes in size of a feedforwardf layer.
    Initializes appropriate number of cores as trainable parameters.
    Accumulates partial sums across the cores
    """

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 rngs: nnx.Rngs,
                 activation_function: Callable,
                 columns_per_core: int = 256
                 ):
        
        self.in_size = in_size
        self.out_size = out_size
        self.activation_function = activation_function
        self.columns_per_core = columns_per_core
        self.rngs = rngs

        # number of cores requires
        self.in_blocks = math.ceil(in_size / columns_per_core)
        self.out_blocks = math.ceil(out_size / columns_per_core)

        # initialize parameters
        initializer = initializers.glorot_normal()
        self.W = nnx.Param(
            initializer(self.rngs.params(), (self.out_blocks, self.in_blocks, self.columns_per_core, self.columns_per_core))
        )

    def required_cores(self) -> int:
        """
        Returns number of cores required to implement this layer
        """
        return self.in_blocks * self.out_blocks

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass. No batch dimension. use vmap
        Assume the x is flat
        """

        # # pad the input if in_size is not a multiple of 256
        # x_padded = jnp.pad(x, pad_width=((0, 0), (0, self.columns_per_core * self.in_blocks - self.in_size)))

        x_reshape = x.reshape(self.in_blocks, self.columns_per_core)

        # compute the partial sums
        y = jnp.einsum('ijkl,jl->ik', self.W, x_reshape)

        # apply activation function
        y = jax.vmap(self.activation_function, in_axes=(0,))(y)

        return y

# if __name__ == "__main__":
#     # testing forward pass
#     rngs = nnx.Rngs(params=0, activations=1, permute=5, default=345)
#     x_test = jax.random.normal(rngs.default(), (10, 1024))
#     activation_function = nnx.relu
#     # activation_function = partial(qrelu, bits=4)
#     test_layer = PartialSumsLayer(
#         in_size=x_test.shape[-1],
#         out_size=512,
#         rngs=rngs,
#         activation_function=activation_function,
#     )

#     # nnx.display(test_layer)
#     y_test = jax.vmap(test_layer, in_axes=(0,))(x_test)
#     print(f"Output shape: {y_test.shape}")
#     print(f"Output: {y_test[0, 0, :10]}")
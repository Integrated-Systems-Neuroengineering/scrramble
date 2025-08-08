import jax
import math
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
from functools import partial

from models import PartialSumsLayer



# define a network with partial sums
class PartialSumsNetwork(nnx.Module):
    """
    Network with partial sums layers.
    """

    def __init__(self,
                 layer_sizes: list,
                 rngs: nnx.Rngs,
                 activation_function: Callable,
                 columns_per_core: int = 256
                ):
        
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.columns_per_core = columns_per_core
        self.rngs = rngs

        # initialize the layers
        self.layers = [
            PartialSumsLayer(
                in_size=i,
                out_size=o,
                rngs=rngs,
                activation_function=activation_function,
                columns_per_core=columns_per_core
            )
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def required_cores(self) -> int:
        """
        Returns number of cores required in the netowork
        """
        cores = [l.required_cores() for l in self.layers]
        return sum(cores)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the network. Assume that x has a batch dimension!
        """

        # resize the image to be (32, 32) for MNIST
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')

        # flatten the first two dimensions
        x = jnp.reshape(x, (x.shape[0], -1))

        for layer in self.layers:
            x = jax.vmap(layer, in_axes=(0,))(x)

        # at the final layer apply population code
        x = x.reshape(x.shape[0], -1)
        x = x[:, :250]
        x = x.reshape(x.shape[0], 10, -1)
        x = jnp.mean(x, axis=-1)

        return x
   
# testing the network
if __name__ == "__main__":
    rngs = nnx.Rngs(params=0, activations=1, permute=5, default=345)
    layer_sizes = [1024, 2048, 512, 256]
    activation_function = nnx.relu
    test_network = PartialSumsNetwork(
        layer_sizes=layer_sizes,
        rngs=rngs,
        activation_function=activation_function,
        columns_per_core=256
    )

    x_test = jax.random.normal(rngs.default(), (10, 32, 32, 1))
    y_test = test_network(x_test)
    print(f"Output shape: {y_test.shape}")
    print(f"Output: {y_test[0, :]}")

    # print number of cores required
    print(f"Number of cores required: {test_network.required_cores()}")
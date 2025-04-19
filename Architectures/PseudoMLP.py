"""
Define a MLP-like architecture mappable to RRAM-cores
"""

import jax
import jax.numpy as jnp
from typing import Callable
import flax
from flax import nnx
from functools import partial
import math

from models import MLPLayer
from utils import clipping_ste

class PseudoMLP(nnx.Module):
    """
    RRAM-mappable architecture with API of a standard MLP
    """

    @staticmethod
    def linear_map(
        x: jax.Array,
        threshold: float = None,
        noise_sd: float = None,
        key: jax.random.key = None
    ):
        return x

    def __init__(self, 
    layers: list, # list of layers
    rngs: nnx.Rngs, # nnx PRNG stream
    dense_activation_fn: Callable, # activation functions used by the RRAM cores
    accumulator_activation_fn: Callable, # activation functions used by the accumulator
    threshold = 0.0, # threshold for binary activation
    noise_sd = 0.05, # standard deviation for noise for binary activation
    ):

        self.layers = layers
        self.rngs = rngs
        self.dense_activation_fn = dense_activation_fn
        self.accumulator_activation_fn = accumulator_activation_fn
        self.threshold = threshold
        self.noise_sd = noise_sd

        # initialize the layers 
        self.mlp_layers = [
            MLPLayer(
                in_size = li,
                out_size = lo,
                rngs = self.rngs,
                dense_activation_fn = self.dense_activation_fn,
                accumulator_activation_fn = self.accumulator_activation_fn,
                threshold = self.threshold,
                noise_sd = self.noise_sd
            ) for li, lo in zip(self.layers[:-1], self.layers[1:-1])
        ]

        # define the last layer
        self.output_layer = MLPLayer(
            in_size = self.layers[-2],
            out_size = self.layers[-1],
            rngs = self.rngs,
            dense_activation_fn = self.dense_activation_fn,
            accumulator_activation_fn = self.linear_map,
            threshold = self.threshold,
            noise_sd = self.noise_sd
        )

        self.mlp_layers.append(self.output_layer)

    def __call__(self, x:jax.Array) -> jax.Array:

        # making first layer 1024-wide to maximally utilize the cores
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest') 
        x = x.reshape(x.shape[0], -1)
        
        for layer in self.mlp_layers:
            x = layer(x)
            
        # use population coding to get the final output
        x = x[:, :250].reshape(x.shape[0], 10, 25)
        pop_logits = jnp.average(x, axis = -1)

        return pop_logits

    def get_num_cores(self):

            """
            Get the number of cores in the model.
            """
            blocks = [math.ceil(l/256) for l in self.layers]
            cores_per_layer = [b_in*b_out + b_out for b_in, b_out in zip(blocks[:-1], blocks[1:])]
            return sum(cores_per_layer)


# testing
def __main__():
    rngs = nnx.Rngs(params=345, activation=67565)
    model = PseudoMLP(
        layers = [1024, 2048, 512, 256],
        rngs = rngs,
        dense_activation_fn = clipping_ste,
        accumulator_activation_fn = clipping_ste,
        threshold = 0.0,
        noise_sd = 0.05
    )

    x = jax.random.normal(rngs.params(), (10, 32, 32, 1))
    y = model(x)
    print(y.shape)
    print(y[0, :10])
    print(model.get_num_cores())

# if __name__ == "__main__":
#     __main__()
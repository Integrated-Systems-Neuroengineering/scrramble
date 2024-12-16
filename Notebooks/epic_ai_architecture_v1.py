"""
Modules for the EPIC AI chip architecture.
Learning is done using binary stochastic version of Trident algorithm
"""

import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import defaultdict
from binary_trident_helper_functions import *



## defining the core module
class EICCore(nn.Module):
    """
    Initializes number of cores and operations that each core performs.

    - Each core receives inputs x which are 256x1 vectors
    - The core splits the inputs into two identical parts
    - Each part is the shuffled randomly.
    - Core then performs W(xp - xn)
    - Returns output after applying the binary activation function.
    """

    ## define the number of rows ans columns
    key: jax.random.PRNGKey
    num_inputs: int
    num_outputs: int

    @staticmethod
    def intercore_connectivity(num_rows, num_cols, alpha, key):
        """
        Defines intercore connectivity matrix.
        Follows a simplified arrangement where each core is connected to a neighbor on the same row or column with some probability alpha.
        """
        V = jnp.zeros((num_rows, num_cols, num_rows, num_cols))
        key, split_key = jax.random.split(key)

        random_conn = jax.random.uniform(split_key, (num_cols, num_cols)) < alpha

        V = V.at[jnp.arange(num_rows), :, jnp.arange(num_rows), :].set(random_conn)

        key, split_key = jax.random.split(key)
        random_conn = jax.random.uniform(split_key, (num_rows, num_rows)) < alpha

        V = V.at[:, jnp.arange(num_cols), :, jnp.arange(num_cols)].set(random_conn)

        return V
    
    @staticmethod
    def split_and_shuffle(input):
        """
        Split input into two parts. Shuffle each part chunk-wise.

        Args:
        1. vector

        Outputs:
        1. xp and xn
        """

        ## get the chunk size
        chunk_size = len(input) // 4

        ## split the input
        xp, xn = input.copy(), input.copy()

        ## generate shuffle indices for each chunk
        num_chunks = 4
        key, subkey = jax.random.split(key)

        def shuffle_chunk(chunk, shuffle_key):
            idx = jax.random.permutation(shuffle_key, chunk_size)
         

    @nn.compact
    def __call__(self, x):
        """
        For every core simply perform matrix-vector product
        """
        uw = self.param(
            "uw", 
            nn.initializers.xavier_uniform(), (self.num_outputs, self.num_inputs)
        )

        # reparametrization to ensure non-negative weights
        W = jnp.square(uw)

        y = jnp.dot(W, x)

        y = custom_binary_gradient(x=y, threshold=0.0, noise_sd=0.1, key=self.key)

        return y
    
    def split_inputs(self, x):
        """
        Split inputs into two parts.
        Shuffle each vector in chunks of 64
        """


    
    def get_weights(self, params):
        """
        Returns weights for visualization
        """
        weights = params["params"]["uw"]
        weights = jnp.square(weights)
        return weights
    

## defining the chip architecture module

    





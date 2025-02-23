# defining the EICDense module
import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import defaultdict
from HelperFunctions.binary_trident_helper_functions import *


class EICDense(nn.Module):
    """
    Pseudo-dense layer using EIC Cores.
    Args:
    in_size: int, number of input neurons
    out_size: int, number of output neurons
    threshold: float, threshold for binary activation
    noise_sd: flaat, standard deviation of noise for binary activation
    key: jax.random.PRNGKey, random key

    Returns:
    x: jnp.ndarray, output of the layer
    """

    in_size: int
    out_size: int

    def setup(self):
        """
        Set up dependent parameters
        """
        self.out_blocks = max(self.out_size//256, 1) # number of blocks required at the output 
        self.in_blocks = max(self.in_size//256, 1) # number of bloacks required at the input


        self.num_cores = self.out_blocks * self.in_blocks # number of cores required
        self.W = self.param(
            "weights",
            lambda key, shape: nn.initializers.xavier_normal()(key, shape),
            (self.out_blocks, self.in_blocks, 256, 256)
        )


    def __call__(self, x):
        """
        Forward pass of the layer
        Args:
        x: jnp.ndarray (batch_size, in_size), input to the layer
        
        Returns:
        x: jnp.ndarray, output of the layer
        """

        assert x.shape[-1] == self.in_size, f"Input shape is incorrect. Got {x.shape[-1]}, expected {self.in_size}"

        x_reshaped = x.reshape(x.shape[0], self.in_blocks, 256) # organize x into blocks of 256 for every batch

        # make sure that the weights are positive
        W_pos= jax.nn.softplus(self.W)

        # quantize weights
        # W_pos = quantize_params(W_pos, bits = 8)

        y = jnp.einsum("ijkl,bjl->bijk", W_pos, x_reshaped)


        return y
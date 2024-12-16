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
    threshold: float
    noise_sd: float
    key: jax.random.key
    activation: callable = None

    def setup(self):
        """
        Set up dependent parameters
        """

        self.out_blocks = self.out_size//256 # number of blocks required at the output 
        self.in_blocks = self.in_size//256 # number of bloacks required at the input

        # if the number of blocks is zero, set it to 1
        if self.in_blocks == 0:
            self.in_blocks = 1
        if self.out_blocks == 0:
            self.out_blocks = 1

        self.num_cores = self.out_blocks * self.in_blocks # number of cores required
        self.W = self.param(
            "weights",
            lambda key, shape: jnp.abs(nn.initializers.xavier_uniform()(key, shape)),
            (self.out_blocks, self.in_blocks, 256, 256)
        )

        # if the activation is None, simply return linear map
        if self.activation is None:
            self.activation = lambda x, threshold, noise_sd, key: x

    def sigmoid_fn(self, x, threshold = 0.0, noise_sd = 0.1, key = jax.random.key(0)):
        """
        Simple sigmoid.
        """
        return jax.nn.sigmoid(x)

    def __call__(self, x):
        """
        Forward pass of the layer
        Args:
        x: jnp.ndarray, input to the layer
        
        Returns:
        x: jnp.ndarray, output of the layer
        """

        assert x.shape == (self.in_size,), "Input shape is incorrect"

        x_reshaped = x.reshape(self.in_blocks, 256) # organize x into blocks of 256

        # make sure that the weights are positive
        W_pos= jnp.square(self.W)

        y = jnp.einsum("ijkl,jl->ijk", W_pos, x_reshaped)

        key, split_key = jax.random.split(self.key)
        y = self.activation(y, threshold = self.threshold, noise_sd = self.noise_sd, key = split_key)

        return y
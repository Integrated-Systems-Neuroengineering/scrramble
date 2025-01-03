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
    activation: callable

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
            lambda key, shape: nn.initializers.xavier_normal()(key, shape),
            (self.out_blocks, self.in_blocks, 256, 256)
        )


    @staticmethod
    def linear_map(x, threshold = 0., noise_sd = 0.1, key = None):
        """
        Linear map
        """
        return x

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

        assert x.shape == (self.in_size,), f"Input shape is incorrect. Got {x.shape}, expected {(self.in_size,)}"

        x_reshaped = x.reshape(self.in_blocks, 256) # organize x into blocks of 256

        # make sure that the weights are positive
        W_pos= jax.nn.relu(self.W)

        y = jnp.einsum("ijkl,jl->ijk", W_pos, x_reshaped)

        # activation_fn = self.activation if self.activation is not None else self.linear_map
        activation_fn = self.sigmoid_fn
        key = self.make_rng("activation")
        y = activation_fn(y, threshold = self.threshold, noise_sd = self.noise_sd, key = key)

        return y
    
    
# testing...
# def __main__():
#     rng = jax.random.key(0)
#     key, subkey = jax.random.split(rng)
#     x = jax.random.normal(key, (1024,))
#     eic = EICDense(in_size = 1024, out_size = 2048, threshold=0., activation = custom_binary_gradient, noise_sd = 0.1)
#     params_eic = eic.init(key, x)
#     print("Initialized EICDense parameters")
#     print(f"Params: {params_eic}")
#     print(f"Params shape: {params_eic['params']['weights'].shape}")
#     y = eic.apply(params_eic, x, rngs = {"activation": subkey})
#     print(f"Output: {y}")
#     print(f"Output shape: {y.shape}")

#     print("TRIAL 2")

#     x = jax.random.normal(key, (1024,))
#     eic = EICDense(in_size = 1024, out_size = 2048, threshold=0., activation = custom_binary_gradient, noise_sd = 0.1)
#     params_eic = eic.init(subkey, x)
#     print("Initialized EICDense parameters")
#     print(f"Params: {params_eic}")
#     print(f"Params shape: {params_eic['params']['weights'].shape}")
#     y2 = eic.apply(params_eic, x, rngs = {"activation": subkey})
#     print(f"Output: {y2}")
#     print(jnp.linalg.norm(y - y2))
#     print(f"Output shape: {y.shape}")


# if __name__ == "__main__":
#     __main__()
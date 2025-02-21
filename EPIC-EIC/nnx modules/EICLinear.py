import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
from EICDense import *
from Accumulator import *
from PermuteBlock import *
from binary_ste import *


class EICLinear(nnx.Module):
    """
    Consolidated EICLinear block
    """

    def __init__(self,
                 in_size: int, # size of input layer
                 out_size: int, # size of the output layer
                 eic_activation_fn: Callable, # EICDense activation function
                 acc_activation_fn: Callable, # Accumulator activation function
                 key: jax.random.key, # PRNG key
                 threshold: float = 0.0, # threshold for binary activation
                 noise_sd: float = 0.05 # standard deviation of noise for binary activation 
                 ):
        self.in_size = in_size
        self.out_size = out_size
        self.eic_activation_fn = eic_activation_fn
        self.acc_activation_fn = acc_activation_fn
        self.key = key
        self.threshold = threshold
        self.noise_sd = noise_sd

        # self.in_blocks = self.in_size//256
        # self.out_blocks = self.out_size//256

        # define the three blocks
        key, subkey = jax.random.split(self.key)
        self.eic_dense = EICDense(self.in_size, self.out_size, key)
        key, subkey = jax.random.split(key)
        self.accumulator = Accumulator(self.out_size, subkey)
        key, subkey = jax.random.split(key)
        self.permute = PermuteBlock(self.in_size)
        
    def __call__(self, x):
        """
        Forward pass of EICLinear block.
        Order of operations: input (x) -> PermuteBlock -> EICDense -> Accumulator 
        Ensure that:
          - before arriving to this module, there is a conv layer -> PermuteBlock. Otherwise, before the first layer add a PermuteBlock
        Args:
         - x: jax.Array, input vector, typically flattened image etc.
        Returns:
            - x: jax.Array, output of the EICLinear block
        """

        x = self.permute(x)
        x = self.eic_dense(x)
        key, subkey = jax.random.split(self.key)
        x = self.eic_activation_fn(x, threshold = self.threshold, noise_sd = self.noise_sd, key = subkey)
        x = self.accumulator(x)
        key, subkey = jax.random.split(key)
        x = self.acc_activation_fn(x, threshold = self.threshold, noise_sd = self.noise_sd, key = subkey)

        return x
        
# test
# key = jax.random.key(123124)
# eic_lin = EICLinear(in_size = 1024, out_size = 2048, eic_activation_fn = binary_ste, acc_activation_fn = binary_ste, key = key)
# x = jax.random.normal(key, (10, 1024))*0.001
# y = eic_lin(x)
# print(y.shape)
# plt.matshow(y.reshape(-1, 128))

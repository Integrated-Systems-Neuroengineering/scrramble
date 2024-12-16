"""
Version 2 of the EPIC AI architecture
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

## define a constraints dictionary
constraints_dict = {
    'num_rows' : 16,
    'num_cols' : 16,
    'total_cores' : 16*16,
    'weight_dim1' : 256,
    'weight_dim2' : 256, ## these are the dimensions of each core
    'core_input_size': 256,
    'core_output_size' : 256,
    'input_slot_size' : 64,
}

# defining the EICDense module
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
    
# define the accumulator module
class Accumulator(nn.Module):
    """
    Accumulating the EICDense outputs. 
    Since the EICDense generates pseudo-feedforward outputs, we use a learnable accumulation matrix that minimizes error
    between the true feedforward output and the EIC output.

    Args:
        in_block_size: int, number of 256-sized blocks. This should be the .shape[0] of the EICDense output
    """

    in_block_size: int
    threshold: float
    noise_sd: float
    key: jax.random.key
    activation: callable = None

    def setup(self):
        """
        Set up the weights for the accumulator
        """

        self.W = self.param(
            "weights",
            nn.initializers.xavier_normal(),
            (self.in_block_size, 256, 256)
        )

        if self.activation is None:
            self.activation = lambda x, threshold, noise_sd, key: x

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the accumulator
        Args:
        x: jnp.ndarray, input to the accumulator
        
        Returns:
        x: jnp.ndarray, output of the accumulator
        """

        assert x.shape[0] == self.in_block_size, "Input shape is incorrect"

        y = jnp.einsum('ijk,imk->ij', self.W, x)
        y = self.activation(y, threshold = self.threshold, noise_sd = self.noise_sd, key = self.key)

        # flatten y before returning
        y = y.reshape(-1)

        return y
    
## define a module to split the input and shuffle it in blocks of 64
class ShuffleBlocks(nn.Module):
    """
    Shuffles an input flat vector into blocks of 64. Mathematically emulating inter-core communication.

    Args:
        subvector_len: int, length of each subvector (typically 256 for EIC core)
        slot_len: int, length of each slot (typically 64 for EIC core)
        key: jax.random.PRNGKey, random key

    Returns:
        (xp - xn): jnp.ndarray, shuffled input vector of shape (input_len,)
    """
    subvector_len: int
    slot_len: int
    key: jax.random.key

    @nn.compact
    def __call__(self, x):
        """
        Shuffle input vector x block-wise 
        Args:
        x: jnp.ndarray, input vector of shape (input_len,)

        Returns:
        x_shuffled: jnp.ndarray, shuffled input vector of shape (input_len,)
        """
        assert self.subvector_len % self.slot_len == 0, "Slot length must be an integer multiple of input_len"

        ## for comments consider x = (1024,) vector

        # determine how many blocks are in the input vector
        num_subvectors = x.shape[0]//self.subvector_len # e.g. 1024//256 = 4 subvectors
        slots_per_input = self.subvector_len//self.slot_len # e.g. 256//64 = 4 slots per input

        # reshape x into a 3D tensor of shape (num_subvectors, slots_per_input, slot_len), e.g. (4, 4, 64)
        x_reshaped = x.reshape(num_subvectors, slots_per_input, self.slot_len)

        # shuffle over slots_per_input dimension

        ## for positive vector
        key, subkey = jax.random.split(self.key)
        keys = jax.random.split(key, num_subvectors)

        shuffled_blocks_pos = [
            x_reshaped[i, jax.random.permutation(keys[i], slots_per_input, independent=True)] for i in range(num_subvectors)
        ]

        xpos = jnp.concatenate([blocks.reshape(-1) for blocks in shuffled_blocks_pos])

        # for negative vector
        key, subkey = jax.random.split(subkey)
        keys = jax.random.split(key, num_subvectors)

        shuffled_blocks_neg = [
            x_reshaped[i, jax.random.permutation(keys[i], slots_per_input, independent=True)] for i in range(num_subvectors)
        ]

        xneg = jnp.concatenate([blocks.reshape(-1) for blocks in shuffled_blocks_neg])

        return xpos - xneg







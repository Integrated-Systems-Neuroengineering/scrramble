import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from HelperFunctions.binary_trident_helper_functions import *

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

    def setup(self):
        """
        Set up the weights for the accumulator
        """

        self.W = self.param(
            "weights",
            nn.initializers.xavier_normal(),
            (self.in_block_size, 256, 256)
        )


    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the accumulator
        Args:
        x: jnp.ndarray, input to the accumulator
        
        Returns:
        x: jnp.ndarray, output of the accumulator
        """

        assert x.shape[1] == self.in_block_size, "Input shape is incorrect"
        # assert x.shape[1] == self.out_block_size, "Input shape is incorrect"

        # ensure positive 
        W_pos = jax.nn.softplus(self.W)
        # W_pos = quantize_params(W_pos, bits = 8)
        
        x = jnp.einsum("bijk->bik", x)
        y = jnp.einsum("ijk,bik->bik", W_pos, x) 

        # flatten y before returning
        y = y.reshape((y.shape[0], -1)) # (batch_size, out_size)

        return y

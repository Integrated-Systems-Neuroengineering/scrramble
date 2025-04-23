import jax
import math
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers
from utils import rram_quantize

class BlockwiseDense(nnx.Module):
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

    def __init__(self, in_size: int, 
                out_size: int,
                rngs: nnx.Rngs,
                num_rows: int = 256,
                num_cols: int = 256,
                quantize: str = 'log',
                quantize_bits: int = 8,
                g_inf: float = 2.0,
                g_min: float = 1e-3,
                tau: float = 0.75
                ):

        self.in_size = in_size
        self.out_size = out_size
        self.rngs = rngs
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.quantize = quantize
        self.quantize_bits = quantize_bits
        self.g_inf = g_inf
        self.g_min = g_min
        self.tau = tau

        # initialize the number of cores required
        self.in_blocks = math.ceil(self.in_size/self.num_cols) # number of blocks required at the input
        self.out_blocks = math.ceil(self.out_size/self.num_rows)  # number of blocks required at the output 
        self.num_cores = self.in_blocks * self.out_blocks

        # initialize the core weights
        # weights reshaped as (out_blocks, in_blocks, num_rows, num_cols)
        # assumes that input is shaped as (batch_size, in_blocks, num_rows)
        glorot_initializer = initializers.glorot_normal()
        self.cores = nnx.Param(glorot_initializer(self.rngs.params(), (self.out_blocks, self.in_blocks, self.num_rows, self.num_cols)))


    def __call__(self, x):
        """
        Forward pass of the layer
        Args:
        x: jnp.ndarray (batch_size, in_size), input to the layer
        
        Returns:
        x: jnp.ndarray, output of the layer
        """

        assert x.shape[-1] == self.in_size, f"Input shape is incorrect. Got {x.shape[-1]}, expected {self.in_size}"

        # pad the input if in_size is not a multiple of 256
        x_padded = jnp.pad(x, pad_width=((0, 0), (0, self.num_rows * self.in_blocks - self.in_size)))

        # organize x into blocks of 256 for every batch
        x_reshaped = x_padded.reshape(x.shape[0], self.in_blocks, self.num_rows) 

        # make sure that the weights are positive: relu and softplus work
        cores = nnx.relu(self.cores.value)#nnx.softplus(self.cores.value)

        # TODO: quantize weights 
        if self.quantize is not None:
            cores = cores.reshape(-1,)
            cores = jax.vmap(rram_quantize, in_axes=(0, None, None, None, None, None))(cores, self.tau, self.g_inf, self.g_min, self.quantize_bits, self.quantize)
            cores = cores.reshape(self.out_blocks, self.in_blocks, self.num_rows, self.num_cols)

        y = jnp.einsum("ijkl,bjl->bijk", cores, x_reshaped)


        return y
    

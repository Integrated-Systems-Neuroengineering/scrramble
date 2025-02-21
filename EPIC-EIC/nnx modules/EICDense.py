import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers

class EICDense(nnx.Module):
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

    def __init__(self, in_size, out_size, key):
        self.in_size = in_size
        self.out_size = out_size

        # initialize the number of cores required
        self.in_blocks = in_size // 256
        self.out_blocks = out_size // 256
        self.num_cores = self.in_blocks * self.out_blocks

        # initialize the core weights
        # weights reshaped as (out_blocks, in_blocks, 256, 256)y
        # assumes that input is shaped as (batch_size, in_blocks, 256)
        glorot_initializer = initializers.glorot_normal()
        self.cores = nnx.Param(glorot_initializer(key, (self.out_blocks, self.in_blocks, 256, 256)))


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
        self.cores = jax.nn.softplus(self.cores)

        y = jnp.einsum("ijkl,bjl->bijk", self.cores, x_reshaped)

        # quantize weights
        # W_pos = quantize_params(W_pos, bits = 8)

        return y
    
# # test
# key = jax.random.key(123124)
# ed = EICDense(in_size = 1024, out_size = 2048, key = key)
# x = jax.random.normal(key, (10, 1024))*0.01
# y = ed(x)
# print(y.shape)
# print(ed.cores.shape)
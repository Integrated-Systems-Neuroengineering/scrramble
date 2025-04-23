# define the accumulator module
import jax
import math
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers
from utils import rram_quantize


class WeightSharingAccumulator(nnx.Module):
    """
    Accumulating the EICDense outputs. 
    Since the EICDense generates pseudo-feedforward outputs, we use a learnable accumulation matrix that minimizes error
    between the true feedforward output and the EIC output.

    Args:
        [OLD] in_block_size: int, number of 256-sized blocks. This should be the .shape[0] of the EICDense output
        out_size: int, size of the output
    """

    def __init__(self, out_size: int,
                 rngs: nnx.Rngs,
                 num_rows: int = 256,
                 num_cols: int = 256,
                 quantize: str = 'log',
                 quantize_bits: int = 8,
                 g_inf: float = 2.0,
                 g_min: float = 1e-3,
                 tau: float = 0.75
                 ):

        self.rngs = rngs
        self.out_size = out_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.quantize = quantize
        self.quantize_bits = quantize_bits
        self.g_inf = g_inf
        self.g_min = g_min
        self.tau = tau

        # this should be the same as the number of output blocks from the EICDense
        self.out_block = math.ceil(out_size/self.num_rows) #max(out_size // 256, 1)  # number of blocks required at the output 

        # set up the params
        glorot_initializer = initializers.glorot_normal()
        self.acc_cores = nnx.Param(glorot_initializer(self.rngs.params(), (self.out_block, self.num_rows, self.num_cols)))



    def __call__(self, x):
        """
        Forward pass of the accumulator
        Args:
        x: jnp.ndarray, input to the accumulator
        
        Returns:
        x: jnp.ndarray, output of the accumulator
        """

        assert x.shape[1] == self.out_block, f"Input shape is incorrect. Got {x.shape[1]}, expected {self.out_block}"
        # assert x.shape[1] == self.out_block_size, "Input shape is incorrect"


        # ensure positive: relu and softplus work
        acc_cores = nnx.relu(self.acc_cores.value)#nnx.softplus(self.acc_cores.value)

        if self.quantize is not None:
            acc_cores = acc_cores.reshape(-1,)
            acc_cores = jax.vmap(rram_quantize, in_axes=(0, None, None, None, None, None))(acc_cores, self.tau, self.g_inf, self.g_min, self.quantize_bits, self.quantize)
            acc_cores = acc_cores.reshape(self.out_block, self.num_rows, self.num_cols)

        y = jnp.einsum('ojk,boik->boj', acc_cores, x)

        # flatten y before returning
        y = y.reshape((y.shape[0], -1)) # (batch_size, out_size)

        return y

# test
# acc = Accumulator(out_size = 2048, key = key)
# y_acc = binary_ste(acc(y), threshold = 0.0, noise_sd = 5e-2, key = key)
# plt.imshow(y_acc.reshape(-1, 128))
# print(y_acc.shape)
# print(acc.acc_cores.shape)

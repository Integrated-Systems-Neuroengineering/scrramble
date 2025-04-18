import jax
import jax.numpy as jnp
from typing import Callable
import flax
from flax import nnx
from functools import partial

from models import BlockwiseDense, WeightSharingAccumulator, PermuteWeightSharing, PermuteBlockwiseDense
from utils import rram_quantize, clipping_ste

"""
Define a test MLP architecture
"""

class MLPLayer(nnx.Module):
    """
    Single MLP layer. Implements ternary activations effectively through input balancing.
    """

    def __init__(self, 
    in_size: int, # size of input
    out_size: int, # size of output
    dense_activation_fn: Callable, # activation functions used by the RRAM cores
    accumulator_activation_fn: Callable, # activation functions used by the accumulator
    rngs: nnx.Rngs, # nnx PRNG stream
    permute_block_size: int = 16, # block size for permutation
    core_input_size: int = 256, # core input size
    threshold: float = 0.0, # threshold for binary activation
    noise_sd: float = 0.05, # standard deviation for noise for binary activation
    quantize: str = 'log', # quantization method
    bits: int = 8, # bits for quantization
    g_inf: float = 2.0, # maximum RRAM conductance per device
    g_min: float = 1e-5, # minimum RRAM conductance per device
    tau: float = 0.2 # time constant for RRAM device
    ):

        self.in_size = in_size
        self.out_size = out_size
        self.dense_activation_fn = dense_activation_fn
        self.accumulator_activation_fn = accumulator_activation_fn
        self.rngs = rngs
        self.threshold = threshold
        self.noise_sd = noise_sd
        self.quantize = quantize
        self.bits = bits
        self.g_inf = g_inf
        self.g_min = g_min
        self.tau = tau

        self.apply_dense_activation = partial(dense_activation_fn, threshold = self.threshold, noise_sd = self.noise_sd)
        self.apply_accumulator_activation = partial(accumulator_activation_fn, threshold = self.threshold, noise_sd = self.noise_sd)

        # define the blocks

        # single input dense block
        self.single_input_cores = BlockwiseDense(
            in_size = self.in_size,
            out_size = self.out_size,
            rngs = self.rngs,
            # quantize = self.quantize,
            # bits = self.bits,
            # g_inf = self.g_inf,
            # g_min = self.g_min,
            # tau = self.tau
        )

        self.accumulator = WeightSharingAccumulator(
            out_size = self.out_size,
            rngs = self.rngs
        )

        self.permute_dense = PermuteBlockwiseDense(
            input_size = self.in_size,
            rngs = self.rngs
        )

        self.permute_accumulator = PermuteWeightSharing(
            rngs = self.rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass
        """

        x = self.permute_dense(x)
        x = self.single_input_cores(x)
        x = self.apply_dense_activation(x, key=self.rngs.activation())
        x = self.permute_accumulator(x)
        x = self.accumulator(x)
        x = self.apply_accumulator_activation(x, key=self.rngs.activation())

        return x


# testing

def __main__():
    rngs = nnx.Rngs(params=0, activation=1)
    x = jax.random.normal(rngs.params(), (10, 1024))
    mlp = MLPLayer(
        in_size = 1024,
        out_size = 2048,
        dense_activation_fn = clipping_ste,
        accumulator_activation_fn = clipping_ste,
        rngs = rngs
    )

    y = mlp(x)
    print(y.shape)
    print(y[0, :10])

# if __name__ == "__main__":
#     __main__()

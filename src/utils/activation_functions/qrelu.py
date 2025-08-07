"""
Updated quantized ReLU function to be used for Post-training Quantization


Created on: 08/06/2025

"""

import jax
import jax.numpy as jnp
from flax import nnx


def qrelu(x: jax.Array,
              bits: int = 8,
              max_value:float = 2.0):
    """
    Quantized ReLU with quantization
    """

    num_levels = 2**bits - 1
    resolution = max_value/num_levels

    # apply ReLU to the input
    x = nnx.relu(x)

    # multiplier
    m = jnp.floor(x/resolution)

    # quantize the input
    x = jnp.clip(m * resolution, 0, max_value)

    return x
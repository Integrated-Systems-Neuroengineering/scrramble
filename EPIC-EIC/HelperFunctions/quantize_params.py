import jax
import jax.numpy as jnp


def quantize_params(params, bits = 8):
    """
    Quantizes the parameters of the model to given number of bits.
    Args:
        params: flax model parameters
        bits: number of bits to quantize to
    Returns:
        quantized_params: quantized flax model parameters
    """

    scale = 2**(bits - 1) - 1
    params = jax.tree.map(
        lambda p: jnp.round(p * scale) / scale, params
    )

    return params

import jax
import jax.numpy as jnp
import math
import flax
from flax import nnx
from functools import partial

def quantized_relu(
    x: float, # input value, set up uses vmap
    bits: int = 8,
    max_value: float = 5.0,
):

    """
    Quantized relu function.
    Args:
        x: input value
        bits: number of bits for quantization
        max_value: maximum value for quantization
    Returns:
        Quantized value of x
    """

    # find the levels
    num_levels = 2**bits - 1

    # find the threshold
    thresholds = jnp.linspace(0, max_value, num_levels + 1)

    # find the corresponding mappings
    levels = nnx.relu(thresholds)

    # determine all indices where the thresholds are less than x
    less_than = levels < x

    # find the index of the largest threshold that is less than x
    idx = jnp.sum(less_than) - 1
    idx = jnp.clip(idx, 0, len(thresholds)-1)

    out = levels[idx]

    return out

@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def quantized_relu_ste(
    x: float,
    bits: int = 8,
    max_value: float = 5.0,
    **kwargs
):

    return quantized_relu(x, bits, max_value)

def quantized_relu_fwd(
    x: float,
    bits: int = 8,
    max_value: float = 2.0,
    **kwargs
):

    primal_out = quantized_relu(x, bits, max_value)
    return primal_out, x

def quantized_relu_bwd(
    bits, max_value, residuals, gradients
):
    x = residuals
    grad = 1.0 #jnp.where(x > 0, 1.0, 0.0)

    return (grad*gradients, )

quantized_relu_ste.defvjp(quantized_relu_fwd, quantized_relu_bwd)


## testing
def __main__():
    rngs = nnx.Rngs(params=0, activation=1, default=46732)
    test_input = jax.random.normal(rngs.params(), (3, 5, 10))

    out = quantized_relu_ste(test_input, bits=8, max_value=1.0)

    print(f"Test output shape = {out.shape}")
    print(f"Test output = {out}")

if __name__ == '__main__':
    __main__()
    


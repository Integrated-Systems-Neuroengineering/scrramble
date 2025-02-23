import jax
import jax.numpy as jnp
from jax import vmap, jit

"""
Implements clippig straight through estimator.
"""

## define binary thresholding function: states [-1, 1]
# def binary_activation(x, threshold, noise_sd, key):
#     """
#     Binary activation function
#     """
#     # key, key2 = jax.random.split(key, 2)

#     # generate noise
#     noise = jax.random.normal(key, shape = x.shape) * noise_sd

#     # inject noise
#     x = x + noise

#     s = jnp.where(
#         x < threshold, 0.0, 1.0
#     )

#     return s

@jax.custom_vjp
def binary_ste(x, threshold, noise_sd, key):
    """
    Binary activation function
    """
    # key, key2 = jax.random.split(key, 2)

    # generate noise
    noise = jax.random.normal(key, shape = x.shape)

    # inject noise
    x = x + noise*noise_sd

    s = jnp.where(
        x < threshold, 0.0, 1.0
    )

    return s

def binary_ste_fwd(x, threshold, noise_sd, key):
    return binary_ste(x, threshold, noise_sd, key), (x, threshold, noise_sd)

def binary_ste_bwd(residuals, gradients):
    x, threshold, noise_sd = residuals
    # key, subkey = jax.random.split(jax.random.key(0))
    grad = jnp.where(jnp.abs(x) < 1.0, 1.0, 0.0)
    return (grad*gradients, None, None, None)

binary_ste.defvjp(binary_ste_fwd, binary_ste_bwd)
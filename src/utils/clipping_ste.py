import jax
import jax.numpy as jnp
from jax import vmap, jit
from flax import nnx



## define binary thresholding function: states [-1, 1]
def binary_activation(x, threshold, noise_sd, key):
    """
    Binary activation function
    """
    # key, key2 = jax.random.split(key, 2)

    # generate noise
    noise = jax.random.normal(key, shape = x.shape) * noise_sd

    # inject noise
    x = x + noise

    s = jnp.where(
        x < threshold, 0.0, 1.0
    )

    return s


@jax.custom_vjp
def clipping_ste(x, threshold, noise_sd, key):
    return binary_activation(x = x, threshold = threshold, noise_sd = noise_sd, key=key)

def clipping_ste_fwd(x, threshold, noise_sd, key):
    return clipping_ste(x, threshold, noise_sd, key), (x, threshold, noise_sd)

def clipping_ste_bwd(residuals, gradients):
    x, threshold, noise_sd = residuals
    grad = jnp.where(jnp.abs(x) < 1.0, 1.0, 0.0) #gaussian_pdf(x = x - threshold, mu = 0, sigma = noise_sd)
    return (grad*gradients, None, None, None)

clipping_ste.defvjp(clipping_ste_fwd, clipping_ste_bwd)
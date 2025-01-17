import jax
import jax.numpy as jnp
from jax import vmap, jit

# print("Modified custom grad to STE")

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

## helper function
@jax.jit
def gaussian_cdf(x, mu, sigma):
    return jax.scipy.stats.norm.cdf(x, loc = mu, scale = sigma)

@jax.jit
def gaussian_pdf(x, mu, sigma):
    return jax.scipy.stats.norm.pdf(x, loc = mu, scale = sigma)

@jax.jit
def bin_expected_state(x, threshold, noise_sd):
    e = gaussian_cdf(x = x - threshold, mu = 0, sigma = noise_sd)
    return e

@jax.custom_vjp
def custom_binary_gradient(x, threshold, noise_sd, key):
    return binary_activation(x = x, threshold = threshold, noise_sd = noise_sd, key = key)

def custom_binary_gradient_fwd(x, threshold, noise_sd, key):
    return custom_binary_gradient(x, threshold, noise_sd, key), (x, threshold, noise_sd)

def custom_binary_gradient_bwd(residuals, gradients):
    x, threshold, noise_sd = residuals
    key, subkey = jax.random.split(jax.random.key(0))
    grad = binary_activation(x=x, threshold=threshold, noise_sd=noise_sd, key=subkey) #gaussian_pdf(x = x - threshold, mu = 0, sigma = noise_sd*10)
    return (grad*gradients, None, None, None)

custom_binary_gradient.defvjp(custom_binary_gradient_fwd, custom_binary_gradient_bwd)
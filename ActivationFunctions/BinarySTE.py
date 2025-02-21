import jax
import jax.numpy as jnp
from jax import vmap, jit

"""
Implements a binary activation function with clipping straight through estimator (STE) gradient estimator.

- Activation function: f(x) = 1 if x > 0 else 0
- Gradient estimator: g = g0 * 1_{|x| < \sigma} where \sigma is a constant (indicator function)
"""


"""
Hardware-aware quantization function for weights.
- Assumes that the RRAM conductance follows saturating exponential rectifier characteristics
- Quantization implemented as zeroth order hold.
- Option to use linear or logarithmic sampling schemes
- Gradient is approximated as the gradient of the ground-truth conductance function which turns out to be a decaying exponential.
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial

def conductance_fn(x, tau = 0.2, g_inf = 1.0, g_min = 1e-3) -> float:
    """
    An approximation of RRAM conductance function.
    RRAM conductance is approximated as a saturating exponential.

    Args:
    1. x: float, input to the conductance function
    2. tau: flaot, time constant, the close to 0, the faster the conductance saturates
    3. g_inf: float, maximum conductance RRAM (in a.u. for the model but can be varied)
    4. g_min: float, minimum conductance of RRAM (in a.u. for the model but can be varied)

    Returns:
    1. g: float, conductance of RRAM
    """
    g = (g_inf - g_min)*(1 - jnp.exp(-x/tau)) + g_min
    return g

def grad_conductance_fn(x, tau=0.2, g_inf=1.0, g_min=1e-3):
    """
    Gradient of the conductance function. Computes gradient of the saturating exponential.
    """
    grad_fn = jax.grad(conductance_fn)
    grad = grad_fn(x, tau=tau, g_inf=g_inf, g_min=g_min)
    return grad


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def rram_quantize(x, tau = 0.2, g_inf = 1.0, g_min = 1e-1, bits = 8, method = 'log', conductance_fn = conductance_fn):
    """
    Quantize the conductance function to a given number of bits.
    Emulates in approximate terms the quantization of the conductance in a real device.
    For start assume that the conducatance is a saturating exponential function of the form (g_inf - g_min)(1 - e^-(t/tau)) + g_min
    where g_inf is the maximum conductance, A_min is the minimum conductance, and tau is the time constant.
    The function will quantize the output of the conductance function to a given number of bits.
    The quantization can be done using either a linear or logarithmic scale.
    The number of bits determines the number of quantization levels.
    Implements a zeroth order hold scheme.

    Parameters:
    1. x: float, input to the conductance function
    2. tau: float, time constant of the conductance function
    3. g_inf: float, maximum conductance
    4. g_min: float, minimum conductance
    5. bits: int, number of bits for quantization
    6. method: str, quantization method ('linear', or 'log')
    7. conductance_fn: Callable, conductance function to be quantized

    Returns:
    1. level: float, highest level <= weight
    """

    # compute the conductance given weights
    g = conductance_fn(x, tau=tau, g_inf=g_inf, g_min=g_min)
    # print(g)

    # compute the quantization thresholds
    partial(jax.jit, static_argnames=['bits', 'method', 'conductance_fn'])
    if method == 'log':
        # assert g_min > 0, "Minimum conductance must be greater than zero."
        thresholds = jnp.logspace(jnp.log10(g_min), jnp.log10(g_inf), 2**bits)
    elif method  == 'linear':
        thresholds = jnp.linspace(g_min, g_inf, 2**bits)
    else:
        raise ValueError("Current support for sampling methods is 'log' and 'linear'")

    # compute the conductances at the samples
    g_samples = conductance_fn(thresholds, tau=tau, g_inf=g_inf, g_min=g_min)

    # find g_samples <= g
    less_than_g = g_samples <= g
    idx = jnp.argmin(less_than_g) - 1

    idx = jnp.maximum(0, idx)

    level = g_samples[idx]

    return level

def rram_quantize_fwd(x, tau, g_inf, g_min, bits, method, conductance_fn):
    primal_out = rram_quantize(x, tau, g_inf, g_min, bits, method, conductance_fn)
    residuals = (x, tau, g_inf, g_min)
    return primal_out, residuals

def rram_quantize_bwd(bits, method, conductance_fn, residuals, gradients):
    x, tau, g_inf, g_min = residuals
    grad_x = grad_conductance_fn(x, tau=tau, g_inf=g_inf, g_min=g_min)
    return (grad_x*gradients, None, None, None)

rram_quantize.defvjp(rram_quantize_fwd, rram_quantize_bwd)
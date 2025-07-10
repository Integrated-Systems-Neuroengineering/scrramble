import jax
import jax.numpy as jnp

# squash activation function
def squash(x, eps=1e-8, axis=-1):
    """
    Squash function for CapsNets
    Args:
    x: input vector.
    Returns:
    squash_x: non-lineariy applied vector
    """

    x_norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    scale = (x_norm**2)/(1 + x_norm**2)

    squash_x = scale * (x / (x_norm + eps))

    return squash_x
## define a module to split the input and shuffle it in blocks of 64
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

class ShuffleBlock(nn.Module):
    """
    Defines a trainable permutation matrix with a temperature parameter.
    This is a proxy for the lookup table. In some sense it describes an idealized LUT.
    Temperature parameter can be very loosely interpreted as "release probability"

    Args:
        input_size: int, size of the input (e.g. (2048,))
    Defines:
        A: jnp.ndarray, trainable permutation matrix
        tau: float, temperature parameter
    Returns:
        y: jnp.ndarray, shuffled input. Basically, y = Ax
    """

    input_size: int
    tau: float 

    def setup(self):
        """
        Set up trainable permutation matrix
        """

        self.Z = self.param(
            'Z',
            nn.initializers.normal(),
            (self.input_size, self.input_size)
        )

    
    def __call__(self, x):
        """
        Soft shuffle the input
        """

        P = jax.nn.softmax(self.Z/self.tau, axis = -1) * jnp.sign(self.Z)
        y = jnp.einsum('ij,j->i', P, x)

        return y
    


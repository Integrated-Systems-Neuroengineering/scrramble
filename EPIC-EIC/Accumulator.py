import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

# define the accumulator module
class Accumulator(nn.Module):
    """
    Accumulating the EICDense outputs. 
    Since the EICDense generates pseudo-feedforward outputs, we use a learnable accumulation matrix that minimizes error
    between the true feedforward output and the EIC output.

    Args:
        in_block_size: int, number of 256-sized blocks. This should be the .shape[0] of the EICDense output
    """

    in_block_size: int
    threshold: float
    noise_sd: float
    key: jax.random.key
    activation: callable = None

    def setup(self):
        """
        Set up the weights for the accumulator
        """

        self.W = self.param(
            "weights",
            nn.initializers.xavier_normal(),
            (self.in_block_size, 256, 256)
        )

        if self.activation is None:
            self.activation = lambda x, threshold, noise_sd, key: x

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the accumulator
        Args:
        x: jnp.ndarray, input to the accumulator
        
        Returns:
        x: jnp.ndarray, output of the accumulator
        """

        assert x.shape[0] == self.in_block_size, "Input shape is incorrect"

        y = jnp.einsum('ijk,imk->ij', self.W, x)
        y = self.activation(y, threshold = self.threshold, noise_sd = self.noise_sd, key = self.key)

        # flatten y before returning
        y = y.reshape(-1)

        return y
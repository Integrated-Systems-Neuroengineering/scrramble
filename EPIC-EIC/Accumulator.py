import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from HelperFunctions.binary_trident_helper_functions import *

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

    @staticmethod
    def linear_map(x, threshold = 0., noise_sd = 0.1, key = None):
        """
        Linear map
        """
        return x


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
        # assert x.shape[1] == self.out_block_size, "Input shape is incorrect"

        # ensure positive 
        W_pos = jax.nn.relu(self.W)
        x = jnp.einsum("ijk->ik", x)
        y = jnp.einsum("ijk,ik->ik", W_pos, x) 


        # y = jnp.einsum('ijk,imk->ij', W_pos, x) #jnp.einsum('imk,ijk->im', W_pos, x) #
        key = self.make_rng("activation")

        activation_fn = self.activation if self.activation is not None else self.linear_map
        y = activation_fn(y, threshold = self.threshold, noise_sd = self.noise_sd, key = key)

        # flatten y before returning
        y = y.reshape(-1)

        return y

# testing...
# def __main__():
#     rng = jax.random.key(42)
#     x = jax.random.normal(rng, (8, 4, 256))
#     acc = Accumulator(
#         in_block_size = x.shape[0],
#         threshold = 0.0,
#         noise_sd = 1.0,
#         activation = custom_binary_gradient
#     )

#     params = acc.init(rng, x)
#     print("Initialized accumulator parameters")
#     print("----------------------------------------")
#     print(f"Output shape: {params["params"]["weights"].shape}")
#     y = acc.apply(params, x, rngs = {"activation": rng})
#     print("----------------------------------------")
#     print(f"Accumulator output {y}, \n Output shape: {y.shape}")


# if __name__ == "__main__":
#     __main__()
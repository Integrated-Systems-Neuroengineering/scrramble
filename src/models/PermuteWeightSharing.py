import jax
import jax.numpy as jnp
import flax
from flax import nnx

class PermuteWeightSharing(nnx.Module):
    """
    Defines a permute block for the accumulator. Performs the following operation
    1. Take in the output of EIC.Dense which is (out_blocks, in_blocks, block_size) e.g. (8, 4, 256) for 1024 -> 2048
    2. Reshapes the input in (out_blocks, in_blocks, rows, columns) e.g. (8, 4, 16, 16)
    3. Applies two independent permutations to ONLY the columns dimension.
    4. Keep one +1 and other as -1
    5. Recombine them into (out_blocks, in_blocks, block_size)
    """

    def __init__(self, 
                #  input_size: int,
                 rngs: nnx.Rngs,
                 num_slots: int = 16,
                 core_input_size: int = 256
                 ):
        
        # self.input_size = input_size
        self.rngs = rngs
        self.num_slots = num_slots

        # define permutation axes
        # self.permute_block_size = 16
        self.core_input_size = core_input_size
        # self.num_slots = self.core_input_size // self.permute_block_size # should be 16 in the latest iteration
        # self.num_subvectors = self.input_size // self.core_input_size # for input_size = 1024, should be 4

        # initialize the temperature parameter
        # self.tau = nnx.Param(1.0) TODO: uncomment this if the old approach is better.

        # generate two independent permutation sequences
        # key = jax.random.key(1245)
        # key1, key2 = jax.random.split(key)
        p1 = jax.random.permutation(self.rngs.params(), self.num_slots)
        p2 = jax.random.permutation(self.rngs.params(), self.num_slots)

        # generate permutation matrices
        m1 = jnp.eye(self.num_slots)#*self.tau
        m2 = jnp.eye(self.num_slots)#*self.tau

        # generate the permutation matrices
        self.Ppos = nnx.Variable(m1[p1])
        # self.Ppos = nnx.Variable(jax.nn.softmax(self.Ppos, axis = -1))
        self.Pneg = nnx.Variable(m2[p2])
        # self.Pneg = nnx.Variable(jax.nn.softmax(self.Pneg, axis = -1))

    def __call__(self, x):
        """
        Applies permutation to input tensor.
        Args:
        x: jnp.ndarray, shape = (batch_size, out_blocks, in_blocks, 256) e.g. (8, 4, 256) for 1-24 - > 2048

        Returns:
        y: jnp.ndarray, shape = (batch_size, out_blocks, in_blocks, 256) e.g. (8, 4, 256) for 1-24 - > 2048
        """

        assert x.shape[-1] == self.core_input_size, f"Input shape is incorrect. Got {x.shape[-1]}, expected {self.core_input_size}"
        # assert self.num_subvectors * self.num_slots * self.permute_block_size == self.input_size, f"Inconsistent metrics!"

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.num_slots, -1)


        xpos = jnp.einsum("ij, ...jk -> ...ik", self.Ppos.value, x)
        xneg = jnp.einsum("ij, ...jk -> ...ik", self.Pneg.value, x)

        xout = xpos - xneg

        xout = xout.reshape(x.shape[0], x.shape[1], x.shape[2], -1)

        return xout

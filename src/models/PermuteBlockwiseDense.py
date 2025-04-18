# define the accumulator module
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers


class PermuteBlockwiseDense(nnx.Module):
    """
    Contains two fixed permutation matrices (pos and neg) to shuffle the input block-wise.
    This block should be used at the beginning of the pipeline so that balanced inputs are fed into the BlockwiseDense layer.
    """

    def __init__(self, 
                 input_size: int,
                 rngs: nnx.Rngs,
                 permute_block_size: int = 16,
                 core_input_size: int = 256,
                 ):
        
        self.input_size = input_size
        self.rngs = rngs

        # define permutation axes
        self.permute_block_size = permute_block_size
        self.core_input_size = core_input_size
        self.num_slots = self.core_input_size // self.permute_block_size # should be 16 in the latest iteration
        self.num_subvectors = self.input_size // self.core_input_size # for input_size = 1024, should be 4

        # initialize the temperature parameter TODO: uncomment this if the old approach is better.
        # self.tau = nnx.Param(1.0)

        # generate two independent permutation sequences
        p1 = jax.random.permutation(self.rngs.params(), self.num_slots)
        p2 = jax.random.permutation(self.rngs.params(), self.num_slots) # jnp.roll(p1, shift = 1) #

        # generate permutation matrices
        m1 = jnp.eye(self.num_slots)#*self.tau
        m2 = jnp.eye(self.num_slots)#*self.tau

        # generate the permutation matrices: TODO: remove the softmax if possible
        self.Ppos = nnx.Variable(m1[p1]) # Trying this approach
        # self.Ppos = nnx.Variable(jax.nn.softmax(self.Ppos, axis = -1))
        self.Pneg = nnx.Variable(m2[p2]) # Trying this appraoch
        # self.Pneg = nnx.Variable(jax.nn.softmax(self.Pneg, axis = -1))

    def __call__(self, x):
        """
        Apply permutations and return (xpos - xneg)
        Args:
        x: jnp.ndarray, input vector. Shape: (batch_size, input_size) e.g. (32, 2048)
        Returns:
        xpos - xneg: jnp.ndarray, difference of permuted inputs. Shape: (batch_size, input_size)
        """

        assert x.shape[-1] == self.input_size, f"Input shape is incorrect. Got {x.shape[-1]}, expected {self.input_size}"
        assert self.num_subvectors * self.num_slots * self.permute_block_size == self.input_size, f"Inconsistent metrics!"


        x = x.reshape(-1, self.num_subvectors, self.num_slots, self.permute_block_size) # first dimension must be the batch size

        xpos = jnp.einsum('ij,bsjp->bsip', self.Ppos.value, x)
        xneg = jnp.einsum('ij,bsjp->bsip', self.Pneg.value, x)

        xout = xpos - xneg

        xout = xout.reshape((-1, self.input_size))

        return xout
    

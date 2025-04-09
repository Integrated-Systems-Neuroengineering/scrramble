# define the accumulator module
import math
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers


class PermuteBlockwiseDense(nnx.Module):
    """
    Contains two fixed permutation matrices (pos and neg) to shuffle the input block-wise.
    This block should be used at the beginning of the pipeline so that balanced inputs are fed into the EICDense/
    """

    def __init__(self, 
                 input_size: int,
                 rngs: nnx.Rngs,
                 ):
        
        self.input_size = input_size
        self.rngs = rngs

        # define permutation axes
        self.permute_block_size = 16
        self.core_input_size = 256
        self.num_slots = math.ceil(self.core_input_size / self.permute_block_size) # should be 16 in the latest iteration
        self.num_subvectors = math.ceil(self.input_size / self.core_input_size) # for input_size = 1024, should be 4

        # initialize the temperature parameter
        self.tau = nnx.Param(1.0)

        # generate two independent permutation sequences
        # key = jax.random.key(1245)
        # key1, key2 = jax.random.split(key)
        p1 = jax.random.permutation(self.rngs.params(), self.num_slots)
        p2 = jax.random.permutation(self.rngs.params(), self.num_slots) # jnp.roll(p1, shift = 1) #

        # generate permutation matrices
        m1 = jnp.eye(self.num_slots)*self.tau
        m2 = jnp.eye(self.num_slots)*self.tau

        # generate the permutation matrices
        self.Ppos = m1[p1]
        self.Ppos = nnx.Variable(jax.nn.softmax(self.Ppos, axis = -1))
        self.Pneg = m2[p2]
        self.Pneg = nnx.Variable(jax.nn.softmax(self.Pneg, axis = -1))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply permutations and return (xpos - xneg)
        Args:
        x: jnp.ndarray, input vector. Shape: (batch_size, input_size) e.g. (32, 2048)
        Returns:
        xpos - xneg: jnp.ndarray, difference of permuted inputs. Shape: (batch_size, input_size)
        """

        assert x.shape[-1] == self.input_size, f"Input shape is incorrect. Got {x.shape[-1]}, expected {self.input_size}"
        assert self.num_subvectors * self.num_slots * self.permute_block_size == self.input_size, f"Inconsistent metrics!"

        # print("new per_d")

        # better to assume single inputs and let nnx.vmap handle batching
        x = x.reshape(self.num_subvectors, self.num_slots, self.permute_block_size)

        xpos = jnp.einsum('ij,sjp->sip', self.Ppos.value, x) # removed batch dimension
        xneg = jnp.einsum('ij,sjp->sip', self.Pneg.value, x) # removed batch dimension

        xout = xpos - xneg

        xout = xout.flatten() # without batch dimension, flattening is enough

        return xout
    
# test
# per = PermuteBlock(input_size = 2048)
# y_per = per(y_acc)
# plt.imshow(y_per.reshape(-1, 256))
# print(y_per.shape)
# # print(per.acc_cores.shape)
# # print(jax.tree.map(jnp.shape, nnx.state(per, nnx.Param)))
# nnx.display(per)
# define the accumulator module
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers


class PermuteBlock(nnx.Module):
    """
    Contains two fixed permutation matrices (pos and neg) to shuffle the input block-wise.
    This block should be used at the beginning of the pipeline so that balanced inputs are fed into the EICDense/
    """

    def __init__(self, 
                 input_size: int
                 ):
        
        # define permutation axes
        self.input_size = input_size
        self.permute_block_size = 16
        self.core_input_size = 256
        self.num_slots = self.core_input_size // self.permute_block_size # should be 16 in the latest iteration
        self.num_subvectors = self.input_size // self.core_input_size # for input_size = 1024, should be 4

        # initialize the temperature parameter
        self.tau = nnx.Param(1.0)

        # generate two independent permutation sequences
        key = jax.random.key(1245)
        key1, key2 = jax.random.split(key)
        p1 = jax.random.permutation(key1, self.num_slots)
        p2 = jax.random.permutation(key2, self.num_slots) # jnp.roll(p1, shift = 1) #

        # generate permutation matrices
        m1 = jnp.eye(self.num_slots)*self.tau
        m2 = jnp.eye(self.num_slots)*self.tau

        # generate the permutation matrices
        self.Ppos = m1[p1]
        self.Ppos = jax.nn.softmax(self.Ppos, axis = -1)
        self.Pneg = m2[p2]
        self.Pneg = jax.nn.softmax(self.Pneg, axis = -1)

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

        x = x.reshape(x.shape[0], self.num_subvectors, self.num_slots, self.permute_block_size) # first dimension must be the batch size

        xpos = jnp.einsum('ij,bsjp->bsip', self.Ppos, x)
        xneg = jnp.einsum('ij,bsjp->bsip', self.Pneg, x)

        xout = xpos - xneg

        xout = xout.reshape((x.shape[0], self.input_size))

        return xout
    
# test
# per = PermuteBlock(input_size = 2048)
# y_per = per(y_acc)
# plt.imshow(y_per.reshape(-1, 256))
# print(y_per.shape)
# # print(per.acc_cores.shape)
# # print(jax.tree.map(jnp.shape, nnx.state(per, nnx.Param)))
# nnx.display(per)
## define a module to split the input and shuffle it in blocks of 64
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn

class ShuffleBlocks(nn.Module):
    """
    Shuffles an input flat vector into blocks of 64. Mathematically emulating inter-core communication.

    Args:
        subvector_len: int, length of each subvector (typically 256 for EIC core)
        slot_len: int, length of each slot (typically 64 for EIC core)
        key: jax.random.PRNGKey, random key

    Returns:
        (xp - xn): jnp.ndarray, shuffled input vector of shape (input_len,)
    """
    subvector_len: int
    slot_len: int
    key: jax.random.key

    @nn.compact
    def __call__(self, x):
        """
        Shuffle input vector x, block-wise 
        Args:
        x: jnp.ndarray, input vector of shape (input_len,)

        Returns:
        x_shuffled: jnp.ndarray, shuffled input vector of shape (input_len,)
        """
        assert self.subvector_len % self.slot_len == 0, "Slot length must be an integer multiple of input_len"

        ## for comments consider x = (1024,) vector

        # determine how many blocks are in the input vector
        num_subvectors = x.shape[0]//self.subvector_len # e.g. 1024//256 = 4 subvectors
        slots_per_input = self.subvector_len//self.slot_len # e.g. 256//64 = 4 slots per input

        # reshape x into a 3D tensor of shape (num_subvectors, slots_per_input, slot_len), e.g. (4, 4, 64)
        x_reshaped = x.reshape(num_subvectors, slots_per_input, self.slot_len)

        # shuffle over slots_per_input dimension

        ## for positive vector
        key, subkey = jax.random.split(self.key)
        keys = jax.random.split(key, num_subvectors)

        shuffled_blocks_pos = [
            x_reshaped[i, jax.random.permutation(keys[i], slots_per_input, independent=True)] for i in range(num_subvectors)
        ]

        xpos = jnp.concatenate([blocks.reshape(-1) for blocks in shuffled_blocks_pos])

        # for negative vector
        key, subkey = jax.random.split(subkey)
        keys = jax.random.split(key, num_subvectors)

        shuffled_blocks_neg = [
            x_reshaped[i, jax.random.permutation(keys[i], slots_per_input, independent=True)] for i in range(num_subvectors)
        ]

        xneg = jnp.concatenate([blocks.reshape(-1) for blocks in shuffled_blocks_neg])

        return xpos - xneg




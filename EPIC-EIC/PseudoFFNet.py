# define the pseudo-feedforward network abstracting the EIC core
import jax
import jax.numpy as jnp
import optax
import flax
import matplotlib.pyplot as plt
from flax import linen as nn
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds
from collections import defaultdict
from EICDense import *
from ShuffleBlocks import *
from Accumulator import *
from HelperFunctions.binary_trident_helper_functions import *
from HelperFunctions.binary_mnist_dataloader import *

class PseudoFFNet(nn.Module):
    """
    Implement the forward pass. Needs to be twaeked to implement layers
    Current implementation:
        Dense layers: [1024, 2048, 256, 10 (256)]
    """

    key: jax.random.PRNGKey 

    @nn.compact
    def __call__(self, x):
        key, subkey = jax.random.split(self.key)
        
        #1024 -> 2048
        sh1 = ShuffleBlocks(
            subvector_len = 256,
            slot_len = 64,
            key = key
        )

        sh1_params = sh1.init(key, x)
        x_shuffled = sh1.apply(sh1_params, x)

        fc1 = EICDense(
            in_size = 1024,
            out_size = 2048,
            threshold = 0.0,
            noise_sd = 0.1,
            key = key,
            activation = custom_binary_gradient
        )

        fc1_params = fc1.init(key, x_shuffled)
        y = fc1.apply(fc1_params, x_shuffled)

        ac1 = Accumulator(
            in_block_size = y.shape[0],
            threshold = 0.0,
            noise_sd = 0.1,
            key = key,
            activation = custom_binary_gradient #lambda x, threshold, noise_sd, key: x
        )

        ac1_params = ac1.init(key, y)
        y = ac1.apply(ac1_params, y)

        # 2048 -> 256
        sh2 = ShuffleBlocks(
            subvector_len = 256,
            slot_len = 64,
            key = key
        )

        sh2_params = sh2.init(key, y)
        y = sh2.apply(sh2_params, y)

        fc2 = EICDense(
            in_size = 2048,
            out_size = 256,
            threshold = 0.0,
            noise_sd = 0.1,
            key = key,
            activation = custom_binary_gradient
        )

        fc2_params = fc2.init(key, y)
        y = fc2.apply(fc2_params, y)

        ac2 = Accumulator(
            in_block_size = y.shape[0],
            threshold = 0.0,
            noise_sd = 0.1,
            key = key,
            activation = custom_binary_gradient
        )

        ac2_params = ac2.init(key, y)
        y = ac2.apply(ac2_params, y)

        # 256 -> 10 (256)
        sh3 = ShuffleBlocks(
            subvector_len = 256,
            slot_len = 64,
            key = key
        )
        sh3_params = sh3.init(key, y)
        y = sh3.apply(sh3_params, y)

        fc3 = EICDense(
            in_size = 256,
            out_size = 10,
            threshold = 0.0,
            noise_sd = 0.1,
            key = key,
            activation = custom_binary_gradient
        )
        fc3_params = fc3.init(key, y)
        y = fc3.apply(fc3_params, y)

        ac3 = Accumulator(
            in_block_size = y.shape[0],
            threshold = 0.0,
            noise_sd = 0.1,
            key = key,
            activation = lambda x, threshold, noise_sd, key: x # linear map
        )

        ac3_params = ac3.init(key, y)
        y = ac3.apply(ac3_params, y)

        return y


# testing the network
def __main__():
    key = jax.random.key(0)
    x = jax.random.normal(key, (1024,))
    # y = toy_pipeline(x, key)
    # print(y.shape)
    # plt.matshow(y.reshape(-1, 256))
    net1 = PseudoFFNet(key = jax.random.key(0))
    rng = jax.random.PRNGKey(0)
    params = net1.init(rng, x)
    y = net1.apply(params, x)   
    print(y.shape)
    plt.matshow(y.reshape(-1, 64))
    plt.show()

if __name__ == "__main__":
    __main__()



        
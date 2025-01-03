import jax
import jax.numpy as jnp
import optax
import flax
import matplotlib.pyplot as plt
from flax import linen as nn
from EICDense import *
from ShuffleBlocks import *
from Accumulator import *
from HelperFunctions.binary_trident_helper_functions import *
from HelperFunctions.binary_mnist_dataloader import *

class EICNet(nn.Module):
    """
    EICNetwork -> Might supersede PseudoFFNet... in progress
    1024 - 2048 - 256 - 10
    """
    def setup(self):
        nsd = 1.
        self.fc1 = EICDense(in_size = 256, out_size = 2048, threshold = 0.0, noise_sd = nsd, activation = custom_binary_gradient) # (8, 4, 256)
        self.ac1 = Accumulator(in_block_size = 2048//256, threshold = 0., noise_sd = nsd, activation = custom_binary_gradient) # (2048,)
        self.fc2 = EICDense(in_size = 2048, out_size = 256, threshold = 0.0, noise_sd = nsd, activation = custom_binary_gradient) 
        self.ac2 = Accumulator(in_block_size = 1, threshold = 0., noise_sd = nsd, activation = None)
        # self.fc3 = EICDense(in_size = 256, out_size = 10, threshold = 0.0, noise_sd = nsd, activation = custom_binary_gradient)
        # self.ac3 = Accumulator(in_block_size = 1, threshold = 0., noise_sd = nsd, activation = None)
        # print(f"Noise SD: {nsd}")

    def __call__(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        # x = self.fc3(x)
        # x = self.ac3(x)
        return x
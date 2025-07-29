"""
ScRRAMBLe-CapsNet with digit reconstruction network for MNIST dataset.
"""

import jax
import math
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
import json
import os
import pickle
from functools import partial
from datetime import date


from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste, squash
# from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist

# -------------------------------------------------------------------
# Define the Reconstruction network
# -------------------------------------------------------------------
class ReconstructionNetwork(nnx.Module):
    def __init__(self,
                 input_size: int,
                 rngs: nnx.Rngs):
        
        # define feedforward layers
        self.fc1 = nnx.Linear(input_size, 5000, rngs=rngs)
        self.fc2 = nnx.Linear(5000, 3000, rngs=rngs)
        self.fc3 = nnx.Linear(3000, 28*28, rngs=rngs)

    def __call__(self, x):

        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        x = nnx.sigmoid(self.fc3(x))

        return x


# -------------------------------------------------------------------
# Define the MNIST CapsNet Model
# -------------------------------------------------------------------
class ScRRAMBLeCapsNetWithReconstruction(nnx.Module):
    """
    ScRRAMBLe CapsNet model for MNIST classification.

    Notes:
    - Currently assumes that the connection probability is the same for all the layers.
    """

    def __init__(
            self,
            input_vector_size: int, # size of flattened input vector
            capsule_size: int, # size of each capsule e.g. 256 (number of columns/rows of a core)
            receptive_field_size: int, # size of each receptive field e.g. 64 (number of columns/rows of a slot)
            connection_probability: float, # fraction of total receptive fields on sender side that each receiving slot/receptive field takes input from
            rngs: nnx.Rngs,
            layer_sizes: list = [20, 10, 10], # number of capsules in each layer of the capsnet. e.g. [20, 10] means 20 capsules in layer 1 and 10 capsules in layer 2
            activation_function: Callable = nnx.relu, # activation function to use in the network
    ):
        
        self.input_vector_size = input_vector_size
        self.capsule_size = capsule_size 
        self.receptive_field_size = receptive_field_size
        self.rngs = rngs
        self.connection_probability = connection_probability
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function

        # calculate the effective capsules in input vector rouded to the nearest integral multiple of capsule size
        self.input_eff_capsules = math.ceil(self.input_vector_size/self.capsule_size)

        # add this element as the first element of layer_sizes
        self.layer_sizes.insert(0, self.input_eff_capsules)

        # define ScRRAMBLe capsules
        self.scrramble_caps_layers = [ScRRAMBLeCapsLayer(
            input_vector_size=self.capsule_size * Nci,
            num_capsules=Nco,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs
        ) for Nci, Nco in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

        # define the reconstruction network
        self.reconstruction_nw = ReconstructionNetwork(
            input_size=self.capsule_size * self.layer_sizes[-1],  # Input size
            rngs=self.rngs
        )


    def __call__(self, x:jax.Array) -> jax.Array:
        """
        Forward pass through the ScRRAMBLe CapsNet
        """

        # resize the image to be (32, 32) for MNIST
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')

        # flatten the first two dimensions
        x = jnp.reshape(x, (x.shape[0], -1))

        # pass the input through the layers
        for layer in self.scrramble_caps_layers:
            x = jax.vmap(layer, in_axes=(0,))(x)
            x = jnp.reshape(x, (x.shape[0], -1))
            shape_x = x.shape
            # x = x.flatten()
            # x = jax.vmap(self.activation_function, in_axes=(0, None, None))(x, 8, 1.0) # 8 bits, 1.0 is the max clipping threshold.
            x = self.activation_function(x)  # Apply the activation function.
            x = jnp.reshape(x, shape_x)

        # add the reconstruction network
        x_recon = x.reshape((x.shape[0], -1))  # Flatten the output for reconstruction.
        x_recon = jax.vmap(self.reconstruction_nw, in_axes=(0,))(x_recon)  # Apply the reconstruction network.


        return x_recon, x
    

# # testing
# def __main__():
#     key = jax.random.key(10)
#     key1, key2, key3, key4 = jax.random.split(key, 4)
#     rngs = nnx.Rngs(params=key1, activations=key2, permute=key3, default=key4)

#     model = ScRRAMBLeCapsNetWithReconstruction(
#         input_vector_size=1024,
#         capsule_size=256,
#         receptive_field_size=64,
#         connection_probability=0.2,
#         rngs=rngs,
#         layer_sizes=[50, 10],  # 20 capsules in the first layer and (translates to sum of layer_sizes cores total)
#         activation_function=nnx.relu
#     )

#     # dummy input
#     x_test = jax.random.normal(rngs.default(), (10, 28, 28, 1))
#     recon, caps_out = model(x_test)
#     print("Reconstruction shape:", recon.shape)
#     print("Capsule output shape:", caps_out.shape)

# if __name__ == "__main__":
#     __main__()

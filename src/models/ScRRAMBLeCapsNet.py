"""
Use this script to train ScRRAMBLe CapsNet on MNIST. 

Created on: 07/02/2025
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
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from datetime import date

from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste
from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

# -------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------
def save_metrics(metrics_dict, filename):
    """
    Save the metrics to a file.
    Args:
        metrics_dict: dict, metrics to save.
        filename: str, name of the file to save the metrics to.
    """

    metrics_dir = "/local_disk/vikrant/scrramble/logs"
    filename = os.path.join(metrics_dir, filename)

    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)

    print(f"Metrics saved to {filename}")

    
def save_model(state, filename):
    """
    Save the model state in a specified file
    """

    checkpoint_dir = "/local_disk/vikrant/scrramble/models"
    filename_ = os.path.join(checkpoint_dir, filename)

    with open(filename_, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"Model saved to {filename_}")

# -------------------------------------------------------------------
# Define the MNIST CapsNet Model
# -------------------------------------------------------------------

class ScRRAMBLeCapsNet(nnx.Module):
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
    ):
        
        self.input_vector_size = input_vector_size
        self.capsule_size = capsule_size
        self.receptive_field_size = receptive_field_size
        self.rngs = rngs
        self.connection_probability = connection_probability
        self.layer_sizes = layer_sizes

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
            x = x.flatten()
            x = jax.vmap(quantized_relu_ste, in_axes=(0, None, None))(x, 8, 1.0) # 8 bits, 1.0 is the max clipping threshold.
            x = jnp.reshape(x, shape_x)

        return x

# testing
def __main__():
    rngs = nnx.Rngs(params=0, activations=1, permute=2, default=3564132)

    model = ScRRAMBLeCapsNet(
        input_vector_size=1024,
        capsule_size=256,
        receptive_field_size=64,
        connection_probability=0.2,
        rngs=rngs,
        layer_sizes=[20, 10]  # 20 capsules in the first layer and

    )

    print(f"Model number of capsules/effective capsules for input: {model.layer_sizes}")

    x = jax.random.normal(rngs.default(), (10, 32, 32, 1))
    out = model(x)

    # print the output shape
    print(f"Output shape: {out.shape}")

    out = jnp.reshape(out, (out.shape[0], model.layer_sizes[-1], -1))

    print(f"Output shape after reshaping: {out.shape}")

    print(f"Some outputs: {out[0, 0, :10]}")

# if __name__ == "__main__":
#     __main__()

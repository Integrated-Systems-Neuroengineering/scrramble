"""
ScRRAMBLE-CIFAR10 Pilot Script.
- Architecture:
ScRRAMBLe Routing between capsules
Skip connection from input to parent capsules
Reconstruction error regularizer

Created on: 07/15/2025

Author: Vikrant Jaltare
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

from utils.activation_functions import quantized_relu_ste, squash
from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations. 

# -------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------
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
# Defining the model
# -------------------------------------------------------------------

class ReconstructionLayer(nnx.Module):
    """
    Feedforward layer that reconstructs the input from parent capsules.
    """

    def __init__(self,
                 *,
                 rngs: nnx.Rngs):
        
        
        self.linear1 = nnx.Linear(in_features=2560, out_features=5000, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=5000, out_features=4000, rngs=rngs)
        self.linear3 = nnx.Linear(in_features=4000, out_features=3072, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through reconstruction layer
        """

        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = nnx.sigmoid(self.linear(3))

        return x



class ScRRAMBLeCIFAR(nnx.Module):

    def __init__(self, 
                input_vector_size: int,
                num_primary_capsules: int,
                num_parent_capsules: int,
                connection_probability: float,
                rngs: nnx.Rngs,
                receptive_field_size: int = 64,
                capsule_size: int = 256,
                activation_function: Callable = nnx.relu,
                fc_layer_sizes = [256, 5000, 3072]):

        self.input_vector_size = input_vector_size
        self.num_primary_capsules = num_primary_capsules
        self.num_parent_capsules = num_parent_capsules
        self.connection_probability = connection_probability
        self.rngs = rngs
        self.receptive_field_size = receptive_field_size
        self.capsule_size = capsule_size

        self.receptive_field_per_capsule = math.ceil(self.capsule_size / self.receptive_field_size)

        self.activation_function = activation_function
        self.fc_layer_sizes = fc_layer_sizes

        # calculate the effective capsules that the input is made of
        self.input_eff_capsules = math.ceil(self.input_vector_size/self.capsule_size)

        # construct input -> primary capsule ScRRAMBLe Layer
        self.primary_caps_layer = ScRRAMBLeCapsLayer(
            input_vector_size=self.input_vector_size,
            num_capsules=self.num_primary_capsules,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs,
        )

        # define ScRRAMBLe projection between input and parent capsules a.k.a Skip Connection
        self.C_input_to_parent = ScRRAMBLe_routing(
            input_cores=self.input_eff_capsules,
            output_cores=self.num_parent_capsules, # number of parent capsules!
            receptive_fields_per_capsule=self.receptive_field_per_capsule,
            connection_probability=self.connection_probability,
            key=self.rngs.params(),
            with_replacement=True
        )

        self.C_input_to_parent = nnx.Variable(self.C_input_to_parent)

        #TODO: Add code for primary -> parent capsules and FC layer

        # routing between primary and parent capsules
        self.parent_capsule_layer = ScRRAMBLeCapsLayer(
            input_vector_size=self.num_orimary_capsules*self.capsule_size,
            num_capsules=self.num_parent_capsules,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs,
        )

        # define the fully connected layer
        self.reconstruction_layer = ReconstructionLayer(rngs=self.rngs)

    @staticmethod
    def spatial_block_reshape(x: jax.Array) -> jax.Array:
        """
        Reshape the input image to a 1-D vector, preserving the spatial relationships. 
        """

        x_blocks = x.reshape(2, 2, 16, 16, 3).reshape(256, 12)
        x_flat = x_blocks.reshape(-1) # 3072

        return x_flat
    
    def skip_connection(self, x:jax.Array) -> jax.Array:
        """
        Skip connection from input to parent capsules
        """
        # reshape x into (input capsules, receptive fields per capsule, receptive field size])
        x_reshaped = x.reshape(self.input_eff_capsules, self.receptive_field_per_capsule, self.receptive_field_size)

        # perform ScrRAMBLe routing
        x_routed = jnp.einsum('ijkl,ijm->klm', self.C_input_to_parent, x_reshaped)

        # flatten it before returning
        x_routed_flat = x_routed.reshape(-1)
        return x_routed_flat
    
    def get_active_capsule(self, x:jax.Array) -> jax.Array:

        """
        Assumes the input arrives from the parent capsule layer in form (parent capsules, receptive fields per capsule, receptive field size)
        """

        x_reshape = x.reshape(x.shape[0], -1)

        # take norm along the last dimension
        norms = jnp.linalg.norm(x_reshape, axis=-1)

        # pick the argmax
        active_capsule = jnp.argmax(norms, axis=-1)

        # construct the mask
        mask = jnp.zeros(self.capsule_size*self.num_parent_capsules)
        mask = mask.at[active_capsule:(active_capsule+1)*self.capsule_size-1].set(1.0)
        # mask = nnx.Variable(mask)

        return mask




    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the CIFAR10 model.
        Args:
            x: jax.Array, input image of shape (B, 32, 32, 3)
        """

        # flatten input using spatial block reshape
        x = jax.vmap(self.spatial_block_reshape, in_axes=(0,))(x)  # (B, 3072)

        # pass through the primary capsule layer
        x = jax.vmap(self.primary_caps_layer, in_axes=(0,))(x)

        # construct the skip connection input
        x_skip = jax.vmap(self.skip_connection, in_axes=(0,))(x)

        # pass through the parent capsule layer
        x = (x_skip + jax.vmap(self.parent_capsule_layer, in_axes=(0,))(x))

        # only consider the most active parent capsule
        x_recon = x.reshape(-1)
        mask = jax.vmap(self.get_active_capsule, in_axes=(0,))(x)
        x_recon = mask * x_recon

        # pass through the reconstruction layer
        x_recon = jax.vmap(self.reconstruction_layer, in_axes=(0,))(x_recon)

        return x_recon, x



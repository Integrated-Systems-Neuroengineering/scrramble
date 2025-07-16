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




        


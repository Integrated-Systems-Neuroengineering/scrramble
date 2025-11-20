"""
ScRRAMBLe + Residual Blocks for CIFAR-10 Dataset

Created on 11/19/2025
Author: Vikrant Jaltare

Best accuracy so far: N/A
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

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns



from utils.activation_functions import quantized_relu_ste, squash
from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_cifar10, fast_scrramble



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

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists.

    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)

    print(f"Metrics saved to {filename}")

    
def save_model(state, filename):
    """
    Save the model state in a specified file
    """

    checkpoint_dir = "/local_disk/vikrant/scrramble/models"
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"Model saved to {filename_}")

# -------------------------------------------------------------------
# Residual Block: outputs shape 2048
# -------------------------------------------------------------------
class ReasidualBlock(nnx.Module):
    """
    Residual block with two convolutional layers and a residual connection.
    Order of operations:
    input -> conv -> relu -> conv + input -> relu
    """

    def __init__(self,
                 kernel_size: tuple[int, int],
                 in_features: int,
                 out_features: int,
                 rngs: nnx.Rngs,
                 padding: str = "SAME",
                 **kwargs):
        
        self.conv1 = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(out_features, rngs=rngs)
        
        # Projection layer: use 1x1 conv when input and output features differ
        self.use_projection = (in_features != out_features)
        if self.use_projection:
            self.projection = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=(1, 1), padding=padding, rngs=rngs)

    def __call__(self, x):
        x_res = x
        
        # Project residual connection if dimensions don't match
        if self.use_projection:
            x_res = self.projection(x_res)
        
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = x + x_res 
        x = nnx.relu(x)
        return x
    
# -------------------------------------------------------------------
# ScRRAMBLE + Res Network
# -------------------------------------------------------------------
class ScRRAMBLeResCIFAR10(nnx.Module):
    """
    ScRRAMBLe + Residual Network for CIFAR-10 classification.
    """
    
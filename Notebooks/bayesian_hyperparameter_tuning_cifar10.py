"""
Test run of Bayesian Hpyerparameter optimization on CIFAR-10 dataset.
Parameters under test:
- learning rate
- batch size
- connection probability
- Primary capsule size
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
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from tqdm import tqdm
from datetime import date
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


from models import ScRRAMBLeCIFAR


from utils.activation_functions import quantized_relu_ste, squash
from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_cifar10



import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

# -------------------------------------------------------------------
# TODO:
#
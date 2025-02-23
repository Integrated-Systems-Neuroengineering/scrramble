"""
Script to train the EIC model.
"""
import sys
import os
from functools import partial
import jax
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from flax.training import train_state
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HelperFunctions")))

# import the modules
from EICNetwork import *
from HelperFunctions.binary_mnist_dataloader import *


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# #---------------------------------------------------------------
# # Define helper functions
# #---------------------------------------------------------------

# # Quantize parameters
# def quantize_params(params, bits = 8):
#     """
#     Quantizes the parameters of the model to given number of bits.
#     Args:
#         params: flax model parameters
#         bits: number of bits to quantize to
#     Returns:
#         quantized_params: quantized flax model parameters
#     """

#     scale = 2**(bits - 1) - 1
#     params = jax.tree.map(
#         lambda p: jnp.round(p * scale) / scale, params
#     )

#     return params

# # metrics
# def compute_metrics(*, logits, labels):
#   loss = cross_entropy_loss(logits=logits, labels=labels)
#   accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
#   metrics = {
#       'loss': loss,
#       'accuracy': accuracy,
#   }
#   return metrics

# #---------------------------------------------------------------
# # Define loss functions
# #---------------------------------------------------------------

# # cross entropy loss
# def cross_entropy_loss(*, logits, labels):
#     one_hot_labels = jax.nn.one_hot(labels, num_classes = 10)
#     loss = optax.softmax_cross_entropy(logits = logits, labels = one_hot_labels).mean()
#     return loss




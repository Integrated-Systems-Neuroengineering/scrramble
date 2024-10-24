import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

"""
Classes containing neuron type and network architecture functions.
Use this file in a jupyter notebook to run experiments.
"""


"""
Neuron parent class

Definitions:
- state_function: Implements ternary state updates.
- expected_state: Compute the expected state for the neuron, used during backpropagation.
- 
"""
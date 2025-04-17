import jax
import jax.numpy as jnp
from typing import Callable
import flax
from flax import nnx

from models import BlockwiseDense, WeightSharingAccumulator, PermuteWeightSharing, PermuteBlockwiseDense
from utils import clipping_ste, rram_quantize

"""
Define a test MLP architecture
"""


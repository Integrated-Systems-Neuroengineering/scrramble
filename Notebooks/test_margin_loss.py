"""
Testing margin loss
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
import math

from utils.loss_functions import margin_loss
from models import ScRRAMBLeCapsNet

rngs = nnx.Rngs(params=0, activations=1, permute=2, default=3564132)
model = ScRRAMBLeCapsNet(
        input_vector_size=1024,
        capsule_size=256,
        receptive_field_size=64,
        connection_probability=0.2,
        rngs=rngs,
        layer_sizes=[20, 10]  # 20 capsules in the first layer and

    )

margin_loss(model, {'image': jax.random.normal(rngs.default(), (10, 32, 32, 1)), 'label': jax.random.randint(rngs.default(), (10,), 0, 10)})
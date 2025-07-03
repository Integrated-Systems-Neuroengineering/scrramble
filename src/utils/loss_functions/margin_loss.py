"""
Loss functions for ScRRAMBLe networks
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
import optax
from functools import partial


# -------------------------------------------
# Margin Loss from Capsule Networks
# -------------------------------------------
def margin_loss(
    model,
    batch,
    m_plus: float = 0.9,
    m_minus: float = 0.1,
    lambda_: float = 0.5
    ):

    caps_output = model(batch['image']) # this output will be in shape (batch_size, num_output_cores (10), slots/receptive fields per core, slot/receptive_field_length)
    print(f"Caps output shape: {caps_output.shape}")

    # the length of the vector encodes probability of a class
    caps_output = caps_output.reshape(caps_output.shape[0], caps_output.shape[1], -1)
    print(f"Caps output reshaped: {caps_output.shape}") # at this point this should be (batch_size, num_output_cores, 256) for the default core length of 256

    caps_output_magnitude = jnp.linalg.norm(caps_output, axis=-1)
    print(f"Caps output magnitude: {caps_output_magnitude.shape}") # this should be (batch_size, num_output_cores (10))

    # create one-hot-encoded labels
    labels = batch['label']
    labels = jax.nn.one_hot(labels, num_classes=caps_output_magnitude.shape[1])
    print(f"Labels shape: {labels.shape}") # this should be (batch_size, num_output_cores)

    # compute the margin loss
    loss = jnp.sum(labels * jax.nn.relu(m_plus - caps_output_magnitude)**2 + lambda_ * (1 - labels) * jax.nn.relu(caps_output_magnitude - m_minus)**2)
    print(f"Loss: {loss}")

    return loss, caps_output_magnitude
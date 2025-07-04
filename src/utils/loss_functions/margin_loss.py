"""
Loss functions for ScRRAMBLe networks
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
import optax
from functools import partial
# from models.ScRRAMBLeCapsLayer import ScRRAMBLeCapsLayer
# from models.ScRRAMBLeCapsNet import ScRRAMBLeCapsNet


# -------------------------------------------
# Margin Loss from Capsule Networks
# -------------------------------------------
def margin_loss(
    model,
    batch,
    num_classes: int = 10,
    m_plus: float = 0.9,
    m_minus: float = 0.1,
    lambda_: float = 0.5
    ):

    caps_output = model(batch['image']) # this output will be in shape (batch_size, num_output_cores (10), slots/receptive fields per core, slot/receptive_field_length)
    print(f"Caps output shape: {caps_output.shape}")

    # the length of the vector encodes probability of a class
    caps_output = caps_output.reshape(caps_output.shape[0], num_classes, -1)
    print(f"Caps output reshaped: {caps_output.shape}") # at this point this should be (batch_size, num_output_cores, 256) for the default core length of 256

    caps_output_magnitude = jnp.linalg.norm(caps_output, axis=-1)
    print(f"Caps output magnitude: {caps_output_magnitude}") # this should be (batch_size, num_output_cores (10))
    print(f"Caps output magnitude shape: {caps_output_magnitude.shape}") # this should be (batch_size, num_output_cores (10))

    # create one-hot-encoded labels
    labels = batch['label']
    labels = jax.nn.one_hot(labels, num_classes=caps_output_magnitude.shape[1])
    print(f"Labels shape: {labels.shape}") # this should be (batch_size, num_output_cores)

    # compute the margin loss
    loss = jnp.sum(labels * jax.nn.relu(m_plus - caps_output_magnitude)**2 + lambda_ * (1 - labels) * jax.nn.relu(caps_output_magnitude - m_minus)**2)
    print(f"Loss: {loss}")

    return loss, caps_output_magnitude


# -------------------------------------------
# Testing
# -------------------------------------------
# def __main__():
#     rngs = nnx.Rngs(params=0, activations=1, permute=2, default=3564132)

#     model = ScRRAMBLeCapsNet(
#         input_vector_size=1024,
#         capsule_size=256,
#         receptive_field_size=64,
#         connection_probability=0.2,
#         rngs=rngs,
#         layer_sizes=[20, 10]  # 20 capsules in the first layer and

#     )

#     print(f"Model number of capsules/effective capsules for input: {model.layer_sizes}")

#     x = jax.random.normal(rngs.default(), (10, 32, 32, 1))
#     out = model(x)

#     out = jnp.reshape(out, (out.shape[0], model.layer_sizes[-1], -1))


# if __name__ == "__main__":
#     __main__()
"""
Define Metric functions for models
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit

def accuracy(params, model, input, targets, keys):
    """
    Compute accuracy of the model.
    Args:
    params: flax model parameters
    model: flax model
    input: jnp.ndarray, input data
    targets: jnp.ndarray, target data
    """

    logits = vmap(lambda img, key: model.apply(params, img, rngs = {"activation": key}))(input, keys)
    pred_labels = jnp.argmax(logits[:, :10], axis = -1)
    acc = jnp.mean(pred_labels == targets)

    return acc

def cross_entropy_loss(logits, targets):
    """
    Compute cross entropy loss.
    Args:
    logits: jnp.ndarray, predicted labels
    targets: jnp.ndarray, targets
    """

    one_hot_labels = jax.nn.one_hot(targets, num_classes = 256)
    probits = jax.nn.log_softmax(logits, axis = -1)
    ce_loss = -jnp.mean(jnp.sum(one_hot_labels * probits, axis = -1))

    return ce_loss


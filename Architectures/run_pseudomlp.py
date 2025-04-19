
import jax
import math
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
import orbax.checkpoint as ocp
import json
import os
import pickle

from PseudoMLP import PseudoMLP
from utils import clipping_ste

import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

import matplotlib.pyplot as plt
import seaborn as sns

# pick the data path
DATA_PATH = f"/local_disk/vikrant/datasets/"

# import the model
layers = [1024, 2048, 1024, 256] # 19 cores setup [1024, 512, 512, 256], 61 cores setup: [1024, 2048, 512, 256]
rngs = nnx.Rngs(params=134, activation=67565)
model = PseudoMLP(
    layers=layers,
    rngs=rngs,
    dense_activation_fn=clipping_ste,
    accumulator_activation_fn=clipping_ste,
    threshold=0.0,
    noise_sd=1e-2
)

print(f"No. of cores: {model.get_num_cores()}")

# ----------------------------------------
# Define data loader
# ----------------------------------------
tf.random.set_seed(0)  # Set the random seed for reproducibility.

train_steps = 10000
eval_every = 1000
batch_size = 256
train_ds: tf.data.Dataset = tfds.load('mnist', split='train', data_dir=DATA_PATH)
test_ds: tf.data.Dataset = tfds.load('mnist', split='test', data_dir=DATA_PATH)

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # normalize train set
test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # Normalize the test set.

# Binarize images: here we use tf.round which converts values <0.5 to 0 and >=0.5 to 1.
binarize = lambda sample: {
    'image': tf.round(sample['image']),
    'label': sample['label'],
}
train_ds = train_ds.map(binarize)
test_ds = test_ds.map(binarize)

# Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
train_ds = train_ds.repeat().shuffle(1024)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

print("Data loaded successfully!")

# ----------------------------------------
# Define the optimizer and loss function
# ----------------------------------------
learning_rate = 1e-3
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

# nnx.display(optimizer)

def loss_fn(model: PseudoMLP, batch):
  logits = model(batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model: PseudoMLP, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: PseudoMLP, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

print("Training and evaluation functions defined successfully!")


# ----------------------------------------
# Training
# ----------------------------------------
metrics_history = {
'train_loss': [],
'train_accuracy': [],
'test_loss': [],
'test_accuracy': [],
}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
  # Run the optimization for one step and make a stateful update to the following:
  # - The train state's model parameters
  # - The optimizer state
  # - The training loss and accuracy batch metrics

  train_step(model, optimizer, metrics, batch)

  if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
    # Log the training metrics.
    for metric, value in metrics.compute().items():  # Compute the metrics.
      metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
    metrics.reset()  # Reset the metrics for the test set.

    # Compute the metrics on the test set after each training epoch.
    for test_batch in test_ds.as_numpy_iterator():
      eval_step(model, metrics, test_batch)

    # Log the test metrics.
    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    metrics.reset()  # Reset the metrics for the next training epoch.

    print("------------------------")
    print(f"Step {step}: Test loss: {metrics_history['test_loss'][-1]}, Accuracy: {metrics_history['test_accuracy'][-1]}")
    print(f"Step {step}: Train loss: {metrics_history['train_loss'][-1]}, Accuracy: {metrics_history['train_accuracy'][-1]}")
    print("------------------------")

best_accuracy = max(metrics_history['test_accuracy'])
print(f"Best accuracy: {best_accuracy}")

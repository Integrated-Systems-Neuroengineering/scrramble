"""
Use this script to train ScRRAMBLE on MNIST.
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
import pandas as pd
from datetime import date

from utils import clipping_ste, intercore_connectivity, load_mnist
from models import ScRRAMBLeLayer


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

# -------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------
def save_metrics(metrics_dict, filename):
    """
    Save the metrics to a file.
    Args:
        metrics_dict: dict, metrics to save.
        filename: str, name of the file to save the metrics to.
    """

    metrics_dir = "/local_disk/vikrant/scrramble/logs"
    filename = os.path.join(metrics_dir, filename)

    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)

    print(f"Metrics saved to {filename}")

    
def save_model(state, filename):
    """
    Save the model state in a specified file
    """

    checkpoint_dir = "/local_disk/vikrant/scrramble/models"
    filename_ = os.path.join(checkpoint_dir, filename)

    with open(filename_, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"Model saved to {filename_}")


# -------------------------------------------------------------------
# Define the MNIST classifier
# ------------------------------------------------------------------
class ScRRAMBLeMNIST(nnx.Module):
    """
    MNIST Classifier using ScRRAMBLE architecture
    """

    def __init__(
        self,
        input_vector_size: int,
        input_cores: int,
        output_cores: int,
        avg_slot_connectivity: int, 
        slots_per_core: int,
        slot_length: int,
        activation: Callable,
        rngs: nnx.Rngs,
        group_size: int,
        core_length: int = 256,
        threshold: float = 0.0,
        noise_sd: float = 0.05,


    ):

        self.input_vector_size = input_vector_size
        self.input_cores = input_cores
        self.output_cores = output_cores
        self.avg_slot_connectivity = avg_slot_connectivity
        self.slots_per_core = slots_per_core
        self.slot_length = slot_length
        self.rngs = rngs
        self.group_size = group_size
        self.core_length = core_length
        self.threshold = threshold
        self.noise_sd = noise_sd
        self.activation = partial(activation, threshold=threshold, noise_sd=noise_sd, key=rngs.activation())

        # define the scrramble layer
        self.scrramble_layer = ScRRAMBLeLayer(
            input_vector_size=input_vector_size,
            input_cores=input_cores,
            output_cores=output_cores,
            avg_slot_connectivity=avg_slot_connectivity,
            slots_per_core=slots_per_core,
            slot_length=slot_length,
            activation=activation,
            rngs=rngs,
            core_length=core_length,
            threshold=threshold,
            noise_sd=noise_sd
        )

    @staticmethod
    def chunkwise_reshape(x):
        """
        Reshape the input in a block-wise manner similar to how a conv layer would process an image
        Args:
            x: jax.Array, input data (image)
        Returns:
            x: jax.Array, flattened data
        """
        # reshape into 32x32
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')

        # reshape again
        x = jnp.reshape(x, (x.shape[0], 8, 8, 4, 4, 1))

        # flatten along the last two dimensions
        x = jnp.reshape(x, (x.shape[0], 8, 8, -1))

        # flatten the first two dimensions
        x = jnp.reshape(x, (x.shape[0], -1))

        return x


    @partial(nnx.jit, static_argnames=['output_coding'])
    def __call__(self, x, output_coding: str = 'population'):
        """
        Forward pass for the ScRRAMBLe MNIST classifier
        Args:
            x: jax.Array, input data. Assumed to be flattened MNIST image. No batch.
            output_coding: str, specifies how the binary output should be interpreted. Choices are: ['population', 'svm', ...]. Only 'population' is implemented so far.
        Returns:
            out: jax.Array, output of classifier with population coding applied. (batch_size, group_size)
        """

        # reshape the image
        # print(x.shape)
        # x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')
        # x = x.reshape(x.shape[0], -1)

        x = self.chunkwise_reshape(x)

        # using vmap do the forward pass
        y = nnx.vmap(self.scrramble_layer, in_axes=0)(x)

        # check if population coding is used
        if output_coding == 'population':

            # truncation
            y_reshaped = y.reshape(y.shape[0], y.shape[1], -1)
            y_reshaped = y_reshaped[..., :250]

            y_reshaped = y_reshaped.reshape(y_reshaped.shape[0], self.group_size, -1)
            y_reshaped = jnp.mean(y_reshaped, axis=-1)
            
            return y_reshaped
            
        else:
            raise NotImplementedError("Non-population coding not implemented yet.")


    def get_params(self):
        """
        Get the number of parameters in the model
        Naive implementation
        """
        Wi = self.scrramble_layer.Wi.value.flatten()
        Wo = self.scrramble_layer.Wo.value.flatten()
        params = Wi.shape[0] + Wo.shape[0]

        return params


    
# -------------------------------------------------------------------
# Loading the dataset
# ------------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 64, # 64 is a good batch size for MNIST
    'train_steps': 20000, # run for longer, 20000 is good!
    'binarize': True, 
    'greyscale': True,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 500,
}

train_ds, test_ds = load_mnist(
    batch_size=dataset_dict['batch_size'],
    train_steps=dataset_dict['train_steps'],
    binarize=dataset_dict['binarize'],
    greyscale=dataset_dict['greyscale'],
    data_dir=dataset_dict['data_dir'],
    seed=dataset_dict['seed'],
    shuffle_buffer=dataset_dict['shuffle_buffer'],
    threshold=dataset_dict['threshold'],
)

# -------------------------------------------------------------------
# Training functions
# -------------------------------------------------------------------
def loss_fn(model: ScRRAMBLeMNIST, batch):
  logits = model(batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits

@nnx.jit
def train_step(model: ScRRAMBLeMNIST, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeMNIST, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.


# -------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

# 16 cores give 1,048,576 parameters
# 20 cores gives 2,097,152 parameters
total_cores = 32
ni = 20
ni, no = [ni, total_cores - ni]

print(f"No. of cores = {ni + no}")

rngs = nnx.Rngs(params=0, activation=1, permute=2)
model = ScRRAMBLeMNIST(
    input_vector_size=32*32,
    input_cores=ni,
    output_cores=no,
    avg_slot_connectivity=12,
    slots_per_core=4,
    slot_length=64,
    activation=clipping_ste,
    rngs=rngs,
    group_size=10,
    core_length=256,
    threshold=0.0,
    noise_sd=0.05 # standard deviation of the noise distribution (typical value = 0.05)
)

print(f"Model parameters: {model.get_params()}")

# optimizers
hyperparameters = {
    'learning_rate': 7e-4, # 1e-3 seems to work well
    'momentum': 0.9, 
    'weight_decay': 1e-4
}

optimizer = nnx.Optimizer(
    model,
    optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
)

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss')
)


metrics_history = {
'train_loss': [],
'train_accuracy': [],
'test_loss': [],
'test_accuracy': [],
}

def train_scrramble_mnist(model = model, 
                            optimizer = optimizer, 
                            metrics = metrics, 
                            train_ds = train_ds, 
                            test_ds = test_ds, 
                            dataset_dict = dataset_dict,
                            save_model_flag = True,
                            save_metrics_flag = True):

    eval_every = dataset_dict['eval_every']
    train_steps = dataset_dict['train_steps']

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        # Run the optimization for one step and make a stateful update to the following:
        # - The train state's model parameters
        # - The optimizer state
        # - The training loss and accuracy batch metrics

        train_step(model, optimizer, metrics, batch)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(float(value))  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            for test_batch in test_ds.as_numpy_iterator():
                eval_step(model, metrics, test_batch)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(float(value))
            metrics.reset()  # Reset the metrics for the next training epoch.
            
            print(f"Step {step}: Test loss: {metrics_history['test_loss'][-1]}, Accuracy: {metrics_history['test_accuracy'][-1]}")

    best_accuracy = max(metrics_history['test_accuracy'])
    print(f"Best accuracy: {best_accuracy}")

    if save_model_flag:
        today = date.today().isoformat()
        filename = f"sscamble_mnist_model_ci_{model.input_cores}_co_{model.output_cores}_acc_{best_accuracy*100:.0f}_{today}.pkl"
        graphdef, state = nnx.split(model)
        save_model(state, filename)

    if save_metrics_flag:
        today = date.today().isoformat()
        filename = f"sscamble_mnist_metrics_ci_{model.input_cores}_co_{model.output_cores}_acc_{best_accuracy*100:.0f}_{today}.pkl"
        save_metrics(metrics_history, filename)

    return model

if __name__ == "__main__":
    model = train_scrramble_mnist(
        model=model,
        optimizer=optimizer,
        metrics=metrics,
        train_ds=train_ds,
        test_ds=test_ds,
        dataset_dict=dataset_dict,
        save_model_flag=False,
        save_metrics_flag=False
    )





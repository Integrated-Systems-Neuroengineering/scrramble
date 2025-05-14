"""
Use this script to run analysis of weight sharing in ScRRAMBLE on MNIST
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
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')
        x = x.reshape(x.shape[0], -1)

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
    
# -------------------------------------------------------------------
# Loading the dataset
# ------------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 64,
    'train_steps': 5000,
    'binarize': True,
    'greyscale': True,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 200,
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
# ------------------------------------------------------------------
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

# number of resamples
n_resamples = 10

# budget of cores
n_cores = 16
ni_list = [8, 10, 12]
no_list = [n_cores - ni for ni in ni_list]
in_out_list = [(ni, no) for ni, no in zip(ni_list, no_list)]

# avg connectivities
avg_conn_list = jnp.arange(1, 10, 1).tolist()

# optimizers
hyperparameters = {
    'learning_rate': 5e-4,
    'momentum': 0.9, 
    'weight_decay': 1e-2
}

# architecture dict
arch_dict = {
    'arch' : [],
    'test_accuracy': [],
    'train_accuracy': [],
    'test_loss' : [],
    'train_loss' : [],
    'avg_slot_connectivity': []
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

# @nnx.jit()
def run_weight_sharing_analysis():
    key1 = jax.random.key(134)

    # architecture loop
    for idx, a in tqdm(enumerate(in_out_list), total=len(in_out_list), desc="Architecture loop"):
        no, ni = a

        # average connectivity loop
        for lam in tqdm(avg_conn_list, total=len(avg_conn_list), desc=r"Connectivity loop"):

            # resample loop
            for r in tqdm(range(n_resamples), desc="Resampling loop"):
                key1, key2, key3 = jax.random.split(key1, 3)
                rng = nnx.Rngs(params=key1, activation=key2, permute=key3)

                # define the model and optimizer
                model = ScRRAMBLeMNIST(
                    input_vector_size=32*32,
                    input_cores=ni,
                    output_cores=no,
                    avg_slot_connectivity=lam,
                    slots_per_core=4,
                    slot_length=64,
                    activation=clipping_ste,
                    rngs=rng,
                    group_size=10,
                    core_length=256,
                    threshold=0.0,
                    noise_sd=0.05
                )

                optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay']))
                metrics = nnx.MultiMetric(accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average('loss'))

                # training loop
                metrics_history = {
                'train_loss': [],
                'train_accuracy': [],
                'test_loss': [],
                'test_accuracy': [],
                }

                eval_every = dataset_dict['eval_every']
                train_steps = dataset_dict['train_steps']

                for step, batch in tqdm(enumerate(train_ds.as_numpy_iterator()), total=len(train_ds), desc="Training loop"):
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

                        # print(f"Step {step}: Test loss: {metrics_history['test_loss'][-1]}, Accuracy: {metrics_history['test_accuracy'][-1]}")

                # get the best metrics 
                best_test_accuracy = max(metrics_history['test_accuracy'])
                best_train_accuracy = max(metrics_history['train_accuracy'])
                best_test_loss = min(metrics_history['test_loss'])
                best_train_loss = min(metrics_history['train_loss'])
                arch_dict['arch'].append(int(idx))
                arch_dict['test_accuracy'].append(float(best_test_accuracy))
                arch_dict['train_accuracy'].append(float(best_train_accuracy))
                arch_dict['test_loss'].append(float(best_test_loss))
                arch_dict['train_loss'].append(float(best_train_loss))
                arch_dict['avg_slot_connectivity'].append(int(lam))
                print(f"Architecture: {in_out_list[idx]}, Avg. connectivity = {lam}, Test accuracy: {best_test_accuracy}, Train accuracy: {best_train_accuracy}")

    # save the architecture dict
    today = date.today().isoformat()
    logs_path = "/local_disk/vikrant/scrramble/logs" # saving in the local_disk
    filename_ = os.path.join(logs_path, f'mnist_arch_dict_cores_{n_cores}_resmp_{n_resamples}_{today}.pkl')
    with open(filename_, 'wb') as f:
        pickle.dump(arch_dict, f)
    

if __name__ == "__main__":
    run_weight_sharing_analysis()


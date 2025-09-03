"""
Sweeping receptive field sizes/slot sizes.
Updated script.

Created on 08/25/2025
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
from datetime import date
import csv

import signal
import sys
import traceback

from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste, squash
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist
from utils.loss_functions import margin_loss


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

# -------------------------------------------------------------------
# Data Logging
# -------------------------------------------------------------------

def setup_csv_logging(num_repeats: int, num_cores: int):
    """Setup CSV file with headers"""
    today = date.today().isoformat()
    logs_path = "/Volumes/export/isn/vikrant/Data/scrramble/logs"
    os.makedirs(logs_path, exist_ok=True)

    csv_filename = os.path.join(logs_path, f'sweep_rf_size_connection_proba_capsnet_cores_{num_cores}_repeats_{num_repeats}_{today}.csv')

    # Write headers
    headers = [
        'rf_size', 'connection_probability', 'repeat_num',
        'test_accuracy', 'test_loss', 
        'valid_accuracy', 'valid_loss',
        'train_accuracy', 'train_loss',
        'best_step', 'num_cores'
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    
    print(f"CSV logging setup at: {csv_filename}")
    return csv_filename

def log_result_to_csv(csv_filename, rf_size, conn_prob, repeat_num, 
                     test_acc, test_loss, valid_acc, valid_loss, 
                     train_acc, train_loss, best_step, num_cores):
    """Append a single result to CSV file"""
    
    row = [
        int(rf_size), float(conn_prob), int(repeat_num),
        float(test_acc), float(test_loss),
        float(valid_acc), float(valid_loss), 
        float(train_acc), float(train_loss),
        int(best_step), int(num_cores)
    ]
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


# -------------------------------------------------------------------
# Define the MNIST CapsNet Model
# -------------------------------------------------------------------

class ScRRAMBLeCapsNet(nnx.Module):
    """
    ScRRAMBLe CapsNet model for MNIST classification.

    Notes:
    - Currently assumes that the connection probability is the same for all the layers.
    """

    def __init__(
            self,
            input_vector_size: int, # size of flattened input vector
            capsule_size: int, # size of each capsule e.g. 256 (number of columns/rows of a core)
            receptive_field_size: int, # size of each receptive field e.g. 64 (number of columns/rows of a slot)
            connection_probability: float, # fraction of total receptive fields on sender side that each receiving slot/receptive field takes input from
            rngs: nnx.Rngs,
            layer_sizes: list = [20, 10, 10], # number of capsules in each layer of the capsnet. e.g. [20, 10] means 20 capsules in layer 1 and 10 capsules in layer 2
            activation_function: Callable = nnx.relu, # activation function to use in the network
    ):
        
        self.input_vector_size = input_vector_size
        self.capsule_size = capsule_size 
        self.receptive_field_size = receptive_field_size
        self.rngs = rngs
        self.connection_probability = connection_probability
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function

        # calculate the effective capsules in input vector rouded to the nearest integral multiple of capsule size
        self.input_eff_capsules = math.ceil(self.input_vector_size/self.capsule_size)

        # add this element as the first element of layer_sizes
        self.layer_sizes.insert(0, self.input_eff_capsules)

        # define ScRRAMBLe capsules
        self.scrramble_caps_layers = [ScRRAMBLeCapsLayer(
            input_vector_size=self.capsule_size * Nci,
            num_capsules=Nco,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs
        ) for Nci, Nco in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]


    def __call__(self, x:jax.Array) -> jax.Array:
        """
        Forward pass through the ScRRAMBLe CapsNet
        """

        # resize the image to be (32, 32) for MNIST
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')

        # flatten the first two dimensions
        x = jnp.reshape(x, (x.shape[0], -1))

        # pass the input through the layers
        for layer in self.scrramble_caps_layers:
            x = jax.vmap(layer, in_axes=(0,))(x)
            x = jnp.reshape(x, (x.shape[0], -1))
            shape_x = x.shape
            x = x.flatten()
            # x = jax.vmap(self.activation_function, in_axes=(0, None, None))(x, 8, 1.0) # 8 bits, 1.0 is the max clipping threshold.
            x = self.activation_function(x)  # Apply the activation function.
            x = jnp.reshape(x, shape_x)

        return x
    

# -------------------------------------------------------------------
# Loading the dataset
# ------------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 64, # 64-100 is a good batch size for MNIST
    'train_steps': 20000, # run for longer, 20000 is good!
    'binarize': True, 
    'greyscale': True,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 2000,
}

# loading the dataset
train_ds, valid_ds, test_ds = load_and_augment_mnist(
    batch_size=dataset_dict['batch_size'],
    train_steps=dataset_dict['train_steps'],
    data_dir=dataset_dict['data_dir'],
    seed=dataset_dict['seed'],
    shuffle_buffer=dataset_dict['shuffle_buffer'],
)


# -------------------------------------------------------------------
# Training functions
# -------------------------------------------------------------------
@nnx.jit
def train_step(model: ScRRAMBLeCapsNet, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, loss_fn: Callable = margin_loss):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeCapsNet, metrics: nnx.MultiMetric, batch, loss_fn: Callable = margin_loss):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

# define metrics
metrics = nnx.MultiMetric(
accuracy=nnx.metrics.Accuracy(),
loss=nnx.metrics.Average('loss')
)

# -------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------- 

# hyperparameters
hyperparameters = {
    'learning_rate': 0.5e-4, # 1e-3 seems to work well
    'momentum': 0.9, 
    'weight_decay': 1e-4
}

# list of rf/slot sizes to sweep over
rf_sizes = jnp.logspace(0, 9, 10, base=2).astype(int).tolist() # should be [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# list of probabilities
connection_probas = [i/10 for i in range(1, 11)]
connection_probas.insert(0, 0.05)
# print(connection_probas)

# initial key
key1 = jax.random.key(0)

# number of resamples
num_resamples = 10

# layers
layers = [10, 10]

# define sweep function
def rf_sweep(log_results: bool = True,
             num_resamples: int = num_resamples,
             key1 : jax.random.key = key1):
    
    if log_results:
        filename = setup_csv_logging(num_resamples, sum(layers))
    else:
        filename = None

    # loop over rf_sizes:
    for i_rf, rf_size in enumerate(rf_sizes):

        # loop over connection probabilities
        for i_p, p in enumerate(connection_probas):

            # loop over resamples
            for n in range(num_resamples):
                print(f"RF Size: {rf_size}, Connection Probability: {p}, Repeat: {n+1}/{num_resamples}")

                # splt keys
                key1, key2, key3, key4 = jax.random.split(key1, 4)
                rngs = nnx.Rngs(params=key1, activations=key2, default=key3, permute=key4)

                # trial load the data again
                # train_ds, valid_ds, test_ds = load_and_augment_mnist(
                #                                             batch_size=dataset_dict['batch_size'],
                #                                             train_steps=dataset_dict['train_steps'],
                #                                             data_dir=dataset_dict['data_dir'],
                #                                             seed=int(234*i_rf + 12*i_p),
                #                                             shuffle_buffer=dataset_dict['shuffle_buffer'],
                #                                         )

                # define the model
                model = ScRRAMBLeCapsNet(
                    input_vector_size=int(1024),
                    capsule_size=256,
                    receptive_field_size=rf_size,
                    connection_probability=p,
                    layer_sizes=layers,
                    activation_function=nnx.relu,
                    rngs=rngs
                )

                # define optimizer
                optimizer = nnx.Optimizer(
                    model,
                    optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
                )

                # define a dict to store results of a single run
                metrics_history = defaultdict(list)

                eval_every = dataset_dict['eval_every']
                train_steps = dataset_dict['train_steps']

                # TRAINING LOOP
                for step, batch in enumerate(train_ds.as_numpy_iterator()):

                    # train the model
                    train_step(model=model, optimizer=optimizer, metrics=metrics, batch=batch)

                    # evaluate the model
                    if step > 0 and (step%eval_every == 0 or step==train_steps-1):

                        # append the step to metrics history
                        metrics_history['step'].append(step)

                        # log training metrics
                        for metric, value in metrics.compute().items():
                            metrics_history[f"train_{metric}"].append(float(value))
                        metrics.reset()

                        # evaluate on the validation set
                        for valid_batch in valid_ds.as_numpy_iterator():
                            # evaluate the model
                            eval_step(model=model, metrics=metrics, batch=valid_batch)

                        # log validation dataset results
                        for metric, value in metrics.compute().items():
                            metrics_history[f"valid_{metric}"].append(float(value))
                        metrics.reset()


                        # evaluate test accuracy for each turn, we later pick the one that corresponds to best valid accuracy
                        for test_batch in test_ds.as_numpy_iterator():
                            # evaluate the model
                            eval_step(model=model, metrics=metrics, batch=test_batch)

                        # log test dataset results
                        for metric, value in metrics.compute().items():
                            metrics_history[f"test_{metric}"].append(float(value))
                        metrics.reset()

                        print(f"Step: {step}, Train Acc: {metrics_history['train_accuracy'][-1]}, Valid Acc: {metrics_history['valid_accuracy'][-1]}, Test Acc: {metrics_history['test_accuracy'][-1]}")

                best_valid_index = int(jnp.argmax(jnp.array(metrics_history['valid_accuracy'])))
                best_step = metrics_history['step'][best_valid_index]

                # save the corresponding accuracies
                test_accuracy = metrics_history['test_accuracy'][best_valid_index]
                test_loss = metrics_history['test_loss'][best_valid_index]
                valid_accuracy = metrics_history['valid_accuracy'][best_valid_index]
                valid_loss = metrics_history['valid_loss'][best_valid_index]
                train_accuracy = metrics_history['train_accuracy'][best_valid_index]
                train_loss = metrics_history['train_loss'][best_valid_index]

                print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")

                # log the results in csv
                if log_results:
                    log_result_to_csv(
                        csv_filename=filename, 
                        rf_size=int(rf_size), 
                        conn_prob=float(p), 
                        repeat_num=int(n+1), 
                        test_acc=float(test_accuracy), 
                        test_loss=float(test_loss), 
                        valid_acc=float(valid_accuracy), 
                        valid_loss=float(valid_loss), 
                        train_acc=float(train_accuracy), 
                        train_loss=float(train_loss), 
                        best_step=int(best_step), 
                        num_cores=int(sum(layers))
                    )

            print("__"*20)

    return model, filename


def __main__():
    model, filename = rf_sweep(log_results=True, num_resamples=num_resamples)
    print("++"*50)
    print("RF Sweep completed.")
    print(f"Results saved to {filename}")
    print("++"*50)

if __name__ == "__main__":
    __main__()


# # check if model definition works
# for i_rf, rf_size in enumerate([1]):
#     for i_p, p in enumerate(connection_probas):
#         print(f"RF Size: {rf_size}, Connection Probability: {p}")
#         key1, key2, key3, key4 = jax.random.split(key1, 4)
#         rngs = nnx.Rngs(params=key1, activations=key2, default=key3, permute=key4)

#         model = ScRRAMBLeCapsNet(
#             input_vector_size=1024,
#             capsule_size=256,
#             receptive_field_size=rf_size,
#             connection_probability=p,
#             layer_sizes=[10, 10, 10],
#             activation_function=nnx.relu,
#             rngs=rngs
#         )

#         print(f"Model created with RF Size: {rf_size}, Connection Probability: {p}")
#         print("--"*20)

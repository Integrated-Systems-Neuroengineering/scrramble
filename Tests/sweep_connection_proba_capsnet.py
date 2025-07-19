"""
Use this script to sweep the connection probability parameter of inter-capsule/core connectivity. 

- Assumes a budget of 60 cores. 50 cores form primary capsules and 10 cores form parent capsules each corresponding to a digit class.
- The simulation uses ReLU activation in the caapsules and the margin loss uses squash function.

Created on: 07/11/2025

# Edited on: 07/18/2025
- Include sweep over different core sizes (20, 30, 60)

Author: Vikrant Jaltare
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

from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste, squash
from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist


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
    'batch_size': 100, # 64-100 is a good batch size for MNIST
    'train_steps': 20000, # run for longer, 20000 is good!
    'binarize': True, 
    'greyscale': True,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 1000,
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

# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

# hyperparameters
hyperparameters = {
    'learning_rate': 0.7e-4, # 1e-3 seems to work well
    'momentum': 0.9, 
    'weight_decay': 1e-4
}

# average connectivities to sweep over
conn_probabilities = jnp.arange(0.1, 1.1, 0.1).tolist()
conn_probabilities.insert(0, 0.05)  # add 0.05 to the list
print(f"Connection probabilities to sweep over: {conn_probabilities}")

# architecture dict
arch_dict = {
    'test_accuracy': [],
    'valid_accuracy' : [],
    'train_accuracy': [],
    'test_loss' : [],
    'valid_loss' : [],
    'train_loss' : [],
    'connection_probability': [],
    'step': [],
    'resamples': [],
    'num_cores': [],
}

num_resamples = 100 # 50 resamples for each connection probability

primary_caps_list = [10, 20, 50]

# define the analysis function
def run_sweep_analysis():

    key1 = jax.random.key(654)

    for pci, primary_caps in enumerate(primary_caps_list):
        print("__"*20)
        print(f"Primary capsules = {primary_caps}")
        print("__"*20)

        for idx, p in tqdm(enumerate(conn_probabilities), total=len(conn_probabilities), desc="Connection probabilities"):
            print(f"Connection probability: {p}")

            # resample loop
            for r in tqdm(range(num_resamples), desc="resampling..."):
                key1, key2, key3, key4 = jax.random.split(key1, 4)
                rngs = nnx.Rngs(params=key1, activations=key2, default=key3, permute=key4)

                # define the model
                model = ScRRAMBLeCapsNet(
                        input_vector_size=1024,
                        capsule_size=256,
                        receptive_field_size=64,
                        connection_probability=p,
                        rngs=rngs,
                        layer_sizes=[primary_caps, 10],  # primary_caps in the first layer and (translates to sum of layer_sizes cores total)
                        activation_function=nnx.relu
                    )
                
                # define the optimizer
                optimizer = nnx.Optimizer(
                                model,
                                optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
                            )
                
                # define metrics logging
                metrics = nnx.MultiMetric(
                            accuracy=nnx.metrics.Accuracy(),
                            loss=nnx.metrics.Average('loss')
                        )
                
                # define dictionary to store the metrics
                metrics_history = {
                                    'train_loss': [],
                                    'train_accuracy': [],
                                    'test_loss': [],
                                    'valid_loss': [],
                                    'valid_accuracy': [],
                                    'test_accuracy': [],
                                    'step': []
                            }
                
                eval_every = dataset_dict['eval_every']
                train_steps = dataset_dict['train_steps']

                # TRAINING LOOP
                for step, batch in enumerate(train_ds.as_numpy_iterator()):
                    # train step
                    train_step(model, optimizer, metrics, batch)

                    # add the metrics
                    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):

                        # append the step to the metrics history
                        metrics_history['step'].append(step)

                        # log the training metrics
                        for metric, value in metrics.compute().items():
                            metrics_history[f"train_{metric}"].append(float(value))
                        metrics.reset() 

                        # EVALUATE ON VALIDATION SET
                        for valid_batch in valid_ds.as_numpy_iterator():
                            eval_step(model, metrics, valid_batch)
                        
                        # log the validation metrics
                        for metric, value in metrics.compute().items():
                            metrics_history[f"valid_{metric}"].append(float(value))
                        metrics.reset()

                        # Evaluate on the test step for EACH step: We later pick the test accuracy and loss corresponding to the best validation accuracy and loss.
                        for test_batch in test_ds.as_numpy_iterator():
                            eval_step(model, metrics, test_batch)

                        # log the test metrics
                        for metric, value in metrics.compute().items():
                            metrics_history[f"test_{metric}"].append(float(value))
                        metrics.reset()

                # print("=="*20)
                # print(f"Test accuracy: {metrics_history['test_accuracy'][-1]}")
                # print(f"Test loss: {metrics_history['test_loss'][-1]}")
                # print("=="*20)

                # pick the index for the best validation accuracy
                best_valid_index = int(jnp.argmax(jnp.array(metrics_history['valid_accuracy'])))
                best_step = metrics_history['step'][best_valid_index]

                # save the best metrics
                test_accuracy = metrics_history['test_accuracy'][best_valid_index]
                test_loss = metrics_history['test_loss'][best_valid_index]
                best_valid_accuracy = metrics_history['valid_accuracy'][best_valid_index]
                best_valid_loss = metrics_history['valid_loss'][best_valid_index]
                best_train_accuracy = metrics_history['train_accuracy'][best_valid_index]
                best_train_loss = metrics_history['train_loss'][best_valid_index]

                print("=="*20)
                print(f"Num cores: {sum(model.layer_sizes) - model.input_eff_capsules}")
                print(f"Test accuracy: {test_accuracy}")
                print(f"Test loss: {test_loss}")
                print("=="*20)

                # append to the arch_dict
                arch_dict['test_accuracy'].append(float(test_accuracy))
                arch_dict['valid_accuracy'].append(float(best_valid_accuracy))
                arch_dict['train_accuracy'].append(float(best_train_accuracy))
                arch_dict['test_loss'].append(float(test_loss))
                arch_dict['valid_loss'].append(float(best_valid_loss))
                arch_dict['train_loss'].append(float(best_train_loss))
                arch_dict['connection_probability'].append(float(p))
                arch_dict['step'].append(int(best_step))
                arch_dict['resamples'].append(int(r))
                arch_dict['num_cores'].append(int(sum(model.layer_sizes) - model.input_eff_capsules))
    
    # save the metrics
    # save the architecture dict
    today = date.today().isoformat()
    logs_path = "/local_disk/vikrant/scrramble/logs" # saving in the local_disk

    # create the logs directory if it doesn't exist
    os.makedirs(logs_path, exist_ok=True)
    
    filename_ = os.path.join(logs_path, f'capsnet_scrramble_relu_caps{sum(model.layer_sizes) - model.input_eff_capsules:d}_{today}.pkl')
    with open(filename_, 'wb') as f:
        pickle.dump(arch_dict, f)


            
if __name__ == "__main__":
    run_sweep_analysis()

    print("++"*30)
    print("Sweep analysis completed.")
    print("++"*30)


            


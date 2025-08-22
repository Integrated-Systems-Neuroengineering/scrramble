"""
Sweep receptive field size variation and its impact on CapsNet performance.

Dataset: MNIST

Created on: 08/18/2025
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

from models import ScRRAMBLeCapsLayer, ScRRAMBLeCapsNetWithReconstruction

from utils.activation_functions import quantized_relu_ste, squash
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
# Loading the dataset
# ------------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 100, # 64 is a good batch size for MNIST
    'train_steps': int(2e4), # run for longer, 20000 is good!
    'binarize': False, 
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

# -------------------------------------------
# Margin Loss from Capsule Networks
# -------------------------------------------

def margin_loss(
    logits,
    labels,
    num_classes: int = 10,
    m_plus: float = 0.9,
    m_minus: float = 0.1,
    lambda_: float = 0.5
    ):

    """
    Margin loss redefined for ScRRAMBLe-CIFAR model. Takes in logits and labels directly.
    """

    caps_output = logits # this output will be in shape (batch_size, num_output_cores (10), slots/receptive fields per core, slot/receptive_field_length)
    # print(f"Caps output shape: {caps_output.shape}")

    # the length of the vector encodes probability of a class
    caps_output = caps_output.reshape(caps_output.shape[0], num_classes, -1)
    # print(f"Caps output reshaped: {caps_output.shape}") # at this point this should be (batch_size, num_output_cores, 256) for the default core length of 256

    # apply squash function along the last axis
    caps_output = squash(caps_output, axis=-1, eps=1e-8)

    caps_output_magnitude = jnp.linalg.norm(caps_output, axis=-1)
    # print(f"Caps output magnitude: {caps_output_magnitude}") # this should be (batch_size, num_output_cores (10))
    # print(f"Caps output magnitude shape: {caps_output_magnitude.shape}") # this should be (batch_size, num_output_cores (10))

    # create one-hot-encoded labels
    # labels = labels
    labels = jax.nn.one_hot(labels, num_classes=caps_output_magnitude.shape[1])
    # print(f"Labels shape: {labels.shape}") # this should be (batch_size, num_output_cores)

    # compute the margin loss
    loss_per_sample = jnp.sum(labels * jax.nn.relu(m_plus - caps_output_magnitude)**2 + lambda_ * (1 - labels) * jax.nn.relu(caps_output_magnitude - m_minus)**2, axis=1)
    loss = jnp.mean(loss_per_sample)

    # print(f"Loss: {loss}")

    return loss, caps_output_magnitude

def loss_fn(model, batch, num_classes=10, m_plus=0.9, m_minus=0.1, lambda_=0.5, regularizer=1e-4):
    """
    Combine margin loss and reconstruction loss.
    Args:
        model: ScRRAMBLeCIFAR model.
        batch: dict, batch of data.
        num_classes: int, number of classes.
        m_plus: float, margin for positive classes.
        m_minus: float, margin for negative classes.
        lambda_: float, regularization parameter.
        regularizer: float, regularization strength.
    """

    # compute the forward pass 
    recon, caps_out = model(batch['image'])
    labels = batch['label']

    # compute margin loss
    margin_loss_val, caps_out_magnitude  = margin_loss(caps_out, labels, num_classes=num_classes, m_plus=m_plus, m_minus=m_minus, lambda_=lambda_)

    # compute the reconstruction error
    batch_ = batch['image']
    reshaped_input = jnp.reshape(batch_, (batch_.shape[0], -1))
    reconstruction_loss = jnp.mean(jnp.square(reshaped_input - recon))

    # compute total loss
    total_loss = margin_loss_val + regularizer*reconstruction_loss

    return total_loss, caps_out_magnitude

@nnx.jit
def train_step(model: ScRRAMBLeCapsNetWithReconstruction, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeCapsNetWithReconstruction, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

# receptive field sizes
rf_sizes = jnp.logspace(0, 8, 9, base=2).astype(int).tolist()

# core budgets
conn_probabilities = jnp.arange(0.1, 1.1, 0.1).tolist()
conn_probabilities.insert(0, 0.05)  # add 0.05 to the list


# rngs = nnx.Rngs(default=0, permute=1, activation=2)
# x_test = jax.random.normal(rngs.default(), (10, 28, 28, 1))

# # first simply test if a forward pass through the model works
# for rf_size in rf_sizes:

#     print(f"Testing receptive field size: {rf_size}")


#     model = ScRRAMBLeCapsNetWithReconstruction(
#                             input_vector_size=1024,
#                             capsule_size=256,
#                             receptive_field_size=rf_size,
#                             connection_probability=0.2,
#                             rngs=rngs,
#                             layer_sizes=[10, 10],  # primary_caps in the first layer and (translates to sum of layer_sizes cores total)
#                             activation_function=nnx.relu
#                         )
    
#     recon, caps_out = model(x_test)
#     print(f"Reconstruction shape: {recon.shape}")
#     print(f"Capsule output shape: {caps_out.shape}")
    
# hyperparameters
hyperparameters = {
    'learning_rate': 0.7e-4, # 1e-3 seems to work well
    'momentum': 0.9, 
    'weight_decay': 1e-4
}

# storing the results
arch_dict = defaultdict(list)

num_repeats = 30

def sweep_rf_size():
    key1 = jax.random.key(235)

    for i_rf, rf_size in enumerate(rf_sizes):
        print("--"*40)
        print(f"RF size: {rf_size}")
        print("--"*40)

        for i_conn, p in enumerate(conn_probabilities):
            print(f"Connection probability: {p}")

            for n in range(num_repeats):

                key1, key2, key3, key4 = jax.random.split(key1, 4)
                rngs = nnx.Rngs(params=key1, activations=key2, default=key3, permute=key4)

                model = ScRRAMBLeCapsNetWithReconstruction(
                            input_vector_size=1024,
                            capsule_size=256,
                            receptive_field_size=rf_size,
                            connection_probability=p,
                            rngs=rngs,
                            layer_sizes=[50, 10],  # 60 capsules total
                            activation_function=nnx.relu
                            )
                
                optimizer = nnx.Optimizer(
                                    model,
                                    optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
                                )

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
                # print(f"Num cores: {sum(model.layer_sizes) - model.input_eff_capsules}")
                print(f"RF size: {rf_size}, Connection probability: {p}, Repeat: {(n+1)}/{num_repeats}")
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
                arch_dict['rf_size'].append(int(rf_size))
                arch_dict['step'].append(int(best_step))
                arch_dict['resamples'].append(int(n + 1))  # n is the current repeat, starting from 0
                arch_dict['num_cores'].append(60)

                    
    # save the metrics
    # save the architecture dict
    today = date.today().isoformat()
    logs_path = "/Volumes/export/isn/vikrant/Data/scrramble/logs" # saving in the local_disk

    # create the logs directory if it doesn't exist
    os.makedirs(logs_path, exist_ok=True)
    
    filename_ = os.path.join(logs_path, f'sweep_rf_size_connection_proba_capsnet_w_recon_{today}.pkl')
    with open(filename_, 'wb') as f:
        pickle.dump(arch_dict, f)


            
if __name__ == "__main__":
    sweep_rf_size()

    print("++"*30)
    print("RF-Sweep Analysis Completed!")
    print("++"*30)
            

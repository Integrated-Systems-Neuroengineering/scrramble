"""
Baselines for comparision with SCRRAMBLe
Baselines:
1. Logistic regression.
2. Feedforward network with same aproximate size as SCRRAMBLe architectures
"""

import jax
import math
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
import os
import pickle
import numpy as np
from functools import partial
from datetime import date

from utils import clipping_ste, intercore_connectivity, plot_connectivity_matrix, load_mnist
from models import ScRRAMBLeLayer, ScRRAMBLeClassifier


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap, BoundaryNorm
# from matplotlib.cm import get_cmap
# import seaborn as sns

# ----------------------------------------------------------------
# Logistic Regression model
# ----------------------------------------------------------------

class BinaryRegressor(nnx.Module):
    """
    Binary logistic model for 'negative control'.
    - Takes in flattened binary images of shape (batch_size, 784)
    - Outputs logits of shape (batch_size, 1)
    """

    def __init__(self, rngs: nnx.Rngs, in_size: int = 784, out_size: int = 10):
        self.rngs = rngs
        self.in_size = in_size
        self.out_size = out_size

        self.linear = nnx.Linear(self.in_size, self.out_size, rngs=self.rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass of the model.
        - Ignore the dimension of batch, we will be using vmap
        """
        x = x.reshape(x.shape[0], self.in_size)
        x = nnx.vmap(self.linear, in_axes=0)(x)
        return nnx.softmax(x)


# ----------------------------------------------------------------
# Feedforward network model
# ----------------------------------------------------------------

class FeedForwardNetwork(nnx.Module):
    """
    Feedforward network for 'positive control'.
    """

    def __init__(self,
        layers: list,
        rngs: nnx.Rngs,
        threshold: float = 0.0,
        noise_sd: float = 0.05,
        activation: Callable = clipping_ste):

        self.rngs = rngs
        self.layers = layers
        self.threshold = threshold
        self.noise_sd = noise_sd

        self.activation = partial(activation, threshold=self.threshold, noise_sd=self.noise_sd, key=self.rngs.activation())

        self.network = [nnx.Linear(li, lo, rngs=self.rngs) for li, lo in zip(layers[:-1], layers[1:-1])]
        self.network.append(nnx.Linear(layers[-2], layers[-1], rngs=self.rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass of the model.
        - Ignore the dimension of batch, we will be using vmap
        """

        # flatten the input
        x = x.reshape(x.shape[0], -1)

        # for layer in self.network[:-1]:
        #     x = self.activation(layer(x))

        for layer in self.network:
            x = self.activation(layer(x))

        # x = nnx.softmax(self.network[-1](x))

        return x

    def get_params(self):
        """
        Get the number of parameters in the model
        """

        parameters = 0
        for layer in self.network:
            w_shape = layer.kernel.shape[0] * layer.kernel.shape[1]
            b_shape = layer.bias.shape[0]
            parameters += w_shape + b_shape
        
        return parameters



# ----------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 128,
    'train_steps': 6000,
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
# -------------------------------------------------------------------
def loss_fn(model, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']).mean()

    return loss, logits

@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

# -------------------------------------------------------------------
# Saving metrics
# -------------------------------------------------------------------
def save_metrics(metrics_dict: dict, filename: str, logs_directory: str = "/local_disk/vikrant/scrramble/logs"):
    """
    Saving metrics to a file
    """

    today = date.today().isoformat()
    test_acc = round(metrics_dict['test_accuracy'][-1]*100)
    filename = f"{filename}_a_{test_acc}_{today}.pkl"
    filepath = os.path.join(logs_directory, filename)

    # if the directory does not exist, create it
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    with open(filepath, "wb") as f:
        pickle.dump(metrics_dict, f)
    
    print(f"Metrics saved to {filepath}")



# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------
# ni = 10
# no = 6
# print(f"Size same as number of cores: {ni+no}")
# print(f"No. params ALMOST as number of cores: {ni+no}")

arch_dict = {
    'ff_layers' : [784, 1150, 1000, 10], # gives 1,059,210; [784, 1150, 1000, 10] has 2063760 parameters
    'threshold' : 0.0,
    'noise_sd' : 0.05,
    'activation': clipping_ste,
}

# optimizer
hyperparameters = {
    'learning_rate': 5e-4,
    'momentum': 0.9, 
    'weight_decay': 1e-4
}

logistic_metrics = {
    'train_accuracy': [],
    'train_loss': [],
    'test_accuracy': [],
    'test_loss': []
}

ff_metrics = {
    'train_accuracy': [],
    'train_loss': [],
    'test_accuracy': [],
    'test_loss': []
}

# ----------------------------------------------------------------
# Data Loading: Logistic regression
# ----------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 128,
    'train_steps': 6000,
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



# initialize the models
key1 = jax.random.key(0)
key1, key2 = jax.random.split(key1)
rngs = nnx.Rngs(params=key1, activation=key2)

logistic_model = BinaryRegressor(rngs=rngs)

key1, key2 = jax.random.split(key1)
ff_model = FeedForwardNetwork(
    layers=arch_dict['ff_layers'],
    rngs=rngs,
    threshold=arch_dict['threshold'],
    noise_sd=arch_dict['noise_sd'],
    activation=arch_dict['activation'],
)

print(f"# parameters in FF network: {ff_model.get_params()}")

logistic_optimizer = nnx.Optimizer(logistic_model, optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'], b1=hyperparameters['momentum']))
ff_optimizer = nnx.Optimizer(ff_model, optax.adamw(learning_rate=hyperparameters['learning_rate'],  weight_decay=hyperparameters['weight_decay'], b1=hyperparameters['momentum']))

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss')
)      


def run_logistic(model=logistic_model, optimizer=logistic_optimizer, metrics=metrics, dataset_dict=dataset_dict, metrics_history=logistic_metrics):
    """
    Run the logistic regression model
    """

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
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            for test_batch in test_ds.as_numpy_iterator():
                eval_step(model, metrics, test_batch)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
            metrics.reset()  # Reset the metrics for the next training epoch.

            print(f"Step {step}: Test loss: {metrics_history['test_loss'][-1]}, Accuracy: {metrics_history['test_accuracy'][-1]}")

    # save the metrics
    filename = f"baseline_logistic_regression_cores_{ni+no}"
    save_metrics(metrics_history, filename)

    return metrics_history

# ----------------------------------------------------------------
# Data Loading: Feedforward network
# ----------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 256,
    'train_steps': 6000,
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

def run_ff(model=ff_model, optimizer=ff_optimizer, metrics=metrics, dataset_dict=dataset_dict, metrics_history=ff_metrics, save_files=False):
    """
    Run the logistic regression model
    """

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

    # save the metrics
    print(f"best test accuracy: {max(metrics_history['test_accuracy'])}")
    if save_files:
        today = date.today().isoformat()
        filename = f"baseline_ff_cores_par_2M"
        save_metrics(metrics_history, filename)
        print(f"Metrics saved to {filename}")

    return metrics_history





# ----------------------------------------------------------------
# Testing the model
# ----------------------------------------------------------------

# def __main__():

#     key1 = jax.random.key(0)
#     key1, key2 = jax.random.split(key1)

#     rngs = nnx.Rngs(params=key1, activation=key2)

#     key1, key2 = jax.random.split(key1)
#     x_test = jax.random.normal(key1, (10, 784))
#     x_test = jax.vmap(lambda x: jnp.where(x>0.5, 1.0, 0.0))(x_test)

#     logistic = BinaryRegressor(rngs)
#     ff = FeedForwardNetwork([784, 256*10, 256*6, 10], rngs)

#     print("Logistic Regressor")
#     out = logistic(x_test)
#     print(out.shape)
#     print(out[0, :10])

#     print("Feedforward Network")
#     out = ff(x_test)
#     print(out.shape)
#     print(out[0, :10])


if __name__ == "__main__":
    # print("Running the logistic regression model")
    # log_metrics = run_logistic()
    print("Running the feedforward network model")
    ff_metrics = run_ff(save_files=True)

    print('done')
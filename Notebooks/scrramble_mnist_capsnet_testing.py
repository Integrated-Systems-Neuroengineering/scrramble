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

from utils import clipping_ste, intercore_connectivity, load_mnist, load_and_augment_mnist, ScRRAMBLe_routing
from utils.activation_functions import quantized_relu_ste, squash
# from models import ScRRAMBLeLayer


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
# ScRRAMBLeLayer class
# -------------------------------------------------------------------
class ScRRAMBLeLayer(nnx.Module):
    """
    Experimental ScRRAMBLe Layer.
    - Defines trainable weights for every core.
    - The weights are organized into input a ouput layers which an be mapped onto cores.
    - Input is assumed to be flattened and will be reshaped inside the module to be fed into correct slots.
    
    Args:

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
        core_length: int = 256,
        threshold: float = 0.0,
        noise_sd: float = 0.05


    ):

        self.core_length = core_length
        self.input_vector_size = input_vector_size
        self.input_eff_cores = self.input_vector_size//core_length
        self.input_cores = input_cores
        self.output_cores = output_cores
        self.avg_slot_connectivity = avg_slot_connectivity
        self.slots_per_core = slots_per_core
        self.slot_length = slot_length
        self.activation = activation
        self.rngs = rngs
        self.threshold = threshold
        self.noise_sd = noise_sd
        # self.activation = partial(self.activation, threshold=self.threshold, noise_sd=self.noise_sd, key=self.rngs.activation())
        self.activation = activation

        # define weights for input cores
        initializer = initializers.glorot_normal()
        self.Wi = nnx.Param(initializer(self.rngs.params(), (self.input_cores, self.slots_per_core, self.slots_per_core, self.slot_length, self.slot_length)))

        # define output weights
        self.Wo = nnx.Param(initializer(self.rngs.params(), (self.output_cores, self.slots_per_core, self.slots_per_core, self.slot_length, self.slot_length)))

        # define connectivity matrix between input vector and input cores: we can trat input as if it came from a layer of cores
        # Ci = intercore_connectivity(
        #     input_cores=self.input_eff_cores,
        #     output_cores=self.input_cores,
        #     slots_per_core=self.slots_per_core,
        #     avg_slot_connectivity=self.avg_slot_connectivity,
        #     key=self.rngs.params()
        # )

        Ci = ScRRAMBLe_routing(
            input_cores=self.input_eff_cores,
            output_cores=self.input_cores,
            receptive_fields_per_capsule=self.slots_per_core,
            connection_probability=0.9,  # average connectivity
            key=self.rngs.params(),
            with_replacement=True
        )

        self.Ci = nnx.Variable(Ci) 

        # define connectivity matrix between input and output cores
        # C_cores = intercore_connectivity(
        #     input_cores=self.input_cores,
        #     output_cores=self.output_cores,
        #     slots_per_core=self.slots_per_core,
        #     avg_slot_connectivity=self.avg_slot_connectivity,
        #     key=self.rngs.params()
        # )

        C_cores = ScRRAMBLe_routing(
            input_cores=self.input_cores,
            output_cores=self.output_cores,
            receptive_fields_per_capsule=self.slots_per_core,
            connection_probability=0.9,  # average connectivity
            key=self.rngs.params(),
            with_replacement=True
        )

        self.C_cores = nnx.Variable(C_cores)

    def __call__(self, x):
        # reshape the input
        x = x.reshape(self.input_eff_cores, self.slots_per_core, self.slot_length)

        # reconstruct the scrambled input
        x = jnp.einsum('ijkl,ijm->klm', self.Ci.value, x)

        # Feed this into the first set of cores
        y1 = jnp.einsum('ijklm,ikm->ijl', self.Wi.value, x)

        # apply the non-linearity
        # y1 = self.activation(y1)
        y1_shape = y1.shape
        y1 = y1.flatten() # last dimesion represents output of a core
        y1 = jax.vmap(self.activation, in_axes=(0, None, None))(y1, 10, 10.0)
        y1 = y1.reshape(y1_shape)  # reshape back to the original shape

        # scramble the input to cores in layers l
        y1 = jnp.einsum('ijkl,ijm->klm', self.C_cores.value, y1)

        # feed the scrambled input into the set of cores in layer l
        y2 = jnp.einsum('ijklm,ikm->ijl', self.Wo.value, y1)

        # apply the non-linearity
        y2_shape = y2.shape
        y2 = y2.flatten() # last dimesion represents output of a core
        y2 = jax.vmap(self.activation, in_axes=(0, None, None))(y2, 10, 10.0)
        y2 = y2.reshape(y2_shape)  # reshape back to the original shape
        return y2


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
        # self.activation = partial(activation, threshold=threshold, noise_sd=noise_sd, key=rngs.activation())
        self.activation = quantized_relu_ste

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
            output_coding: str, specifies how the binary output should be interpreted. Choices are: ['population', 'capsule', ...]. Only 'population' is implemented so far.
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
            
        elif output_coding == 'capsule':
            pass


        
        
        else:
            raise NotImplementedError(f"Output coding of type {output_coding} is unsupported.")


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
    'batch_size': 128, # 64 is a good batch size for MNIST
    'train_steps':  500, # run for longer, 20000 is good!
    'binarize': True, 
    'greyscale': True,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 20,
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
    # activation=clipping_ste,
    activation=quantized_relu_ste,
    rngs=rngs,
    group_size=10,
    core_length=256,
    threshold=0.0,
    noise_sd=0.05 # standard deviation of the noise distribution (typical value = 0.05)
)

print(f"Model parameters: {model.get_params()}")

# optimizers
hyperparameters = {
    'learning_rate': 0.5e-4, # 1e-3 seems to work well
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





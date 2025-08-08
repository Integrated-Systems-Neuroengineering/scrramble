"""
Use this script to train ScRRAMBLe CapsNet on MNIST. 

Created on: 07/02/2025

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

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg')  # Use a non-interactive backend for matplotlib.

from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste, squash
# from utils.loss_functions import margin_loss
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

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists.

    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)

    print(f"Metrics saved to {filename}")

    
def save_model(state, filename):
    """
    Save the model state in a specified file
    """

    checkpoint_dir = "/local_disk/vikrant/scrramble/models"
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"Model saved to {filename_}")


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def qrelu(x: float, bits: int = 8, max_value: float = 2.0):
    # Forward pass: quantize
    x_relu = jnp.maximum(x, 0.0)  # ReLU
    x_clipped = jnp.minimum(x_relu, max_value)  # Clip to max_value
    
    # Quantize
    num_levels = 2**bits - 1
    scale = num_levels / max_value
    quantized = jnp.round(x_clipped * scale) / scale
    
    return quantized

def qrelu_fwd(x: float, bits: int = 8, max_value: float = 2.0):
    result = qrelu(x, bits, max_value)
    return result, x

def qrelu_bwd(bits, max_value, residuals, gradients):
    x = residuals
    # Straight-through: pass gradient if input would be in valid range
    mask = (x > 0) & (x <= max_value)
    grad = jnp.where(mask, 1.0, 0.0)
    return (grad * gradients,)

qrelu.defvjp(qrelu_fwd, qrelu_bwd)

# -------------------------------------------------------------------
# Define the Reconstruction network
# -------------------------------------------------------------------
class ReconstructionNetwork(nnx.Module):
    def __init__(self,
                 input_size: int,
                 rngs: nnx.Rngs):
        
        # define feedforward layers
        self.fc1 = nnx.Linear(input_size, 5000, rngs=rngs)
        self.fc2 = nnx.Linear(5000, 3000, rngs=rngs)
        self.fc3 = nnx.Linear(3000, 28*28, rngs=rngs)

    def __call__(self, x):

        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        x = nnx.sigmoid(self.fc3(x))

        return x


# -------------------------------------------------------------------
# Define the MNIST CapsNet Model
# -------------------------------------------------------------------
class ScRRAMBLeCapsNetWithReconstruction(nnx.Module):
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

        # define the reconstruction network
        self.reconstruction_nw = ReconstructionNetwork(
            input_size=self.capsule_size * self.layer_sizes[-1],  # Input size
            rngs=self.rngs
        )


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
            # x = x.flatten()
            # x = jax.vmap(self.activation_function, in_axes=(0, None, None))(x, 8, 1.0) # 8 bits, 1.0 is the max clipping threshold.
            x = self.activation_function(x)  # Apply the activation function.
            x = jnp.reshape(x, shape_x)

        # add the reconstruction network
        x_recon = x.reshape((x.shape[0], -1))  # Flatten the output for reconstruction.
        x_recon = jax.vmap(self.reconstruction_nw, in_axes=(0,))(x_recon)  # Apply the reconstruction network.


        return x_recon, x
    
# -------------------------------------------------------------------
# Loading the dataset
# ------------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 100, # 64 is a good batch size for MNIST
    'train_steps': int(2e4), # run for longer, 20000 is good!
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

# test if the dataset is loaded correctly by checking the shapes of the datasets
# print(f"Train dataset shape: {train_ds.element_spec['image'].shape}, {train_ds.element_spec['label'].shape}")
# print(f"Valid dataset shape: {valid_ds.element_spec['image'].shape}, {valid_ds.element_spec['label'].shape}")
# print(f"Test dataset shape: {test_ds.element_spec['image'].shape}, {test_ds.element_spec['label'].shape}")

# -------------------------------------------------------------------
# Training functions
# -------------------------------------------------------------------

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
key = jax.random.key(10)
key1, key2, key3, key4 = jax.random.split(key, 4)
rngs = nnx.Rngs(params=key1, activations=key2, permute=key3, default=key4)

model = ScRRAMBLeCapsNetWithReconstruction(
    input_vector_size=1024,
    capsule_size=256,
    receptive_field_size=64,
    connection_probability=0.2,
    rngs=rngs,
    layer_sizes=[40, 10],  # 20 capsules in the first layer and (translates to sum of layer_sizes cores total)
    activation_function=nnx.relu
)

# optimizers
hyperparameters = {
    'learning_rate': 0.8e-4, # 1e-3 seems to work well
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
'valid_loss': [],
'valid_accuracy': [],
'test_accuracy': [],
'step': []
}

# -------------------------------------------------------------------
# Training function
# -------------------------------------------------------------------
def train_scrramble_capsnet_mnist(
        model: ScRRAMBLeCapsNetWithReconstruction = model,
        optimizer: nnx.Optimizer = optimizer,
        train_ds: tf.data.Dataset = train_ds,
        valid_ds: tf.data.Dataset = valid_ds,
        dataset_dict: dict = dataset_dict,
        save_model_flag: bool = False,
        save_metrics_flag: bool = False,
):
    
    eval_every = dataset_dict['eval_every']
    train_steps = dataset_dict['train_steps']

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        # Run the optimization for one step and make a stateful update to the following:
        # - The train state's model parameters
        # - The optimizer state
        # - The training loss and accuracy batch metrics

        train_step(model, optimizer, metrics, batch)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
            metrics_history['step'].append(step)  # Record the step.
            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(float(value))  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the validation set after each training epoch.
            for valid_batch in valid_ds.as_numpy_iterator():
                eval_step(model, metrics, valid_batch)

            # Log the validation metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'valid_{metric}'].append(float(value))
            metrics.reset()  # Reset the metrics for the next training epoch.

            print(f"Step {step}: Valid loss: {metrics_history['valid_loss'][-1]}, Accuracy: {metrics_history['valid_accuracy'][-1]}")

    best_accuracy = max(metrics_history['valid_accuracy'])
    print(f"Best accuracy: {best_accuracy}")

    # find the test set accuracy
    for test_batch in test_ds.as_numpy_iterator():
        eval_step(model, metrics, test_batch)
        # print the metrics
    for metric, value in metrics.compute().items():
        metrics_history[f'test_{metric}'].append(float(value))
    metrics.reset()  # Reset the metrics for the next training epoch.

    print("="*50)
    print(f"Test loss: {metrics_history['test_loss'][-1]}, Test accuracy: {metrics_history['test_accuracy'][-1]}")
    print("="*50)

    if save_model_flag:
        today = date.today().isoformat()
        filename = f"sscamble_mnist_capsnet_recon_capsules{(sum(model.layer_sizes)-model.input_eff_capsules):d}_acc_{metrics_history['test_accuracy'][-1]*100:.0f}_{today}.pkl"
        graphdef, state = nnx.split(model)
        save_model(state, filename)

    if save_metrics_flag:
        today = date.today().isoformat()
        filename = f"sscamble_mnist_capsnet_recon_capsules{(sum(model.layer_sizes)-model.input_eff_capsules):d}_acc_{metrics_history['test_accuracy'][-1]*100:.0f}_{today}.pkl"
        save_metrics(metrics_history, filename)

    return model

if __name__ == "__main__":
    model = train_scrramble_capsnet_mnist(
        model=model,
        optimizer=optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        dataset_dict=dataset_dict,
        save_model_flag=False,
        save_metrics_flag=False,

    )

    # see how the reconstruction works

    # pick a test batch
    model.eval()
    test_batch = next(iter(test_ds.as_numpy_iterator()))
    recon, caps_out = model(test_batch['image'])

    # plot the reconstructed images and their predicted labels
    fig, ax = plt.subplots(8, 8, figsize=(8, 8), dpi=120)
    for i, axs in zip(jnp.arange(64), ax.ravel()):
        im = axs.imshow(recon[i, :].reshape(28, 28), cmap='gray')
        axs.axis('off')

    plt.tight_layout()
    plt.show()



# # testing
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

#     # print the output shape
#     print(f"Output shape: {out.shape}")

#     out = jnp.reshape(out, (out.shape[0], model.layer_sizes[-1], -1))

#     print(f"Output shape after reshaping: {out.shape}")

#     print(f"Some outputs: {out[0, 0, :10]}")

# if __name__ == "__main__":
#     __main__()

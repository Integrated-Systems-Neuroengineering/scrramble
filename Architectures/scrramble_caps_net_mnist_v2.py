"""
Use this script to train ScRRAMBLe CapsNet on MNIST. 

Created on: 07/08/25
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

# Alternative implementation of quantized ReLU with straight-through estimator
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
# Define the MNIST CapsNet Model
# -------------------------------------------------------------------
class ScRRAMBLeCapsNet(nnx.Module):

    def __init__(self, 
                 num_capsules_per_layer: list,
                 input_vector_size: int,
                 rngs : nnx.Rngs,
                 capsule_size: int =256,
                 receptive_field_size: int = 64,
                 connection_probabilities_per_layer: list = [0.2, 0.2],
                 activation_function: callable = qrelu):
        
        self.num_capsules_per_layer = num_capsules_per_layer
        self.input_vector_size = input_vector_size
        self.capsule_size = capsule_size
        self.receptive_field_size = receptive_field_size
        self.connection_probabilities_per_layer = connection_probabilities_per_layer
        self.activation_function = activation_function
        self.rngs = rngs
        self.receptive_fields_per_core = math.ceil(self.input_vector_size / self.receptive_field_size)

        # compute how many many effective capsules the input is made up of
        self.input_effective_capsules = math.ceil(self.input_vector_size / self.capsule_size)

        # insert the number of effective capsules at the input layer
        self.num_capsules_per_layer.insert(0, self.input_effective_capsules)

        print(self.num_capsules_per_layer)

        # make sure that the length of the connection probabilities matches the number of layers
        assert len(self.connection_probabilities_per_layer) == len(self.num_capsules_per_layer) - 1, \
            "Length of connection probabilities must match number of layers - 1"

        # define the layers
        self.scrramble_caps_layers = []

        # define the layers using list comprehension
        for p, Nci, Nco in zip(self.connection_probabilities_per_layer, self.num_capsules_per_layer[:-1], self.num_capsules_per_layer[1:]):
            self.scrramble_caps_layers.append(
                ScRRAMBLeCapsLayer(
                    input_vector_size=Nci * self.capsule_size,
                    num_capsules=Nco,
                    capsule_size=self.capsule_size,
                    receptive_field_size=self.receptive_field_size,
                    connection_probability=p,
                    rngs=self.rngs
                )    
            )

        print(f"ScRRAMBLeCapsNet initialized with {len(self.scrramble_caps_layers)} layers.")

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass
        """

        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten the input

        for layer in self.scrramble_caps_layers:
            x = jax.vmap(layer, in_axes=(0,))(x)
            x = jnp.reshape(x, (x.shape[0], -1))  
            shape_x = x.shape
            x = x.flatten()
            x = jax.vmap(self.activation_function, in_axes=(0, None, None))(x, 8, 1.5)
            x = jnp.reshape(x, shape_x)

        return x


# -------------------------------------------------------------------
# loading MNIST dataset
# -------------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 32, # 64 is a good batch size for MNIST
    'train_steps': 5000, # run for longer, 20000 is good!
    'binarize': True, 
    'greyscale': True,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 500,
}


train_ds, valid_ds, test_ds = load_and_augment_mnist(
    batch_size=dataset_dict['batch_size'],
    train_steps=dataset_dict['train_steps'],
    data_dir=dataset_dict['data_dir'],
    seed=dataset_dict['seed'],
    shuffle_buffer=dataset_dict['shuffle_buffer'],
)

# print the sizes of the datasets
print("Train dataset element spec:", train_ds.element_spec)
print("Valid dataset element spec:", valid_ds.element_spec)
print("Test dataset element spec:", test_ds.element_spec)
# print(f"Train dataset size: {len(train_ds)}")
# print(f"Validation dataset size: {len(valid_ds)}")
# print(f"Test dataset size: {len(test_ds)}")

# -------------------------------------------------------------------
# Training functions
# -------------------------------------------------------------------
# -------------------------------------------
# Margin Loss from Capsule Networks
# -------------------------------------------
def margin_loss(
    model: ScRRAMBLeCapsNet,
    batch,
    num_classes: int = 10,
    m_plus: float = 0.9,
    m_minus: float = 0.1,
    lambda_: float = 0.5
    ):

    caps_output = model(batch['image']) # this output will be in shape (batch_size, num_output_cores (10), slots/receptive fields per core, slot/receptive_field_length)
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
    labels = batch['label']
    labels = jax.nn.one_hot(labels, num_classes=caps_output_magnitude.shape[1])
    # print(f"Labels shape: {labels.shape}") # this should be (batch_size, num_output_cores)

    # compute the margin loss
    loss_per_sample = jnp.sum(labels * jax.nn.relu(m_plus - caps_output_magnitude)**2 + lambda_ * (1 - labels) * jax.nn.relu(caps_output_magnitude - m_minus)**2, axis=1)
    loss = jnp.mean(loss_per_sample)

    # print(f"Loss: {loss}")

    return loss, caps_output_magnitude

@nnx.jit
def train_step(model: ScRRAMBLeCapsNet, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, loss_fn: Callable = margin_loss):
    """ Train model for single step"""

    # define gradient funtction
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    # compute the loss and gradients
    (loss, logits), grads = grad_fn(model, batch)

    # update the metrics
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

    # update the parameters
    optimizer.update(grads)

@nnx.jit
def eval_step(model: ScRRAMBLeCapsNet, metrics: nnx.MultiMetric, batch, loss_fn: Callable = margin_loss):
    """Evaluate model for a single step"""
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

rngs = nnx.Rngs(params=0, activations=1, permute=2, default=3564132)
model = ScRRAMBLeCapsNet(
    input_vector_size=32*32,
    num_capsules_per_layer=[20, 10],  # 20 capsules in the first layer and 10 in the second
    capsule_size=256,
    receptive_field_size=64,
    connection_probabilities_per_layer=[0.2, 0.2],  # Connection
    rngs=rngs,
    activation_function=quantized_relu_ste  # Using quantized ReLU with straight-through estimator
)

# optimizer details
hyperparameters = {
    'learning_rate': 1e-4, # 1e-3 seems to work well
    'momentum': 0.95, 
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



# Training loop
def training_caps_net(
        model : ScRRAMBLeCapsNet = model,
        optimizer: nnx.Optimizer = optimizer,
        train_ds: tf.data.Dataset = train_ds,
        valid_ds: tf.data.Dataset = valid_ds,
        dataset_dict: dict = dataset_dict,
        save_model_flag: bool = False,
        save_metrics_flag: bool = False
):
    
    eval_every = dataset_dict['eval_every']
    train_steps = dataset_dict['train_steps']

    for step, batch in tqdm(enumerate(train_ds.as_numpy_iterator()), total=train_steps):
        train_step(model=model, optimizer=optimizer, metrics=metrics, batch=batch)

        if step == 0:
            for metric, value in metrics.compute().items():
                print(f"Initial {metric}: {value}")
        
        if (step > 0) and (step%eval_every == 0 or step == train_steps - 1):

            for metric, value in metrics.compute().items():
                metrics_history[f'train_{metric}'].append(value)

            metrics_history['step'].append(step)
            metrics.reset()

            # compute validation loss and accuracy
            for valid_batch in valid_ds.as_numpy_iterator():
                eval_step(model=model, metrics=metrics, batch=valid_batch)

            # log validation metrics
            for metric, value in metrics.compute().items():
                metrics_history[f'valid_{metric}'].append(value)
            metrics.reset()

            # print out the metrics
            print(f"Step {step}, Valid loss: {metrics_history['valid_loss'][-1]}, Valid accuracy: {metrics_history['valid_accuracy'][-1]}")

    # compute the best accuracy from the validation set
    best_accuracy = max(metrics_history['valid_accuracy'])
    print(f"Best validation accuracy: {best_accuracy}, at step {metrics_history['step'][metrics_history['valid_accuracy'].index(best_accuracy)]}")
      
    if save_model_flag:
        today = date.today().isoformat()
        filename = f"sscamble_mnist_model_ci_{model.input_cores}_co_{model.output_cores}_acc_{best_accuracy*100:.0f}_{today}.pkl"
        graphdef, state = nnx.split(model)
        save_model(state, filename)

    if save_metrics_flag:
        today = date.today().isoformat()
        filename = f"sscamble_mnist_metrics_ci_{model.input_cores}_co_{model.output_cores}_acc_{best_accuracy*100:.0f}_{today}.pkl"
        save_metrics(metrics_history, filename)

if __name__ == "__main__":
    training_caps_net()


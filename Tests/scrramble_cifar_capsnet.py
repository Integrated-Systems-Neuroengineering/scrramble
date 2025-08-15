
"""
Testing script for ScRRAMBLe CapsNet on CIFAR-10 dataset.
This script sets up the model, loads the dataset, and runs training and evaluation.

Created on: 08/14/2025
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
import seaborn as sns



from utils.activation_functions import quantized_relu_ste, squash
from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_cifar10



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

# ---------------------------------------------------------------
# ScRRAMBLeCAPSLayer without padding: To be modified in ScRRAMBLeCapsLayer
# ---------------------------------------------------------------
class ScRRAMBLeCapsLayer(nnx.Module):
    """
    Experimental Capsule module with ScRRAMBLe Routing.
    Defines a set of capsules with receptive fields.
    Routing is done through ScRRAMBLe.

    A few analogies for using intercore_connectivity function that implements ScRRAMBLe.
    1. input_cores: number of capsules needed. Calculate as (input vector size) / (capsule size).
    2. output_cores: number of capsules to be routed to. Calculate as (output vector size) / (capsule size).
    3. slots_per_core: number of receptive fields per capsule. Take as a given integer. e.g. if capsule size is 256, 4 slots_per_core would mean that each capsule has 4 receptive fields of size 64.
    4. avg_slot_connectivity: lambda parameter. Same as before. But consider connectivity to a receptive field instead of a slot. slot == receptive field in this context.
    """

    def __init__(self,
                 input_vector_size: int, # size of flattened input vector
                 num_capsules: int, # treat this as number of cores that will be used but it doesn't have to be that
                 capsule_size: int, # size of each capsule e.g. 256 (number of columns/rows of a core)
                 receptive_field_size: int, # size of each receptive field e.g. 64 (number of columns/rows of a slot)
                 connection_probability: float, # fraction of total receptive fields on sender side that each receiving slot/receptive field takes input from
                 rngs: nnx.Rngs
                 ):
        
        self.input_vector_size = input_vector_size
        self.num_capsules = num_capsules
        self.capsule_size = capsule_size
        self.receptive_field_size = receptive_field_size
        self.rngs = rngs
        self.connection_probability = connection_probability

        # compute the number of receptive fields per capsule
        self.receptive_fields_per_capsule = math.ceil(self.capsule_size / self.receptive_field_size) # rounded up to the nearest integer

        # compute number of effective capsules coming from the input vector
        self.input_eff_capsules = math.ceil(self.input_vector_size / self.capsule_size) # rounded up to the nearest integer

        # initialize the ScRRAMBLe connectivity matrix
        # Ci = intercore_connectivity(
        #     input_cores=self.input_eff_capsules,
        #     output_cores=self.num_capsules,
        #     slots_per_core=self.receptive_fields_per_capsule,
        #     avg_slot_connectivity=self.avg_receptive_field_connectivity,
        #     key=self.rngs.params()
        # ) 

        Ci = ScRRAMBLe_routing(
            input_cores=self.input_eff_capsules,
            output_cores=self.num_capsules,
            receptive_fields_per_capsule= self.receptive_fields_per_capsule,
            connection_probability=self.connection_probability,
            key=self.rngs.params(),
            with_replacement=True
        )

        self.Ci = nnx.Variable(Ci)

        # initialize the weights on the capsules
        initializer = initializers.glorot_normal()
        self.Wi = nnx.Param(initializer(self.rngs.params(), (self.num_capsules, self.receptive_fields_per_capsule, self.receptive_fields_per_capsule, self.receptive_field_size, self.receptive_field_size))) # e.g. (10, 4, 4, 64, 64)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the capsule layer with ScRRAMBLe routing
        Args:
        x: jax.Array. flattened input, No batch dimension. Shape should be (input_vector_size,). e.g. (1000,)
        """

        # # pad the input with zeros if the length is not a multiple of capsule size
        # if x.shape[0]%self.capsule_size != 0:
        #     x_padded = jnp.pad(x, (0, self.input_eff_capsules*self.capsule_size - x.shape[0]), mode='constant', constant_values=0)
        # else:
        #     x_padded = x
        
        # reshape input into (input_eff_capsules, receptive_fields_per_capsule, receptive_field_size)
        x_reshaped = x.reshape(self.input_eff_capsules, self.receptive_fields_per_capsule, self.receptive_field_size)

        # ScRRAMBLe Routing to the cores
        x_routed = jnp.einsum('ijkl,ijm->klm', self.Ci, x_reshaped)

        y = jnp.einsum('ijklm,ikm->ijl', self.Wi, x_routed)

        return y

    # visualizing connectivity
    def visualize_connectivity(self) -> jax.Array:
        """
        Function returns a jax.Array Wc describing connectivity between neurons in one layer of the network.
        Args:
        1. learned_capsule_weights: jax.Array: Make sure that the shape is (self.Wi.shape[0], self.receptive_fields_per_capsule, self.receptive_fields_per_capsule, self.receptive_field_size, self.receptive_field_size) (5D tensor)
        2. Routing matrix: taken from the intercore_connectivity function. The shape should be (output cores, output slots, input cores, input slots)

        Returns:
        Wc: jax.Array of shape (num output neurons, num input neurons) where Wc[i, j] is the weight from input neuron j to output neuron i.
        """

        # find number of neurons
        num_output_neurons = self.num_capsules * self.capsule_size
        num_input_neurons = self.input_eff_capsules * self.capsule_size

        # initialize the giant connectivity matrix
        Wc = jnp.zeros((num_output_neurons, num_input_neurons))

        # set up for loops
        for co in range(self.num_capsules):
            for so in range(self.receptive_fields_per_capsule):
                for ci in range(self.input_eff_capsules):
                    for si in range(self.receptive_fields_per_capsule):
                        # print(f"co = {co}, so = {so}, ci = {ci}, si = {si}")
                        # get routing weight
                        r = float(self.Ci[co, so, ci, si])

                        if r == 0:
                            continue
                        else:
                            W_dense = r*self.Wi[co, so , si, :, :]
                            # print(W_dense.shape)
                            # print(co*self.capsule_size + so*self.receptive_field_size)
                            # print(co*self.capsule_size + (so+1)*self.receptive_field_size)
                            # print(self.capsule_size*ci + self.receptive_field_size*si)
                            # print(self.capsule_size*ci + (si+1)*self.receptive_field_size)
                            # print(Wc[(co*self.capsule_size + so*self.receptive_field_size):(co*self.capsule_size + (so+1)*self.receptive_field_size), (self.capsule_size*ci + self.receptive_field_size*si):(self.capsule_size*ci + (si+1)*self.receptive_field_size)].shape)

                            Wc = Wc.at[(co*self.capsule_size + so*self.receptive_field_size):(co*self.capsule_size + (so+1)*self.receptive_field_size), (self.capsule_size*ci + self.receptive_field_size*si):(self.capsule_size*ci + (si+1)*self.receptive_field_size)].set(W_dense)

        return Wc

# ---------------------------------------------------------------
# CIFAR10 calssifier
# ---------------------------------------------------------------
class ConvPreprocessing(nnx.Module):
    """
    Convolutional preprocessing layer for CIFAR10
    """

    def __init__(self,
                 rngs: nnx.Rngs,
                 channels: int,
                 kernel_size: tuple,
                 strides: int,
                 padding: str = 'VALID',
                 mask = None,
                #  activation: Callable = nnx.relu,
                 **kwargs
                 ):

        
        self.conv_block = nnx.Conv(
            in_features=3,
            out_features=channels,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            mask=mask,
            rngs=rngs

        )

    def __call__(self, x:jax.Array) -> jax.Array:
        x = self.conv_block(x)

        return x

class ScRRAMBLeCIFAR(nnx.Module):

    def __init__(self,
                capsule_sizes: list,
                rngs: nnx.Rngs,
                connection_probability: float = 0.2,
                receptive_field_size: int = 128,
                kernel_size: tuple = (9, 9),
                channels: int = 64,
                strides: int = 3,
                padding: str = 'VALID',
                mask = None,
                capsule_size: int = 256,
                activation_function: Callable = nnx.gelu,
                **kwargs):
        
        # add conv preprocessing layer
        self.conv_preprocessing = ConvPreprocessing(
            rngs=rngs,
            channels=channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            mask=mask
        )

        # output for conv preprocessing should be (B, 8, 8, 64)
        input_vector_size = 8 * 8 * channels
        input_eff_capsules = math.ceil(input_vector_size / capsule_size)
        capsule_sizes.insert(0, input_eff_capsules)



        self.receptive_field_size = receptive_field_size
        self.capsule_size = capsule_size
        self.activation_function = activation_function

        self.receptive_field_per_capsule = math.ceil(self.capsule_size / self.receptive_field_size)

        # # conv block -> primary capsules
        # self.primary_caps_layer = ScRRAMBLeCapsLayer(
        #     input_vector_size=input_vector_size,
        #     num_primary_capsules=num_primary_capsules,
        #     num_parent_capsules=num_parent_capsules,
        #     rngs=rngs,
        #     receptive_field_size=receptive_field_size,
        #     connection_probability=connection_probability,

        # )

        # # parimary capsules -> parent capsules
        # self.parent_capsule_layer = ScRRAMBLe

        self.scrramble_caps_layers = [ScRRAMBLeCapsLayer(
            input_vector_size=self.capsule_size * Nci,
            num_capsules=Nco,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=connection_probability,
            rngs=rngs
        ) for Nci, Nco in zip(capsule_sizes[:-1], capsule_sizes[1:])]

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.
        Args:
        x: jax.Array/np.array, shape (B, 32, 32, 3) for CIFAR10 images.
        """

        # conv block
        x = jax.vmap(self.conv_preprocessing, in_axes=(0,))(x)  # (B, 8, 8, 64)

        # apply gelu 
        x = nnx.gelu(x)

        # flatten the input an pass through ScRRAMBLe
        x = x.reshape(x.shape[0], -1)  # (B, 8*8*64)
        # print(x.shape)

        for layer in self.scrramble_caps_layers:
            x = jax.vmap(layer, in_axes=(0,))(x)
            x = self.activation_function(x)

        return x



    @staticmethod
    def spatial_block_reshape(x: jax.Array) -> jax.Array:
        """
        Reshape the input image to a 1-D vector, preserving the spatial relationships. 
        """

        x_blocks = x.reshape(8, 8, 4, 4, 3).reshape(64, 48)
        x_flat = x_blocks.reshape(-1) # 3072

        return x_flat
    
    def skip_connection(self, x:jax.Array) -> jax.Array:
        """
        Skip connection from input to parent capsules
        """
        # reshape x into (input capsules, receptive fields per capsule, receptive field size])
        x_reshaped = x.reshape(self.input_eff_capsules, self.receptive_field_per_capsule, self.receptive_field_size)

        # perform ScrRAMBLe routing
        x_routed = jnp.einsum('ijkl,ijm->klm', self.C_input_to_parent, x_reshaped)

        # flatten it before returning
        x_routed_flat = x_routed.reshape(-1)
        return x_routed_flat
    
    def get_active_capsule(self, x:jax.Array) -> jax.Array:

        """
        Assumes the input arrives from the parent capsule layer in form (parent capsules, receptive fields per capsule, receptive field size)
        """

        x_reshape = x.reshape(x.shape[0], -1)

        # take norm along the last dimension
        norms = jnp.linalg.norm(x_reshape, axis=-1)

        # pick the argmax
        active_capsule = jnp.argmax(norms, axis=-1)

        # construct the mask
        mask = jnp.zeros(self.capsule_size*self.num_parent_capsules)
        mask = mask.at[active_capsule:(active_capsule+1)*self.capsule_size-1].set(1.0)
        # mask = nnx.Variable(mask)

        return mask

# ---------------------------------------------------------------
# Loading the CIFAR-10 dataset
# ---------------------------------------------------------------

data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 100, # 64 is a good batch size for CIFAR-10
    'train_steps': 5000, # run for longer, 30000 is good for CIFAR-10
    'eval_every': 1000, # evaluate every 1000 steps
    'binarize': False,  # CIFAR-10 is usually kept as RGB
    'greyscale': False,  # CIFAR-10 is RGB by default
    'data_dir': data_dir,
    'seed': 101,
    'quantize_flag': False,  # whether to quantize the images
    'quantize_bits': False,  # number of bits to quantize the images
    'num_rotations': 4,  # for every image, rotate it by
    'shuffle_buffer': 1024,  # shuffle buffer size
    }

train_ds, valid_ds, test_ds = load_cifar10(
        batch_size=dataset_dict['batch_size'],
        train_steps=dataset_dict['train_steps'],
        data_dir=dataset_dict['data_dir'],
        seed=dataset_dict['seed'],
        shuffle_buffer=dataset_dict['shuffle_buffer'],
        augmentation=True,
        greyscale=dataset_dict['greyscale'],
        quantize_flag=dataset_dict['quantize_flag'],
        quantize_bits=dataset_dict['quantize_bits'],
        num_rotations=dataset_dict['num_rotations'],
    )

# ---------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------
@nnx.jit
def train_step(model: ScRRAMBLeCIFAR,
               optimizer: nnx.Optimizer,
               metrics: nnx.MultiMetric,
               batch,
               loss_fn: Callable = margin_loss,
               ):
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeCIFAR, metrics: nnx.MultiMetric, batch, loss_fn: Callable = margin_loss):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

@nnx.jit
def pred_step(model: ScRRAMBLeCIFAR, batch):
    """
    Prediction step
    """

    caps_out = model(batch['image'])

    # reshape
    out = jnp.reshape(caps_out, (caps_out.shape[0], 10, -1))

    # take vector sizes along the final dimension
    out = jnp.linalg.norm(out, axis=-1)

    # take argmax along the second dimension to get the predicted class

    out = jnp.argmax(out, axis=-1)

    return out

# ---------------------------------------------------------------
# Model and hyperparameters
# ---------------------------------------------------------------
# hyperparameters
hyperparameters = {
    'learning_rate': 1e-4, # 1e-3 seems to work well
    'momentum': 0.9, 
    'weight_decay': 1e-4
}

# model
model_parameters = {
    'capsule_sizes': [90, 10],
    'rngs': nnx.Rngs(default=0, permute=1, params=2, activation=3),
    'connection_probability': 0.2,
    'receptive_field_size': 64,
    'kernel_size': (9, 9),
    'channels': 64,
    'strides': 3,
    'padding': 'VALID',
    'mask': None,
    'capsule_size': 256,
    'activation_function': nnx.gelu,
}

model = ScRRAMBLeCIFAR(**model_parameters)

optimizer = nnx.Optimizer(
    model,
    optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
)

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss')
)

metrics_history = defaultdict(list)

# -------------------------------------------------------------------
# Training function
# -------------------------------------------------------------------
def train_scrramble_capsnet_mnist(
        model: ScRRAMBLeCIFAR = model,
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

    return model, metrics_history


if __name__ == "__main__":
    # train the model
    model, metrics_history = train_scrramble_capsnet_mnist(
        model=model,
        optimizer=optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        dataset_dict=dataset_dict,
        save_model_flag=False,
        save_metrics_flag=False
    )

    labels_dict = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
        }
    
    test_batch = next(iter(test_ds.as_numpy_iterator()))
    predictions = pred_step(model, test_batch)
 
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    for i, axs in enumerate(ax.ravel()):
        axs.imshow(test_batch['image'][i])
        axs.set_title(f"Predicted: {labels_dict[predictions[i]]}\nTrue: {labels_dict[test_batch['label'][i]]}")
        axs.axis('off')

    plt.tight_layout()
    plt.show()
    



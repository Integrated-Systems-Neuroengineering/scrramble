"""
ScRRAMBLe + Residual Blocks for CIFAR-10 Dataset

Created on 11/19/2025
Author: Vikrant Jaltare

Best accuracy so far: Res 64x2->128x2->256x6 and gelu	50-10 (Layer norm)	0.08-0.1	3.00E-04	83.75%	0.153	5.00E+04	64	64
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
from utils import ScRRAMBLe_routing, intercore_connectivity, load_cifar10, fast_scrramble



import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

# -------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Residual Block: outputs shape 2048
# -------------------------------------------------------------------
class ResidualBlock(nnx.Module):
    """
    Residual block with two convolutional layers and a residual connection.
    Order of operations:
    input -> conv -> relu -> conv + input -> relu
    """

    def __init__(self,
                 kernel_size: tuple[int, int],
                 in_features: int,
                 out_features: int,
                #  stride: tuple[int, int],
                 rngs: nnx.Rngs,
                 padding: str = "SAME",
                 activation_fn: Callable = nnx.relu,
                 **kwargs):
        
        self.conv1 = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=kernel_size, padding=padding, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(out_features, rngs=rngs)
        self.activation_fn = activation_fn
        
        # # Projection layer: use 1x1 conv when input and output features differ
        # self.use_projection = (in_features != out_features)
        # if self.use_projection:
        #     self.projection = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=(1, 1), padding=padding, rngs=rngs)

    def __call__(self, x):
        x_res = x
        
        # # Project residual connection if dimensions don't match
        # if self.use_projection:
        #     x_res = self.projection(x_res)
        
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = x + x_res
        x = self.activation_fn(x)
        return x

# ---------------------------------------------------------------
# ScRRAMBLeCAPSLayer
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

        # Ci = ScRRAMBLe_routing(
        #     input_cores=self.input_eff_capsules,
        #     output_cores=self.num_capsules,
        #     receptive_fields_per_capsule= self.receptive_fields_per_capsule,
        #     connection_probability=self.connection_probability,
        #     key=self.rngs.params(),
        #     with_replacement=True
        # )

        Ci = fast_scrramble(
            num_destination_cores=self.num_capsules,
            num_source_cores=self.input_eff_capsules,
            core_size=self.capsule_size,
            slot_size=self.receptive_field_size,
            key=self.rngs.params(),
            proba=self.connection_probability,
            verify_balanced_flag=False
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
        x_routed = jnp.einsum('ijkl,ijm->klm', self.Ci.value, x_reshaped)

        y = jnp.einsum('ijklm,ikm->ijl', self.Wi.value, x_routed)

        return y


# -------------------------------------------------------------------
# ScRRAMBLE + Res Network
# -------------------------------------------------------------------
class ScRRAMBLeResCIFAR10(nnx.Module):
    """
    ScRRAMBLe + Residual Network for CIFAR-10 classification.
    """

    def __init__(self,
                capsule_sizes: list,
                rngs: nnx.Rngs,
                connection_probabilities: list,
                receptive_field_size: int = 64,
                kernel_size: tuple = (3, 3),
                channels: int = 64,
                padding: str = 'SAME',
                capsule_size: int = 256,
                activation_function: Callable = nnx.relu,
                **kwargs):
        
        self.activation_function = activation_function
        

        # add the projection block
        self.projection_block1 = nnx.Conv(in_features=3, out_features=64, kernel_size=(1, 1), padding='SAME', rngs=rngs)

        # add first stage of 3 res blocks with filter sizes 64
        self.res1 = nnx.List([
            ResidualBlock(in_features=64, out_features=64, kernel_size=(3, 3), padding=padding, rngs=rngs, activation_fn=activation_function)
            for _ in range(2)
        ])

        # add projection block 2 with 64 -> 128 channels
        self.projection_block2 = nnx.Conv(in_features=64, out_features=128, kernel_size=(1, 1), padding='SAME', rngs=rngs)

        # add the second stage of 3 res blocks with filter sizes 128
        self.res2 = nnx.List([
            ResidualBlock(in_features=128, out_features=128, kernel_size=(3, 3), padding=padding, rngs=rngs, activation_fn=activation_function)
            for _ in range(3)
        ])

        # add projection block 3 with 128 -> 256 channels
        self.projection_block3 = nnx.Conv(in_features=128, out_features=256, kernel_size=(1, 1), padding='SAME', rngs=rngs)

        # add third stage of 3 res blocks with filter sizes 256
        self.res3 = nnx.List([
            ResidualBlock(in_features=256, out_features=256, kernel_size=(3, 3), padding=padding, rngs=rngs, activation_fn=activation_function)
            for _ in range(6)
        ])

        # add max pool layer
        self.max_pool = partial(nnx.max_pool, window_shape=(2,2), strides=(2,2))

        # add avg pool layer
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2,2), strides=(2,2))

        # TODO: add an rms norm layer
        self.rms_norms = nnx.List(
            [nnx.LayerNorm(capsule_size*c, rngs=rngs) for c in capsule_sizes[:-1]]
        )

        # add a random-fixed projection matrix as a preprocessing for ScRRAMBLe Layers
        output_dim = 2048
        initializer = initializers.glorot_normal()
        self.M = nnx.Variable(
            initializer(rngs.params(), (output_dim, 4096)) # output dim can be changed! prefer a multiple of 256
        )

        capsule_sizes.insert(0, math.ceil(output_dim/capsule_size))  # insert input capsule size at the beginning

        # adding the ScRRAMBLe layers
        self.scrramble_caps_layers = nnx.List([ScRRAMBLeCapsLayer(
            input_vector_size=capsule_size * Nci,
            num_capsules=Nco,
            capsule_size=capsule_size,
            receptive_field_size=receptive_field_size,
            connection_probability=pi,
            rngs=rngs
        ) for Nci, Nco, pi in zip(capsule_sizes[:-1], capsule_sizes[1:], connection_probabilities)]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the ScRRAMBLe + ResNet model.
        Args:
            x: jax.Array. Input image batch. Shape should be (batch_size, height, width, channels).
        Returns:
            jax.Array. Output logits. Shape should be (batch_size, num_classes).
        """

        # pass through projection block 1
        x = self.projection_block1(x)

        # pass through first stage of res blocks
        for res_block in self.res1:
            x = res_block(x)
        x = self.max_pool(x)

        # pass through projection block 2
        x = self.projection_block2(x)

        # pass through second stage of res blocks
        for res_block in self.res2:
            x = res_block(x)
        x = self.max_pool(x)

        # pass through projection block 3
        x = self.projection_block3(x)

        # pass through third stage of res blocks
        for res_block in self.res3:
            x = res_block(x)
        x = self.avg_pool(x)

        # flatten the output
        x = x.reshape((x.shape[0], -1))  # shape: (batch_size, height*width*channels)

        # project using fixed random matrix M
        x = jnp.einsum('ij, bj -> bi', self.M, x) # shape: (batch_size, output_dim)

        # for layer in self.scrramble_caps_layers:
        #     x = jax.vmap(layer, in_axes=(0,))(x)
        #     x = self.activation_function(x)

        # comment this part when not using norms
        for layer, rms_norm in zip(self.scrramble_caps_layers[:-1], self.rms_norms):
            x = jax.vmap(layer, in_axes=(0,))(x)
            x = self.activation_function(x)
            x_shape = x.shape
            x = x.reshape((x_shape[0], -1))  # flatten before passing to next layer
            x = rms_norm(x)
            x = x.reshape(x_shape)  # reshape back to original shape

        # final layer without norm
        layer = self.scrramble_caps_layers[-1]
        x = jax.vmap(layer, in_axes=(0,))(x)

        return x
    
    
# ---------------------------------------------------------------
# Loading the CIFAR-10 dataset
# ---------------------------------------------------------------

data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 64, # 64 is a good batch size for CIFAR-10
    'train_steps': 50000, # run for longer, 30000 is good for CIFAR-10
    'eval_every': 1000, # evaluate every 1000 steps
    'binarize': False,  # CIFAR-10 is usually kept as RGB
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
        quantize_flag=dataset_dict['quantize_flag'],
        quantize_bits=dataset_dict['quantize_bits'],
        num_rotations=dataset_dict['num_rotations'],
    )

# ---------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------
@nnx.jit
def train_step(model: ScRRAMBLeResCIFAR10,
               optimizer: nnx.Optimizer,
               metrics: nnx.MultiMetric,
               batch,
               loss_fn: Callable = margin_loss,
               ):
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(model, grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeResCIFAR10, metrics: nnx.MultiMetric, batch, loss_fn: Callable = margin_loss):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

@nnx.jit
def pred_step(model: ScRRAMBLeResCIFAR10, batch):
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
    'learning_rate': 3e-4, # 1e-3 seems to work well
    'momentum': 0.9, 
    'weight_decay': 1e-4
}

# model
model_parameters = {
    'capsule_sizes': [50, 10],
    'rngs': nnx.Rngs(default=0, permute=1, params=2, activation=3),
    'connection_probabilities': [0.1, 0.1],
    'receptive_field_size': 64, 
    'capsule_size': 256,
    'activation_function': nnx.gelu,
}

model = ScRRAMBLeResCIFAR10(**model_parameters)
nnx.display(model)

optimizer = nnx.Optimizer(
    model,
    optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay']),
    wrt=nnx.Param
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
        model: ScRRAMBLeResCIFAR10 = model,
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

# ----------------------------------------------------------------------
# TODO: Write a script to sweep over connection densities and slot sizes
# ----------------------------------------------------------------------
def sweep_params():
    return

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

    # labels_dict = {
    #     0: "airplane",
    #     1: "automobile",
    #     2: "bird",
    #     3: "cat",
    #     4: "deer",
    #     5: "dog",
    #     6: "frog",
    #     7: "horse",
    #     8: "ship",
    #     9: "truck"
    #     }
    
    # test_batch = next(iter(test_ds.as_numpy_iterator()))
    # predictions = pred_step(model, test_batch)
 
    # fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    # for i, axs in enumerate(ax.ravel()):
    #     axs.imshow(test_batch['image'][i])
    #     axs.set_title(f"Predicted: {labels_dict[int(predictions[i])]}\nTrue: {labels_dict[int(test_batch['label'][i])]}")
    #     axs.axis('off')

    # plt.tight_layout()
    # plt.show()
    



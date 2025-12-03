"""
ScRRAMBLe-MLP mixer architecture for CIFAR-10 dataset
Outlook:
- Consider patched of images split across channels. For instance for CIFAR 10 (32, 32, 3) image can be broken down into 16, 8x8 patches of
3 chanels each.
-  Feed each patch into an input register of a core.
- Slot size shoulbe ve the same as falttened patch size, e.g. for 8x8 patch, slot size should be 64 and so on.
- Can apply RMS/Layer norm, use skip connections as needed similar to the MLP-Mixer paper.

Created on: 12/01/2025
Author: Vikrant Jaltare
"""
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".60"  # Use 60% of GPU memory.

import jax
import math
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
import json

from einops import rearrange, reduce, repeat
from einops.layers.flax import Rearrange, Reduce
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

cifar10_labels_dict = {
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
# ScRRAMBLe-Reshape Block
# -------------------------------------------------------------------
class ScRRAMBLePatching(nnx.Module):
    """
    Patching the input and routing it to the cores
    """

    def __init__(self, 
                 patch_size: tuple[int, int],
                 rngs: nnx.Rngs,
                 output_cores: int,
                 connection_proba: float,
                 capsule_size: int = 256,
                 input_shape: tuple[int, int, int] = (32, 32, 3),
                 **kwargs):
        
        self.patch_size = patch_size
        self.capsule_size = capsule_size
        self.input_shape = input_shape
        
        # compute the number of patches
        self.num_patches_h = input_shape[0] // patch_size[0]
        self.num_patches_w = input_shape[1] // patch_size[1]

        # compute the receptive field size / slot size
        self.rf_size = patch_size[0] * patch_size[1]

        # compute rfs/slots per core
        self.rfs_per_capsule = self.capsule_size // self.rf_size

        # compute effective cores required
        self.num_cores = input_shape[2] * self.num_patches_h * self.num_patches_w // self.rfs_per_capsule

        # define ScRRAMBLe routing
        self.C = fast_scrramble(
            num_destination_cores=output_cores,
            num_source_cores=self.num_cores,
            core_size=self.capsule_size,
            slot_size=self.rf_size,
            key=rngs.params(),
            proba=connection_proba,
        )

        self.C = nnx.Variable(self.C)

    def __call__(self, x):
        # rearrange the input
        x = rearrange(x, 'b (p1 h1) (p2 w2) c -> b p1 p2 h1 w2 c', p1=self.num_patches_h, p2=self.num_patches_w)
        x = rearrange(x, 'b p1 p2 h1 w2 c -> b h1 w2 (p1 p2 c)', p1=self.num_patches_h, p2=self.num_patches_w)
        x = rearrange(x, 'b h1 w2 (ci si) -> b ci si h1 w2', ci=self.num_cores, si=self.rfs_per_capsule)
        x = jnp.einsum('bcshw, ijkl -> bklhw', x, self.C)
        # x = rearrange(x, '(p1 h1) (p2 w2) c -> p1 p2 h1 w2 c', p1=self.num_patches_h, p2=self.num_patches_w)
        # x = rearrange(x, 'p1 p2 h1 w2 c -> h1 w2 (p1 p2 c)', p1=self.num_patches_h, p2=self.num_patches_w)
        # x = rearrange(x, 'h1 w2 (ci si) -> ci si h1 w2', ci=self.num_cores, si=self.rfs_per_capsule)
        # x = jnp.einsum('cshw, ijkl -> klhw', x, self.C)
        return x

# -------------------------------------------------------------------
# ScRRAMBLe MLP Mixer block
# -------------------------------------------------------------------
class ScRRAMBLeMLPMixerBlock(nnx.Module):
    """
    ScRRAMBLe MLP Mixer Block. Stackable module.
    Flow:
    ScRRAMBLePatching -> N blocks of ScRRAMBLeMLPMixerBlock -> Loss
    """

    def __init__(
            self,
            input_capsules: int,
            output_capsules: int,
            rngs: nnx.Rngs,
            connection_probability: float,
            patch_size: tuple[int, int],
            capsule_size: int = 256,
            slot_size: int = 64,
            activation_fn: Callable = nnx.gelu,
    ):
        self.slot_size = slot_size
        self.slots_per_core = math.ceil(capsule_size / slot_size)
        self.activation_fn = activation_fn
        self.patch_size = patch_size

        # define weights of cores(input cores)
        initializer = initializers.glorot_normal()
        self.Wi = nnx.Param(
            initializer(rngs.params(), (input_capsules, slot_size, slot_size, self.slots_per_core, self.slots_per_core))
        )

        # define layer norm
        self.layer_norm = nnx.LayerNorm(input_capsules*capsule_size, rngs=rngs)

        # define routing to next layer (output cores)
        self.Co = fast_scrramble(
            num_destination_cores=output_capsules,
            num_source_cores=input_capsules,
            core_size=capsule_size,
            slot_size=capsule_size // self.slots_per_core,
            key=rngs.params(),
            proba=connection_probability,
        )

        self.Co = nnx.Variable(self.Co)

    def __call__(self, x):
        # self.Wi.value = rearrange(self.Wi.value, 'ci (ls1 n1) (ls2 n2) -> ci ls1 ls2 n1 n2', ls1=self.slot_size, ls2=self.slot_size, n1=self.slots_per_core, n2=self.slots_per_core)
        x = rearrange(x, 'ci p1 h w -> ci p1 (h w)')
        x = jnp.einsum('ijklm, imk -> ilj', self.Wi, x)
        x = self.activation_fn(x)
        x_shape = x.shape
        x = x.reshape(-1,)
        x = self.layer_norm(x)
        x = x.reshape(x_shape)
        x = jnp.einsum('ijkl, ijm -> klm', self.Co, x)
        x = rearrange(x, 'ci p1 (h w) -> ci p1 h w', h=self.patch_size[0], w=self.patch_size[1])
        return x


# -------------------------------------------------------------------
# ScRRAMBLe MLP Mixer Network
# -------------------------------------------------------------------
class ScRRAMBLeMLPMixerNetwork(nnx.Module):
    """
    Chaining the MLP blocks to get a network.
    """

    def __init__(self,
                 cores_per_layer: list,
                 rngs: nnx.Rngs,
                 connection_probability: float, # for now let's simplify and only keep one probability for all layers
                 patch_size: tuple[int, int],
                 input_shape: tuple[int, int, int],
                 slot_size: int,
                 core_size: int,
                 activation_fn: Callable = nnx.gelu,
                 num_classes=10
                 ):
        
        self.activation_fn = activation_fn

        # define the patching layer
        self.patch_layer = ScRRAMBLePatching(
            patch_size=patch_size,
            rngs=rngs,
            output_cores=cores_per_layer[0],
            connection_proba=connection_probability,
            capsule_size=core_size,
            input_shape=input_shape
        )

        # define the MLP mixer blocks
        self.mlp_mix_blocks = nnx.List([
            ScRRAMBLeMLPMixerBlock(
                input_capsules=ci,
                output_capsules=co,
                rngs=rngs,
                connection_probability=connection_probability,
                patch_size=patch_size,
                capsule_size=core_size,
                slot_size=slot_size,
                activation_fn=self.activation_fn
            )
            for ci, co in zip(cores_per_layer[:-1], cores_per_layer[1:])
        ])

        # add a layer norm
        self.layer_norm_list = nnx.List([
            nnx.LayerNorm(co * core_size, rngs=rngs)
            for co in cores_per_layer[1:]
        ])

        # add a classifier layer
        self.classifier = nnx.Linear(
            in_features=cores_per_layer[-1] * core_size,
            out_features=num_classes,
            rngs=rngs
        )

    def __call__(self, x):
        # patch the input and prepare for scrrambling
        x = self.patch_layer(x)

        # pass through the MLP mixer blocks
        for mlp_block, layer_norm in zip(self.mlp_mix_blocks, self.layer_norm_list):
            x_res = jax.vmap(mlp_block, in_axes=(0,))(x)
            x = x + x_res  # skip connection, should work as long as there are equal number of cores in and out
            x_shape = x.shape
            x_flat = x.reshape(x.shape[0], -1)
            x_flat = jax.vmap(layer_norm, in_axes=(0,))(x_flat)
            x = x_flat.reshape(x_shape)

        # pass through the classifier
        logits = self.classifier(x.reshape(x.shape[0], -1))
        return logits

# ---------------------------------------------------------------
# Classifier block
# ---------------------------------------------------------------
class Classifier(nnx.Module):
    """
    Define a linear classifier  
    """

    def __init__(self,
                 input_size: int,
                 num_classes: int,
                 rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            in_features=input_size,
            out_features=num_classes,
            rngs=rngs
        )

    def __call__(self, x):
        # fltten the input across non-batch dimensions
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x
    
# ---------------------------------------------------------------
# Loading the CIFAR-10 dataset
# ---------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 256, # 64 is a good batch size for CIFAR-10
    'train_steps': 30000, # run for longer, 30000 is good for CIFAR-10
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
def loss_fn(
        model: ScRRAMBLeMLPMixerNetwork,
        batch
):
    """
    Apply a softmax cross entropy loss w/ integer labels across the classifier output.
    """

    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=batch['label']
    ).mean()

    return loss, logits


@nnx.jit
def train_step(
        model: ScRRAMBLeMLPMixerNetwork,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch,
        loss_fn: Callable = loss_fn
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(model, grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeMLPMixerNetwork, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

@nnx.jit
def predict_step(model: ScRRAMBLeMLPMixerNetwork, batch, labels_dict : dict = cifar10_labels_dict):
    """
    Return the predicted class
    """
    logits = model(batch['image'])
    logits = nnx.softmax(logits, axis=-1)
    predicted_classes = jnp.argmax(logits, axis=-1)
    predicted_class_names = [labels_dict[int(i)] for i in predicted_classes]
    return predicted_classes, predicted_class_names

# ---------------------------------------------------------------
# Model and hyperparameters
# ---------------------------------------------------------------
# hyperparameters
hyperparameters = {
    'learning_rate': 5e-4, # 1e-3 seems to work well
    'momentum': 0.9, 
    'weight_decay': 1e-4
}

model_config = {
    'cores_per_layer': [50]*3,  # number of cores per layer
    'connection_probability': 0.1,  # connection probability for ScRRAMBLe routing
    'patch_size': (4, 4),  # patch size
    'input_shape': (32, 32, 3),  # CIFAR-10 input shape
    'slot_size': 16,  # slot size
    'core_size': 256,  # core size
    'num_classes': 10,  # number of classes in CIFAR-10
    'rngs': nnx.Rngs(default=0, permute=1, params=2, activation=3)
}

model = ScRRAMBLeMLPMixerNetwork(**model_config)
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
def train_scrramble_mlp_mix(
        model: ScRRAMBLeMLPMixerNetwork = model,
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
    model, metrics_history = train_scrramble_mlp_mix(
        model=model,
        optimizer=optimizer,
        train_ds=train_ds,
        valid_ds=valid_ds,
        dataset_dict=dataset_dict,
        save_model_flag=False,
        save_metrics_flag=False
    )


# -------------------------------------------------------------------
# Testbench
# -------------------------------------------------------------------
# rngs = nnx.Rngs(0)
# x = jnp.ones((10, 32, 32, 3))
# print(f"Input shape = {x.shape}")
# model = ScRRAMBLeMLPMixerNetwork(
#     cores_per_layer=[20, 20, 20],
#     rngs=rngs,
#     connection_probability=0.2,
#     patch_size=(8, 8),
#     input_shape=(32, 32, 3),
#     slot_size=64,
#     core_size=256,
# )
# nnx.display(model)
# y = model(x)
# print(f"Output shape = {y.shape}")

# layer = ScRRAMBLePatching(patch_size=(8, 8), rngs=rngs, output_cores=20, connection_proba=0.2)
# y = layer(x)
# print(f"Patching Output Shape: {y.shape}")
# C = layer.C.value
# print(f"Scramble Matrix Shape Patching: {C.shape}")
# mlp_block = ScRRAMBLeMLPMixerBlock(input_capsules=20, output_capsules=10, rngs=rngs, connection_probability=0.2, patch_size=(8, 8))
# z = jax.vmap(mlp_block, in_axes=(0,))(y)
# print(f"MLP Mixer Block Output Shape: {z.shape}")

# mlp_block2 = ScRRAMBLeMLPMixerBlock(input_capsules=10, output_capsules=10, rngs=rngs, connection_probability=0.2, patch_size=(8, 8))
# z2 = jax.vmap(mlp_block2, in_axes=(0,))(z)
# print(f"MLP Mixer Block2 Output Shape: {z2.shape}")


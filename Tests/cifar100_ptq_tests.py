"""
Inference on ScRRAMBLexResNet model on CIFAAR-10 dataset.


Created on: 01/03/2026
"""

import argparse
import csv
import fcntl

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
from datetime import date, datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns



from utils.activation_functions import quantized_relu_ste, squash
from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_cifar10, fast_scrramble, load_cifar10_augment, load_cifar100_augment



import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

today = date.today().isoformat()

MODEL_PATH = f"/Volumes/export/isn/vikrant/Data/scrramble/models"
data_dir = "/local_disk/vikrant/datasets"

def save_metrics(metrics_dict, filename):
    """
    Save the metrics to a file.
    Args:
        metrics_dict: dict, metrics to save.
        filename: str, name of the file to save the metrics to.
    """

    metrics_dir = "/local_disk/vikrant/scrramble/logs"
    os.makedirs(metrics_dir, exist_ok=True)  # Ensure the directory exists.
    filename = os.path.join(metrics_dir, filename)

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists.

    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)

    print(f"Metrics saved to {filename}")

def save_result_to_csv(result, csv_path):
    # Create directory if needed
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if file exists to write headers
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        # Lock file for thread-safe writing
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        try:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(result)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# --------------------------------------------------------------
# Loading the model
# --------------------------------------------------------------
# payload = pickle.load(open(os.path.join(MODEL_PATH, "scrramble_resnet20_cifar10_model_2026-01-03.pkl"), "rb"))
# configs = payload['configs']
# trained_state = payload['state']

# print("Configs...")
# print(configs)
# print("---------------------------------------------------")
# print(f"MODEL LOADED FROM: {MODEL_PATH}/scrramble_resnet20_cifar10_model_2026-01-03.pkl")
# print("---------------------------------------------------")

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
        x = self.batch_norm(x)
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
        x_routed = jnp.einsum('ijkl,ijm->klm', self.Ci, x_reshaped) #.value is deprecated

        y = jnp.einsum('ijklm,ikm->ijl', self.Wi, x_routed) #.value is deprecated

        return y


# -------------------------------------------------------------------
# ScRRAMBLE + Res Network
# -------------------------------------------------------------------
class ScRRAMBLeResCIFAR100(nnx.Module):
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
        self.capsule_sizes = capsule_sizes
        

        # add the projection block
        self.projection_block1 = nnx.Conv(in_features=3, out_features=64, kernel_size=(1, 1), padding='SAME', rngs=rngs)

        # add first stage of 3 res blocks with filter sizes 64
        self.res1 = nnx.List([
            ResidualBlock(in_features=64, out_features=64, kernel_size=(3, 3), padding=padding, rngs=rngs, activation_fn=activation_function)
            for _ in range(3)
        ])

        # add projection block 2 with 64 -> 128 channels
        self.projection_block2 = nnx.Conv(in_features=64, out_features=128, kernel_size=(1, 1), padding='SAME', rngs=rngs)

        # add the second stage of 3 res blocks with filter sizes 128
        self.res2 = nnx.List([
            ResidualBlock(in_features=128, out_features=128, kernel_size=(3, 3), padding=padding, rngs=rngs, activation_fn=activation_function)
            for _ in range(4)
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

        self.capsule_sizes.insert(0, math.ceil(output_dim/capsule_size))  # insert input capsule size at the beginning

        # adding the ScRRAMBLe layers
        self.scrramble_caps_layers = nnx.List([ScRRAMBLeCapsLayer(
            input_vector_size=capsule_size * Nci,
            num_capsules=Nco,
            capsule_size=capsule_size,
            receptive_field_size=receptive_field_size,
            connection_probability=pi,
            rngs=rngs
        ) for Nci, Nco, pi in zip(self.capsule_sizes[:-1], self.capsule_sizes[1:], connection_probabilities)]
        )

        # add a final classifier layer
        self.classifier = nnx.Linear(in_features=capsule_size * capsule_sizes[-1], out_features=100, rngs=rngs)  # 100 classes for CIFAR-100

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

        x = x.reshape((x.shape[0], -1))  # flatten before classifier
        x = self.classifier(x)

        return x

# ---------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------
# add ths loss function
def loss_fn(model: ScRRAMBLeResCIFAR100, batch: dict):
    logits = model(batch['image'])
    # print(f"logits.shape: {logits.shape}")
    # print(f"batch['label'].shape: {batch['label'].shape}")
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels=batch['label']).mean(
    )

    return loss, logits

@nnx.jit
def train_step(model: ScRRAMBLeResCIFAR100,
               optimizer: nnx.Optimizer,
               metrics: nnx.MultiMetric,
               batch,
               loss_fn: Callable = loss_fn,
               ):
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
    optimizer.update(model, grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeResCIFAR100, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

@nnx.jit
def pred_step(model: ScRRAMBLeResCIFAR100, batch):
    """
    Prediction step
    """

    logits = model(batch['image'])

    logits = nnx.softmax(logits, axis=-1)

    # take argmax along the second dimension to get the predicted class
    out = jnp.argmax(logits, axis=-1)

    return out

# -------------------------------------------------------
# Quantization function
# -------------------------------------------------------
def qrelu_ptq(x: jax.Array,
              bits: int = 8,
              max_value:float = 2.0):
    """
    Quantized ReLU with quantization
    """

    num_levels = 2**bits - 1
    resolution = max_value/num_levels

    # apply ReLU to the input
    x = nnx.relu(x)

    # multiplier
    m = jnp.floor(x/resolution)

    # quantize the input
    x = jnp.clip(m * resolution, 0, max_value)

    return x

def qgelu_ptq(x: jax.Array,
              bits: int = 8,
              max_value:float = 2.0):
    """
    Quantized GELU with quantization
    """

    num_levels = 2**bits - 1
    resolution = max_value/num_levels

    # apply ReLU to the input
    x = nnx.gelu(x)

    # multiplier
    m = jnp.floor(x/resolution)

    # quantize the input
    x = jnp.clip(m * resolution, 0, max_value)

    return x


# -------------------------------------------------------
# Looping over 
# -------------------------------------------------------
bits_list = jnp.arange(1, 31).astype(int)
results_dict = defaultdict(list)

key1 = jax.random.key(10)

max_val=100.0

for i, bits in tqdm(enumerate(bits_list), total=len(bits_list), desc="Bits sweep"):
    print("--"*40)
    print(f"Running inference for {bits} bits")
    print("--"*40)

    results_dict['bits'].append(bits.item())
    # print(results_dict)

    dataset_dict = {
        'batch_size': 64,  # 64 is a good batch size for CIFAR-10
        'train_steps': 1000,  # run for longer, 30000 is good for CIFAR-10
        'eval_every': 1000,  # evaluate every 1000 steps
        'binarize': False,  # CIFAR-10 is usually kept as RGB
        'data_dir': data_dir,
        'seed': int(0 + bits.item()),
        'quantize_flag': False,  # whether to quantize the images
        'quantize_bits': False,  # number of bits to quantize the images
        'num_rotations': 4,  # for every image, rotate it by
        'shuffle_buffer': 1024,  # shuffle buffer size  
        }

    # train_ds, valid_ds, test_ds = load_cifar10(
    #         batch_size=dataset_dict['batch_size'],
    #         train_steps=dataset_dict['train_steps'],
    #         data_dir=dataset_dict['data_dir'],
    #         seed=dataset_dict['seed'],
    #         shuffle_buffer=dataset_dict['shuffle_buffer'],
    #         augmentation=True,
    #         quantize_flag=dataset_dict['quantize_flag'],
    #         quantize_bits=dataset_dict['quantize_bits'],
    #         num_rotations=dataset_dict['num_rotations'],
    #     )

    train_ds, valid_ds, test_ds = load_cifar100_augment(
            batch_size=dataset_dict['batch_size'],
            train_steps=dataset_dict['train_steps'],
            data_dir=dataset_dict['data_dir'],
            seed=dataset_dict['seed'],
            shuffle_buffer=dataset_dict['shuffle_buffer'],
            augmentation=True,
            training=False,
    )

    activation_function = partial(qgelu_ptq, bits=bits, max_value=max_val)

    results_dict['max_value'].append(max_val)

    payload = pickle.load(open(os.path.join(MODEL_PATH, "scrramble_resnet20_cifar100_model_2026-01-03.pkl"), "rb"))
    configs = payload['configs']
    trained_state = payload['state']

    print("Configs...")
    print(configs)

    # loading the model
    key1, key2, key3, key4 = jax.random.split(key1, 4)
    rngs = nnx.Rngs(params=key1, activations=key2, permute=key3, default=key4)


    model = ScRRAMBLeResCIFAR100(
        capsule_sizes=configs['capsule_sizes'],
        connection_probabilities=configs['connection_probabilities'],
        receptive_field_size=configs['receptive_field_size'],
        capsule_size=configs['capsule_size'],
        rngs=rngs,
        activation_function=activation_function
    )

    graphdef, _ = nnx.split(model)
    # add the trained state to the new graphdef
    model = nnx.merge(graphdef, trained_state)

    # nnx.display(model)

    # eval the model
    accuracy_per_batch = []

    for test_batch in test_ds.as_numpy_iterator():
        out = pred_step(model, test_batch)
        accuracy_per_batch.append(float(jnp.mean(out == test_batch['label'])))

    mean_acc = jnp.mean(jnp.array(accuracy_per_batch)).item()

    del model

    results_dict['test_accuracy'].append(mean_acc)

    print(f"Test Accuracy at {bits} bits: {mean_acc*100:.2f}%")

    # save results to csv
    filepath = f"/Volumes/export/isn/vikrant/Data/scrramble/logs/scrramblexres_cifar100_ptq_gelu_sweep_results_{today}.csv"
    results_dict_single = {
        'bits': bits.item(),
        'max_value': max_val,
        'test_accuracy': mean_acc,
    }
    save_result_to_csv(results_dict_single, filepath)



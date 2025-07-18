"""
ScRRAMBLE-CIFAR10 Pilot Script.
- Architecture:
ScRRAMBLe Routing between capsules
Skip connection from input to parent capsules
Reconstruction error regularizer

Created on: 07/15/2025

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

# from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste, squash
# from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist, load_cifar10


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
# Defining the model
# -------------------------------------------------------------------

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

        # pad the input with zeros if the length is not a multiple of capsule size
        if False: #x.shape[0]%self.capsule_size != 0:
            x_padded = jnp.pad(x, (0, self.input_eff_capsules*self.capsule_size - x.shape[0]), mode='constant', constant_values=0)
        else:
            x_padded = x
        
        # reshape input into (input_eff_capsules, receptive_fields_per_capsule, receptive_field_size)
        x_reshaped = x_padded.reshape(self.input_eff_capsules, self.receptive_fields_per_capsule, self.receptive_field_size)

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


## testing
def __main__():
    rngs = nnx.Rngs(params=0, activation=1, default=46732)
    test_input = jax.random.normal(rngs.params(), (10, 1000))
    test_net = ScRRAMBLeCapsLayer(
        input_vector_size=1000,
        num_capsules=4,
        capsule_size=256,
        receptive_field_size=64,
        connection_probability=0.8,
        rngs=rngs
    )

    nnx.display(test_net)

    test_out = jax.vmap(test_net, in_axes=(0,))(test_input)

    # testing the output shape
    print(f"Test output shape = {test_out.shape}")
    test_out = test_out.flatten()
    test_out = jax.vmap(quantized_relu_ste, in_axes=(0, None, None))(test_out, 8, 1.0)
    print(f"Some outputs = {test_out[:10]}")

    # testing the connectivity visualization
    Wc = test_net.visualize_connectivity()
    print(f"Connectivity matrix shape = {Wc.shape}")
    # print some values
    print(f"Some connectivity values = {Wc[:4, :4]}")

# if __name__ == "__main__":
#     __main__()


class ReconstructionLayer(nnx.Module):
    """
    Feedforward layer that reconstructs the input from parent capsules.
    """

    def __init__(self,
                 rngs: nnx.Rngs,
                 input_features: int = 2560,
                 ):
        
        
        self.linear1 = nnx.Linear(in_features=input_features, out_features=500, rngs=rngs)
        # self.linear2 = nnx.Linear(in_features=4000, out_features=4000, rngs=rngs)
        self.linear3 = nnx.Linear(in_features=500, out_features=3072, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through reconstruction layer
        """

        x = nnx.relu(self.linear1(x))
        # x = nnx.relu(self.linear2(x))
        x = nnx.sigmoid(self.linear3(x))

        return x



class ScRRAMBLeCIFAR(nnx.Module):

    def __init__(self, 
                input_vector_size: int,
                num_primary_capsules: int,
                num_parent_capsules: int,
                connection_probability: float,
                rngs: nnx.Rngs,
                receptive_field_size: int = 64,
                capsule_size: int = 256,
                activation_function: Callable = nnx.relu,
                fc_layer_sizes = [256, 5000, 3072]):

        self.input_vector_size = input_vector_size
        self.num_primary_capsules = num_primary_capsules
        self.num_parent_capsules = num_parent_capsules
        self.connection_probability = connection_probability
        self.rngs = rngs
        self.receptive_field_size = receptive_field_size
        self.capsule_size = capsule_size

        self.receptive_field_per_capsule = math.ceil(self.capsule_size / self.receptive_field_size)

        self.activation_function = activation_function
        self.fc_layer_sizes = fc_layer_sizes

        # calculate the effective capsules that the input is made of
        self.input_eff_capsules = math.ceil(self.input_vector_size/self.capsule_size)

        # construct input -> primary capsule ScRRAMBLe Layer
        self.primary_caps_layer = ScRRAMBLeCapsLayer(
            input_vector_size=self.input_vector_size,
            num_capsules=self.num_primary_capsules,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs,
        )

        # define ScRRAMBLe projection between input and parent capsules a.k.a Skip Connection
        self.C_input_to_parent = ScRRAMBLe_routing(
            input_cores=self.input_eff_capsules,
            output_cores=self.num_primary_capsules, # number of parent capsules!
            receptive_fields_per_capsule=self.receptive_field_per_capsule,
            connection_probability=self.connection_probability,
            key=self.rngs.params(),
            with_replacement=True
        )

        self.C_input_to_parent = nnx.Variable(self.C_input_to_parent)

        # routing between primary and parent capsules
        self.parent_capsule_layer = ScRRAMBLeCapsLayer(
            input_vector_size=self.num_primary_capsules*self.capsule_size,
            num_capsules=self.num_parent_capsules,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs,
        )

        # define the fully connected layer
        self.reconstruction_layer = ReconstructionLayer(rngs=self.rngs)

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




    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the CIFAR10 model.
        Args:
            x: jax.Array, input image of shape (B, 32, 32, 3)
        """

        # flatten input using spatial block reshape
        x = jax.vmap(self.spatial_block_reshape, in_axes=(0,))(x)  # (B, 3072)

        # construct the skip connection input
        x_skip = jax.vmap(self.skip_connection, in_axes=(0,))(x)

        # pass through the primary capsule layer
        x = jax.vmap(self.primary_caps_layer, in_axes=(0,))(x)

        # pass through the parent capsule layer
        parent_input = x + x_skip.reshape(x_skip.shape[0], self.num_primary_capsules, self.receptive_field_per_capsule, self.receptive_field_size) 
        x = jax.vmap(self.parent_capsule_layer, in_axes=(0,))(parent_input)
        x = x.reshape(x.shape[0], -1)
        # x = (x_skip + parent_out)

        # only consider the most active parent capsule
        # x_recon = x.reshape(-1)
        # mask = jax.vmap(self.get_active_capsule, in_axes=(0,))(x)
        # x_recon = mask * x_recon

        # pass through the reconstruction layer
        x_recon = jax.vmap(self.reconstruction_layer, in_axes=(0,))(x)

        return x_recon, x


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



def loss_fn(model, batch, num_classes=10, m_plus=0.9, m_minus=0.1, lambda_=0.5, regularizer=5e-4):
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
    reshaped_input = jax.vmap(model.spatial_block_reshape, in_axes=(0,))(batch['image'])  # (B, 3072)
    reconstruction_loss = jnp.mean(jnp.square(reshaped_input - recon))

    # compute total loss
    total_loss = margin_loss_val + regularizer*reconstruction_loss

    return total_loss, caps_out_magnitude


@nnx.jit
def train_step(model: ScRRAMBLeCIFAR, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeCIFAR, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
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

# -------------------------------------------------------------------
# Loading the dataset
# ------------------------------------------------------------------
data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 30, # 64 is a good batch size for MNIST
    'train_steps': 20000, # run for longer, 20000 is good!
    'binarize': True, 
    'greyscale': False,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 1000,
    'quantize_flag': True,
    'quantize_bits': 8,  # number of bits to quantize the images
    'num_rotations': 4,  # for every image, rotate it by 90 degrees
    'augmentation': True,  # whether to augment the images
}

train_ds, valid_ds, test_ds = load_cifar10(
    batch_size=dataset_dict['batch_size'],
    train_steps=dataset_dict['train_steps'],
    data_dir=dataset_dict['data_dir'],
    seed=dataset_dict['seed'],
    shuffle_buffer=dataset_dict['shuffle_buffer'],
    augmentation=dataset_dict['augmentation'],
    greyscale=dataset_dict['greyscale'],
    quantize_flag=dataset_dict['quantize_flag'],  # CIFAR-10 is usually kept as RGB
    quantize_bits=dataset_dict['quantize_bits'],
    num_rotations= dataset_dict['num_rotations'],
)

model_params = {
    'input_vector_size': 3072,  # CIFAR-10 images are 32x32x3 = 3072
    'num_primary_capsules': 20,  # number of primary capsules
    'num_parent_capsules': 10,  # number of parent capsules
    'connection_probability': 0.2,  # connection probability between capsules

}

# create the model
rngs = nnx.Rngs(params=0, activations=1, permute=2, default=2345)
model = ScRRAMBLeCIFAR(
    input_vector_size=model_params['input_vector_size'],
    num_primary_capsules=model_params['num_primary_capsules'],
    num_parent_capsules=model_params['num_parent_capsules'],
    connection_probability=model_params['connection_probability'],
    rngs=rngs,
)

# test
batch = next(iter(train_ds))
recon, x = model(jnp.array(batch['image']))

#print the shapes
print(f"Reconstruction shape: {recon.shape}")  # Should be (B, 3072)
print(f"Capsule output shape: {x.shape}")  # Should be (B, 2560)

# TODO: finish writing the training loop
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


def train_cifar10(
        model,
        train_ds,
        valid_ds,
        test_ds,
        hyperparameters,
        dataset_dict,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,

):
    eval_every = dataset_dict['eval_every']
    train_steps = dataset_dict['train_steps']

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

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        metrics_history['step'].append(int(step))

        train_step(model, optimizer, metrics, batch)





"""
ScRRAMBLe-CapsNet with no padding.
- To be used for CIFAR10 dataset.
- Make sure that the Conv Preprocessing 
- Note that this script has its own CapsLayer implementation.
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
from functools import partial
from datetime import date

from utils.activation_functions import quantized_relu_ste, squash
# from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist

# -------------------------------------------------------------------
# Defining the Conv Preprocessing Layer
# -------------------------------------------------------------------

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
    
# -------------------------------------------------------------------
# ScRRAMBLe CapsLayer without padding
# -------------------------------------------------------------------
class ScRRAMBLeCapsLayerNoPad(nnx.Module):
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
    

# -------------------------------------------------------------------
# Define the ScRRAMBLe-CapsNet Model for CIFAR10
# -------------------------------------------------------------------

class ScRRAMBLeCIFAR(nnx.Module):

    def __init__(self,
                capsule_sizes: list,
                rngs: nnx.Rngs,
                connection_probability: float = 0.2,
                receptive_field_size: int = 64,
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

        self.scrramble_caps_layers = [ScRRAMBLeCapsLayerNoPad(
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

        # apply gelu/relu 
        x = nnx.relu(x)

        # flatten the input an pass through ScRRAMBLe
        x = x.reshape(x.shape[0], -1)  # (B, 8*8*64)
        # print(x.shape)

        for layer in self.scrramble_caps_layers:
            x = jax.vmap(layer, in_axes=(0,))(x)
            x = self.activation_function(x)

        return x

# test the model
if __name__ == "__main__":
    capsule_size = [5, 10]
    rngs = nnx.Rngs(default=0, activation=1, params=9, dropout=2, permute=456)
    test_model = ScRRAMBLeCIFAR(
    capsule_sizes=capsule_size,
    rngs=rngs,
    connection_probability=0.2,
    receptive_field_size=64,
            )

    nnx.display(test_model)
    x_test = jax.random.normal(rngs.default(), (9, 32, 32, 3))
    out = test_model(x_test)
    print(f"Output shape: {out.shape}")
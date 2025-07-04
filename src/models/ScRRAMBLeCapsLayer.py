"""
Single Capsule Network Layer using ScRRAMBLe routing.
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
import math
from flax.nnx.nn import initializers
from collections import defaultdict
from typing import Callable
from utils import intercore_connectivity, ScRRAMBLe_routing
from utils.loss_functions import margin_loss
from utils.activation_functions import quantized_relu_ste


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
        if x.shape[0]%self.capsule_size != 0:
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

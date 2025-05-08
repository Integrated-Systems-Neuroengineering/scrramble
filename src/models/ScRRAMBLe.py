import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx.nn import initializers
from collections import defaultdict
from functools import partial
from typing import Callable
from utils import clipping_ste, intercore_connectivity

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
        core_length: int = 256


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
        self.activation = partial(self.activation, threshold=0.0, noise_sd=0.05, key=self.rngs.activation())

        # define weights for input cores
        initializer = initializers.glorot_normal()
        self.Wi = nnx.Param(initializer(self.rngs.params(), (self.input_cores, self.slots_per_core, self.slots_per_core, self.slot_length, self.slot_length)))

        # define output weights
        self.Wo = nnx.Param(initializer(self.rngs.params(), (self.output_cores, self.slots_per_core, self.slots_per_core, self.slot_length, self.slot_length)))

        # define connectivity matrix between input vector and input cores: we can trat input as if it came from a layer of cores
        Ci = intercore_connectivity(
            input_cores=self.input_eff_cores,
            output_cores=self.input_cores,
            slots_per_core=self.slots_per_core,
            avg_slot_connectivity=self.avg_slot_connectivity,
            key=self.rngs.params()
        )

        self.Ci = nnx.Variable(Ci) 

        # define connectivity matrix between input and output cores
        C_cores = intercore_connectivity(
            input_cores=self.input_cores,
            output_cores=self.output_cores,
            slots_per_core=self.slots_per_core,
            avg_slot_connectivity=self.avg_slot_connectivity,
            key=self.rngs.params()
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
        y1 = self.activation(y1)

        # scramble the input to cores in layers l
        y1 = jnp.einsum('ijkl,ijm->klm', self.C_cores.value, y1)

        # feed the scrambled input into the set of cores in layer l
        y2 = jnp.einsum('ijklm,ikm->ijl', self.Wo.value, y1)

        # apply the non-linearity

        return self.activation(y2)

# define a population coding module
class ScRRAMBLeClassifier(nnx.Module):
    """
    Test module with population coding.
    """
    def __init__(self, population_coding: bool, group_size: int, **kwargs):
        self.scrramble_layer = ScRRAMBLeLayer(**kwargs)
        self.population_coding = population_coding
        self.group_size = group_size

    def __call__(self, x):

        out = nnx.vmap(self.scrramble_layer, in_axes=0, out_axes=0)(x) # apply vmap across the batch dimension
        # flatten the output
        flat_out = out.reshape(out.shape[0], -1)

        # truncate the output so that final dimension is divisible by the group size
        chunk_size = math.floor(flat_out.shape[1]/self.group_size)
        flat_out = flat_out[:, :chunk_size*self.group_size]

        # reshape to average 
        out_res = flat_out.reshape(flat_out.shape[0], self.group_size, -1)

        out = jnp.mean(out_res, axis=-1)
            
        return out


# test the ScRRAMBLe layer
def __main__():
    rngs = nnx.Rngs(params=1, activation=2)
    x_test = jax.random.normal(key=rngs.params(), shape=(512,))

    test_layer = ScRRAMBLeLayer(
        input_vector_size=x_test.shape[0],
        input_cores=5,
        output_cores=4,
        avg_slot_connectivity=1,
        slots_per_core=4,
        slot_length=64,
        rngs=rngs,
        activation=clipping_ste
    )

    nnx.display(test_layer)

    test_out = test_layer(x_test)
    print(test_out.shape)
    print(test_out[0, 0, :])

# if __name__ == "__main__":
#     __main__()
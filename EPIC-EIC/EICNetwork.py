# define the network module
# import the chip parameters from chip_params.json file

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from EICLinear import *
from Block import *
import matplotlib.pyplot as plt
import json
        

class EICNetwork(Block):
    """
    - Contains the current best, EIC-mappable architecture for the network.
    - For independent training of block, the first layer is asomed to be a usual conv layer. This can for example, be replaced with the PIC-end input.
    - Note that I'm including custom functions for linear, relu and sigmoid. They keep the same signature as the binary_ste function.
    - In case the binary stochastic activation implemented with binary_ste fails to train, we can use sigmoid for now and later replace it with a custom activation function.
    - The EICNetwork is a sequence of EICLinear blocks.

    * Architecture:
    - Pseudo-Feedforward layer: Mathematical approximation of a feedforward layer implemented using constraints of the EIC core.
    - Initial conv + avg pool takes (batch_size, 32, 32, 1) image and converts it to (batch_size, 16, 16, 1) tensor which is then resized to (batch_size, 256).
    - Layer sizes: 256 -> 2048 -> 512 -> 256 (10).
    - NOTE: This architecture is a bit different on the input layer (excel sheet has 1024) because I absorbed the 1024 in conv layer just for reference to NN2. This only changes throughput.
    - Activation: Binary with gradient clipping straight-through estimator. In case of low performance, we might tweak the STE.
    """

    @staticmethod
    def linear_activation(x, threshold = None, noise_sd = None, key = None):
        return x
    
    @staticmethod
    def relu_activation(x, threshold = None, noise_sd = None, key = None):
        """
        In case the binary activation doesn't work, we can use a ReLU activation
        """
        return jax.nn.relu(x)
    
    @staticmethod
    def sigmoid_activation(x, threshold = None, noise_sd = None, key = None):
        """
        In case the binary activation doesn't work, we can use a sigmoid activation
        """
        return jax.nn.sigmoid(x)

    def __init__(self, 
                layer_sizes: list, # list of feedforward layer sizes e.g. [1024, 2048, 512, 256]
                eic_activation_fn: Callable, # EICDense activation function
                acc_activation_fn: Callable, # Accumulator activation function
                key: jax.random.key, # PRNG key
                chip_params: dict = None # dictionary of chip parameters
                ):
        
        """
        Args:
        - layer_sizes: list, list of feedforward layer sizes e.g. [1024, 2048, 512, 256]
        - eic_activation_fn: Callable, EICDense activation function
        - acc_activation_fn: Callable, Accumulator activation function
        - key: jax.random.key, PRNG key
        - chip_params: dict, dictionary of chip parameters
        """
        
        self.layer_sizes = layer_sizes
        self.eic_activation_fn = eic_activation_fn
        self.acc_activation_fn = acc_activation_fn
        self.key = key
        self.chip_params = chip_params

        # define the layers
        self.first_layers = [EICLinear(in_size = in_size, 
                                 out_size = out_size, 
                                 eic_activation_fn = self.eic_activation_fn, 
                                 acc_activation_fn = self.acc_activation_fn, 
                                 key = self.key) 
                                 for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:-1])]
        
        self.last_layer = EICLinear(
            in_size = self.layer_sizes[-2],
            out_size = self.layer_sizes[-1],
            eic_activation_fn = self.eic_activation_fn,
            acc_activation_fn = self.linear_activation,
            key = self.key
        )

        self.conv = nnx.Conv(in_features=1, out_features=1, kernel_size=(3,3), strides=1, padding=1, rngs=nnx.Rngs(params=1134, dropout=78978))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass of the EICNetwork.
        Order of operations: Conv -> AvgPool -> EICLinear layers
        Args:
            - x: jax.Array, input vector, typically flattened image etc.
        Returns:
            - x: jax.Array, output of the EICNetwork
        """

        # Resize input to (batch_size, 32, 32, 1): emulating the 1024-long input that we anticipate
        x = jax.image.resize(x, shape=(x.shape[0], 32, 32, 1), method="bilinear")
        print(x.shape)

        # Apply dummy convolution and average pooling
        x = self.conv(x)
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID') 
        
        # Flatten the output to match the input size of the first linear layer
        x = x.reshape(x.shape[0], -1)

        print(x.shape)

        # apply EIC layers
        for layer in self.first_layers:
            x = layer(x)

        x = self.last_layer(x)           
        
        return x
    
    def get_num_cores(self):
        layers = self.layer_sizes

        # compute number of cores needed
        cores_per_param = [max(l1//256, 1) * max(l2//256, 1) for l1, l2 in zip(layers[:-1], layers[1:])]
        dense_cores = sum(cores_per_param)
        acc_cores = [max(l2//256, 1) for l2 in layers[1:]]
        acc_cores = sum(acc_cores)

        tot_cores = dense_cores + acc_cores # store this is return parameters

        # same a secondary metrics list
        size_metrics = {
            'dense_cores': dense_cores,
            'accumulator_cores': acc_cores,
            'total_cores': tot_cores,
        }

        return size_metrics


    def get_latency(self):
        """
        Returns latency of the EIC network in ns
        Requires the chip parameters to be defined in the chip_params dictionary.
        """

        # duration_per_inference = 4.13e-8 # s
        num_layers = len(self.layer_sizes) # number of layers in the network
        # print(f"Number of layers: {num_layers}")
        latency = (self.chip_params['duration_per_inference'] * (1 + num_layers))*1e9
        print(f"Latency in ns: {latency}")
        return latency
    
    def get_energy(self):
        """
        Estimates the energy consumption of the proposed network in TOP/J
        Requires the chip parameters to be defined in the chip_params dictionary.
        """
        num_cores = self.get_num_cores()
        num_cores = num_cores['total_cores']
        col_ops_per_inference = num_cores * self.chip_params['rram_num_cols']
        energy_per_inference = col_ops_per_inference * (self.chip_params['rram_energy_per_col_op'] + self.chip_params['ctrl_energy_per_col_op'] + self.chip_params['comparator_energy_per_col_op'] + self.chip_params['bus_energy_per_col_op']) + self.chip_params['sram_static_power']*self.chip_params['duration_per_inference']
        ops_per_inference = col_ops_per_inference*self.chip_params['rram_num_rows']
        # print(f"Ops per inference: {ops_per_inference}")
        energy = (energy_per_inference*1e3)/ops_per_inference * 1e12 # TOP/J
        print(f"Energy in TOP/J: {energy}")
        return energy
    
    def get_operations(self):
        """
        Returns throughput in TOPS/s
        Requires the chip parameters to be defined in the chip_params dictionary.
        """
        num_cores = self.get_num_cores()
        num_cores = num_cores['total_cores']
        # print(f"Number of cores: {num_cores}")
        col_ops_per_inference = num_cores * self.chip_params['rram_num_cols']
        ops_per_inference = col_ops_per_inference*self.chip_params['rram_num_rows']
        # print(f"Operations per inference: {ops_per_inference}")
        duration_per_inference = self.chip_params['duration_per_inference']
        throughput = ops_per_inference/duration_per_inference * 1e-12 # TOPS/s
        print(f"Throughput in TOPS/s: {throughput}")
        return throughput
    

# test
im_ = jnp.ones((10, 28, 28, 1))
key = jax.random.key(23412)
chip_params = json.load(open("ChipParams/chip_params.json", "r"))
eic_net = EICNetwork(layer_sizes = [256, 4096, 2048, 256], eic_activation_fn = binary_ste, acc_activation_fn = binary_ste, key = key, chip_params = chip_params)
out_ = eic_net(im_)
lat = eic_net.__getitem__("latency")
ene = eic_net.__getitem__("energy")
thr = eic_net.__getitem__("operations")
print(out_.shape)
print(out_)
plt.imshow(out_.reshape(-1, 32))
plt.colorbar()
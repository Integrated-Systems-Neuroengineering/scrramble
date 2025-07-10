"""
Testing margin loss
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
import math
from functools import partial
from typing import Callable
from utils.loss_functions import margin_loss
from models import ScRRAMBLeCapsLayer
from utils.activation_functions import quantized_relu_ste
from utils import load_and_augment_mnist


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
class ScRRAMBLeCapsNet(nnx.Module):
    """
    ScRRAMBLe CapsNet model for MNIST classification.

    Notes:
    - Currently assumes that the connection probability is the same for all the layers.
    """

    def __init__(
            self,
            input_vector_size: int, # size of flattened input vector
            capsule_size: int, # size of each capsule e.g. 256 (number of columns/rows of a core)
            receptive_field_size: int, # size of each receptive field e.g. 64 (number of columns/rows of a slot)
            connection_probability: float, # fraction of total receptive fields on sender side that each receiving slot/receptive field takes input from
            rngs: nnx.Rngs,
            layer_sizes: list = [20, 10, 10], # number of capsules in each layer of the capsnet. e.g. [20, 10] means 20 capsules in layer 1 and 10 capsules in layer 2
            activation_function: Callable = qrelu, # activation function to use in the network
    ):
        
        self.input_vector_size = input_vector_size
        self.capsule_size = capsule_size 
        self.receptive_field_size = receptive_field_size
        self.rngs = rngs
        self.connection_probability = connection_probability
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function

        # calculate the effective capsules in input vector rouded to the nearest integral multiple of capsule size
        self.input_eff_capsules = math.ceil(self.input_vector_size/self.capsule_size)

        # add this element as the first element of layer_sizes
        self.layer_sizes.insert(0, self.input_eff_capsules)

        # define ScRRAMBLe capsules
        self.scrramble_caps_layers = [ScRRAMBLeCapsLayer(
            input_vector_size=self.capsule_size * Nci,
            num_capsules=Nco,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs
        ) for Nci, Nco in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]


    def __call__(self, x:jax.Array) -> jax.Array:
        """
        Forward pass through the ScRRAMBLe CapsNet
        """

        # resize the image to be (32, 32) for MNIST
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')

        # flatten the first two dimensions
        x = jnp.reshape(x, (x.shape[0], -1))

        # pass the input through the layers
        for layer in self.scrramble_caps_layers:
            x = jax.vmap(layer, in_axes=(0,))(x)
            x = jnp.reshape(x, (x.shape[0], -1))
            shape_x = x.shape
            x = x.flatten()
            x = jax.vmap(self.activation_function, in_axes=(0, None, None))(x, 8, 1.5) # 8 bits, 1.0 is the max clipping threshold.
            x = jnp.reshape(x, shape_x)

        return x

rngs = nnx.Rngs(params=0, activations=1, permute=2, default=3564132)
model = ScRRAMBLeCapsNet(
        input_vector_size=1024,
        capsule_size=256,
        receptive_field_size=64,
        connection_probability=0.2,
        rngs=rngs,
        layer_sizes=[20, 10]  # 20 capsules in the first layer and

    )

# load the data
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

# loading the dataset
train_ds, valid_ds, test_ds = load_and_augment_mnist(
    batch_size=dataset_dict['batch_size'],
    train_steps=dataset_dict['train_steps'],
    data_dir=dataset_dict['data_dir'],
    seed=dataset_dict['seed'],
    shuffle_buffer=dataset_dict['shuffle_buffer'],
)

# use 1 batch for testing
dummy_input = next(iter(train_ds))['image']
dummy_labels = next(iter(train_ds))['label']

loss_fn = margin_loss

# take gradient of the loss function
grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
 
# compute the gradient
(loss, logits), grads = grad_fn(model, {'image': dummy_input, 'label': dummy_labels})

print(f"Loss: {loss}")
print(f"Logits: {logits}")
# print(f"Grads: {grads}")

# Check if gradients are non-zero
# total_grad_norm = 0
# for param in jax.tree_leaves(grads):
#     total_grad_norm += jnp.sum(jnp.abs(param))
# print(f"Total gradient norm: {total_grad_norm}")

"""
Testing margin loss with one batch - forward pass and gradients
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
    """Quantized ReLU with straight-through estimator"""
    x_relu = jnp.maximum(x, 0.0)
    x_clipped = jnp.minimum(x_relu, max_value)
    
    num_levels = 2**bits - 1
    scale = num_levels / max_value
    quantized = jnp.round(x_clipped * scale) / scale
    
    return quantized

def qrelu_fwd(x: float, bits: int = 8, max_value: float = 2.0):
    result = qrelu(x, bits, max_value)
    return result, x

def qrelu_bwd(bits, max_value, residuals, gradients):
    x = residuals
    mask = (x > 0) & (x <= max_value)
    grad = jnp.where(mask, 1.0, 0.0)
    return (grad * gradients,)

qrelu.defvjp(qrelu_fwd, qrelu_bwd)


class ScRRAMBLeCapsNet(nnx.Module):
    """ScRRAMBLe CapsNet model for MNIST classification."""

    def __init__(
        self,
        input_vector_size: int,
        capsule_size: int,
        receptive_field_size: int,
        connection_probability: float,
        rngs: nnx.Rngs,
        layer_sizes: list = [20, 10],
        activation_function: Callable = qrelu,
    ):
        
        self.input_vector_size = input_vector_size
        self.capsule_size = capsule_size 
        self.receptive_field_size = receptive_field_size
        self.rngs = rngs
        self.connection_probability = connection_probability
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function

        self.input_eff_capsules = math.ceil(self.input_vector_size/self.capsule_size)
        self.layer_sizes.insert(0, self.input_eff_capsules)

        self.scrramble_caps_layers = [ScRRAMBLeCapsLayer(
            input_vector_size=self.capsule_size * Nci,
            num_capsules=Nco,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs
        ) for Nci, Nco in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the ScRRAMBLe CapsNet"""
        
        # Convert TF tensor to JAX array if needed
        if hasattr(x, 'numpy'):
            x = jnp.array(x.numpy())
        elif not isinstance(x, jnp.ndarray):
            x = jnp.array(x)
        
        # Resize and flatten
        x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')
        x = jnp.reshape(x, (x.shape[0], -1))

        # Pass through layers
        for layer in self.scrramble_caps_layers:
            x = jax.vmap(layer, in_axes=(0,))(x)
            x = jnp.reshape(x, (x.shape[0], -1))
            
            # Apply activation
            shape_x = x.shape
            x = x.flatten()
            x = jax.vmap(self.activation_function, in_axes=(0, None, None))(x, 8, 1.5)
            x = jnp.reshape(x, shape_x)

        return x


def test_margin_loss():
    """Test margin loss with one batch"""
    
    # Initialize model
    rngs = nnx.Rngs(params=0, activations=1, permute=2, default=3564132)
    model = ScRRAMBLeCapsNet(
        input_vector_size=1024,
        capsule_size=256,
        receptive_field_size=64,
        connection_probability=0.2,
        rngs=rngs,
        layer_sizes=[20, 10]
    )

    # Load dataset
    data_dir = "/local_disk/vikrant/datasets"
    dataset_dict = {
        'batch_size': 32,
        'train_steps': 100,  # We only need one batch
        'binarize': True, 
        'greyscale': True,
        'data_dir': data_dir,
        'seed': 101,
        'shuffle_buffer': 1024,
        'threshold': 0.5,
        'eval_every': 500,
    }

    train_ds, _, _ = load_and_augment_mnist(
        batch_size=dataset_dict['batch_size'],
        train_steps=dataset_dict['train_steps'],
        data_dir=dataset_dict['data_dir'],
        seed=dataset_dict['seed'],
        shuffle_buffer=dataset_dict['shuffle_buffer'],
    )

    # Get one batch
    batch = next(iter(train_ds))
    
    # Convert to JAX arrays
    batch_jax = {
        'image': jnp.array(batch['image']),
        'label': jnp.array(batch['label'])
    }
    
    print(f"Batch shapes - Image: {batch_jax['image'].shape}, Label: {batch_jax['label'].shape}")
    print(f"Label values: {batch_jax['label']}")

    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    try:
        output = model(batch_jax['image'])
        print(f"Model output shape: {output.shape}")
        print(f"Output stats: min={jnp.min(output):.4f}, max={jnp.max(output):.4f}, mean={jnp.mean(output):.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return

    # Test loss computation
    print("\n--- Testing Loss Computation ---")
    try:
        loss, logits = margin_loss(model, batch_jax)
        print(f"Loss: {loss}")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits stats: min={jnp.min(logits):.4f}, max={jnp.max(logits):.4f}")
        
        if jnp.isnan(loss):
            print("WARNING: Loss is NaN!")
        else:
            print("Loss computation successful!")
            
    except Exception as e:
        print(f"Loss computation failed: {e}")
        return

    # Test gradient computation
    print("\n--- Testing Gradient Computation ---")
    try:
        grad_fn = nnx.value_and_grad(margin_loss, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch_jax)
        
        print(f"Gradient computation successful!")
        print(f"Loss: {loss}")
        
        # Check gradient statistics
        total_grad_norm = 0
        param_count = 0
        for param in jax.tree.leaves(grads):
            if param is not None:
                total_grad_norm += jnp.sum(jnp.abs(param))
                param_count += param.size
        
        print(f"Total gradient norm: {total_grad_norm:.6f}")
        print(f"Average gradient magnitude: {total_grad_norm/param_count:.8f}")
        
        if total_grad_norm == 0:
            print("WARNING: All gradients are zero!")
        elif jnp.isnan(total_grad_norm):
            print("WARNING: Gradients contain NaN!")
        else:
            print("Gradients look healthy!")
            
    except Exception as e:
        print(f"Gradient computation failed: {e}")
        return

    print("\n--- Test Complete ---")


if __name__ == "__main__":
    test_margin_loss()
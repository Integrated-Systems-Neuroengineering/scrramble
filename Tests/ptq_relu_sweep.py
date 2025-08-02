"""
Inference performance on quantized ReLU layers.
- Loading pretrained models.


Created on: 07/31/2025 
"""

import jax
import math
import jax.numpy as jnp
import optax
import flax
from flax import nnx
import json
import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Callable
from functools import partial
from tqdm import tqdm
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste
from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist, plot_connectivity_matrix
from models import ScRRAMBLeCapsNetWithReconstruction


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

today = date.today().isoformat()

model_path = f"/local_disk/vikrant/scrramble/models/sscamble_mnist_capsnet_recon_capsules60_acc_99_2025-07-28.pkl"
training_metrics_path = f" /local_disk/vikrant/scrramble/logs/sscamble_mnist_capsnet_recon_capsules60_acc_99_2025-07-28.pkl"

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

# -------------------------------------------------------
# Quantization function
# -------------------------------------------------------
def qrelu_ptq(x: jax.Array,
              bits: int = 3,
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

# -------------------------------------------------------
# Loss function
# -------------------------------------------------------

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
    batch_ = batch['image']
    reshaped_input = jnp.reshape(batch_, (batch_.shape[0], -1))
    reconstruction_loss = jnp.mean(jnp.square(reshaped_input - recon))

    # compute total loss
    total_loss = margin_loss_val + regularizer*reconstruction_loss

    return total_loss, caps_out_magnitude

# -------------------------------------------------------
# Eval functions
# -------------------------------------------------------
@nnx.jit
def pred_step(model: ScRRAMBLeCapsNetWithReconstruction, batch):
    """
    Prediction step for the ScRRAMBLe CapsNet model.
    """
    # Forward pass through the model
    recon_out, caps_out = model(batch['image'])
    
    # reshape
    out = jnp.reshape(caps_out, (caps_out.shape[0], 10, -1))

    # take vector sizes along hte final dimension
    out = jnp.linalg.norm(out, axis=-1)

    # take argmax along the second dimension to get the predicted class
    out = jnp.argmax(out, axis=-1)
    
    return out

@nnx.jit
def eval(model: ScRRAMBLeCapsNetWithReconstruction, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
    """
    Evaluation step
    """

    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
)
# -------------------------------------------------------
# Load pretrained model
# -------------------------------------------------------
# key = jax.random.key(10)
# key1, key2, key3, key4 = jax.random.split(key, 4)
# rngs = nnx.Rngs(params=key1, activations=key2, permute=key3, default=key4)

# model = ScRRAMBLeCapsNetWithReconstruction(
#     input_vector_size=1024,
#     capsule_size=256,
#     receptive_field_size=64,
#     connection_probability=0.2,
#     rngs=rngs,
#     layer_sizes=[50, 10],  # 20 capsules in the first layer and (translates to sum of layer_sizes cores total)
#     activation_function=qrelu_ptq
# )

# loaded_state = pickle.load(open(model_path, "rb"))
# graphdef, old_state = nnx.split(model)
# model = nnx.merge(graphdef, loaded_state)

# -------------------------------------------------------
# Load the data
# -------------------------------------------------------

data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 100, # 64 is a good batch size for MNIST
    'train_steps': int(2e4), # run for longer, 20000 is good!
    'binarize': True, 
    'greyscale': True,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 1000,
}

# loading the dataset
train_ds, valid_ds, test_ds = load_and_augment_mnist(
    batch_size=dataset_dict['batch_size'],
    train_steps=dataset_dict['train_steps'],
    data_dir=dataset_dict['data_dir'],
    seed=dataset_dict['seed'],
    shuffle_buffer=dataset_dict['shuffle_buffer'],
)

# -------------------------------------------------------
# Looping over 
# -------------------------------------------------------
bits_list = jnp.arange(1, 33).tolist()
logs = defaultdict(list)

key1 = jax.random.key(10)


for i, bits in tqdm(enumerate(bits_list), total=len(bits_list), desc="Bits sweep"):
    print("--"*40)
    print(f"Running inference for {bits} bits")
    print("--"*40)

    logs['bits'].append(int(bits))

    # loading the dataset
    train_ds, valid_ds, test_ds = load_and_augment_mnist(
        batch_size=dataset_dict['batch_size'],
        train_steps=dataset_dict['train_steps'],
        data_dir=dataset_dict['data_dir'],
        seed=dataset_dict['seed'],
        shuffle_buffer=dataset_dict['shuffle_buffer'],
    )

    activation_fn = partial(qrelu_ptq, bits=bits, max_value=2.0)

    # loading the model
    key1, key2, key3, key4 = jax.random.split(key1, 4)
    rngs = nnx.Rngs(params=key1, activations=key2, permute=key3, default=key4)

    model = ScRRAMBLeCapsNetWithReconstruction(
        input_vector_size=1024,
        capsule_size=256,
        receptive_field_size=64,
        connection_probability=0.2,
        rngs=rngs,
        layer_sizes=[50, 10], 
        activation_function=activation_fn
    )

    loaded_state = pickle.load(open(model_path, "rb"))
    graphdef, old_state = nnx.split(model)
    model = nnx.merge(graphdef, loaded_state)

    accuracy_per_batch = []

    for test_batch in test_ds.as_numpy_iterator():
        out = pred_step(model, test_batch)
        accuracy_per_batch.append(float(jnp.mean(out == test_batch['label'])))

    logs['test_accuracy'].append(float(jnp.mean(jnp.array(accuracy_per_batch))))

        # # evaluate the model
        # for test_batch in test_ds.as_numpy_iterator():
        #     eval(model, metrics, test_batch)

        # # metrics
        # for metric, value in metrics.compute().items():
        #     logs[f'test_{metric}'].append(float(value))
        # metrics.reset()  # Reset the metrics for the next training epoch.

    print(f"Test accuracy for {bits} bits: {logs['test_accuracy'][-1]}")

## TODO: Add a save script
save_metrics(logs, f"ptq_relu_cores{sum(model.layer_sizes)-model.input_eff_capsules}_{today}.pkl")

print("++"*50)
print(f"Saved at {os.path.join('/local_disk/vikrant/scrramble/logs', f'ptq_relu_cores{sum(model.layer_sizes)-model.input_eff_capsules}_{today}.pkl')}")
print("++"*50)



    

fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(logs['bits'], logs['test_accuracy'], marker='o')
ax.plot(logs['bits'], logs['test_accuracy'], marker='o', linestyle='-', color='b')
ax.set_xlabel('Number of Bits')
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy vs Number of Bits')
plt.show()

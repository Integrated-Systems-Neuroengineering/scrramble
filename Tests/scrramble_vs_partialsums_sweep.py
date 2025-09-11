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

from models import ScRRAMBLeCapsNetWithReconstruction, PartialSumsNetwork

from utils.activation_functions import quantized_relu_ste, squash, qrelu
# from utils.loss_functions import margin_loss
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

today = date.today().isoformat()

# -------------------------------------------------------
# Load the models: Change the directories to use appropriate/latest models.
# -------------------------------------------------------
scrramble50_state = pickle.load(open("/local_disk/vikrant/scrramble/models/sscamble_mnist_capsnet_recon_capsules50_acc_99_2025-09-10.pkl", "rb"))
ps50_state = pickle.load(open("/local_disk/vikrant/scrramble/models/partial_sums_full_precision_model_cores_50_acc_99_2025-09-10.pkl", "rb"))

# -------------------------------------------------------
# Auxilliary functions
# -------------------------------------------------------

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
# Eval functions
# -------------------------------------------------------

@nnx.jit
def pred_step_partialsums(model: PartialSumsNetwork, batch):
  logits = model(batch['image'])
  return logits.argmax(axis=1)

# writing the pred function
@nnx.jit
def pred_step_scrramble(model: ScRRAMBLeCapsNetWithReconstruction, batch):
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
max_vals_list = jnp.logspace(-6, 5, 30, base=2).tolist()
scrramble_logs = defaultdict(list)
ps_logs = defaultdict(list)

key1 = jax.random.key(10)

# -------------------------------------------------------
# Entropy sweep for 8 bits
# -------------------------------------------------------

# for i, bits in tqdm(enumerate(bits_list), total=len(bits_list), desc="Bits sweep"):
# for i, max_val in tqdm(enumerate(max_vals_list), total=len(max_vals_list), desc="Entropy sweep"):
#     print("--"*40)
#     print(f"Running inference for {max_val} bits")
#     print("--"*40)

#     activation_fn = partial(qrelu, bits=8, max_value=max_val)

#     # loading the model
#     key1, key2, key3, key4 = jax.random.split(key1, 4)
#     rngs = nnx.Rngs(params=key1, activations=key2, permute=key3, default=key4)

#     scrramble50 = ScRRAMBLeCapsNetWithReconstruction(
#     input_vector_size=1024,
#     capsule_size=256,
#     receptive_field_size=64,
#     connection_probability=0.2,
#     rngs=rngs,
#     layer_sizes=[40, 10],  # 20 capsules in the first layer and (translates to sum of layer_sizes cores total)
#     activation_function=activation_fn
#     ) 

#     graphdef, untrained_state = nnx.split(scrramble50)
#     scrramble50 = nnx.merge(graphdef, scrramble50_state)

#     partialsums50 = PartialSumsNetwork(
#     layer_sizes=[1024, 2048, 512, 256],
#     rngs=rngs,
#     activation_function=activation_fn,
#     columns_per_core=256
#         )
#     graphdef, untrained_state = nnx.split(partialsums50)
#     partialsums50 = nnx.merge(graphdef, ps50_state)

#     # evaluate the scrramble model
#     accuracy_per_batch = []
#     for test_batch in test_ds.as_numpy_iterator():
#         # ScrRAMBLE model
#         out = pred_step_scrramble(scrramble50, test_batch)
#         accuracy_per_batch.append(float(jnp.mean(out == test_batch['label'])))

#     scrramble_logs['test_accuracy'].append(float(jnp.mean(jnp.array(accuracy_per_batch))))
#     scrramble_logs['max_val'].append(float(max_val))

#     print(f"ScrRAMBLE model accuracy for {max_val} bits: {scrramble_logs['test_accuracy'][-1]}")

#     # evaluate partial sums model
#     accuracy_per_batch = []
#     for test_batch in test_ds.as_numpy_iterator():
#         # Partial Sums model
#         out = pred_step_partialsums(partialsums50, test_batch)
#         accuracy_per_batch.append(float(jnp.mean(out == test_batch['label'])))

#     ps_logs['test_accuracy'].append(float(jnp.mean(jnp.array(accuracy_per_batch))))
#     ps_logs['max_val'].append(float(max_val))

#     print(f"Partial Sums model accuracy for {max_val} bits: {ps_logs['test_accuracy'][-1]}")

# # save the logs
# save_metrics(scrramble_logs, f"scrramble_ptq_entropy_sweep_cores_50_bits_8_{today}.pkl")
# save_metrics(ps_logs, f"partial_sums_ptq_entropy_sweep_cores_50_bits_8_{today}.pkl")

# print("++"*50)
# print(f"ScRRAMBLe Logs Saved at: {os.path.join('/local_disk/vikrant/scrramble/logs', f'scrramble_ptq_entropy_sweep_cores_50_bits_8_{today}.pkl')}")
# print(f"Partial Sums Logs Saved at: {os.path.join('/local_disk/vikrant/scrramble/logs', f'partial_sums_ptq_entropy_sweep_cores_50_bits_8_{today}.pkl')}")
# print("++"*50)

# -------------------------------------------------------
# Bits sweep for max value = 10.0
# -------------------------------------------------------
for i, bits in tqdm(enumerate(bits_list), total=len(bits_list), desc="Bits sweep"):
    print("--"*40)
    print(f"Running inference for {bits} bits")
    print("--"*40)

    max_val=10.0

    activation_fn = partial(qrelu, bits=bits, max_value=max_val)

    # loading the model
    key1, key2, key3, key4 = jax.random.split(key1, 4)
    rngs = nnx.Rngs(params=key1, activations=key2, permute=key3, default=key4)

    scrramble50 = ScRRAMBLeCapsNetWithReconstruction(
    input_vector_size=1024,
    capsule_size=256,
    receptive_field_size=64,
    connection_probability=0.2,
    rngs=rngs,
    layer_sizes=[40, 10],  # 20 capsules in the first layer and (translates to sum of layer_sizes cores total)
    activation_function=activation_fn
    ) 

    graphdef, untrained_state = nnx.split(scrramble50)
    scrramble50 = nnx.merge(graphdef, scrramble50_state)

    partialsums50 = PartialSumsNetwork(
    layer_sizes=[1024, 2048, 512, 256],
    rngs=rngs,
    activation_function=activation_fn,
    columns_per_core=256
        )
    graphdef, untrained_state = nnx.split(partialsums50)
    partialsums50 = nnx.merge(graphdef, ps50_state)

    # evaluate the scrramble model
    accuracy_per_batch = []
    for test_batch in test_ds.as_numpy_iterator():
        # ScrRAMBLE model
        out = pred_step_scrramble(scrramble50, test_batch)
        accuracy_per_batch.append(float(jnp.mean(out == test_batch['label'])))

    scrramble_logs['test_accuracy'].append(float(jnp.mean(jnp.array(accuracy_per_batch))))
    scrramble_logs['max_val'].append(float(max_val))
    scrramble_logs['bits'].append(int(bits))


    print(f"ScrRAMBLE model accuracy for {bits} bits: {scrramble_logs['test_accuracy'][-1]}")

    # evaluate partial sums model
    accuracy_per_batch = []
    for test_batch in test_ds.as_numpy_iterator():
        # Partial Sums model
        out = pred_step_partialsums(partialsums50, test_batch)
        accuracy_per_batch.append(float(jnp.mean(out == test_batch['label'])))

    ps_logs['test_accuracy'].append(float(jnp.mean(jnp.array(accuracy_per_batch))))
    ps_logs['max_val'].append(float(max_val))
    ps_logs['bits'].append(int(bits))


    print(f"Partial Sums model accuracy for {bits} bits: {ps_logs['test_accuracy'][-1]}")

# save the logs
save_metrics(scrramble_logs, f"scrramble_ptq_bits_sweep_cores_50_maxval_{max_val}_{today}.pkl")
save_metrics(ps_logs, f"partial_sums_ptq_bits_sweep_cores_50_maxval_{max_val}_{today}.pkl")

print("++"*50)
print(f"ScRRAMBLe Logs Saved at: {os.path.join('/local_disk/vikrant/scrramble/logs', f'scrramble_ptq_bits_sweep_cores_50_maxval_{max_val}_{today}.pkl')}")
print(f"Partial Sums Logs Saved at: {os.path.join('/local_disk/vikrant/scrramble/logs', f'partial_sums_ptq_bits_sweep_cores_50_maxval_{max_val}_{today}.pkl')}")
print("++"*50)






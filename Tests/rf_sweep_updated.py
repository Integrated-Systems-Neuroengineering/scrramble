"""
Sweeping receptive field sizes/slot sizes.
Updated script.

Created on 08/25/2025
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
import csv

import signal
import sys
import traceback

from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste, squash
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist
from utils.loss_functions import margin_loss


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

# -------------------------------------------------------------------
# Data Logging
# -------------------------------------------------------------------

def setup_csv_logging(num_repeats: int):
    """Setup CSV file with headers"""
    today = date.today().isoformat()
    logs_path = "/Volumes/export/isn/vikrant/Data/scrramble/logs"
    os.makedirs(logs_path, exist_ok=True)

    csv_filename = os.path.join(logs_path, f'sweep_rf_size_connection_proba_capsnet_repeats_{num_repeats}_{today}.csv')

    # Write headers
    headers = [
        'rf_size', 'connection_probability', 'repeat_num',
        'test_accuracy', 'test_loss', 
        'valid_accuracy', 'valid_loss',
        'train_accuracy', 'train_loss',
        'best_step', 'num_cores'
    ]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    
    print(f"CSV logging setup at: {csv_filename}")
    return csv_filename

def log_result_to_csv(csv_filename, rf_size, conn_prob, repeat_num, 
                     test_acc, test_loss, valid_acc, valid_loss, 
                     train_acc, train_loss, best_step, num_cores):
    """Append a single result to CSV file"""
    
    row = [
        int(rf_size), float(conn_prob), int(repeat_num),
        float(test_acc), float(test_loss),
        float(valid_acc), float(valid_loss), 
        float(train_acc), float(train_loss),
        int(best_step), int(num_cores)
    ]
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


# -------------------------------------------------------------------
# Exception Handling: TODO
# -------------------------------------------------------------------



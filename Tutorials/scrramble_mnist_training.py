"""
ScRRAMBLe-CapsNet framework with a feedforward reconstruction network for MNIST classification. 
- The architecture uses input-balanced intercore routing with CapsNet framework to handle outputs per core.
- Model can be saved and reshaped into (..., capsule_size, capsule_size) for visualization of the learned connectivity and weights.
- ScRRAMBLe_routing provdes a way to visualize the routing per layer and visualizing the weights.

Example run commany: python3 scrramble_mnist_training.py --connection_density 0.2 --slot_size 64 --resample 1 --train_steps 50000


"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import argparse
import fcntl
import csv

import jax
import math
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
import json
import pickle
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from datetime import date

import matplotlib.pyplot as plt
import matplotlib as mpl

from models import ScRRAMBLeCapsLayer

from utils.activation_functions import quantized_relu_ste, squash
from utils import ScRRAMBLe_routing, intercore_connectivity, load_and_augment_mnist


import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

# -------------------------------------------------------------------
# Parsing arguments
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="ScRRAMBLe-CapsNet on MNIST with reconstruction")

    # parameters to sweep
    parser.add_argument("--connection_density", type=float, required=True) # e.g. 0.2
    parser.add_argument("--slot_size", type=int, required=True) # e.g. 64
    parser.add_argument("--resample", type=int, required=True) # e.g. 30
    parser.add_argument("--seed", type=int, default=101, help="Random seed for reproducibility")

    # model parameters
    parser.add_argument("--capsule_sizes", nargs="+", type=int, default=[50, 10])
    parser.add_argument("--capsule_size", type=int, default=256)

    # hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=int(5e4))
    parser.add_argument("--eval_every", type=int, default=1000)

    # dataset parameters
    parser.add_argument("--augmentation", action='store_true', help="Whether to apply data augmentation") # whether to apply data augmentation
    parser.add_argument("--plot_reconstruction", action='store_true', help="Whether to plot reconstructed images") # whether to plot reconstructed images

    # output files
    parser.add_argument("--data_dir", type=str, default=None) # change this as needed
    parser.add_argument("--results_dir", type=str, default="../results/") # Change this as needed
    parser.add_argument("--save_results", action='store_true', help="Save results to CSV")
    parser.add_argument("--results", type=str)
    parser.add_argument("--save_metrics", action="store_true", help="Save metrics to JSON") # whether to save the metrics history
    parser.add_argument("--metrics_file", type=str)
    parser.add_argument("--save_model", action="store_true", help="Save Model to pickle") # whether to save the model
    parser.add_argument("--model_file", type=str)

    return parser.parse_args()
# -------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------
def save_result_to_csv(result, csv_path):
    # Create directory if needed
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if file exists to write headers
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        # Lock file for thread-safe writing
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        try:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(result)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def save_payload(state, configs, filename):
    """
    Save model and parameters to a pickle.
    state: nnx.Model state
    configs: dict, configuration parameters
    """

    payload = {
        'configs' : configs,
        'state': state
    }

    checkpoint_dir = "../checkpoint" #"/local_disk/vikrant/scrramble/models"
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"Model saved to {filename_}")

# -------------------------------------------------------------------
# Define the Reconstruction network
# -------------------------------------------------------------------
class ReconstructionNetwork(nnx.Module):
    def __init__(self,
                 input_size: int,
                 rngs: nnx.Rngs):
        
        # define feedforward layers
        self.fc1 = nnx.Linear(input_size, 5000, rngs=rngs)
        self.fc2 = nnx.Linear(5000, 3000, rngs=rngs)
        self.fc3 = nnx.Linear(3000, 28*28, rngs=rngs)

    def __call__(self, x):

        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        x = nnx.sigmoid(self.fc3(x))

        return x


# -------------------------------------------------------------------
# Define the MNIST CapsNet Model
# -------------------------------------------------------------------
class ScRRAMBLeCapsNetWithReconstruction(nnx.Module):
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
            activation_function: Callable = nnx.relu, # activation function to use in the network
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
        self.scrramble_caps_layers = nnx.List(
            [ScRRAMBLeCapsLayer(
            input_vector_size=self.capsule_size * Nci,
            num_capsules=Nco,
            capsule_size=self.capsule_size,
            receptive_field_size=self.receptive_field_size,
            connection_probability=self.connection_probability,
            rngs=self.rngs
        ) for Nci, Nco in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        )

        # define the reconstruction network
        self.reconstruction_nw = ReconstructionNetwork(
            input_size=self.capsule_size * self.layer_sizes[-1],  # Input size
            rngs=self.rngs
        )


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
            x = self.activation_function(x)  # Apply the activation function.
            x = jnp.reshape(x, shape_x)

        # add the reconstruction network
        x_recon = x.reshape((x.shape[0], -1))  # Flatten the output for reconstruction.
        x_recon = jax.vmap(self.reconstruction_nw, in_axes=(0,))(x_recon)  # Apply the reconstruction network.


        return x_recon, x
    

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

    # the length of the vector encodes probability of a class
    caps_output = caps_output.reshape(caps_output.shape[0], num_classes, -1)

    # apply squash function along the last axis
    caps_output = squash(caps_output, axis=-1, eps=1e-8)

    caps_output_magnitude = jnp.linalg.norm(caps_output, axis=-1)

    # create one-hot-encoded labels
    labels = jax.nn.one_hot(labels, num_classes=caps_output_magnitude.shape[1])

    # compute the margin loss
    loss_per_sample = jnp.sum(labels * jax.nn.relu(m_plus - caps_output_magnitude)**2 + lambda_ * (1 - labels) * jax.nn.relu(caps_output_magnitude - m_minus)**2, axis=1)
    loss = jnp.mean(loss_per_sample)


    return loss, caps_output_magnitude

def loss_fn(model, batch, num_classes=10, m_plus=0.9, m_minus=0.1, lambda_=0.5, regularizer=1e-4):
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

@nnx.jit
def train_step(model: ScRRAMBLeCapsNetWithReconstruction, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(model, grads)  # In-place updates.

@nnx.jit
def eval_step(model: ScRRAMBLeCapsNetWithReconstruction, metrics: nnx.MultiMetric, batch, loss_fn: Callable = loss_fn):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.

# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------
def train_scrramble_capsnet_mnist(
        model: ScRRAMBLeCapsNetWithReconstruction,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        dataset_dict: dict,
        resample: int,
        metrics_history: dict,
        save_model_flag: bool = False, # as needed to save model
        save_metrics_flag: bool = False, # as needed to save metrics history
):
    
    eval_every = dataset_dict['eval_every']
    train_steps = dataset_dict['train_steps']

    key1 = jax.random.key(10)

    for step, batch in enumerate(train_ds.as_numpy_iterator()):

        metrics_history['step'].append(step)  # Record the step.

        train_step(model, optimizer, metrics, batch)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
            metrics_history['step'].append(step)  # Record the step.
            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(float(value))  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the validation set after each training epoch.
            for valid_batch in valid_ds.as_numpy_iterator():
                eval_step(model, metrics, valid_batch)

            # Log the validation metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'valid_{metric}'].append(float(value))
            metrics.reset()  # Reset the metrics for the next training epoch.

            print(f"Step {step}: Valid loss: {metrics_history['valid_loss'][-1]}, Accuracy: {metrics_history['valid_accuracy'][-1]}")

    # find the test set accuracy
    for test_batch in test_ds.as_numpy_iterator():
        eval_step(model, metrics, test_batch)
        # print the metrics
    for metric, value in metrics.compute().items():
        metrics_history[f'test_{metric}'].append(float(value))
    metrics.reset()  # Reset the metrics for the next training epoch.

    print("="*50)
    print(f"Resample: {resample}, Test loss: {metrics_history['test_loss'][-1]}, Test accuracy: {metrics_history['test_accuracy'][-1]}")
    print("="*50)

    if save_model_flag:
        today = date.today().isoformat()
        filename = f"sscamble_mnist_capsnet_recon_capsules{(sum(model.layer_sizes)-model.input_eff_capsules):d}_acc_{metrics_history['test_accuracy'][-1]*100:.0f}_{today}.pkl"
        graphdef, state = nnx.split(model)
        save_payload(state=state, configs=dataset_dict, filename=filename)

    if save_metrics_flag:
        today = date.today().isoformat()
        filename = f"sscamble_mnist_capsnet_recon_perf_metrics_capsules{(sum(model.layer_sizes)-model.input_eff_capsules):d}_acc_{metrics_history['test_accuracy'][-1]*100:.0f}_{today}.pkl"
        save_payload(state=metrics_history, configs=dataset_dict, filename=filename)
        # save_metrics(metrics_history, filename)

    return model

# -------------------------------------------------------------------
# Training function
# -------------------------------------------------------------------
def main():
    args = parse_args()

    print("--"*50)
    print(f"TRAINING CONFIGURATION: \n")
    print(f"Core Sizes: {args.capsule_sizes} ")
    # print(f"Model Core sizes: {model.capsule_sizes} ")
    print(f"Connection Density: {args.connection_density} ")
    print(f"Slot Size: {args.slot_size} ")
    print(f"Resample: {args.resample} ")
    print(f"Batch Size: {args.batch_size} ")

    seed = args.seed
    num_resamples = args.resample

    dataset_dict = {
    'batch_size': args.batch_size, # 64 is a good batch size for MNIST
    'train_steps': args.train_steps, # run for longer, 20000 is good!
    'data_dir': args.data_dir,
    'augmentation': args.augmentation,
    'seed': seed,
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
        augmentation=dataset_dict['augmentation'], # keep True
    )

    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'valid_loss': [],
        'valid_accuracy': [],
        'test_accuracy': [],
        'step': [],
        'resample': []
        }


    for r in tqdm(range(num_resamples), desc="Resamples"):

        model_parameters = {
            'input_vector_size': 1024, # MNIST is resized to 32x32 and flattened, so 1024
            'capsule_sizes': args.capsule_sizes,
            'rngs': nnx.Rngs(default=int(seed + r), permute=int(seed + r + 1), params=int(seed + r + 2), activation=int(seed + r + 3)),
            'connection_probability': args.connection_density,
            'receptive_field_size': args.slot_size,
            'layer_sizes': args.capsule_sizes,
            'capsule_size': args.capsule_size,
            'activation_function': nnx.relu,
        }

        model = ScRRAMBLeCapsNetWithReconstruction(
            input_vector_size=model_parameters['input_vector_size'],
            capsule_size=model_parameters['capsule_size'],
            receptive_field_size=model_parameters['receptive_field_size'],
            connection_probability=model_parameters['connection_probability'],
            rngs=model_parameters['rngs'],
            layer_sizes=model_parameters['layer_sizes'],
            activation_function=model_parameters['activation_function'],
        )

        # hyperparameters
        hyperparameters = {
            'learning_rate': args.learning_rate, # 1e-3 seems to work well
            'momentum': args.momentum, 
            'weight_decay': args.weight_decay
        }

        # optimizer
        optimizer = nnx.Optimizer(
            model,
            optax.adamw(learning_rate=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay']),
            wrt=nnx.Param
        )

        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average('loss')
        )

        # call the training function
        trained_model = train_scrramble_capsnet_mnist(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            dataset_dict=dataset_dict,
            resample=r,
            metrics_history=metrics_history,
            save_model_flag=args.save_model,
            save_metrics_flag=args.save_metrics
        )

    # if reconstruction is enabled, plot reconstructed images
    if args.plot_reconstruction:
        timestamp = date.today().isoformat()
        trained_model.eval()

        # get a batch of test images
        test_batch = next(test_ds.as_numpy_iterator())
        recon, _ = trained_model(test_batch['image'])
        recon = recon.reshape((-1, 28, 28))

        # plot original images on top and reconstructed on bottom for the first 10 images in test dataset
        fig, axes = plt.subplots(2, 10, figsize=(16, 8))
        for i in range(10):
            axes[0, i].imshow(test_batch['image'][i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title(f"Original: {test_batch['label'][i]}")
            axes[0, i].axis('off')

            axes[1, i].imshow(recon[i], cmap='gray')
            axes[1, i].set_title(f"Reconstructed")
            axes[1, i].axis('off')

        plt.tight_layout()
        os.makedirs("../plots/", exist_ok=True)
        fig.savefig(f"../plots/mnist_reconstruction_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()

        print("--"*50)
        print(f"Reconstruction plot saved to ../plots/mnist_reconstruction_{timestamp}.png")
        print("--"*50)








if __name__ == "__main__":
    main()
    
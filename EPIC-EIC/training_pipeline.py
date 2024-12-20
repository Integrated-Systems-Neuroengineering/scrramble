"""
Define a training pipeline for EPIC-EIC model.
"""

import jax
import jax.numpy as jnp
import optax
import flax
from tqdm import tqdm
import matplotlib.pyplot as plt
from flax import linen as nn
import tensorflow as tf
import tensorflow_datasets as tfds
from EICDense import *
from ShuffleBlocks import *
from Accumulator import *
from PseudoFFNet import *
from HelperFunctions.binary_trident_helper_functions import *
from HelperFunctions.binary_mnist_dataloader import *
from HelperFunctions.metric_functions import *

DATA_PATH = f"/Users/vikrantjaltare/OneDrive - UC San Diego/Datasets/"

def make_batches(inputs, targets, batch_size):
    """
    Make data batches for training.
    """

    num_samples = len(inputs)

    for i in range(0, num_samples, batch_size):
        yield inputs[i:i+batch_size], targets[i:i+batch_size]


def EIC_training_pipeline(
        data_path = DATA_PATH,
        batch_size = 128,
        learning_rate = 1e-3,
        epochs = 2
):

    """
    Training pipeline for EPIC-EIC model.
    """

    # initialize the model
    rng = jax.random.key(0)
    model = PseudoFFNet()
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(model)

    # load the data
    (train_inputs, train_labels), (val_inputs, val_labels), (test_inputs, test_labels) = load_and_process_mnist(data_path, binarize = True)

    params = model.init(rng, train_inputs[0].reshape(-1,))

    # training loop
    for epoch in tqdm(range(epochs)):
        print("_________________________________________________")
        print(f"Epoch: {epoch+1}")

        # compute accuracy, if epoch = 1 we get the baseline accuracy
        eval_keys = jax.random.split(rng, train_inputs.shape[0])
        train_acc = accuracy(params, model, train_inputs, train_labels, eval_keys)
        val_acc = accuracy(params, model, val_inputs, val_labels, eval_keys)
        
        print(f"Train Accuracy: {train_acc*100:.2f}%, Validation Accuracy: {val_acc*100:.2f}%")
        print("_________________________________________________")

        # training
        for batch in make_batches(train_inputs, train_labels, batch_size):
            batch_images, batch_labels = batch
            keys = jax.random.split(rng, batch_images.shape[0])

            def loss_fn(params):
                logits = vmap(lambda img, key: model.apply(params, img, rngs = {"activation": key}))(batch_images, keys)
                return cross_entropy_loss(logits, batch_labels)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
    
    # copmute the test accuracy
    test_acc = accuracy(params, model, test_inputs, test_labels)
    print("_________________________________________________")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("_________________________________________________")


    return params

def __main__():
    params = EIC_training_pipeline()
    print(params)


if __name__ == "__main__":
    __main__()
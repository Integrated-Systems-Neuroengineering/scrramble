import jax
import math
import jax.numpy as jnp
from typing import Callable
from functools import partial

import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

data_dir = "/local_disk/vikrant/datasets"
dataset_dict = {
    'batch_size': 64, # 64 is a good batch size for MNIST
    'train_steps': 20000, # run for longer, 20000 is good!
    'binarize': True, 
    'greyscale': True,
    'data_dir': data_dir,
    'seed': 101,
    'shuffle_buffer': 1024,
    'threshold' : 0.5, # binarization threshold, not to be confused with the threshold in the model
    'eval_every': 500,
}


def load_and_augment_mnist(
    batch_size: int,
    train_steps: int,
    binarize: bool,
    greyscale: bool,
    data_dir: str,
    seed: int,
    shuffle_buffer: int,
    threshold: float, # this is the binarization threshold, not to be confused with the threshold in the model
    augmentation: bool = True,
    aug_translate: bool = True,
    num_translations: int = 4,
    max_shift: int = 3,
    quantize_bits: int = 8,
    # aug_rotate: float = 10 # degrees
):

    """
    Load and augment the MNIST dataset
    """

    tf.random.set_seed(seed)

    # do train-valid-test split
    train_ds, valid_ds, test_ds = tfds.load('mnist', split=['train[:50000]', 'train[50000:]', 'test'], data_dir=data_dir)

    ## adding helper function

    # normalize
    def _normalize(sample):
        img = tf.cast(sample['image'], tf.float32) / 255.0
        return {'image': img, 'label': sample['label']}

    # translate by a few pixels
    def _translate(image, dx, dy):
        """
        Translate image by dx, dy pixels
        """

        if dx != 0:
            image = tf.roll(image, shift=int(dx), axis=1)
        else:
            pass

        if dy != 0:
            image = tf.roll(image, shift=int(dy), axis=0)
        else:
            pass

        return image
    
    #  a function that puts together the augmented dataset
    def create_augmented_dataset(sample, num_translations=num_translations):

        image = sample['image']
        label = sample['label']

        images = [image]
        labels = [label]

        for _ in range(num_translations):
            # sample dx from a uniform
            dx = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
            dy = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
            translated_image = _translate(image, dx ,dy)
            images.append(translated_image)
            labels.append(label)

        # stack the images
        stacked_images = tf.stack(images)
        stacked_labels = tf.stack(labels)

        aug_dataset = tf.data.Dataset.from_tensor_slices({
            'image': stacked_images,
            'label': stacked_labels
        })

        return aug_dataset

    # check for flags and apply augmentation
    if augmentation:
        train_ds = train_ds.map(create_augmented_dataset).flat_map(lambda x: x)
        valid_ds = valid_ds.map(create_augmented_dataset).flat_map(lambda x: x)

    # normalize
    def _normalize(sample):
        img = tf.cast(sample['image'], tf.float32) / 255.0
        return {'image': img, 'label': sample['label']}

    train_ds = train_ds.map(_normalize)
    valid_ds = valid_ds.map(_normalize)
    test_ds = test_ds.map(_normalize)

    # # apply binarization
    # if binarize:
    #     binarize_mask = lambda s: {
    #         'image': tf.where(s['image'] > threshold, 1.0, 0.0),
    #         'label': s['label']
    #     }

    #     train_ds = train_ds.map(binarize_mask)
    #     valid_ds = valid_ds.map(binarize_mask)
    #     test_ds = test_ds.map(binarize_mask)
    # else:
    #     pass

    def _quantization(sample):
        """
        Quantize the image to given number of bits.
        """

        img = sample['image']
        img = tf.quantization.fake_quant_with_min_max_args(
            img,
            min=0.0,
            max=1.0,
            num_bits=quantize_bits
        )

        return {'image': img, 'label': sample['label']}
    
    # apply the quantization function
    train_ds = train_ds.map(_quantization)
    valid_ds = valid_ds.map(_quantization)
    test_ds = test_ds.map(_quantization)


    # TODO: apply greyscale if more than 3 channels

    # Prepare datasets
    train_ds = train_ds.repeat().shuffle(shuffle_buffer)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, valid_ds, test_ds


# test
def __main__():

    train_ds, valid_ds, test_ds = load_and_augment_mnist(
        batch_size=dataset_dict['batch_size'],
        train_steps=dataset_dict['train_steps'],
        binarize=dataset_dict['binarize'],
        greyscale=dataset_dict['greyscale'],
        data_dir=dataset_dict['data_dir'],
        seed=dataset_dict['seed'],
        shuffle_buffer=dataset_dict['shuffle_buffer'],
        threshold=dataset_dict['threshold'],
        augmentation=True,
    )

if __name__ == "__main__":
    __main__()
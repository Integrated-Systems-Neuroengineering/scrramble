import jax
import math
import jax.numpy as jnp
from typing import Callable
from functools import partial

import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

DATA_PATH = "/local_disk/vikrant/datasets"

def load_cifar10(
    batch_size: int,
    train_steps: int,
    data_dir: str,
    seed: int,
    shuffle_buffer: int,
    augmentation: bool = True,
    quantize_flag: bool = False, # whether to quantize the images
    quantize_bits: int = 8,
    num_rotations: int = 4, # for every image, rotate it by 0, 90, 180, 270 degrees
):
    
    """
    Load and augment CIFAR-10 Dataset
    """

    # set the seed
    tf.random.set_seed(seed)

    # load and split the dataset into train-valid-test
    train_ds, valid_ds, test_ds = tfds.load('cifar10', split=['train[:45000]', 'train[45000:]', 'test'], data_dir=data_dir)

    # normalize the images to 0-1
    def _normalize(sample):
        img = tf.cast(sample['image'], tf.float32) / 255.0
        return {'image': img, 'label': sample['label']}

    # rotate the image by a multiple of 90 degrees
    def _rotate_90(image, k):
        """ Rotate the image by k*90 degrees"""
        return tf.image.rot90(image, k=k)

    # define quantization function
    def _quantize(sample, bits=quantize_bits):
        """
        Quantize an image to a given number of bits
        """
        image = sample['image']
        image = tf.quantization.fake_quant_with_min_max_args(
            image,
            min=0.0,
            max=1.0,
            num_bits=bits
        )
        
        return {'image': image, 'label': sample['label']}
    
    # define augmentation function
    def create_augmented_dataset(sample):
        """
        Created augmented CIFAR10 dataset
        """
        image = sample['image']
        label = sample['label']

        images = []
        labels = []

        # include all num_rotations rotations
        for k in range(num_rotations):
            rotated_image = _rotate_90(image, k)
            images.append(rotated_image)
            labels.append(label)

        # stack the images and labels
        stacked_images = tf.stack(images)
        stacked_labels = tf.stack(labels)

        # create augmented dataset
        augmented_dataset = tf.data.Dataset.from_tensor_slices(
            {
                'image': stacked_images,
                'label': stacked_labels
            }
        )

        return augmented_dataset
    
    # apply normalization
    train_ds = train_ds.map(_normalize)
    valid_ds = valid_ds.map(_normalize)
    test_ds = test_ds.map(_normalize)
    
    # apply quantization
    if quantize_flag:
        train_ds = train_ds.map(_quantize)
        valid_ds = valid_ds.map(_quantize)
        test_ds = test_ds.map(_quantize)

    # augment the dataset
    if augmentation:
        train_ds = train_ds.flat_map(create_augmented_dataset)
        valid_ds = valid_ds.flat_map(create_augmented_dataset)

    # Prep datasets for training
    train_ds = train_ds.repeat().shuffle(shuffle_buffer)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, valid_ds, test_ds

# def load_cifar10(
#     batch_size: int,
#     train_steps: int,
#     data_dir: str,
#     seed: int,
#     shuffle_buffer: int,
#     augmentation: bool = True,
#     quantize_flag: bool = True, # whether to quantize the images
#     quantize_bits: int = 8,
#     num_rotations: int = 4, # for every image, rotate it by 0, 90, 180, 270 degrees
# ):
    
#     """
#     Load and augment CIFAR-10 Dataset
#     """

#     # set the seed
#     tf.random.set_seed(seed)

#     # load and split the dataset into train-valid-test
#     train_ds, valid_ds, test_ds = tfds.load('cifar10', split=['train[:45000]', 'train[45000:]', 'test'], data_dir=data_dir)

#     # normalize the images to 0-1
#     def _normalize(sample):
#         img = tf.cast(sample['image'], tf.float32) / 255.0
#         return {'image': img, 'label': sample['label']}
    
#     # define a function to greyscale the images
#     def _greyscale(sample):
#         """Convert RGB image to greyscale"""
#         image = sample['image']
#         if len(image.shape) == 3 and image.shape[-1] == 3:
#             return tf.image.rgb_to_grayscale(image)
#         return image

#     # rotate the image by a multiple of 90 degrees
#     def _rotate_90(image, k):
#         """ Rotate the image by k*90 degrees"""
#         return tf.image.rot90(image, k=k)

#     # define quantization function
#     def _quantize(sample, bits=quantize_bits):
#         """
#         Quantize an image to a given number of bits
#         """

#         image = sample['image']
#         image = tf.quantization.fake_quant_with_min_max_args(
#             image,
#             min=0.0,
#             max=1.0,
#             num_bits=bits
#         )
        
#         return {'image': image, 'label': sample['label']}
    
#     # define augmentation function
#     def create_augmented_dataset(sample):
#         """
#         Created augmented CIFAR10 dataset
#         """

#         image = sample['image']
#         label = sample['label']

#         images = []
#         labels = []

#         # include all num_rotations rotations
#         for k in range(num_rotations):
#             rotated_image = _rotate_90(image, k)
#             images.append(rotated_image)
#             labels.append(label)

#         # stack the images and labels
#         stacked_images = tf.stack(images)
#         stacked_labels = tf.stack(labels)

#         # create augmented dataset
#         augmented_dataset = tf.data.Dataset.from_tensor_slices(
#             {
#                 'image': stacked_images,
#                 'label': stacked_labels
#             }
#         )

#         return augmented_dataset
    
#     # apply normalization
#     train_ds = train_ds.map(_normalize)
#     valid_ds = valid_ds.map(_normalize)
#     test_ds = test_ds.map(_normalize)

#     # # greyscale
#     # if greyscale:
#     #     train_ds = train_ds.map(_greyscale)
#     #     valid_ds = valid_ds.map(_greyscale)
#     #     test_ds = test_ds.map(_greyscale)
    
#     # apply quantization
#     if quantize_flag:
#         train_ds = train_ds.map(_quantize)
#         valid_ds = valid_ds.map(_quantize)
#         test_ds = test_ds.map(_quantize)

#     # augment the dataset
#     if augmentation:
#         train_ds = train_ds.map(create_augmented_dataset).flat_map(lambda x: x)
#         valid_ds = valid_ds.map(create_augmented_dataset).flat_map(lambda x: x)

#     # Prep datasets for training
#     train_ds = train_ds.repeat().shuffle(shuffle_buffer)
#     train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
#     valid_ds = valid_ds.batch(batch_size, drop_remainder=True).prefetch(1)
#     test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

#     return train_ds, valid_ds, test_ds

# # Test function
# def __main__():

#     dataset_dict = {
#     'batch_size': 64, # 64 is a good batch size for CIFAR-10
#     'train_steps': 30000, # run for longer, 30000 is good for CIFAR-10
#     'binarize': False,  # CIFAR-10 is usually kept as RGB
#     'greyscale': False,  # CIFAR-10 is RGB by default
#     'data_dir': DATA_PATH,
#     'seed': 101,
#     'quantize_flag': True,  # whether to quantize the images
#     'quantize_bits': 8,  # number of bits to quantize the images
#     'num_rotations': 4,  # for every image, rotate it by
#     'shuffle_buffer': 1024,  # shuffle buffer size
#     }

#     train_ds, valid_ds, test_ds = load_cifar10(
#         batch_size=dataset_dict['batch_size'],
#         train_steps=dataset_dict['train_steps'],
#         data_dir=dataset_dict['data_dir'],
#         seed=dataset_dict['seed'],
#         shuffle_buffer=dataset_dict['shuffle_buffer'],
#         augmentation=True,
#         greyscale=dataset_dict['greyscale'],
#         quantize_flag=dataset_dict['quantize_flag'],
#         quantize_bits=dataset_dict['quantize_bits'],
#         num_rotations=dataset_dict['num_rotations'],
#     )

    
#     # Print some info about the datasets
#     print("Datasets created successfully!")
    
#     # Test a batch
#     for batch in train_ds.take(1):
#         print(f"Training batch shape: {batch['image'].shape}")
#         print(f"Training labels shape: {batch['label'].shape}")
#         break
    
#     for batch in valid_ds.take(1):
#         print(f"Validation batch shape: {batch['image'].shape}")
#         print(f"Validation labels shape: {batch['label'].shape}")
#         break
    
#     for batch in test_ds.take(1):
#         print(f"Test batch shape: {batch['image'].shape}")
#         print(f"Test labels shape: {batch['label'].shape}")
#         break


# # if __name__ == "__main__":
# #     __main__()







# # OLD CODE. TODO: Remove after updating

# # def load_cifar10(batch_size: int, train_steps: int, binarize: bool = True, greyscale: bool = True, data_dir: str = DATA_PATH, seed: int = 0, threshold: float = 0.5, shuffle_buffer: int = 1024):
# #     """
# #     Load CIFAR-10 dataset
# #     """

# #     tf.random.set_seed(seed)

# #     train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], data_dir=data_dir)

# #     # normalize
# #     def _normalize(sample):
# #         img = tf.cast(sample['image'], tf.float32) / 255.0

# #         if greyscale:
# #             img = tf.reduce_mean(img, axis=-1, keepdims=True)
        
# #         return {'image': img, 'label': sample['label']}

# #     train_ds = train_ds.map(_normalize)
# #     test_ds = test_ds.map(_normalize)

# #     # binarize
# #     if binarize:
# #         binarize_mask = lambda s: {
# #             'image': tf.where(s['image'] > threshold, 1.0, 0.0),
# #             'label': s['label']
# #         }

# #         train_ds = train_ds.map(binarize_mask)
# #         test_ds = test_ds.map(binarize_mask)

# #     # shuffle the dataset
# #     train_ds = train_ds.repeat().shuffle(shuffle_buffer)
# #     train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
# #     test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

# #     return train_ds, test_ds

# # def __main__():
# #     batch_size = 128
# #     train_steps = 5000
# #     eval_every = 50
# #     train_ds, test_ds = load_cifar10(batch_size = batch_size, train_steps = train_steps, binarize=True)
# #     batch = next(iter(train_ds))
# #     print(batch['image'].shape)

# # if __name__ == "__main__":
# #     __main__()
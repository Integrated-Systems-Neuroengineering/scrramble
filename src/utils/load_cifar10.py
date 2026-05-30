from typing import Callable
from functools import partial
import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.


def load_cifar10(
    batch_size: int,
    train_steps: int,
    seed: int,
    shuffle_buffer: int,
    augmentation: bool = True,
    quantize_flag: bool = False, # whether to quantize the images
    quantize_bits: int = 8,
    num_rotations: int = 4, # for every image, rotate it by 0, 90, 180, 270 degrees
    data_dir = None
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


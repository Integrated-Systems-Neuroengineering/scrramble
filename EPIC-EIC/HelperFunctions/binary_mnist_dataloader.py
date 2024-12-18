import jax
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

DATA_PATH = f"/Users/vikrantjaltare/OneDrive - UC San Diego/Datasets/"

## testing the load mnist function
DATA_PATH = f"/Users/vikrantjaltare/OneDrive - UC San Diego/Datasets/"

def load_and_process_mnist(data_path, binarize):
    """
    Pipeline to load, binarize, resize and batch MNIST dataset.
    Args:
        data_path: str, path to the data
        batch_size: int, batch size
        binarize: bool, whether to binarize the data or not
    Returns:
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels): tuple, tuple, tuple of jnp.ndarray
    """

    # load the dataset
    ds_train, ds_test = tfds.load(
        'mnist',
        split = ['train', 'test'],
        shuffle_files = True,
        as_supervised = True,
        data_dir = data_path
    )

    # normalize and convert to numpy
    def normalize_image(image, label):
        image = tf.cast(image, tf.float32)/ 255.0

        if binarize:
            image = tf.where(image < 0.5, 0.0, 1.0)

        return image, label
    
    train_data = ds_train.map(normalize_image)
    test_data = ds_test.map(normalize_image)

    # separate images and samples
    train_images, train_labels = zip(*train_data)
    test_images, test_labels = zip(*test_data)

    # convert tuples to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # pad the images with zeros to get 32x32 images
    train_images = jnp.pad(train_images, pad_width = ((0, 0), (2, 2), (2, 2), (0, 0)), mode = 'constant', constant_values = 0)
    test_images = jnp.pad(test_images, pad_width = ((0, 0), (2, 2), (2, 2), (0, 0)), mode = 'constant', constant_values = 0)

    # concatenate to create a single dataset
    images = np.concatenate([train_images, test_images], axis = 0)
    labels = np.concatenate([train_labels, test_labels], axis = 0)

    # split the data 50k-10k-10k
    train_images, val_images, test_images = images[:50000], images[50000:60000], images[60000:]
    train_labels, val_labels, test_labels = labels[:50000], labels[50000:60000], labels[60000:]

    # reshape the images
    train_images = train_images.reshape(-1, 1024)
    val_images = val_images.reshape(-1, 1024)
    test_images = test_images.reshape(-1, 1024)

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

def get_train_batches(train_images, train_labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds = ds.shuffle(len(train_images)).batch(batch_size).prefetch(1)
    return tfds.as_numpy(ds)










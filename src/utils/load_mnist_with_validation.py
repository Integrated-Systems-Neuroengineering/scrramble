import tensorflow as tf
import tensorflow_datasets as tfds

DATA_PATH = "/local_disk/vikrant/datasets"

def load_mnist_with_validation(batch_size: int, train_steps: int, train_split: float = 5e4/6e4, validation_split: float = 1e4/6e4, binarize: bool = True, greyscale: bool = True, data_dir: str = DATA_PATH, seed: int = 0, threshold: float = 0.5, shuffle_buffer: int = 1024):
    """
    Load CIFAR-10 dataset
    """

    tf.random.set_seed(seed)

    train_ds, valid_ds, test_ds = tfds.load('mnist', split=['train[:50000]', 'train[50000:]', 'test'], data_dir=data_dir)

    # normalize
    def _normalize(sample):
        img = tf.cast(sample['image'], tf.float32) / 255.0
        return {'image': img, 'label': sample['label']}

    train_ds = train_ds.map(_normalize)
    valid_ds = valid_ds.map(_normalize)
    test_ds = test_ds.map(_normalize)

    # binarize
    if binarize:
        binarize_mask = lambda s: {
            'image': tf.where(s['image'] > threshold, 1.0, 0.0),
            'label': s['label']
        }

        train_ds = train_ds.map(binarize_mask)
        test_ds = test_ds.map(binarize_mask)

    # shuffle the dataset
    train_ds = train_ds.repeat().shuffle(shuffle_buffer)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, valid_ds, test_ds

def __main__():
    batch_size = 100
    train_steps = 5000
    eval_every = 50
    # train_ds, test_ds = load_mnist(batch_size = batch_size, train_steps = train_steps, binarize=True)
    # batch = next(iter(train_ds))
    # print(batch['image'].shape)

    # Fix the function call to get all three datasets
    train_ds, valid_ds, test_ds = load_mnist_with_validation(batch_size=batch_size, train_steps=train_steps, binarize=True)

    # Print dataset sizes
    print(f"Train dataset: {train_steps * batch_size} examples ({train_steps} batches of {batch_size})")

    # Count validation examples
    valid_count = sum(1 for _ in valid_ds)
    print(f"Validation dataset: {valid_count} examples")

    # Count test batches and calculate total examples
    test_batches = sum(1 for _ in test_ds)
    print(f"Test dataset: {test_batches * batch_size} examples ({test_batches} batches of {batch_size})")

# if __name__ == "__main__":
#     __main__()
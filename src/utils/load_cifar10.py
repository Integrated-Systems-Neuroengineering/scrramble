import tensorflow as tf
import tensorflow_datasets as tfds

DATA_PATH = "/local_disk/vikrant/datasets"

def load_cifar10(batch_size: int, train_steps: int, binarize: bool = True, greyscale: bool = True, data_dir: str = DATA_PATH, seed: int = 0, threshold: float = 0.5, shuffle_buffer: int = 1024):
    """
    Load CIFAR-10 dataset
    """

    tf.random.set_seed(seed)

    train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], data_dir=data_dir)

    # normalize
    def _normalize(sample):
        img = tf.cast(sample['image'], tf.float32) / 255.0

        if greyscale:
            img = tf.reduce_mean(img, axis=-1, keepdims=True)
        
        return {'image': img, 'label': sample['label']}

    train_ds = train_ds.map(_normalize)
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

    return train_ds, test_ds

def __main__():
    batch_size = 128
    train_steps = 5000
    eval_every = 50
    train_ds, test_ds = load_cifar10(batch_size = batch_size, train_steps = train_steps, binarize=True)
    batch = next(iter(train_ds))
    print(batch['image'].shape)

# if __name__ == "__main__":
#     __main__()
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
import flax
from flax import nnx


def create_classification_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    n_informative: int,
    random_seed: int = 101,
    test_size: float = 0.2,
    binarize: bool = True

    ):

    """
    Create a synthetic classification dataset using sklearn.
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        random_state=random_seed
    )

    # z-score X
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # binarize
    if binarize:
        X = jnp.where(X>0.0, 1.0, 0.0)


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    return X_train, y_train, X_test, y_test

# create a batch iterator
def batch_iterator(X, y, rngs, batch_size):
    """
    Create a batch iterator for the dataset
    """

    # permute the dataset
    perm = jax.random.permutation(rngs.permute(), X.shape[0])
    X = X[perm]
    y = y[perm]

    # find number of steps in an epoch
    steps_per_epoch = X.shape[0]//batch_size

    for i in range(steps_per_epoch):
        start_idx = i*batch_size
        end_idx = start_idx + batch_size

        yield X[start_idx:end_idx], y[start_idx:end_idx]

    # TODO : add a final batch with the remaining samples


# testing
def __main__():
    X_train, y_train, X_test, y_test = create_classification_dataset(
        n_samples=1000,
        n_features=512,
        n_classes=2,
        n_informative=round(0.4*512),
    )



    b_iterator = batch_iterator(X_train, y_train, nnx.Rngs(permute=0), batch_size=32)
    for x, y in b_iterator:
        print(x.shape, y.shape)
        print(x[0, :5])
        print(y[0])

# if __name__ == "__main__":
#     __main__()

# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
# # check the data
# print(X_train[0, :5])
# print(y_train[0])
# print(X_test[0, :5])
# print(y_test[0])

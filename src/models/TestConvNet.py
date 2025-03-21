import jax
import flax
import jax.numpy as jnp
from flax import nnx
from utils import clipping_ste
from functools import partial

class TestConvNet(nnx.Module):
    """A simple conv net"""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3,3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3,3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2,2), strides=(2,2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)
        
        if activation = 'binary':
            self.activation = partial(clipping_ste, threshold=0.0, noise_sd=0.1, key=rngs.activation())
        else:
            self.activation = nnx.relu


    def __call__(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

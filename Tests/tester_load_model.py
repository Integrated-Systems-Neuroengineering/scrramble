import jax
import jax.numpy as jnp
import flax
from flax import nnx
import pickle
import os

model_path = os.path.join("/Volumes/export/isn/vikrant/Data/scrramble/models", "cifar10_tester.pkl")
loaded_model = pickle.load(open(model_path, 'rb'))
# graphdef, state = nnx.split(loaded_model)
nnx.display(loaded_model)
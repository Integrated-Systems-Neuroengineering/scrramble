
import jax
import jax.numpy as jnp

class Neuron:
    def __init__(self, std=0.01):
        self.std = std

    @jax.jit
    def state_function(self, x, threshold):
        """
        Implements ternary state updates.
        Threshold is a two-element array.
        """
        theta1, theta2 = threshold
        state = jnp.where(x < theta1, -1, jnp.where(x > theta2, 1, 0))
        return state

    @jax.jit
    def expected_state(self, y_tilde, threshold, std):
        """
        Compute the expected state for the neuron, used during backpropagation.
        """
        return jax.nn.sigmoid((y_tilde - threshold) / std)  # Placeholder, adjust as needed

    @jax.jit
    def generate_gaussian_noise(self, key, size):
        """
        Generate random Gaussian noise with the given size.
        """
        return self.std * jax.random.normal(key, shape=size)


from functools import partial
import jax
import jax.numpy as jnp

class Network:
    def __init__(self, sizes, thresholds, noise_sd):
        self.sizes = sizes
        self.thresholds = thresholds
        self.noise_sd = noise_sd

    def init_network_params(self, key):
        """
        Initialize the weights and biases for all the layers.
        """
        keys = jax.random.split(key, len(self.sizes))
        params = [self.random_layer_params(n_in, n_out, k) for n_in, n_out, k in zip(self.sizes[:-1], self.sizes[1:], keys)]
        return params

    def random_layer_params(self, input_size, output_size, key):
        """
        Initialize the weights and biases for a single layer.
        """
        scale = jnp.sqrt(2 / (input_size + output_size))
        w_key, b_key = jax.random.split(key)
        W = jax.random.normal(w_key, (output_size, input_size)) * scale
        b = jax.random.normal(b_key, (output_size, )) * scale
        return W, b

    def predict(self, params, image, thresholds, key):
        """
        Forward pass through the network.
        """
        key = jax.random.PRNGKey(key)
        activations = image

        for i, (W, b) in enumerate(params[:-1]):  # iterate through all but final layer
            key, noise_key = jax.random.split(key, 2)
            y_tilde = jnp.dot(activations, W.T) + b
            activations = Neuron().expected_state(y_tilde, thresholds, self.noise_sd)
            noise = self.noise_sd * jax.random.normal(noise_key, activations.shape)
            outputs = y_tilde + noise
            activations = Neuron().state_function(outputs, thresholds)

        out_W, out_b = params[-1]
        logits = jnp.dot(activations, out_W.T) + out_b
        return logits

    def batched_predict(self, params, images, thresholds, key):
        return jax.vmap(self.predict, in_axes=(None, 0, None, None))(params, images, thresholds, key)

    def accuracy(self, params, images, targets, thresholds, key):
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(self.batched_predict(params, images, thresholds, key), axis=1)
        return jnp.mean(predicted_class == target_class)

    def cross_entropy_loss(self, params, images, targets, thresholds, key):
        logits = self.batched_predict(params, images, thresholds, key)
        log_softmax = jax.nn.log_softmax(logits)
        return -jnp.sum(log_softmax * targets) / targets.shape[0]

    def update(self, params, x, y, thresholds, key, lr):
        loss_fn = partial(self.cross_entropy_loss, thresholds=thresholds, key=key)
        grads = jax.grad(loss_fn)(params, x, y)
        return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]

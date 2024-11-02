# TriDeNT (Tri-state Stochastic Neural Network)

## Overview
Deep neural networks with discrete activations may be useful for efficient implementation. To that extent, binary and ternary neural networks have been shown to perform well on benchmarking tasks like MNIST [insert some refs]. However, it seems that the optimization techniques either use local learning rules or utilize approximations of gradients in a backpropagation like setting [(Mostafa et al. 2017, Front. Neurosci.)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00496/full). With added noise in the inputs, it might be possible to potentially leverage stochastic resonance to compute gradients. This would be useful for backpropagation and end-to-end training of neural networks with discrete states. In this analysis, we focus on ternary neural networks in which the activations $y \in \{-1, 0, 1\}$

## Structure of the workflow
### Theory of ternary neurons with stochastic inputs and gradient-based learning rule. This is being referred to by __TriDeNT__.
In this iteration, the neurons will have ternary activations. The weights and gradients are designed to be full precision. This network will be used for initial benchmarking on MNIST (and maybe a few more datasets...)

### Discretizing gradients
In this iteration, we would discretize the gradients. The idea is similar to the forward pass where full precision gradients will potnetially serve as a stochastic intensity for the gradient value to sample. Gradients will be either ternary or binary.

### Discretizing weights and gradients
In the final iteration, we can experiment with having discrete weights or at least low precision weights to be trained using discrete gradients.

## Theory for TriDeNT with full precision weights and gradients

Consider a ternary neuron with the ternary activation function.
Let thresholds be: $\theta = \{\theta_-, \theta_+\}$ 
$$
h(y) = \begin{cases}

-1 & \text{if } y \leq \theta_- \\
0  & \text{if } \theta_- < y < \theta_+ \\
+1  & \text{if } y \geq \theta_+ \\
\end{cases}
$$

Consider the noise-free inputs
$$\tilde{y} = \mathbf{W} x + b$$

Let $\xi \sim \mathcal{N}(0, \sigma)$ be the additive noise to this input. We thus define
$$y = \tilde{y} + \xi$$

Note that to determine the sate of a neuron in the network, the __noisy__ version of the input is used as argument to the $h(y)$ function. 

This allows us to define a probability distribution of neuron's state, marginal on the __noise-free__ inputs arriving to it. Let $P(S = s_i | \tilde{y})$ be this probability density function.
Here we assume $S \in \{s_1, s_2, \dots, s_k\}$ to be $k$ possible states. For ternary neurons $S \in \{-1, 0, 1\}$.
Let $\theta \in \{\theta_0, \theta_1, \dots, \theta_k\}$ be the thresholds. Note that $\theta_0$ and $\theta_k$ are assumed to be $\mp \infty$ respectively.

Given a known noise distribution $\xi$ (which we are assuming to be gaussian for now),

$$\begin{align}
P(S = s_i | \tilde{y}) &= P(y \in \{\theta_{i - 1}, \theta_i\}) \\

&= \int_{\theta_{i-1}}^{\theta_i} P(y | \tilde{y}) dt

\end{align}
$$

Let $F(t)$ be the CDF of $P(S | \tilde{y})$, then

$$\int_{\theta_{i-1}}^{\theta_i} P(y | \tilde{y}) dy = F(\theta_i) - F(\theta_{i-1})$$

We can subsequently define expected state conditional on noise-free inputs as

$$\mathbf{E}[(S | \tilde{y})] = \sum_{i = 1}^k s_i (F(\theta_i) - F(\theta_{i-1})$$

For gradient-based optimization, we can take the derivative of the expected with changes in the input as

$$\frac{d}{d\tilde{y}} \mathbf{E}(S | \tilde{y}) = \frac{d}{d\tilde{y}} \sum_{i = 1}^k s_i \int_{\theta_{i-1}}^{\theta_i} P(y | \tilde{y}) dy$$

Which in general case simplifies to
$$\frac{d}{d\tilde{y}} \mathbf{E}(S | \tilde{y}) =  \sum_{i = 1}^k s_i \Big( P(y = \theta_i | \tilde{y}) - P(y = \theta_{i - 1} | \tilde{y})\Big)$$

#### Deriving closed-form expression for ternary case with gaussian noise
(TBD...)
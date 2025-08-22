"""
Fast implementation of ScRRAMBLe routing for ScRRAMBLeCapsLayer.

Created on 08/21/2025
Author: Vikrant Jaltare
"""

import jax
import jax.numpy as jnp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import get_cmap
import seaborn as sns

## ScRRAMBLe with JAX. Construct from the perspective of source slots -> destination slots.
def fast_scrramble(
    num_destination_cores = 10,
    num_source_cores = 20,
    core_size = 256,
    slot_size = 1,
    key = jax.random.key(0),
    proba = 0.2,
    verify_balanced_flag=True,
):
    """
    ScRRAMBLe Routing.
    Logic:
    - Define an empty tensor Cijkl with indices source core ci, source slot si, destination core cj, destination slot sj.
    - For every (ci, si) pick a binomial random sample beta with probability p, n = number of total destination slots.
    - Select b destination slots, randomly.
    - In Cijkl add 1 to the corresponding indices (ci, si, cj, sj).
    - For balancing, simply shuffle the final dimention of Cijkl. (along sj)
    - Add -1 to the new shuffled indices
    """

    slots_per_core = core_size // slot_size

    # initialize the connectivity matrix C
    C = jnp.zeros((num_source_cores, slots_per_core, num_destination_cores, slots_per_core))

    # ----------------------
    # Positive Connections
    # ----------------------

    # pick beta values for connections
    key, subkey = jax.random.split(key)
    betas = jax.random.binomial(key=key, n=num_destination_cores*slots_per_core, p=proba, shape=(num_source_cores, slots_per_core))
    max_beta = jnp.max(betas)
    max_beta = int(max_beta)
    # print(f"Max beta = {max_beta}")

    # create a mask for connections
    mask = jnp.arange(max_beta)[None, None, :] < betas[:, :, None]
    ci_idx, si_idx, beta_idx = jnp.where(mask)

    # generate random destination locations
    key, subkey = jax.random.split(subkey)
    destination_core_indices = jax.random.choice(key=key, a=num_destination_cores, shape=(num_source_cores, slots_per_core, max_beta), replace=True)
    key, subkey = jax.random.split(subkey)
    destination_slot_indices = jax.random.choice(key=key, a=slots_per_core, shape=(num_source_cores, slots_per_core, max_beta), replace=True)

    # find the indices of the destination locations
    cj_idx = destination_core_indices[ci_idx, si_idx, beta_idx]
    # print(cj_idx)
    sj_idx = destination_slot_indices[ci_idx, si_idx, beta_idx]
    # print(sj_idx, sj_idx.shape)

    # add positive connections
    C = C.at[ci_idx, si_idx, cj_idx, sj_idx].add(1)

    # ----------------------
    # Negative connections
    # ----------------------

    # find all positive connections and their weights
    ci_pos, si_pos, cj_pos, sj_pos = jnp.where(C > 0)
    weights = C[ci_pos, si_pos, cj_pos, sj_pos]
    # print(weights, weights.shape)

    # create repeats of connections the same number of times as the weights
    ci_neg_all = jnp.repeat(ci_pos, weights.astype(int))
    si_neg_all = jnp.repeat(si_pos, weights.astype(int))
    cj_neg_all = jnp.repeat(cj_pos, weights.astype(int))

    # total negative connections required
    total_neg_connections = jnp.sum(weights)

    # generate random destination slots for all negative connections
    key, subkey = jax.random.split(key)
    sj_neg_all = jax.random.choice(key=key, a=slots_per_core, shape=(int(total_neg_connections), ), replace=True)

    C = C.at[ci_neg_all, si_neg_all, cj_neg_all, sj_neg_all].add(-1)


    def verify_balanced_connectivity(C):
        """
        Verify if the connectivity matrix makes sense
        - Each core must have equal number of positive and negative connections on an average
        - Returs a distribution of connections per core.
        - Distribution should be a delta zero with ideally no outliers
        """

        assert len(C.shape) == 4, "C must be a 4D tensor"

        balance_list = []

        # check for cores taking inputs
        for co in range(C.shape[2]):
            conn = jnp.sum(C[:, :, co, :])
            balance_list.append(conn)

        perc = jnp.percentile(a=jnp.array(balance_list), q=95)
        if perc == 0:
            pass
        else:
            raise ValueError(f"Connectivity matrix is not balanced. 95th percentile is {perc}")

    if verify_balanced_flag:
        verify_balanced_connectivity(C)


    return C


def __main__():
    # Example usage
    C = fast_scrramble(
        num_destination_cores=10,
        num_source_cores=20,
        core_size=256,
        slot_size=1,
        key=jax.random.key(0),
        proba=0.9,
        verify_balanced_flag=True
    )

    # print(f"Connectivity matrix shape: {C.shape}")
    # print(f"Connectivity matrix: {C}")

    # Visualize the connectivity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(C.reshape((C.shape[0] * C.shape[1], C.shape[2] * C.shape[3])), cmap='coolwarm', annot=False)
    plt.title("ScRRAMBLe Connectivity Matrix")
    plt.xlabel("Destination Slots")
    plt.ylabel("Source Slots")
    plt.show()


if __name__ == "__main__":
    __main__()


    

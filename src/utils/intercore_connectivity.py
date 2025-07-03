import jax
import jax.numpy as jnp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import get_cmap
import seaborn as sns

# ------------------------------------------------------------------
# define a function to return a tensor with connectivity constraints
# ------------------------------------------------------------------

def intercore_connectivity(
    input_cores: int,
    output_cores: int,
    slots_per_core: int,
    avg_slot_connectivity: int,
    key: jax.random.key,
    balanced: bool = True
    ):

    """
    Returns a tensor C with shape (input_cores, total_input_slots, output_cores, total_output_slots).
    C[ci, si, co, so] can be [-2, -1, 0, 1, 2].
    1: connection to positive slots
    -1: connection to negative slots
    0: no connection
    -2: rare but repeated connection to negative slot
    2: rare but repeated connection to positive slot
    """

    # define the constants
    total_input_slots = input_cores * slots_per_core
    total_output_slots = output_cores * slots_per_core

    # define the lists of (core, slot) tuples
    Li = [(ci, si) for ci in range(input_cores) for si in range(slots_per_core)]
    Lo = [(co, so) for co in range(output_cores) for so in range(slots_per_core)]
    assert len(Li) == total_input_slots, "Incorrect number of input slots, check input_cores and slots_per_core values"
    assert len(Lo) == total_output_slots, "Incorrect number of output slots, check output_cores and slots_per_core values"
    
    # print(Li)
    # print(Lo)

    # define the positive and negative mapping dicts
    pos_mapping = defaultdict(list)
    neg_mapping = defaultdict(list)

    # construct the positive mapping
    for (co, so) in Lo:
        
        # pick a random number of slots it can receive input from
        key, subkey = jax.random.split(key)
        beta = jax.random.poisson(key=key, lam=avg_slot_connectivity)

        # pick random slots from Li
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(key=key, a=len(Li), shape=(beta,), replace=True) # allow for repeated connections
        connections = [Li[i] for i in idx]
        if beta > 0:
            pos_mapping[(co, so)] = connections
        else:
            pos_mapping[(co, so)] = []


    # print(f"Positive mapping: {pos_mapping}")

    # construct the negative mapping
    for k, v in pos_mapping.items():
        (co, so) = k
        # check if there are any connections
        if len(v) == 0:
            continue
        else:
            # print(v, len(v))
            for ti in v:
                # print(ti)
                # choose some slots where negative connections can be made
                key, subkey = jax.random.split(key)
                gamma = jax.random.choice(key=key, a=slots_per_core, shape=(1,), replace=True)[0].item()
                neg_mapping[ti].append((co, gamma))

    # print(f"Negative mapping: {neg_mapping}")

    # construct the tensor for positive mappings
    C = jnp.zeros((input_cores, slots_per_core, output_cores, slots_per_core))

    for k, v in pos_mapping.items():
        (co, so) = k

        for ti in v:
            (ci, si) = ti
            C = C.at[ci, si, co, so].add(1)

    # construct the tensor for negative mappings
    for k, v in neg_mapping.items():
        (ci, si) = k

        if len(v) == 0:
            print(f"zero -ve mappings from core {k}")
            continue
        else:
            for to in v:
                (co, so) = to
                C = C.at[ci, si, co, so].add(-1)

    def verify_connectivity(C):
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

    if balanced:
        verify_connectivity(C)

    return C

# ------------------------------------------------------------------
# ScRRAMBLe connectivity with binomial sampling
# ------------------------------------------------------------------
def ScRRAMBLe_routing(
    input_cores: int,
    output_cores: int,
    receptive_fields_per_capsule: int,
    connection_probability: float,
    key: jax.random.key,
    with_replacement: bool = True,
    balanced: bool = True
    ):

    """
    Returns a tensor C with shape (input_cores, total_input_slots, output_cores, total_output_slots).
    C[ci, si, co, so] can be [-2, -1, 0, 1, 2].
    1: connection to positive slots
    -1: connection to negative slots
    0: no connection
    -2: rare but repeated connection to negative slot
    2: rare but repeated connection to positive slot

    """

    # define the constants
    total_input_slots = input_cores * receptive_fields_per_capsule
    total_output_slots = output_cores * receptive_fields_per_capsule

    # define the lists of (core, slot) tuples
    Li = [(ci, si) for ci in range(input_cores) for si in range(receptive_fields_per_capsule)]
    Lo = [(co, so) for co in range(output_cores) for so in range(receptive_fields_per_capsule)]
    assert len(Li) == total_input_slots, "Incorrect number of input slots, check input_cores and slots_per_core values"
    assert len(Lo) == total_output_slots, "Incorrect number of output slots, check output_cores and slots_per_core values"

    # find the number of available receptive fields that can SEND their outputs
    num_receptive_fields = input_cores*receptive_fields_per_capsule
    
    
    # print(Li)
    # print(Lo)

    # define the positive and negative mapping dicts
    pos_mapping = defaultdict(list)
    neg_mapping = defaultdict(list)

    # construct the positive mapping
    for (co, so) in Lo:
        # pick a random number of slots/receptive fields it can receive input from
        key, subkey = jax.random.split(key)
        # beta = jax.random.poisson(key=key, lam=avg_slot_connectivity)
        beta = int(jax.random.binomial(key=key, n=num_receptive_fields, p=connection_probability))

        # pick random slots from Li
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(key=key, a=len(Li), shape=(beta,), replace=with_replacement) # allow/don't allow for repeated connections
        connections = [Li[i] for i in idx]
        if beta > 0:
            pos_mapping[(co, so)] = connections
        else:
            pos_mapping[(co, so)] = []


    # print(f"Positive mapping: {pos_mapping}")

    # construct the negative mapping
    for k, v in pos_mapping.items():
        (co, so) = k
        # check if there are any connections
        if len(v) == 0:
            continue
        else:
            # print(v, len(v))
            for ti in v:
                # print(ti)
                # choose some slots where negative connections can be made
                key, subkey = jax.random.split(key)
                gamma = jax.random.choice(key=key, a=receptive_fields_per_capsule, shape=(1,), replace=with_replacement)[0].item()
                neg_mapping[ti].append((co, gamma))

    # print(f"Negative mapping: {neg_mapping}")

    # construct the tensor for positive mappings
    C = jnp.zeros((input_cores, receptive_fields_per_capsule, output_cores, receptive_fields_per_capsule))

    for k, v in pos_mapping.items():
        (co, so) = k

        for ti in v:
            (ci, si) = ti
            C = C.at[ci, si, co, so].add(1)

    # construct the tensor for negative mappings
    for k, v in neg_mapping.items():
        (ci, si) = k

        if len(v) == 0:
            print(f"zero -ve mappings from core {k}")
            continue
        else:
            for to in v:
                (co, so) = to
                C = C.at[ci, si, co, so].add(-1)

    def verify_connectivity(C):
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

    if balanced:
        verify_connectivity(C)

    return C

# --------------------------------------------------------
# visualization function
# --------------------------------------------------------
def plot_connectivity_matrix(C, title=r"$C_{ijkl}$", figsize=(5, 5), dpi=110):
    """
    Plot a connectivity matrix with discrete colormap for values [-2, -1, 0, 1, 2].
    
    Parameters:
    -----------
    C : numpy.ndarray
        The connectivity matrix to plot
    title : str, optional
        Title for the plot, default is "$C_{ijkl}$"
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        Figure resolution
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    
    # Reshape the input if needed
    if len(C.shape) > 2:
        # Assuming C has shape like (cores_i, slots_i, cores_j, slots_j)
        rows = C.shape[0] * C.shape[1]
        cols = C.shape[2] * C.shape[3]
        data = C.reshape(rows, cols)
    else:
        data = C
    
    # Define the discrete values and boundaries
    values = np.array([-2, -1, 0, 1, 2])
    bounds = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    
    # Create a colormap with 5 colors from the coolwarm colormap
    base_cmap = plt.colormaps['coolwarm']
    colors = base_cmap(np.linspace(0, 1, 5))
    cmap = ListedColormap(colors)
    
    # Create a norm to map the data values to colormap indices
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title)
    im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')
    
    # Add a colorbar with no outline and no tick marks
    cbar = plt.colorbar(im, ticks=values, shrink=0.8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(which='both', length=0, width=0, pad=3)
    cbar.ax.minorticks_off()
    
    # Remove plot spines
    sns.despine(fig, top=True, right=True, left=True, bottom=True)
    
    return fig, ax




def __main__():
    key = jax.random.key(4)
    key, subkey = jax.random.split(key)
    test_run = intercore_connectivity(input_cores = 2, output_cores = 2, slots_per_core = 4, avg_slot_connectivity=4, key=key)
    print(test_run)

    fig, ax = plot_connectivity_matrix(test_run, title=r"$C_{ijkl}$", figsize=(5, 5), dpi=110)
    plt.show()

# if __name__ == "__main__":
#     __main__()

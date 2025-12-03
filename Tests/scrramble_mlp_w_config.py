"""
TODO: Come back to this script if MLP Mixer idea works well. It provides a cleaner code!
"""

# -------------------------------------------------------------------
# Base Configuration Class
# -------------------------------------------------------------------
class ScRRAMBLeConfig:
    """
    Shared configuration for ScRRAMBLe components.
    Not an nnx.Module, just a config container.
    """
    def __init__(
        self,
        patch_size: tuple[int, int],
        capsule_size: int = 256,
        slot_size: int = 64,
        input_shape: tuple[int, int, int] = (32, 32, 3),
    ):
        self.patch_size = patch_size
        self.capsule_size = capsule_size
        self.slot_size = slot_size
        self.input_shape = input_shape
        
        # Derived quantities
        self.num_patches_h = input_shape[0] // patch_size[0]
        self.num_patches_w = input_shape[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.rf_size = patch_size[0] * patch_size[1]
        self.slots_per_core = capsule_size // slot_size


# -------------------------------------------------------------------
# Base ScRRAMBLe Module (with shared config)
# -------------------------------------------------------------------
class ScRRAMBLeBase(nnx.Module):
    """
    Base class for ScRRAMBLe modules that share configuration.
    """
    def __init__(self, config: ScRRAMBLeConfig, rngs: nnx.Rngs):
        self.config = config
        self.rngs = rngs
        
        # Store commonly used values as attributes
        self.patch_size = config.patch_size
        self.capsule_size = config.capsule_size
        self.slot_size = config.slot_size
        self.slots_per_core = config.slots_per_core
        self.num_patches_h = config.num_patches_h
        self.num_patches_w = config.num_patches_w
        self.rf_size = config.rf_size


# -------------------------------------------------------------------
# ScRRAMBLe Patching (inherits from base)
# -------------------------------------------------------------------
class ScRRAMBLePatching(ScRRAMBLeBase):
    """
    Patching the input and routing it to the cores.
    Inherits common configuration from ScRRAMBLeBase.
    """
    def __init__(
        self,
        config: ScRRAMBLeConfig,
        rngs: nnx.Rngs,
        output_cores: int,
        connection_proba: float,
    ):
        super().__init__(config, rngs)
        
        self.output_cores = output_cores
        self.connection_proba = connection_proba
        
        # Compute effective input cores
        self.rfs_per_capsule = self.capsule_size // self.rf_size
        self.num_cores = (
            config.input_shape[2] * 
            self.num_patches_h * 
            self.num_patches_w // 
            self.rfs_per_capsule
        )
        
        # Define ScRRAMBLe routing
        C = fast_scrramble(
            num_destination_cores=output_cores,
            num_source_cores=self.num_cores,
            core_size=self.capsule_size,
            slot_size=self.rf_size,
            key=rngs.params(),
            proba=connection_proba,
        )
        self.C = nnx.Variable(C)
    
    def __call__(self, x):
        # Rearrange the input
        x = rearrange(
            x, 
            'b (p1 h1) (p2 w2) c -> b p1 p2 h1 w2 c',
            p1=self.num_patches_h,
            p2=self.num_patches_w
        )
        x = rearrange(
            x,
            'b p1 p2 h1 w2 c -> b h1 w2 (p1 p2 c)',
            p1=self.num_patches_h,
            p2=self.num_patches_w
        )
        x = rearrange(
            x,
            'b h1 w2 (ci si) -> b ci si h1 w2',
            ci=self.num_cores,
            si=self.rfs_per_capsule
        )
        x = jnp.einsum('bcshw, ijkl -> bklhw', x, self.C.value)
        return x


# -------------------------------------------------------------------
# ScRRAMBLe MLP Mixer Block (inherits from base)
# -------------------------------------------------------------------
class ScRRAMBLeMLPMixerBlock(ScRRAMBLeBase):
    """
    ScRRAMBLe MLP Mixer Block. Stackable module.
    Inherits common configuration from ScRRAMBLeBase.
    """
    def __init__(
        self,
        config: ScRRAMBLeConfig,
        rngs: nnx.Rngs,
        input_capsules: int,
        output_capsules: int,
        connection_probability: float,
        activation_fn: Callable = nnx.gelu,
    ):
        super().__init__(config, rngs)
        
        self.input_capsules = input_capsules
        self.output_capsules = output_capsules
        self.activation_fn = activation_fn
        
        # Define weights of cores (input cores)
        initializer = initializers.glorot_normal()
        self.Wi = nnx.Param(
            initializer(
                rngs.params(),
                (input_capsules, self.slot_size, self.slot_size, 
                 self.slots_per_core, self.slots_per_core)
            )
        )
        
        # Define layer norm
        self.layer_norm = nnx.LayerNorm(
            input_capsules * self.capsule_size,
            rngs=rngs
        )
        
        # Define routing to next layer (output cores)
        Co = fast_scrramble(
            num_destination_cores=output_capsules,
            num_source_cores=input_capsules,
            core_size=self.capsule_size,
            slot_size=self.capsule_size // self.slots_per_core,
            key=rngs.params(),
            proba=connection_probability,
        )
        self.Co = nnx.Variable(Co)
    
    def __call__(self, x):
        # Apply weights
        x = rearrange(x, 'ci p1 h w -> ci p1 (h w)')
        x = jnp.einsum('ijklm, imk -> ilj', self.Wi.value, x)
        x = self.activation_fn(x)
        
        # Layer norm
        x_shape = x.shape
        x = x.reshape(-1)
        x = self.layer_norm(x)
        x = x.reshape(x_shape)
        
        # Route to next layer
        x = jnp.einsum('ijkl, ijm -> klm', self.Co.value, x)
        x = rearrange(
            x,
            'ci p1 (h w) -> ci p1 h w',
            h=self.patch_size[0],
            w=self.patch_size[1]
        )
        return x


# -------------------------------------------------------------------
# ScRRAMBLe MLP Mixer Network
# -------------------------------------------------------------------
class ScRRAMBLeMLPMixerNetwork(nnx.Module):
    """
    Full ScRRAMBLe MLP Mixer network.
    Chains patching + multiple mixer blocks.
    """
    def __init__(
        self,
        config: ScRRAMBLeConfig,
        rngs: nnx.Rngs,
        num_blocks: int,
        capsule_sizes: list[int],  # e.g., [20, 15, 10, 10]
        connection_probabilities: list[float],
        activation_fn: Callable = nnx.gelu,
    ):
        self.config = config
        self.num_blocks = num_blocks
        
        # Patching layer
        self.patching = ScRRAMBLePatching(
            config=config,
            rngs=rngs,
            output_cores=capsule_sizes[0],
            connection_proba=connection_probabilities[0]
        )
        
        # Stack of mixer blocks
        self.mixer_blocks = nnx.List([
            ScRRAMBLeMLPMixerBlock(
                config=config,
                rngs=rngs,
                input_capsules=capsule_sizes[i],
                output_capsules=capsule_sizes[i+1],
                connection_probability=connection_probabilities[i+1],
                activation_fn=activation_fn,
            )
            for i in range(num_blocks)
        ])
    
    def __call__(self, x):
        # Apply patching
        x = self.patching(x)
        
        # Apply mixer blocks with vmap
        for block in self.mixer_blocks:
            x = jax.vmap(block, in_axes=(0,))(x)
        
        return x


# -------------------------------------------------------------------
# Usage Example
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Create shared configuration
    config = ScRRAMBLeConfig(
        patch_size=(8, 8),
        capsule_size=256,
        slot_size=64,
        input_shape=(32, 32, 3)
    )
    
    print(f"Config summary:")
    print(f"  Patches: {config.num_patches_h}x{config.num_patches_w} = {config.num_patches}")
    print(f"  Capsule size: {config.capsule_size}")
    print(f"  Slots per core: {config.slots_per_core}")
    
    # Create network
    rngs = nnx.Rngs(0)
    network = ScRRAMBLeMLPMixerNetwork(
        config=config,
        rngs=rngs,
        num_blocks=2,
        capsule_sizes=[20, 10, 10],  # patching->20, block1->10, block2->10
        connection_probabilities=[0.2, 0.2, 0.2],
        activation_fn=nnx.gelu,
    )
    
    # Test
    x = jnp.ones((10, 32, 32, 3))
    print(f"\nInput shape: {x.shape}")
    
    y = network(x)
    print(f"Output shape: {y.shape}")
    
    # Also test individual components
    print("\n--- Testing individual components ---")
    patching = ScRRAMBLePatching(config, rngs, output_cores=20, connection_proba=0.2)
    y_patch = patching(x)
    print(f"After patching: {y_patch.shape}")
    
    mixer = ScRRAMBLeMLPMixerBlock(
        config, rngs, input_capsules=20, output_capsules=10, connection_probability=0.2
    )
    y_mixer = jax.vmap(mixer, in_axes=(0,))(y_patch)
    print(f"After mixer block: {y_mixer.shape}")
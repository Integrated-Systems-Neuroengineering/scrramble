## using the block module
import jax
from flax import nnx

class Block(nnx.Module):
    """Base class for accelerator circuit blocks."""
    
    def get_latency(self) -> float:
        """Returns the latency of this block (in seconds)."""
        raise NotImplementedError("Must implement a calculation for latency of"
                                  " this block.")

    def get_operations(self) -> int:
        """Returns the number of operations performed by this block (in ops)."""
        raise NotImplementedError("Must implement a calculation for number of"
                                  " operations performed by this block.")
    
    def get_energy(self) -> float:
        """Returns the energy consumption of this block (in Joules)."""
        raise NotImplementedError("Must implement a calculation for energy"
                                  " consumption of this block.")

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the block (to be implemented by subclasses).
        
            note: Block is expected to reshape its own input to the correct shape.
                e.g.: x = x.reshape(self.input_shape)
        """
        raise NotImplementedError("Must implement a forward pass for this block.")
    
    def __getitem__(self, key: str):
        """Syntactic sugar to call the latency, operations, or energy methods
            useful for traversing the connection tree for various calculations.
        """
        if key == "latency":
            return self.get_latency()
        elif key == "operations" or key == "ops":
            return self.get_operations()
        elif key == "energy":
            return self.get_energy()
        else:
            raise KeyError(f"Indexing key {key} not supported.")
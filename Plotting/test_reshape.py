import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

def chunkwise_reshape(x):
    """
    Reshape the input in a block-wise manner with 4x4 chunks
    Args:
        x: jax.Array, input data (image)
    Returns:
        x: jax.Array, flattened data
    """
    # reshape into 32x32
    x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')

    # reshape into 8x8 grid of 4x4 chunks
    x = jnp.reshape(x, (x.shape[0], 8, 8, 4, 4, 1))

    # flatten each 4x4 chunk
    x = jnp.reshape(x, (x.shape[0], 8, 8, -1))  # -1 = 4*4*1 = 16

    # flatten all chunks together
    x = jnp.reshape(x, (x.shape[0], -1))  # -1 = 8*8*16 = 1024

    return x

def vanilla_flatten(x):
    """Standard flattening - just concatenate all pixels row by row"""
    # reshape into 32x32 first (to match chunkwise)
    x = jax.image.resize(x, (x.shape[0], 32, 32, 1), method='nearest')
    return jnp.reshape(x, (x.shape[0], -1))

def create_test_image():
    """Create a test image with a clear pattern to see the difference"""
    # Create a 32x32 image with different patterns in each quadrant
    img = np.zeros((32, 32, 1))
    
    # Quadrant 1: Horizontal stripes
    img[0:16, 0:16, 0] = np.tile([1, 0], (16, 8))
    
    # Quadrant 2: Vertical stripes  
    img[0:16, 16:32, 0] = np.tile([[1], [0]], (8, 16))
    
    # Quadrant 3: Checkerboard
    for i in range(16, 32):
        for j in range(0, 16):
            img[i, j, 0] = (i + j) % 2
    
    # Quadrant 4: Gradient
    img[16:32, 16:32, 0] = np.linspace(0, 1, 16).reshape(1, -1)
    
    return img

def visualize_chunk_ordering(flattened_data, title):
    """Visualize how the flattened data would look if reshaped back to 32x32"""
    # Reshape flattened data back to 32x32 to see the ordering
    reshaped = flattened_data.reshape(32, 32)
    return reshaped

# Create test image
test_img = create_test_image()
test_batch = jnp.array([test_img])  # Add batch dimension

# Apply both flattening methods
chunkwise_flat = chunkwise_reshape(test_batch)
vanilla_flat = vanilla_flatten(test_batch)

# Convert back to numpy for visualization
chunkwise_flat_np = np.array(chunkwise_flat[0])  # Remove batch dimension
vanilla_flat_np = np.array(vanilla_flat[0])

print(f"Original image shape: {test_img.shape}")
print(f"Chunkwise flattened shape: {chunkwise_flat_np.shape}")
print(f"Vanilla flattened shape: {vanilla_flat_np.shape}")
print(f"Are they the same? {np.array_equal(chunkwise_flat_np, vanilla_flat_np)}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original image
axes[0, 0].imshow(test_img[:,:,0], cmap='gray')
axes[0, 0].set_title('Original Test Image\n(Different patterns in each quadrant)')
axes[0, 0].set_xlabel('32x32 pixels')

# Chunkwise flattened data (first 100 elements)
axes[0, 1].plot(chunkwise_flat_np[:100])
axes[0, 1].set_title('Chunkwise Flattened (first 100 elements)')
axes[0, 1].set_xlabel('Element index')
axes[0, 1].set_ylabel('Pixel value')

# Vanilla flattened data (first 100 elements)
axes[0, 2].plot(vanilla_flat_np[:100])
axes[0, 2].set_title('Vanilla Flattened (first 100 elements)')
axes[0, 2].set_xlabel('Element index')
axes[0, 2].set_ylabel('Pixel value')

# Show the ordering difference more clearly
# Reshape flattened data back to 32x32 to see how pixels are ordered
chunkwise_reordered = visualize_chunk_ordering(chunkwise_flat_np, "Chunkwise")
vanilla_reordered = visualize_chunk_ordering(vanilla_flat_np, "Vanilla")

axes[1, 0].imshow(chunkwise_reordered, cmap='gray')
axes[1, 0].set_title('Chunkwise: How pixels are ordered\n(reshaped back to 32x32)')

axes[1, 1].imshow(vanilla_reordered, cmap='gray')
axes[1, 1].set_title('Vanilla: How pixels are ordered\n(reshaped back to 32x32)')

# Difference between the two approaches
difference = np.abs(chunkwise_flat_np - vanilla_flat_np)
axes[1, 2].plot(difference)
axes[1, 2].set_title('Absolute Difference\n|Chunkwise - Vanilla|')
axes[1, 2].set_xlabel('Element index')
axes[1, 2].set_ylabel('Absolute difference')

plt.tight_layout()
plt.show()

# Print some statistics
print(f"\nMax difference between methods: {difference.max():.6f}")
print(f"Mean difference: {difference.mean():.6f}")
print(f"Number of different elements: {np.sum(difference > 1e-10)}")

# Show a small example of how 4x4 chunks are processed
print("\n" + "="*50)
print("EXAMPLE: How chunkwise processing works")
print("="*50)

# Take a small 8x8 example to show chunk processing
small_example = np.arange(64).reshape(8, 8)
print("Small 8x8 example (values 0-63):")
print(small_example)

# Simulate the chunk processing on this small example
reshaped_small = small_example.reshape(2, 2, 4, 4)  # 2x2 grid of 4x4 chunks
print(f"\nAfter reshaping to (2, 2, 4, 4):")
print("Chunk [0,0]:")
print(reshaped_small[0, 0])
print("Chunk [0,1]:")
print(reshaped_small[0, 1])
print("Chunk [1,0]:")
print(reshaped_small[1, 0])
print("Chunk [1,1]:")
print(reshaped_small[1, 1])

# Show flattening order
chunk_flattened = reshaped_small.reshape(2, 2, -1)
final_flattened = chunk_flattened.reshape(-1)
print(f"\nChunkwise flattened order (first 20 elements):")
print(final_flattened[:20])

vanilla_small = small_example.reshape(-1)
print(f"Vanilla flattened order (first 20 elements):")
print(vanilla_small[:20])
import numpy as np

def create_dummy_npz(output_path, num_images=10, image_size=(64, 64, 3)):
    """
    Create a dummy .npz file with random pixel values.
    """
    images = np.random.uniform(-1, 1, (num_images, *image_size)).astype(np.float32)
    np.savez(output_path, arr_0=images)
    print(f"Dummy .npz file saved to {output_path}")

# Example usage
create_dummy_npz("dummy_data.npz")

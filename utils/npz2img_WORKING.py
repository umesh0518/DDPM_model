import numpy as np
from PIL import Image
import os

def npz_to_png(npz_file, output_folder):
    # Load .npz file
    data = np.load(npz_file)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through arrays in .npz
    for key, array in data.items():
        # Check if the array has 4 dimensions
        if array.ndim == 4:
            # Iterate over the first dimension (e.g., 3 images)
            for i, sub_array in enumerate(array):
                # Debug: Check the range of values
                print(f"{key}_img_{i}: Min = {sub_array.min()}, Max = {sub_array.max()}")

                # Normalize only if the range is not [0, 255]
                if sub_array.max() > 255 or sub_array.min() < 0:
                    normalized_array = (255 * (sub_array - sub_array.min()) / (sub_array.max() - sub_array.min())).astype(np.uint8)
                else:
                    normalized_array = sub_array.astype(np.uint8)

                # Convert to Image
                image = Image.fromarray(normalized_array)

                # Save each image with a unique name
                output_path = os.path.join(output_folder, f"{key}_img_{i}.png")
                image.save(output_path)
                print(f"Saved {key}_img_{i} as {output_path}")
        else:
            print(f"Skipping {key}: Unsupported shape {array.shape}")



# Example usage
# npz_file =  r"c:\Users\uyadav\Desktop\openaimodel\openai_samples_cosine_learnedsigma1000600x64x64x3.npz"  # Replace with your .npz file
# output_folder = "output_images_nov29_cosine"  # Replace with your desired output folder
# npz_to_png(npz_file, output_folder)


# Example usage
npz_file =  r"c:\Users\uyadav\Desktop\openaimodel\openai_samples_linear1000600x64x64x3.npz"  # Replace with your .npz file
output_folder = "output_images_nov29_linear"  # Replace with your desired output folder
npz_to_png(npz_file, output_folder)

import os
import random
from PIL import Image
import torch
import lpips
import numpy as np

# Function to calculate Diversity Score (DS) using LPIPS
def calculate_diversity_score(fake_dir, num_samples=50):
    model = lpips.LPIPS(net='alex')  # Use AlexNet backbone for perceptual similarity
    print("Calculating Diversity Score...")

    # Load all image paths in the generated (fake) directory
    image_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total images available: {len(image_paths)}")

    # Check if there are enough images to sample
    if len(image_paths) < 2:
        print("Not enough images to calculate Diversity Score. At least 2 images are required.")
        return None

    # Randomly sample 50 images (or fewer if less than 50 images are available)
    sampled_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    print(f"Number of images sampled: {len(sampled_paths)}")

    # Load and preprocess sampled images
    images = [Image.open(path).convert('RGB') for path in sampled_paths]
    images = [img.resize((256, 256), Image.NEAREST) for img in images]  # Resize to 256x256
    tensors = [torch.tensor(np.array(img).transpose(2, 0, 1)).float() / 255.0 for img in images]
    tensors = [t.unsqueeze(0) for t in tensors]  # Add batch dimension

    # Calculate pairwise LPIPS distances
    distances = []
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            dist = model(tensors[i], tensors[j]).item()
            distances.append(dist)

    # Check if distances were calculated
    if len(distances) == 0:
        print("No distances could be calculated. Check the input images.")
        return None

    # Compute diversity score as the average perceptual distance
    diversity_score = sum(distances) / len(distances)
    print(f"Diversity Score (DS): {diversity_score}")
    return diversity_score


# Directories for fake images
resized_fake_dir_cosine = "resized_fake_images_cosine"

print("Cosine:")
# Calculate Diversity Score for cosine with 50 random samples
diversity_score_value = calculate_diversity_score(resized_fake_dir_cosine, num_samples=50)

resized_fake_dir_linear = "resized_fake_images_linear"

# Uncomment below to calculate for linear images
# print("Linear:")
# diversity_score_value = calculate_diversity_score(resized_fake_dir_linear, num_samples=50)

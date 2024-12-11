import os
from PIL import Image
import torch
import lpips
import numpy as np

# Function to resize images
def resize_images(input_dir, output_dir, target_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.NEAREST)
            img.save(os.path.join(output_dir, filename))

# Function to calculate Diversity Score (DS) using LPIPS
def calculate_diversity_score(fake_dir):
    model = lpips.LPIPS(net='alex')  # Use AlexNet backbone for perceptual similarity
    print("Calculating Diversity Score...")

    # Load all image paths in the generated (fake) directory
    image_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total images available: {len(image_paths)}")

    # Check if there are enough images to process
    if len(image_paths) < 2:
        print("Not enough images to calculate Diversity Score. At least 2 images are required.")
        return None

    # Load and preprocess all images
    images = [Image.open(path).convert('RGB') for path in image_paths]
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
resized_fake_dir_cosine_only = r"output_images_dec7_cosineonly_learned_signma_false"
resized_fake_dir_cosine_learned_sigma = r"output_images_nov29_cosine_learnedsigma"
resized_fake_dir_linear_only = r"output_images_nov29_linear"

# Calculate Diversity Scores for all images
print("Cosine only(All Images):")
calculate_diversity_score(resized_fake_dir_cosine_only)

print("Cosine and learn sigma(All Images):")
calculate_diversity_score(resized_fake_dir_cosine_learned_sigma)

print("Linear only (All Images):")
calculate_diversity_score(resized_fake_dir_linear_only)

# print("Linear (All Images):")
# calculate_diversity_score(resized_fake_dir_linear_only)

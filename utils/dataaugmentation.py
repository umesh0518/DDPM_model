import os
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa

# Directory paths
input_dir = "DDPM_model/melanoma_resized64"
output_dir = "AugmentedData/64size"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define augmenters to apply individually to each image
augmenters = [
    iaa.Fliplr(1.0),  # Horizontal flip
    iaa.Flipud(1.0),  # Vertical flip
    iaa.Affine(rotate=(-15, 15)),  # Small rotation between -15 and 15 degrees
    iaa.Affine(scale=(0.9, 1.1)),  # Small scaling between 90% and 110%
    iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),  # Mild Gaussian noise
    iaa.GaussianBlur(sigma=(0, 0.5)),  # Mild Gaussian blur
    iaa.Multiply((0.9, 1.1)),  # Adjust brightness by a small factor
    iaa.LinearContrast((0.9, 1.1)),  # Mild contrast adjustment
    iaa.Crop(percent=(0, 0.1)),  # Random cropping up to 10%
    iaa.PerspectiveTransform(scale=(0.01, 0.05)),  # Small perspective change
]

def apply_augmenters_separately(image, augmenters):
    """Apply each augmenter individually to the input image."""
    augmented_images = []
    for augmenter in augmenters:
        augmented_image = augmenter(image=image)
        augmented_images.append(augmented_image)
    return augmented_images

def save_separate_augmentations(image, base_filename, output_dir, augmenters):
    """Save each individually augmented image."""
    augmented_images = apply_augmenters_separately(image, augmenters)
    for idx, aug_image in enumerate(augmented_images):
        aug_image_pil = Image.fromarray(aug_image)
        aug_image_pil.save(os.path.join(output_dir, f"{base_filename}_aug{idx}.jpg"))

# Process each image in the dataset and apply individual augmentations
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)
        image = np.array(image)  # Convert to numpy for imgaug

        # Generate and save augmentations for each original image
        base_filename = os.path.splitext(filename)[0]
        save_separate_augmentations(image, base_filename, output_dir, augmenters)

print("Data augmentation complete. Check the output directory for augmented images.")

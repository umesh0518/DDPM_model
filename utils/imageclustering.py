import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Function to load images in a fixed order or match similarity
from skimage.metrics import structural_similarity as ssim

# Function to load random samples from a folder
def load_random_images(folder, num_samples=25):
    image_files = os.listdir(folder)
    sampled_files = random.sample(image_files, min(len(image_files), num_samples))
    images = [Image.open(os.path.join(folder, file)) for file in sampled_files]
    return images

# Function to find similar images based on SSIM
def find_similar_images(base_folder, comparison_folders, num_samples=5):
    image_files = sorted(os.listdir(base_folder))
    sampled_files = random.sample(image_files, min(len(image_files), num_samples))
    base_image_paths = [os.path.join(base_folder, img) for img in sampled_files]

    similar_images = []
    for folder in comparison_folders:
        folder_images = sorted(os.listdir(folder))
        matched_images = []

        for base_img_path in base_image_paths:
            base_img = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE)
            best_match = None
            best_score = -1

            for file in folder_images:
                comp_img_path = os.path.join(folder, file)
                comp_img = cv2.imread(comp_img_path, cv2.IMREAD_GRAYSCALE)

                if base_img.shape == comp_img.shape:
                    score, _ = ssim(base_img, comp_img, full=True)
                    if score > best_score:
                        best_score = score
                        best_match = comp_img_path

            matched_images.append(best_match)

        similar_images.append(matched_images)

    return base_image_paths, similar_images

# 1. Representative Image Grid (Collage Format)
# def create_image_grid(images, title="Image Grid", grid_size=(5, 5)):
#     fig, axes = plt.subplots(*grid_size, figsize=(10, 10))
#     fig.suptitle(title, fontsize=16)
#     for ax, img in zip(axes.ravel(), images):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.show()

# 3. Distribution-Based Comparison (Histograms of Mean Pixel Intensity)
# def compare_image_distributions(folders):
#     plt.figure(figsize=(10, 5))
#     for folder in folders:
#         mean_intensities = []
#         for file in os.listdir(folder):
#             img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
#             mean_intensities.append(np.mean(img))
#         plt.hist(mean_intensities, bins=30, alpha=0.5, label=f"{os.path.basename(folder)}")
#     plt.title("Image Mean Pixel Intensity Distribution")
#     plt.xlabel("Mean Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()

# 4. Side-by-Side Comparison of Images with Matched Images
def side_by_side_comparison_with_matches(base_folder, comparison_folders, num_samples=5, model_labels=None):
    if model_labels is None:
        model_labels = [f"Model {i+1}" for i in range(len(comparison_folders))]

    base_image_paths, matched_images = find_similar_images(base_folder, comparison_folders, num_samples)
    fig, axes = plt.subplots(num_samples, len(comparison_folders) + 1, figsize=(12, num_samples * 2.5), gridspec_kw={'wspace': 0.05})

    for i in range(num_samples):
        # Plot base images
        base_img = Image.open(base_image_paths[i])
        axes[i, 0].imshow(base_img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title(model_labels[0], fontsize=12)  # Updated to use first model label

        # Plot matched images
        for j, folder_images in enumerate(matched_images):
            matched_img = Image.open(folder_images[i])
            axes[i, j + 1].imshow(matched_img)
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(model_labels[j + 1], fontsize=12)  # Updated for remaining model labels

    plt.show()

# Example Usage
folders = ['resized_fake_images_linear_only', 'resized_fake_images_cosine_only', 'resized_fake_images_cosine_leanedsigna_true']
model_labels = ['Linear Noise Scheduler', 'Cosine Noise Scheduler', 'Cosine with Learned Sigma True']
# create_image_grid(load_random_images(folders[0], 25), title="Model 1 Samples")
# compare_image_distributions(folders)
side_by_side_comparison_with_matches(folders[0], folders[1:], num_samples=5, model_labels=model_labels)

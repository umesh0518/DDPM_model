import os
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load ResNet model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
model.eval()

def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).squeeze().numpy()
    return features

# Load all images from folders
def load_images_from_folder(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Paths to image folders
real_folder = r"C:\Users\uyadav\Downloads\ddpm\improved-diffusion\test_images"
fake_folder = r"C:\Users\uyadav\Downloads\ddpm\utils\output_images_nov29_cosine"

# Get image paths
real_image_paths = load_images_from_folder(real_folder)
fake_image_paths = load_images_from_folder(fake_folder)

# Extract features
print("Extracting features for real images...")
real_features = np.array([extract_features(path) for path in real_image_paths])
print("Extracting features for fake images...")
generated_features = np.array([extract_features(path) for path in fake_image_paths])

# Combine features
features = np.concatenate([real_features, generated_features])
labels = np.array([0] * len(real_features) + [1] * len(generated_features))

# Debugging features
print("Checking for invalid values in features...")
if np.isnan(features).any():
    print("NaN values detected in features!")
if np.isinf(features).any():
    print("Inf values detected in features!")
if np.std(features, axis=0).sum() == 0:
    print("Zero variance detected across features!")

# Clean features
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# Dynamically set n_components for PCA preprocessing
n_components = min(features.shape[0], features.shape[1])  # min(n_samples, n_features)
pca_preprocess = PCA(n_components=n_components - 1)  # Use one less component to avoid issues
features_reduced = pca_preprocess.fit_transform(features)

# Apply PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_reduced)

# Adjust t-SNE perplexity for datasets
perplexity = min(30, features_reduced.shape[0] - 1)  # Perplexity < n_samples
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
tsne_result = tsne.fit_transform(features_reduced)

# Visualize t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], label='Real', alpha=0.7, marker='o')
plt.scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], label='Generated', alpha=0.7, marker='^')
plt.legend()
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

# Visualize PCA
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], label='Real', alpha=0.7, marker='o')
plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], label='Generated', alpha=0.7, marker='^')
plt.legend()
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

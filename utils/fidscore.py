import os
from PIL import Image
import torch
from pytorch_fid import fid_score
import lpips
import numpy as np

# Function to calculate FID Score
def calculate_fid(real_dir, fake_dir):
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048,
        num_workers=0  # Disable multiprocessing
    )
    print(f"FID Score: {fid_value}")
    return fid_value

# Function to resize images to 299x299 (required for FID and diversity calculations)
def resize_images(input_dir, output_dir, target_size=(299, 299)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img.save(os.path.join(output_dir, filename))

# Function to calculate Diversity Score (DS) using LPIPS
def calculate_diversity_score(fake_dir):
    model = lpips.LPIPS(net='alex')  # Use AlexNet backbone for perceptual similarity
    print("Calculating Diversity Score...")

    # Load all images in the generated (fake) directory
    image_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [Image.open(path).convert('RGB') for path in image_paths]

    # Debugging: Check number of images loaded
    print(f"Number of images loaded for diversity calculation: {len(images)}")

    # Check if there are enough images to calculate diversity
    if len(images) < 2:
        print("Not enough images to calculate Diversity Score. At least 2 images are required.")
        return None

    # Resize all images to the same size (optional, avoid interpolation if identical)
    images = [img.resize((256, 256), Image.NEAREST) for img in images]

    # Convert images to tensors and normalize
    tensors = [torch.tensor(np.array(img).transpose(2, 0, 1)).float() / 255.0 for img in images]
    tensors = [t.unsqueeze(0) for t in tensors]  # Add batch dimension

    # Calculate pairwise LPIPS distances
    distances = []
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            # Debugging: Check if tensors are identical
            if torch.equal(tensors[i], tensors[j]):
                print(f"Images {i} and {j} are identical.")
                distances.append(0.0)  # Append 0 for identical images
            else:
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



# Directories for real and generated images
# real_dir = r"C:\Users\Nitro\Downloads\IDDPM_model\DDPM_model\fake_images"  # real image directory
# fake_dir = r"C:\Users\Nitro\Downloads\IDDPM_model\DDPM_model\real_images"  # generated image directory

real_dir = r"C:\Users\uyadav\Downloads\ddpm\utils\output_images"  # real image directory
fake_dir = r"C:\Users\uyadav\Downloads\ddpm\utils\realimages"

# Temporary directories for resized images
resized_real_dir = "resized_real_images"
resized_fake_dir = "resized_fake_images"

# Resize images for FID calculation
resize_images(real_dir, resized_real_dir)
resize_images(fake_dir, resized_fake_dir)

# Calculate FID Score
fid_score_value = calculate_fid(resized_real_dir, resized_fake_dir)

# Calculate Diversity Score
diversity_score_value = calculate_diversity_score(resized_fake_dir)
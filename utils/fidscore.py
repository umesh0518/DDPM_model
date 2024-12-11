import os
from PIL import Image
import torch
from pytorch_fid import fid_score

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

# Function to resize images to 299x299 (required for FID calculations)
def resize_images(input_dir, output_dir, target_size=(299, 299)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img.save(os.path.join(output_dir, filename))

# Directories for real and generated images
real_dir = r"C:\Users\uyadav\Downloads\ddpm\improved-diffusion\test_images"

# Temporary directory for resized real images
resized_real_dir = "resized_real_images_test"

# Check if resized real images already exist
if not os.path.exists(resized_real_dir) or not os.listdir(resized_real_dir):
    print("Resizing real images...")
    resize_images(real_dir, resized_real_dir)
else:
    print("Resized real images already exist, skipping...")
    
# Process for cosine only generated images
fake_dir_cosine_only = r"C:\Users\uyadav\Downloads\ddpm\utils\output_images_dec7_cosineonly_learned_signma_false"
resized_fake_dir_cosine_only = "resized_fake_images_cosine_only"

print("Processing cosine only images...")
resize_images(fake_dir_cosine_only, resized_fake_dir_cosine_only)
calculate_fid(resized_real_dir, resized_fake_dir_cosine_only)

# Process for cosine with learned sigma generated images
fake_dir_cosine_learnedSignma = r"C:\Users\uyadav\Downloads\ddpm\utils\output_images_nov29_cosine_learnedsigma"
resized_fake_dir_cosine_learnedSignma = "resized_fake_images_cosine_leanedsigna_true"

print("Processing cosine and learned signma True images...")
resize_images(fake_dir_cosine_learnedSignma, resized_fake_dir_cosine_learnedSignma)
calculate_fid(resized_real_dir, resized_fake_dir_cosine_learnedSignma)

# Process for linear only generated images
fake_dir_linear_only = r"C:\Users\uyadav\Downloads\ddpm\utils\output_images_nov29_linear"
resized_fake_dir_linear_only = "resized_fake_images_linear_only"

print("Processing linear only images...")
resize_images(fake_dir_linear_only, resized_fake_dir_linear_only)
calculate_fid(resized_real_dir, resized_fake_dir_linear_only)

# Process for linear and learned signma generated images
# fake_dir_linear_only = r"C:\Users\uyadav\Downloads\ddpm\utils\output_images_nov29_linear_andLeanered signma"
# resized_fake_dir_linear_only = "resized_fake_images_linear_only"

# print("Processing linear and learned signma images...")
# resize_images(fake_dir_linear_only, resized_fake_dir_linear_only)
# calculate_fid(resized_real_dir, resized_fake_dir_linear_only)

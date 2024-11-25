import os
import shutil
from PIL import Image
import torch
from pytorch_fid import fid_score

def calculate_fid(real_dir, fake_dir):
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048,
        num_workers=0  # Disable multiprocessing
    )
    print(f"FID Score: {fid_value}")

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
real_dir = r"C:\Users\uyadav\Downloads\ddpm\utils\output_images"  # real image directory
fake_dir = r"C:\Users\uyadav\Downloads\ddpm\utils\realimages"  # generated image directory

# Temporary directories for resized images
resized_real_dir = "resized_real_images"
resized_fake_dir = "resized_fake_images"


resize_images(real_dir, resized_real_dir)
resize_images(fake_dir, resized_fake_dir)

calculate_fid(resized_real_dir, resized_fake_dir)

import os
import shutil
import random

# Paths to your dataset directory and desired output directories
dataset_dir = "melanoma_resize_bcndataset"
train_dir = "train_images_bcn"
test_dir = "test_images_bcn"

# Create the train and test directories if they don’t exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Set split ratio
split_ratio = 0.2  # 20% for testing

# List all image files in the dataset directory
all_files = [f for f in os.listdir(dataset_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

# Shuffle the files to randomize selection
random.shuffle(all_files)

# Calculate the split index
split_index = int(len(all_files) * split_ratio)

# Split files into testing and training lists
test_files = all_files[:split_index]
train_files = all_files[split_index:]

# Move files to the respective directories
for file_name in train_files:
    shutil.copy(os.path.join(dataset_dir, file_name), os.path.join(train_dir, file_name))

for file_name in test_files:
    shutil.copy(os.path.join(dataset_dir, file_name), os.path.join(test_dir, file_name))

print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")
print("Data split complete!")

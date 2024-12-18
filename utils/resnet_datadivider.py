import os
import shutil
import random

# Define paths
original_data_dir = "./melanoma_classification"  
output_data_dir = "./data"

# Define categories
categories = ["melanoma", "non_melanoma"]

# Create folders
for split in ["train", "val", "test"]:
    for category in categories:
        os.makedirs(os.path.join(output_data_dir, split, category), exist_ok=True)

def split_data(source_dir, target_dir, split_ratio):
    files = os.listdir(source_dir)
    random.shuffle(files)

    train_split = int(split_ratio[0] * len(files))
    val_split = int(split_ratio[1] * len(files))

    train_files = files[:train_split]
    val_files = files[train_split:train_split + val_split]
    test_files = files[train_split + val_split:]

    for file_list, split in zip([train_files, val_files, test_files], ["train", "val", "test"]):
        for file in file_list:
            shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, split, os.path.basename(source_dir), file))

# Split each category
split_ratios = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test
for category in categories:
    split_data(os.path.join(original_data_dir, category), output_data_dir, split_ratios)

print("Data splitting complete.")

import os
import shutil

# Define the root directory where images are stored
image_root_dir = "archive"  # Change this to your dataset root folder

# Function to rename files by adding category from their directory structure
def rename_files_with_category(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                
                # Extract category from directory structure
                parts = img_path.split(os.sep)
                category = "_".join(parts[-5:-3])  # Combine the last two meaningful directory names
                
                # Create new filename
                new_filename = f"{category}_{file}"
                new_filepath = os.path.join(root, new_filename)
                
                # Rename file
                shutil.move(img_path, new_filepath)
                print(f"Renamed: {img_path} -> {new_filepath}")

# Run the renaming function
rename_files_with_category(image_root_dir)

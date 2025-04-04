import os
import shutil
from PIL import Image

# ===== CONFIGURATION SECTION =====
# Path to the source directory containing the organized images
SOURCE_FOLDER = "../imagesDatasetFixed"

# Path to the destination directory for processed images
DESTINATION_FOLDER = "../imagesDatasetCropped3"
# ===== END CONFIGURATION SECTION =====

def create_destination_structure(source_path, dest_path):
    """
    Create the same folder structure in the destination as in the source.
    """
    # Walk through the source directory structure
    for root, dirs, _ in os.walk(source_path):
        # Calculate the relative path
        rel_path = os.path.relpath(root, source_path)
        
        # Skip if we're in the root directory
        if rel_path == '.':
            continue
        
        # Create the same directory in the destination
        dest_dir = os.path.join(dest_path, rel_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            print(f"Created directory: {dest_dir}")

def process_images(source_path, dest_path):
    """
    Process images by:
    1. Cropping to the bottom half
    2. Renaming based on folder structure with incrementing counter
    """
    # Create counters dictionary to track number of images per folder
    counters = {}
    
    # Track total images processed
    total_processed = 0
    
    # Walk through all files in the source directory
    for root, _, files in os.walk(source_path):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        if not image_files:
            continue
        
        # Get relative path for naming convention
        rel_path = os.path.relpath(root, source_path)
        if rel_path == '.':
            continue
        
        # Initialize counter for this folder if not already done
        if rel_path not in counters:
            counters[rel_path] = 1
        
        # Process all image files in this directory
        for img_file in image_files:
            try:
                # Construct the full file paths
                src_img_path = os.path.join(root, img_file)
                
                # Open the image and get its dimensions
                with Image.open(src_img_path) as img:
                    width, height = img.size
                    
                    # Crop to right half
                    # This takes the full height but only the right half of the width
                    # Format is (left, top, right, bottom)
                    right_half = img.crop((width // 2, 0, width, height))
                    
                    # Create new filename based on folder structure
                    # Replace directory separators with hyphens
                    folder_name = rel_path.replace(os.path.sep, '-')
                    
                    # Create new filename
                    new_filename = f"{folder_name}-{counters[rel_path]}.jpg"
                    
                    # Construct destination path
                    dest_dir = os.path.join(dest_path, rel_path)
                    dest_img_path = os.path.join(dest_dir, new_filename)
                    
                    # Save the cropped image
                    right_half.save(dest_img_path, "JPEG")
                    
                    # Increment the counter for this folder
                    counters[rel_path] += 1
                    total_processed += 1
                    
                    if total_processed % 50 == 0:
                        print(f"Processed {total_processed} images...")
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    # Print results
    print(f"\nProcessing complete!")
    print(f"Total images processed: {total_processed}")
    print("Images per folder:")
    for folder, count in counters.items():
        print(f"  {folder}: {count - 1} images")  # -1 because we increment after processing

def main():
    # Create base destination folder if it doesn't exist
    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)
    
    # Create the same folder structure in the destination
    create_destination_structure(SOURCE_FOLDER, DESTINATION_FOLDER)
    
    # Process all the images
    process_images(SOURCE_FOLDER, DESTINATION_FOLDER)

if __name__ == "__main__":
    main()
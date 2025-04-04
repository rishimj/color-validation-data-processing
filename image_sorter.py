import os
import shutil
import re
from typing import Dict, List, Tuple


# ===== CONFIGURATION SECTION =====
# Modify these variables to set your image ranges

# Path to the folder containing the images
SOURCE_FOLDER = "/Users/rishimanimaran/Downloads/Photos-001 (1)"

# Path to create the sorted folder structure
DESTINATION_FOLDER = "/Users/rishimanimaran/Documents/College/junior-year/spring-2025/cs-3312/color-validation-app-spring/imagesDatasetFixed"

# Medium Cherry ranges (format: "START-END")
MEDIUM_CHERRY_RANGES = {
    "out-of-range-too-light": "3771-3795",
    "in-range-light": "3798-3825",
    "in-range-standard": "3827-3853",
    "in-range-dark": "3855-3886",
    "out-of-range-too-dark": "3888-3912"
}

# Desert Oak ranges (format: "START-END")
DESERT_OAK_RANGES = {
    "out-of-range-too-light": "4113-4147",
    "in-range-light": "4194-4205",
    "in-range-standard": "4207-4240",
    "in-range-dark": "4242-4320",
    "out-of-range-too-dark": "4328-4368"
}

# Graphite Walnut ranges (format: "START-END")
GRAPHITE_WALNUT_RANGES = {
    "out-of-range-too-light": "3926-3934",
    "in-range-light": "3936-3984",
    "in-range-standard": "3986-4033",
    "in-range-dark": "4035-4071",
    "out-of-range-too-dark": "4073-4111"
}
# ===== END CONFIGURATION SECTION =====


def create_folder_structure(base_path: str) -> None:
    """Create the folder structure for sorting images."""
    colors = ["medium-cherry", "desert-oak", "graphite-walnut"]
    categories = [
        "out-of-range-too-light",
        "in-range-light",
        "in-range-standard",
        "in-range-dark",
        "out-of-range-too-dark"
    ]
    
    for color in colors:
        color_path = os.path.join(base_path, color)
        if not os.path.exists(color_path):
            os.makedirs(color_path)
        
        for category in categories:
            category_path = os.path.join(color_path, category)
            if not os.path.exists(category_path):
                os.makedirs(category_path)


def parse_range_str(range_str: str) -> List[int]:
    """Parse a range string like '3759-3800' into a list of integers."""
    try:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    except ValueError:
        raise ValueError(f"Invalid range format: {range_str}. Expected format: '3759-3800'")


def sort_images(
    source_folder: str,
    base_path: str,
    medium_cherry_ranges: Dict[str, str],
    desert_oak_ranges: Dict[str, str],
    graphite_walnut_ranges: Dict[str, str]
) -> Tuple[int, List[str]]:
    """
    Sort images from source_folder into the appropriate folders based on the specified ranges.
    
    Returns:
        Tuple containing:
        - Number of images sorted
        - List of images that weren't sorted (if any)
    """
    # Compile a pattern to match image files and extract the number
    pattern = re.compile(r'IMG_(\d+)\.JPG', re.IGNORECASE)
    
    # Create dictionaries mapping image numbers to their destination folders
    destination_map = {}
    
    # Process ranges for each color and category
    for category, range_str in medium_cherry_ranges.items():
        for num in parse_range_str(range_str):
            destination_map[num] = os.path.join(base_path, "medium-cherry", category)
    
    for category, range_str in desert_oak_ranges.items():
        for num in parse_range_str(range_str):
            destination_map[num] = os.path.join(base_path, "desert-oak", category)
    
    for category, range_str in graphite_walnut_ranges.items():
        for num in parse_range_str(range_str):
            destination_map[num] = os.path.join(base_path, "graphite-walnut", category)
    
    # Sort the images
    count = 0
    not_sorted = []
    
    for filename in os.listdir(source_folder):
        match = pattern.match(filename)
        if match:
            image_num = int(match.group(1))
            
            # Check if the image number is in our mapping
            if image_num in destination_map:
                source_path = os.path.join(source_folder, filename)
                dest_path = os.path.join(destination_map[image_num], filename)
                
                # Copy the file to its destination
                shutil.copy2(source_path, dest_path)
                count += 1
            else:
                not_sorted.append(filename)
    
    return count, not_sorted


def main():
    # Create the folder structure
    create_folder_structure(DESTINATION_FOLDER)
    
    # Filter out any empty ranges
    filtered_medium_cherry = {k: v for k, v in MEDIUM_CHERRY_RANGES.items() if v}
    filtered_desert_oak = {k: v for k, v in DESERT_OAK_RANGES.items() if v}
    filtered_graphite_walnut = {k: v for k, v in GRAPHITE_WALNUT_RANGES.items() if v}
    
    # Sort the images
    count, not_sorted = sort_images(
        SOURCE_FOLDER,
        DESTINATION_FOLDER,
        filtered_medium_cherry,
        filtered_desert_oak,
        filtered_graphite_walnut
    )
    
    print(f"Sorted {count} images.")
    
    if not_sorted:
        print(f"Warning: {len(not_sorted)} images were not sorted because they weren't in any specified range.")
        if len(not_sorted) <= 10:
            print("Unsorted images:", not_sorted)
        else:
            print("First 10 unsorted images:", not_sorted[:10])


if __name__ == "__main__":
    main()
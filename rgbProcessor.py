import os
import numpy as np
from PIL import Image
import itertools
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random

# ============= CONFIGURATION =============
# Set your dataset path here
DATASET_PATH = "/Users/rishimanimaran/Documents/College/junior-year/spring-2025/cs-3312/color-validation-app-spring/images-dataset-4.0/medium-cherry"
# ========================================

def calculate_euclidean_distance(img_path1, img_path2, resize_to=(300, 300)):
    """
    Calculate the RGB Euclidean distance between two images.
    
    Args:
        img_path1: Path to first image
        img_path2: Path to second image
        resize_to: Tuple (width, height) to resize images for comparison
        
    Returns:
        float: The average RGB Euclidean distance
    """
    try:
        # Load images
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        
        # Resize images to a standard size for comparison
        img1 = img1.resize(resize_to)
        img2 = img2.resize(resize_to)
        
        # Convert images to RGB if they aren't already
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        # Convert to numpy arrays for efficient calculation
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        
        # Calculate squared differences for each RGB channel
        r_diff = (img1_array[:,:,0].astype(float) - img2_array[:,:,0].astype(float)) ** 2
        g_diff = (img1_array[:,:,1].astype(float) - img2_array[:,:,1].astype(float)) ** 2
        b_diff = (img1_array[:,:,2].astype(float) - img2_array[:,:,2].astype(float)) ** 2
        
        # Sum the channel differences for each pixel
        pixel_diff = np.sqrt(r_diff + g_diff + b_diff)
        
        # Return the average difference across all pixels
        return np.mean(pixel_diff)
    except Exception as e:
        print(f"Error comparing {os.path.basename(img_path1)} and {os.path.basename(img_path2)}: {str(e)}")
        return np.nan

def get_image_paths_from_category(category_path, max_images=None):
    """
    Get paths to all images in a category folder.
    
    Args:
        category_path: Path to the category folder
        max_images: Maximum number of images to include (optional, for sampling)
        
    Returns:
        list: List of image paths
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_paths = [
        os.path.join(category_path, f) for f in os.listdir(category_path)
        if os.path.isfile(os.path.join(category_path, f)) and
        os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    # Optionally limit the number of images
    if max_images and len(image_paths) > max_images:
        image_paths = random.sample(image_paths, max_images)
    
    return image_paths

def calculate_category_distances(dataset_path, max_images_per_category=None, sample_pairs=None):
    """
    Calculate average RGB Euclidean distances between all category pairs.
    
    Args:
        dataset_path: Path to the dataset root folder
        max_images_per_category: Maximum number of images to use from each category
        sample_pairs: Number of random image pairs to sample for large categories
        
    Returns:
        tuple: (raw_distances_df, normalized_distances_df)
    """
    categories = [
        "out-of-range-too-light",
        "in-range-light",
        "in-range-standard",
        "in-range-dark",
        "out-of-range-too-dark"
    ]
    
    # Dictionary to store image paths for each category
    category_image_paths = {}
    
    # Get paths to all images from each category
    print("Finding images in each category...")
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            category_image_paths[category] = get_image_paths_from_category(
                category_path, max_images=max_images_per_category)
            print(f"  Found {len(category_image_paths[category])} images in {category}")
        else:
            print(f"Warning: Category path not found: {category_path}")
            category_image_paths[category] = []
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=categories, columns=categories)
    
    # Calculate average distance between each pair of categories
    print("\nCalculating average distances between categories...")
    
    for cat1, cat2 in itertools.product(categories, categories):
        image_paths1 = category_image_paths[cat1]
        image_paths2 = category_image_paths[cat2]
        
        if not image_paths1 or not image_paths2:
            results.loc[cat1, cat2] = np.nan
            continue
        
        # Generate pairs of image paths to compare
        if cat1 == cat2:
            # For same category, only compare each pair once
            pairs = [(path1, path2)
                    for i, path1 in enumerate(image_paths1)
                    for j, path2 in enumerate(image_paths2)
                    if i < j]
        else:
            pairs = list(itertools.product(image_paths1, image_paths2))
        
        # If there are too many pairs, sample a smaller subset
        if sample_pairs and len(pairs) > sample_pairs:
            pairs = random.sample(pairs, sample_pairs)
        
        print(f"  Comparing {cat1} and {cat2}: {len(pairs)} image pairs")
        
        # Calculate distances
        distances = []
        for path1, path2 in tqdm(pairs, desc=f"{cat1} vs {cat2}"):
            distance = calculate_euclidean_distance(path1, path2)
            distances.append(distance)
        
        # Calculate average distance
        if distances:
            avg_distance = np.nanmean(distances)  # Ignore NaN values
        else:
            avg_distance = np.nan
        
        results.loc[cat1, cat2] = avg_distance
        print(f"  Average distance between {cat1} and {cat2}: {avg_distance:.2f}")
    
    # Also calculate normalized version (0-100 scale)
    normalized_results = results.copy()
    for i in range(len(normalized_results.index)):
        for j in range(len(normalized_results.columns)):
            value = normalized_results.iloc[i, j]
            if not np.isnan(value):
                normalized_results.iloc[i, j] = min(100, value / 2.55)
    
    return results, normalized_results

def plot_heatmap(distances_df, title, output_path=None):
    """
    Plot a heatmap of the distances between categories.
    
    Args:
        distances_df: DataFrame containing distances
        title: Title for the heatmap
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(distances_df.values, cmap='viridis')
    plt.colorbar(im, label='Average RGB Euclidean Distance')
    
    # Add labels
    plt.xticks(np.arange(len(distances_df.columns)), distances_df.columns, rotation=45, ha='right')
    plt.yticks(np.arange(len(distances_df.index)), distances_df.index)
    
    # Add text annotations
    for i in range(len(distances_df.index)):
        for j in range(len(distances_df.columns)):
            value = distances_df.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > np.nanmean(distances_df.values.flatten()) else 'black'
                plt.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color)
    
    plt.title(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def main():
    # Use the globally defined dataset path
    dataset_path = DATASET_PATH
    
    print(f"Using dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: The path {dataset_path} does not exist.")
        return
    
    # Ask user for sampling parameters to handle large datasets
    max_images = input("Enter maximum number of images to use from each category (leave blank for all): ").strip()
    max_images = int(max_images) if max_images.isdigit() else None
    
    sample_pairs = input("Enter maximum number of image pairs to compare per category pair (leave blank for all): ").strip()
    sample_pairs = int(sample_pairs) if sample_pairs.isdigit() else None
    
    # Calculate distances between categories
    raw_distances, normalized_distances = calculate_category_distances(
        dataset_path, 
        max_images_per_category=max_images,
        sample_pairs=sample_pairs
    )
    
    # Print results as a table
    print("\nRaw Results - Average RGB Euclidean Distances:")
    print(raw_distances.round(2))
    
    print("\nNormalized Results (0-100 scale):")
    print(normalized_distances.round(2))
    
    # Save results to CSV
    raw_csv = os.path.join(dataset_path, "category_distances_raw.csv")
    raw_distances.to_csv(raw_csv)
    
    normalized_csv = os.path.join(dataset_path, "category_distances_normalized.csv")
    normalized_distances.to_csv(normalized_csv)
    
    print(f"\nResults saved to: {raw_csv} and {normalized_csv}")
    
    # Create and save heatmap visualizations
    raw_plot = os.path.join(dataset_path, "category_distances_raw_heatmap.png")
    plot_heatmap(raw_distances, "Average RGB Euclidean Distance Between Categories", raw_plot)
    
    normalized_plot = os.path.join(dataset_path, "category_distances_normalized_heatmap.png")
    plot_heatmap(normalized_distances, "Normalized RGB Euclidean Distance (0-100 scale)", normalized_plot)
    
    print(f"Heatmaps saved to: {raw_plot} and {normalized_plot}")
    
    # Print key comparisons focusing on in-range-standard
    print("\nKey Comparisons from in-range-standard:")
    if "in-range-standard" in raw_distances.index:
        for category in raw_distances.columns:
            if category != "in-range-standard":
                raw_dist = raw_distances.loc["in-range-standard", category]
                norm_dist = normalized_distances.loc["in-range-standard", category]
                print(f"- Standard vs {category}: {raw_dist:.2f} (normalized: {norm_dist:.2f})")

if __name__ == "__main__":
    main()
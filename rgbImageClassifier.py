import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
from scipy.spatial.distance import euclidean

# ============= CONFIGURATION =============
# Set your dataset path here
DATASET_PATH = "/Users/rishimanimaran/Documents/College/junior-year/spring-2025/cs-3312/color-validation-app-spring/images-dataset-4.0/medium-cherry"
# Categories in order (must match the CSV columns)
CATEGORIES = [
    "out-of-range-too-light",
    "in-range-light",
    "in-range-standard", 
    "in-range-dark",
    "out-of-range-too-dark"
]
# Path to your existing reference profiles CSV
REFERENCE_PROFILES_CSV = "/Users/rishimanimaran/Documents/College/junior-year/spring-2025/cs-3312/color-validation-app-spring/data_processing/category_distances_normalized_medium_cherry.csv"
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
        print(f"Error comparing images: {str(e)}")
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
        np.random.seed(42)  # For reproducibility
        image_paths = np.random.choice(image_paths, max_images, replace=False).tolist()
    
    return image_paths

def calculate_image_distribution(input_image_path, dataset_path, max_images_per_category=None, normalize=True):
    """
    Calculate the average RGB Euclidean distance between the input image and
    all images in each category.
    
    Args:
        input_image_path: Path to the input image
        dataset_path: Path to the dataset root folder
        max_images_per_category: Maximum number of images to use from each category
        normalize: Whether to normalize distances to 0-100 scale
        
    Returns:
        pandas.Series: Average distances to each category
    """
    if not os.path.exists(input_image_path):
        raise ValueError(f"Input image path does not exist: {input_image_path}")
    
    # Dictionary to store distances
    distances = {cat: [] for cat in CATEGORIES}
    
    print("Calculating distances between input image and each category...")
    
    for category in CATEGORIES:
        category_path = os.path.join(dataset_path, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Category path not found: {category_path}")
            continue
            
        category_images = get_image_paths_from_category(category_path, max_images_per_category)
        print(f"  Processing {len(category_images)} images from {category}...")
        
        for image_path in tqdm(category_images, desc=f"{category}"):
            distance = calculate_euclidean_distance(input_image_path, image_path)
            distances[category].append(distance)
    
    # Calculate average distance for each category
    avg_distances = {}
    for category, dist_list in distances.items():
        if dist_list:
            avg_distances[category] = np.nanmean(dist_list)
        else:
            avg_distances[category] = np.nan
    
    # Convert to pandas Series
    distance_profile = pd.Series(avg_distances)
    
    # Normalize if requested
    if normalize:
        normalized_profile = pd.Series({
            category: min(100, distance / 2.55) if not np.isnan(distance) else np.nan
            for category, distance in avg_distances.items()
        })
        return normalized_profile
    else:
        return distance_profile

def load_reference_profiles(csv_path):
    """
    Load reference category profiles from CSV.
    
    Args:
        csv_path: Path to the CSV with category distance profiles
        
    Returns:
        pandas.DataFrame: Reference profiles
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Reference profiles CSV not found: {csv_path}")
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_path, index_col=0)
        print(f"Loaded reference profiles with shape: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading reference profiles: {str(e)}")

def classify_image(image_profile, reference_profiles):
    """
    Classify the image by finding the most similar category profile.
    
    Args:
        image_profile: Series with distances to each category
        reference_profiles: DataFrame with reference category profiles
        
    Returns:
        tuple: (predicted_category, similarity_scores)
    """
    # Calculate similarity score (using Euclidean distance between profiles)
    similarity_scores = {}
    
    for category in reference_profiles.index:
        category_profile = reference_profiles.loc[category]
        
        # Calculate distance between profiles (lower = more similar)
        profile_distance = euclidean(
            image_profile.fillna(0),  # Replace NaN with 0 for calculation
            category_profile.fillna(0)
        )
        
        # Convert to similarity score (higher = more similar)
        similarity_scores[category] = 1 / (1 + profile_distance)
    
    # Find the most similar category
    similarity_series = pd.Series(similarity_scores)
    predicted_category = similarity_series.idxmax()
    
    return predicted_category, similarity_series

def plot_distribution_comparison(image_profile, reference_profiles, predicted_category, 
                                output_path=None):
    """
    Create a bar chart comparing the image's distance profile with reference profiles.
    
    Args:
        image_profile: Series with the image's distances to each category
        reference_profiles: DataFrame with reference category profiles
        predicted_category: The predicted category
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    categories = image_profile.index
    x = np.arange(len(categories))
    width = 0.35
    
    # Plot the image profile
    plt.bar(x - width/2, image_profile, width, label='Test Image Profile', color='#2196F3')
    
    # Plot the predicted category profile
    if predicted_category in reference_profiles.index:
        category_profile = reference_profiles.loc[predicted_category]
        plt.bar(x + width/2, category_profile, width, 
                label=f'{predicted_category} Reference Profile', color='#4CAF50')
    
    # Add labels and legend
    plt.xlabel('Categories')
    plt.ylabel('Normalized Euclidean Distance (0-100)')
    plt.title('RGB Distance Profile Comparison')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Add text annotation with prediction
    text = f"Predicted Category: {predicted_category}"
    plt.figtext(0.5, 0.01, text, ha='center', fontsize=12, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def visualize_image_with_result(input_image_path, predicted_category, output_path=None):
    """
    Create a visualization of the input image with its classification result.
    
    Args:
        input_image_path: Path to the input image
        predicted_category: The predicted category
        output_path: Path to save the plot (optional)
    """
    try:
        # Load the image
        img = Image.open(input_image_path)
        
        # Set up the figure
        plt.figure(figsize=(10, 6))
        
        # Display the image
        plt.imshow(img)
        plt.axis('off')
        
        # Add classification result
        plt.title(f"Classified as: {predicted_category}", fontsize=16, pad=20)
        
        # Set background color based on category
        if 'too-dark' in predicted_category.lower() or 'too_dark' in predicted_category.lower():
            plt.gca().set_facecolor('#d32f2f')  # Dark red
        elif 'dark' in predicted_category.lower():
            plt.gca().set_facecolor('#ff9800')  # Orange
        elif 'standard' in predicted_category.lower():
            plt.gca().set_facecolor('#4caf50')  # Green
        elif 'light' in predicted_category.lower():
            plt.gca().set_facecolor('#2196f3')  # Blue
        elif 'too-light' in predicted_category.lower() or 'too_light' in predicted_category.lower():
            plt.gca().set_facecolor('#9c27b0')  # Purple
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error visualizing image: {str(e)}")

def plot_all_reference_profiles(reference_profiles, output_path=None):
    """
    Create a line chart showing all reference profiles for comparison.
    
    Args:
        reference_profiles: DataFrame with reference category profiles
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Get categories
    categories = reference_profiles.columns
    
    # Plot each profile as a line
    for category in reference_profiles.index:
        plt.plot(categories, reference_profiles.loc[category], marker='o', label=category)
    
    # Add labels and legend
    plt.xlabel('Categories')
    plt.ylabel('Normalized Euclidean Distance (0-100)')
    plt.title('Reference Category Profiles')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def main():
    # Set up command line arguments
    testImage = "/Users/rishimanimaran/Documents/College/junior-year/spring-2025/cs-3312/color-validation-app-spring/images-dataset-4.0/medium-cherry/out-of-range-too-light/medium-cherry-out-of-range-too-light-1.jpg"
    
    parser = argparse.ArgumentParser(description='Classify a wood veneer image based on RGB Euclidean distance profile.')
    parser.add_argument('--image', type=str,default =testImage, help='Path to the input image')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, help='Path to the dataset')
    parser.add_argument('--references', type=str, default=REFERENCE_PROFILES_CSV, help='Path to reference profiles CSV')
    parser.add_argument('--max_images', type=int, default=20, help='Maximum images to use per category')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--show_profiles', action='store_true', help='Show all reference profiles')
    
    # Add this argument
    parser.add_argument('--validate', action='store_true', 
                        help='Run validation on the entire dataset')
    
    args = parser.parse_args()
    
    # Add this condition after parsing args
    if args.validate:
        validate_classifier_accuracy(args.dataset, args.references, args.max_images)
        return
    
    
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found at {args.image}")
        return
    
    # Create output directory if specified
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load reference profiles
    print(f"Loading reference profiles from: {args.references}")
    reference_profiles = load_reference_profiles(args.references)
    
    # Optionally show all reference profiles
    if args.show_profiles:
        if args.output:
            profiles_plot = os.path.join(args.output, "reference_profiles.png")
            plot_all_reference_profiles(reference_profiles, profiles_plot)
            print(f"Reference profiles plot saved to: {profiles_plot}")
        else:
            plot_all_reference_profiles(reference_profiles)
    
    # Calculate distance profile for the input image
    print(f"\nProcessing input image: {args.image}")
    image_profile = calculate_image_distribution(
        args.image, args.dataset, args.max_images, normalize=True
    )
    
    # Print the results
    print("\nDistance Profile (Normalized 0-100):")
    print(image_profile.round(2))
    
    # Save the profile
    if args.output:
        profile_csv = os.path.join(args.output, "image_profile.csv")
        image_profile.to_csv(profile_csv)
        print(f"Profile saved to: {profile_csv}")
    
    # Classify the image
    predicted_category, similarity_scores = classify_image(image_profile, reference_profiles)
    
    print(f"\nPredicted Category: {predicted_category}")
    print("\nSimilarity Scores (higher = more similar):")
    print(similarity_scores.sort_values(ascending=False).round(4))
    
    # Plot comparison with the predicted category profile
    if args.output:
        comparison_plot = os.path.join(args.output, "profile_comparison.png")
        plot_distribution_comparison(
            image_profile, reference_profiles, predicted_category, comparison_plot
        )
        print(f"Comparison plot saved to: {comparison_plot}")
        
        # Create visualization of the input image with result
        image_viz = os.path.join(args.output, "classified_image.png")
        visualize_image_with_result(args.image, predicted_category, image_viz)
        print(f"Image visualization saved to: {image_viz}")
    else:
        plot_distribution_comparison(image_profile, reference_profiles, predicted_category)
        visualize_image_with_result(args.image, predicted_category)
    
    print("\nClassification complete!")
    
'''
added stuff
'''

def extract_true_category(image_path):
    """
    Extract the true category from the image path.
    Assumes directory structure like: .../medium-cherry/out-of-range-too-light/image.jpg
    
    Args:
        image_path: Path to the image
        
    Returns:
        str: True category name
    """
    parts = image_path.split(os.sep)
    # The category is usually the second-to-last directory in the path
    if len(parts) >= 2:
        return parts[-2]  # Returns "out-of-range-too-light" from the example path
    return "unknown"

def validate_classifier_accuracy(dataset_path, reference_profiles_csv, max_images_per_category=None):
    """
    Run through all images in the dataset and check if predictions match true categories.
    
    Args:
        dataset_path: Path to the dataset directory
        reference_profiles_csv: Path to reference profiles CSV
        max_images_per_category: Limit images per category (optional)
        
    Returns:
        dict: Accuracy statistics
    """
    print(f"Starting validation on dataset: {dataset_path}")
    
    # Load reference profiles
    reference_profiles = load_reference_profiles(reference_profiles_csv)
    
    # Statistics
    stats = {
        "total": 0,
        "correct": 0,
        "by_category": {}
    }
    
    # Find all image files in the dataset
    all_images = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} images to validate")
    
    # Process each image
    for img_path in tqdm(all_images, desc="Validating images"):
        # Extract true category
        true_category = extract_true_category(img_path)
        
        # Initialize category stats if not exists
        if true_category not in stats["by_category"]:
            stats["by_category"][true_category] = {"total": 0, "correct": 0}
        
        try:
            # Calculate distance profile
            image_profile = calculate_image_distribution(
                img_path, dataset_path, max_images_per_category, normalize=True
            )
            
            # Classify the image
            predicted_category, _ = classify_image(image_profile, reference_profiles)
            
            # Update statistics
            stats["total"] += 1
            stats["by_category"][true_category]["total"] += 1
            
            if predicted_category == true_category:
                stats["correct"] += 1
                stats["by_category"][true_category]["correct"] += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Calculate accuracy percentages
    if stats["total"] > 0:
        stats["accuracy"] = stats["correct"] / stats["total"] * 100
        
        for category, cat_stats in stats["by_category"].items():
            if cat_stats["total"] > 0:
                cat_stats["accuracy"] = cat_stats["correct"] / cat_stats["total"] * 100
    
    # Print results
    print("\n===== VALIDATION RESULTS =====")
    print(f"Total images: {stats['total']}")
    print(f"Correctly classified: {stats['correct']}")
    print(f"Overall accuracy: {stats.get('accuracy', 0):.2f}%")
    
    print("\nAccuracy by category:")
    for category, cat_stats in sorted(stats["by_category"].items()):
        if cat_stats["total"] > 0:
            print(f"  {category}: {cat_stats.get('accuracy', 0):.2f}% ({cat_stats['correct']}/{cat_stats['total']})")
    
    return stats

if __name__ == "__main__":
    main()
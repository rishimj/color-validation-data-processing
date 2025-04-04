import os
import sys
import numpy as np
from PIL import Image
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from collections import defaultdict

# --------- Configuration (modify as needed) ---------
# Base directory of the dataset
BASE_DIR = "/Users/rishimanimaran/Documents/College/junior-year/spring-2025/cs-3312/color-validation-app-spring/images-dataset-4.0"

# Path to your reference profiles CSV
REFERENCE_PROFILES_CSV = "category_distances_normalized_medium_cherry.csv"

# --------- Helper Functions ---------
def get_true_class(image_path):
    """Extract the true class from the image path."""
    parts = image_path.split('/')
    # Find the category in the path (second-to-last directory)
    if len(parts) >= 2:
        return parts[-3]  # Assuming directory structure: .../category/subcategory/image.jpg
    return "unknown"

def get_true_subcategory(image_path):
    """Extract the subcategory from the image path."""
    parts = image_path.split('/')
    if len(parts) >= 2:
        return parts[-2]  # Assuming directory structure: .../category/subcategory/image.jpg
    return "unknown"

def calculate_average_rgb(image_path):
    """Calculate the average RGB values for an image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        return img_array.mean(axis=(0,1))
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.array([0, 0, 0])

def calculate_image_distances(image_path, dataset_paths):
    """Calculate distances between test image and all images in dataset."""
    test_rgb = calculate_average_rgb(image_path)
    distances = []
    
    for dataset_img_path in dataset_paths:
        dataset_rgb = calculate_average_rgb(dataset_img_path)
        distance = euclidean(test_rgb, dataset_rgb)
        distances.append((dataset_img_path, distance))
    
    return distances

def calculate_category_distances(image_path, dataset_paths_by_category):
    """Calculate average distance to each category."""
    category_distances = {}
    
    for category, paths in dataset_paths_by_category.items():
        if not paths:
            continue
            
        distances = calculate_image_distances(image_path, paths)
        avg_distance = np.mean([d[1] for d in distances])
        category_distances[category] = avg_distance
    
    return category_distances

def normalize_distances(distances):
    """Normalize distances to 0-100 scale."""
    min_dist = min(distances.values())
    max_dist = max(distances.values())
    
    if max_dist == min_dist:
        return {k: 0 for k in distances.keys()}
    
    normalized = {}
    for category, distance in distances.items():
        normalized[category] = ((distance - min_dist) / (max_dist - min_dist)) * 100
    
    return normalized

def classify_using_reference_profiles(test_image_distances, reference_profiles):
    """Classify image by comparing its distance profile to reference profiles."""
    best_match = None
    best_score = float('inf')
    similarity_scores = {}
    
    # Normalize the test image distances
    test_normalized = normalize_distances(test_image_distances)
    
    # Compare normalized test profile to each reference profile
    for category, reference_profile in reference_profiles.items():
        # Calculate profile similarity (lower is better)
        profile_diff = 0
        for cat in reference_profile.keys():
            if cat in test_normalized:
                profile_diff += abs(reference_profile[cat] - test_normalized[cat])
        
        similarity_scores[category] = profile_diff
        
        if profile_diff < best_score:
            best_score = profile_diff
            best_match = category
    
    return best_match, similarity_scores, test_normalized

def load_reference_profiles(csv_path):
    """Load reference profiles from CSV file."""
    try:
        df = pd.read_csv(csv_path, index_col=0)
        # Convert DataFrame to nested dictionary
        profiles = {}
        for idx, row in df.iterrows():
            profiles[idx] = row.to_dict()
        return profiles
    except Exception as e:
        print(f"Error loading reference profiles: {e}")
        return None

def get_all_image_paths(base_dir):
    """Get all image paths organized by category."""
    image_paths_by_category = defaultdict(list)
    all_image_paths = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                category = get_true_class(full_path)
                image_paths_by_category[category].append(full_path)
                all_image_paths.append(full_path)
    
    return image_paths_by_category, all_image_paths

# --------- Main Script ---------
def run_validation(verbose=False):
    """Run validation on the entire dataset.
    
    Args:
        verbose: If True, prints more detailed information during processing
    """
    print(f"Starting validation on dataset: {BASE_DIR}")
    print(f"Time started: {time.strftime('%H:%M:%S')}")
    
    # Load reference profiles
    reference_profiles = load_reference_profiles(REFERENCE_PROFILES_CSV)
    if not reference_profiles:
        print(f"Could not load reference profiles from {REFERENCE_PROFILES_CSV}. Exiting.")
        return
    
    print(f"Loaded reference profiles for {len(reference_profiles)} categories.")
    
    # Get all image paths
    image_paths_by_category, all_image_paths = get_all_image_paths(BASE_DIR)
    
    # Statistics tracking
    stats = {
        "total_images": 0,
        "correct_predictions": 0,
        "errors": 0,
        "category_results": defaultdict(lambda: {"total": 0, "correct": 0}),
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "timing": {"start": time.time(), "total_processing_time": 0}
    }
    
    # Track all examples with incorrect predictions
    incorrect_predictions = []
    
    # Process each image
    print(f"Found {len(all_image_paths)} images to process.")
    
    # Use tqdm if available for progress bar, otherwise fall back to simple printing
    try:
        from tqdm import tqdm
        image_iterator = tqdm(all_image_paths, desc="Processing images")
    except ImportError:
        print("Note: Install tqdm package for better progress visualization")
        image_iterator = all_image_paths
        
    for img_path in image_iterator:
        stats["total_images"] += 1
        
        # Print progress updates even without tqdm
        if not 'tqdm' in sys.modules and stats["total_images"] % 10 == 0:
            print(f"Processed {stats['total_images']}/{len(all_image_paths)} images ({stats['total_images']/len(all_image_paths):.1%})...")
        
        try:
            # Get the true class from the path
            true_class = get_true_class(img_path)
            true_subcategory = get_true_subcategory(img_path)
            
            # Time the prediction and show current file being processed
            start_time = time.time()
            if not 'tqdm' in sys.modules:  # Only print if not using tqdm
                print(f"Processing image {stats['total_images']}: {os.path.basename(img_path)}", end="\r")
            
            # Calculate distances to each category
            category_distances = calculate_category_distances(img_path, image_paths_by_category)
            
            # Print detailed info for every 50th image or if verbose
            if stats["total_images"] % 50 == 0:
                print(f"\nDetailed analysis for {img_path}:")
                print(f"  True category: {true_class}")
                print(f"  Raw category distances: {category_distances}")
            
            # Classify using reference profiles
            predicted_class, similarity_scores, normalized_distances = classify_using_reference_profiles(
                category_distances, reference_profiles
            )
            
            # Continue detailed output for milestone images
            if stats["total_images"] % 50 == 0:
                print(f"  Normalized distances: {normalized_distances}")
                print(f"  Predicted category: {predicted_class}")
                print(f"  Correct: {predicted_class == true_class}")
                print(f"  Current accuracy: {stats['correct_predictions']/stats['total_images']:.2%}")
            
            prediction_time = time.time() - start_time
            stats["timing"]["total_processing_time"] += prediction_time
            
            # Update statistics
            stats["category_results"][true_class]["total"] += 1
            stats["confusion_matrix"][true_class][predicted_class] += 1
            
            if predicted_class == true_class:
                stats["correct_predictions"] += 1
                stats["category_results"][true_class]["correct"] += 1
            else:
                # Track incorrect predictions
                incorrect_predictions.append({
                    "image_path": img_path,
                    "true_class": true_class,
                    "true_subcategory": true_subcategory,
                    "predicted_class": predicted_class,
                    "similarity_scores": similarity_scores,
                    "normalized_distances": normalized_distances
                })
        except Exception as e:
            stats["errors"] += 1
            print(f"Error processing {img_path}: {str(e)}")
    
    # Calculate final statistics
    stats["timing"]["end"] = time.time()
    stats["timing"]["total_duration"] = stats["timing"]["end"] - stats["timing"]["start"]
    
    if stats["total_images"] > 0:
        stats["accuracy"] = stats["correct_predictions"] / stats["total_images"]
        
        # Calculate per-category accuracy
        for category, results in stats["category_results"].items():
            if results["total"] > 0:
                results["accuracy"] = results["correct"] / results["total"]
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDATION RESULTS SUMMARY")
    print("="*50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Overall accuracy: {stats.get('accuracy', 0):.2%}")
    print(f"Total processing time: {stats['timing']['total_duration']:.2f} seconds")
    print(f"Average time per image: {stats['timing']['total_processing_time'] / max(1, stats['total_images']):.4f} seconds")
    
    print("\nCategory Accuracies:")
    for category, results in sorted(stats["category_results"].items()):
        if results["total"] > 0:
            print(f"  {category}: {results.get('accuracy', 0):.2%} ({results['correct']}/{results['total']})")
    
    # Save results to file
    results_dir = os.path.join(os.path.dirname(BASE_DIR), "validation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(results_dir, f"validation_results_{timestamp}.json")
    
    # Convert defaultdict to regular dict for JSON serialization
    serializable_stats = {
        "total_images": stats["total_images"],
        "correct_predictions": stats["correct_predictions"],
        "errors": stats["errors"],
        "accuracy": stats.get("accuracy", 0),
        "category_results": {k: dict(v) for k, v in stats["category_results"].items()},
        "confusion_matrix": {k: dict(v) for k, v in stats["confusion_matrix"].items()},
        "timing": dict(stats["timing"]),
        "incorrect_predictions": incorrect_predictions[:100]  # Limit to first 100 for file size
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_stats, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Generate a visualization of the confusion matrix
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert confusion matrix to numpy array
        categories = sorted(stats["category_results"].keys())
        confusion_array = np.zeros((len(categories), len(categories)))
        
        for i, true_cat in enumerate(categories):
            for j, pred_cat in enumerate(categories):
                confusion_array[i, j] = stats["confusion_matrix"][true_cat][pred_cat]
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_array, annot=True, fmt='g', xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(results_dir, f"confusion_matrix_{timestamp}.png")
        plt.savefig(plot_file)
        print(f"Confusion matrix visualization saved to: {plot_file}")
        
        # Additional visualization - incorrect predictions analysis
        if incorrect_predictions:
            plt.figure(figsize=(14, 8))
            
            # Group incorrect predictions by true category
            incorrect_by_category = defaultdict(list)
            for pred in incorrect_predictions:
                incorrect_by_category[pred["true_class"]].append(pred)
            
            # Plot distribution of incorrect predictions
            categories = list(incorrect_by_category.keys())
            counts = [len(incorrect_by_category[cat]) for cat in categories]
            
            plt.bar(categories, counts)
            plt.xlabel('True Category')
            plt.ylabel('Number of Incorrect Predictions')
            plt.title('Distribution of Incorrect Predictions by Category')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            error_plot_file = os.path.join(results_dir, f"incorrect_distribution_{timestamp}.png")
            plt.savefig(error_plot_file)
            print(f"Incorrect predictions distribution saved to: {error_plot_file}")
        
    except ImportError:
        print("Matplotlib or seaborn not available. Skipping visualizations.")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate wood veneer classification using reference profiles')
    parser.add_argument('--verbose', action='store_true', help='Show more detailed output during processing')
    parser.add_argument('--dataset', type=str, default=BASE_DIR, help='Path to the dataset directory')
    parser.add_argument('--references', type=str, default=REFERENCE_PROFILES_CSV, 
                        help='Path to the reference profiles CSV file')
    
    args = parser.parse_args()
    
    # Update global variables if provided in arguments
    if args.dataset:
        BASE_DIR = args.dataset
    if args.references:
        REFERENCE_PROFILES_CSV = args.references
    
    # Show runtime configuration
    print(f"Configuration:")
    print(f"  Dataset path: {BASE_DIR}")
    print(f"  Reference profiles: {REFERENCE_PROFILES_CSV}")
    print(f"  Verbose mode: {'On' if args.verbose else 'Off'}")
    print("-" * 50)
    
    # Run validation
    run_validation(verbose=args.verbose)
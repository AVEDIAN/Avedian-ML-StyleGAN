import os
import argparse
import torch
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import torchvision.transforms as transforms
import shutil
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_images(path, max_images=None):
    """Load images from directory"""
    image_files = []
    for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
        image_files.extend(glob.glob(os.path.join(path, f"**/*.{ext}"), recursive=True))
    
    if not image_files:
        raise ValueError(f"No images found in {path}")
    
    print(f"Found {len(image_files)} images in {path}")
    
    if max_images and len(image_files) > max_images:
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(image_files)
        image_files = image_files[:max_images]
        print(f"Using {max_images} random images")
    
    return image_files

def get_inception_features(image_files, batch_size=16, return_images=False):
    """Extract features using Inception v3"""
    from torchvision.models import inception_v3
    
    # Load model
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove final FC layer
    model = model.to(device)
    model.eval()
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Extract features
    features = []
    valid_files = []
    processed_images = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting features"):
            batch_files = image_files[i:i+batch_size]
            batch = []
            batch_valid_files = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    if return_images:
                        processed_images.append(img.copy())
                    img_tensor = preprocess(img).unsqueeze(0)
                    batch.append(img_tensor)
                    batch_valid_files.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            if not batch:
                continue
                
            batch = torch.cat(batch, dim=0).to(device)
            feat = model(batch)
            features.append(feat.cpu().numpy())
            valid_files.extend(batch_valid_files)
    
    if not features:
        raise ValueError("No valid features extracted")
        
    features = np.concatenate(features, axis=0)
    
    if return_images:
        return features, valid_files, processed_images
    else:
        return features, valid_files

def find_most_similar_images(real_features, fake_features, real_files, fake_files, top_n=5):
    """Find the most similar images between real and fake sets"""
    # Calculate pairwise distances
    distances = cdist(real_features, fake_features, 'euclidean')
    
    # Find closest matches
    most_similar_pairs = []
    
    # For each real image, find the closest fake
    for i in range(len(real_features)):
        closest_fake_indices = np.argsort(distances[i])[:top_n]
        for j in closest_fake_indices:
            similarity = distances[i][j]
            most_similar_pairs.append((real_files[i], fake_files[j], similarity, i, j))
    
    # Sort by similarity (distance)
    most_similar_pairs.sort(key=lambda x: x[2])
    
    return most_similar_pairs[:top_n]

def plot_similar_pairs(similar_pairs, real_images, fake_images, output_file='similar_pairs.png'):
    """Plot the most similar pairs side by side"""
    plt.figure(figsize=(15, 3 * len(similar_pairs)))
    
    for i, (real_file, fake_file, similarity, real_idx, fake_idx) in enumerate(similar_pairs):
        plt.subplot(len(similar_pairs), 2, i*2+1)
        plt.imshow(real_images[real_idx])
        plt.title(f"Real: {os.path.basename(real_file)}")
        plt.axis('off')
        
        plt.subplot(len(similar_pairs), 2, i*2+2)
        plt.imshow(fake_images[fake_idx])
        plt.title(f"Fake: {os.path.basename(fake_file)}\nDistance: {similarity:.4f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved comparison to {output_file}")
    plt.close()

def copy_similar_pairs(similar_pairs, output_dir):
    """Copy the most similar pairs to an output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (real_file, fake_file, similarity, _, _) in enumerate(similar_pairs):
        real_name = os.path.basename(real_file)
        fake_name = os.path.basename(fake_file)
        
        real_out = os.path.join(output_dir, f"{i+1}_real_{real_name}")
        fake_out = os.path.join(output_dir, f"{i+1}_fake_{fake_name}")
        
        shutil.copy(real_file, real_out)
        shutil.copy(fake_file, fake_out)
        
        # Create a text file with similarity info
        with open(os.path.join(output_dir, f"{i+1}_info.txt"), "w") as f:
            f.write(f"Real image: {real_file}\n")
            f.write(f"Fake image: {fake_file}\n")
            f.write(f"Similarity distance: {similarity:.4f}\n")
    
    print(f"Copied {len(similar_pairs)} similar pairs to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Find most similar images between real and fake datasets")
    parser.add_argument("--real", required=True, help="Directory with real images")
    parser.add_argument("--fake", required=True, help="Directory with generated images")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-images", type=int, default=500, help="Max images per category to analyze")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top similar pairs to find")
    parser.add_argument("--output-dir", default="similar_pairs", help="Directory to save similar pairs")
    parser.add_argument("--plot", action="store_true", help="Generate plot of similar pairs")
    
    args = parser.parse_args()
    
    try:
        # Load images
        real_image_files = load_images(args.real, args.max_images)
        fake_image_files = load_images(args.fake, args.max_images)
        
        # Extract features
        print("\nExtracting features from real images...")
        real_features, real_files, real_images = get_inception_features(real_image_files, args.batch_size, return_images=True)
        
        print("\nExtracting features from fake images...")
        fake_features, fake_files, fake_images = get_inception_features(fake_image_files, args.batch_size, return_images=True)
        
        # Find most similar pairs
        print("\nFinding most similar image pairs...")
        similar_pairs = find_most_similar_images(real_features, fake_features, real_files, fake_files, args.top_n)
        
        # Display results
        print("\nTop similar pairs:")
        for i, (real_file, fake_file, similarity, _, _) in enumerate(similar_pairs):
            print(f"{i+1}. Real: {os.path.basename(real_file)} - Fake: {os.path.basename(fake_file)} - Distance: {similarity:.4f}")
        
        # Save results
        copy_similar_pairs(similar_pairs, args.output_dir)
        
        # Plot if requested
        if args.plot:
            plot_similar_pairs(similar_pairs, real_images, fake_images, 
                              output_file=os.path.join(args.output_dir, "similar_pairs.png"))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
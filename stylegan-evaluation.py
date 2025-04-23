import os
import argparse
import torch
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import torchvision.transforms as transforms

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
        random.shuffle(image_files)
        image_files = image_files[:max_images]
        print(f"Using {max_images} random images")
    
    return image_files

def get_inception_features(image_files, batch_size=16):
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
    logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting features"):
            batch_files = image_files[i:i+batch_size]
            batch = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = preprocess(img).unsqueeze(0)
                    batch.append(img)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            if not batch:
                continue
                
            batch = torch.cat(batch, dim=0).to(device)
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    if not features:
        raise ValueError("No valid features extracted")
        
    features = np.concatenate(features, axis=0)
    return features

def calculate_inception_score(features, splits=10):
    """Calculate inception score from features"""
    from scipy.stats import entropy
    from torch.nn import functional as F
    
    # Convert to probabilities
    probs = F.softmax(torch.tensor(features), dim=1).numpy()
    
    # Split the images into N groups
    N = features.shape[0]
    split_size = N // splits
    
    scores = []
    for i in range(splits):
        part = probs[i * split_size:(i + 1) * split_size]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)

def calculate_fid(real_features, fake_features):
    """Calculate FID between real and fake features"""
    from scipy import linalg
    
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    
    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    
    # Calculate sqrt of product between cov
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate score
    score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return score

def main():
    parser = argparse.ArgumentParser(description="Simple StyleGAN Evaluation")
    parser.add_argument("--real", required=True, help="Directory with real images")
    parser.add_argument("--fake", required=True, help="Directory with generated images")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-images", type=int, default=200, help="Max images per category")
    parser.add_argument("--metrics", default="isc,fid", help="Metrics to compute (comma-separated: isc,fid)")
    
    args = parser.parse_args()
    metrics = args.metrics.lower().split(',')
    
    try:
        # Load images
        real_images = load_images(args.real, args.max_images)
        fake_images = load_images(args.fake, args.max_images)
        
        # Extract features for fake images
        fake_features = get_inception_features(fake_images, args.batch_size)
        
        # Store results
        results = {}
        
        # Calculate Inception Score
        if 'isc' in metrics:
            try:
                isc_mean, isc_std = calculate_inception_score(fake_features)
                results['inception_score_mean'] = isc_mean
                results['inception_score_std'] = isc_std
            except Exception as e:
                print(f"Error calculating Inception Score: {e}")
        
        # Calculate FID
        if 'fid' in metrics:
            try:
                # Extract features for real images
                real_features = get_inception_features(real_images, args.batch_size)
                
                # Calculate FID
                fid = calculate_fid(real_features, fake_features)
                results['frechet_inception_distance'] = fid
            except Exception as e:
                print(f"Error calculating FID: {e}")
        
        # Print results in the exact format requested
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
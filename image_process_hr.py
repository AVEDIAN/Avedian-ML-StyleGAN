import boto3
import cv2
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageEnhance

def get_optimal_kernel_size(image):
    """Determine optimal kernel size based on image resolution"""
    height, width = image.shape[:2]
    kernel_size = max(9, min(int(min(height, width) * 0.02) // 2 * 2 + 1, 21))
    return kernel_size

def get_optimal_threshold(gray_image):
    """Determine optimal threshold using Otsu's method and image statistics"""
    mean_val = np.mean(gray_image)
    std_val = np.std(gray_image)
    otsu_thresh, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = max(10, min(otsu_thresh * 0.5, mean_val - std_val))
    return adjusted_thresh

def get_optimal_inpaint_radius(mask):
    """Determine optimal inpainting radius based on detected hair width"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 6  # default value
    
    widths = []
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        widths.append(min(w, h))
    
    avg_width = np.mean(widths) if widths else 6
    return max(3, min(int(avg_width * 1.5), 10))

def detect_hairs(gray_image):
    """
    Detects if there are significant hair-like structures in the image.
    Returns True if hairs are detected, False otherwise.
    """
    try:
        # Get optimal kernel size for this image
        kernel_size = get_optimal_kernel_size(gray_image)
        kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))
        
        # Apply blackhat to detect dark structures
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        
        # Apply Gaussian blur to reduce noise
        gaussian_kernel = max(3, min(kernel_size // 3, 9))
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        bhg = cv2.GaussianBlur(blackhat, (gaussian_kernel, gaussian_kernel), cv2.BORDER_DEFAULT)
        
        # Get threshold
        threshold = get_optimal_threshold(bhg)
        _, mask = cv2.threshold(bhg, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate the percentage of pixels that might be hair
        hair_pixel_percentage = (np.sum(mask == 255) / mask.size) * 100
        
        # If more than 0.25% of pixels are detected as potential hair, return True
        return hair_pixel_percentage > 0.25
        
    except Exception as e:
        print(f"Error in hair detection: {e}")
        return False

def read_image(image_path: str) -> Optional[cv2.Mat]:
    """
    Lee una imagen desde la ruta especificada.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error al leer la imagen: {image_path}")
            return None
        return img
    except Exception as e:
        print(f"Error al leer la imagen {image_path}: {e}")
        return None

def crop_square(img_array):
    """
    crop image to square (1:1) aspect ratio, preserving the smallest dimension.
    """
    height, width = img_array.shape[:2]
    target_size = min(height, width)
    
    if width > height:
        start_x = (width - height) // 2
        start_y = 0
        cropped = img_array[:, start_x:start_x+height]
    else:
        start_x = 0
        start_y = (height - width) // 2
        cropped = img_array[start_y:start_y+width, :]
   
    return cropped

def enhanced_resize(img_array, target_size, quality_level='high'):
    """
    Enhanced image resizing with multiple quality options.
    
    Args:
        img_array: NumPy array of the image
        target_size: Tuple of (width, height)
        quality_level: 'basic', 'high', or 'ultra'
        
    Returns:
        Resized image array
    """
    # Handle downsampling vs upsampling differently
    h, w = img_array.shape[:2]
    target_w, target_h = target_size
    is_downsampling = (target_w < w) or (target_h < h)
    
    if quality_level == 'basic':
        # Standard cubic interpolation
        return cv2.resize(img_array, target_size, interpolation=cv2.INTER_CUBIC)
    
    elif quality_level == 'high':
        # Convert to PIL for better processing
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        
        # Apply different techniques based on whether we're upsampling or downsampling
        if is_downsampling:
            # For downsampling, first apply slight Gaussian blur to prevent aliasing
            sigma = 0.3  # Subtle blur
            kernel_size = max(3, min(int(sigma * 3) * 2 + 1, 5))
            if kernel_size % 2 == 0:
                kernel_size += 1
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
            img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            
            # Use Lanczos for downsampling (better than cubic)
            img_pil = img_pil.resize(target_size, Image.LANCZOS)
        else:
            # For upsampling, use BICUBIC which preserves more detail
            img_pil = img_pil.resize(target_size, Image.BICUBIC)
            
            # Apply subtle sharpening to enhance details
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(1.3)  # Moderate sharpening
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    elif quality_level == 'ultra':
        # Progressive resizing for highest quality
        # This performs the resize in multiple steps for better quality
        
        # Convert to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        
        if is_downsampling:
            # For large downsampling, use progressive approach
            current_w, current_h = w, h
            
            # Calculate how many steps based on resize ratio
            resize_ratio = min(target_w / w, target_h / h)
            
            if resize_ratio < 0.5:  # Significant downsampling
                steps = 3
                
                # Perform progressive downsampling
                for i in range(steps):
                    # Calculate intermediate size
                    intermediate_ratio = 1.0 - (1.0 - resize_ratio) * ((i + 1) / steps)
                    intermediate_w = int(w * intermediate_ratio)
                    intermediate_h = int(h * intermediate_ratio)
                    
                    # Apply appropriate preprocessing for each step
                    if i < steps - 1:
                        # Apply slight blur before each downsample step except the last
                        sigma = 0.5 * (1.0 - intermediate_ratio)
                        kernel_size = max(3, min(int(sigma * 4) * 2 + 1, 7))
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
                        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                    
                    # Resize with high-quality Lanczos
                    img_pil = img_pil.resize((intermediate_w, intermediate_h), Image.LANCZOS)
                    
                    # Convert back to array for possible next step
                    img_array = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                # For modest downsampling, single step with Lanczos is sufficient
                img_pil = img_pil.resize(target_size, Image.LANCZOS)
        else:
            # For upsampling, enhance before resize for better details
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(1.1)  # Subtle contrast boost
            
            # High-quality upsampling
            img_pil = img_pil.resize(target_size, Image.BICUBIC)
            
            # Post-processing for upsampled images
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(1.5)  # Stronger sharpening for upsampled images
            
            # Subtle color enhancement
            enhancer = ImageEnhance.Color(img_pil)
            img_pil = enhancer.enhance(1.05)  # Very subtle color enhancement
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    else:
        # Fallback to cubic interpolation
        return cv2.resize(img_array, target_size, interpolation=cv2.INTER_CUBIC)

def get_and_process_images(s3_folder: str, local_download_folder: str, target_size, hr: bool, muestra: int = 0, crop: bool = False, resize_quality='high') -> None:
    """
    Downloads, processes, and deletes images one by one from S3.
    
    Args:
        s3_folder: S3 folder path
        local_download_folder: Local folder for downloading and processing
        target_size: Target image size tuple (width, height)
        hr: Whether to perform hair removal
        muestra: Number of images to process (0 for all)
        crop: Whether to crop images to square
        resize_quality: 'basic', 'high', or 'ultra' resize quality
    """
    s3_bucket_name = "avedian-ml"
    images_processed = 0
    hairs_detected = 0
    
    try:
        s3 = boto3.client('s3')
        local_folder = Path(local_download_folder)
        local_folder.mkdir(parents=True, exist_ok=True)
        
        processed_dir = local_folder / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        continuation_token = None
        while True:
            if continuation_token:
                response = s3.list_objects_v2(
                    Bucket=s3_bucket_name, 
                    Prefix=s3_folder, 
                    ContinuationToken=continuation_token
                )
            else:
                response = s3.list_objects_v2(
                    Bucket=s3_bucket_name, 
                    Prefix=s3_folder
                )
                
            for obj in response.get('Contents', []):
                if muestra != 0 and images_processed >= muestra:
                    print(f"Processed {images_processed} images, found hair in {hairs_detected} images")
                    return
                    
                key = obj['Key']
                if key.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Download single image
                        local_path = local_folder / Path(key).name
                        s3.download_file(s3_bucket_name, key, str(local_path))
                        
                        # Process image
                        img_array = read_image(str(local_path))
                        if img_array is None:
                            local_path.unlink(missing_ok=True)
                            continue
                        
                        if crop:
                            img_array = crop_square(img_array)
                        
                        if hr:
                            # Convert to grayscale for hair detection
                            grayScale = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            
                            # Only proceed with hair removal if hairs are detected
                            if detect_hairs(grayScale):
                                hairs_detected += 1
                                # Get optimal kernel size
                                kernel_size = get_optimal_kernel_size(img_array)
                                kernel = cv2.getStructuringElement(1, (kernel_size, kernel_size))
                                # Black hat filter
                                blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
                                # Gaussian filter
                                gaussian_kernel = max(3, min(kernel_size // 3, 9))
                                if gaussian_kernel % 2 == 0:
                                    gaussian_kernel += 1
                                bhg = cv2.GaussianBlur(blackhat, (gaussian_kernel, gaussian_kernel), cv2.BORDER_DEFAULT)
                                # Binary thresholding (MASK)
                                threshold = get_optimal_threshold(bhg)
                                ret, mask = cv2.threshold(bhg, threshold, 255, cv2.THRESH_BINARY)
                                # Get optimal inpainting radius
                                inpaint_radius = get_optimal_inpaint_radius(mask)
                                # Replace pixels of the mask
                                img_array = cv2.inpaint(img_array, mask, inpaint_radius, cv2.INPAINT_TELEA)
                            else:
                                print(f"No hair detected in: {Path(key).name}")
                        
                        # Use the enhanced resizing function instead of cv2.resize
                        output_path = processed_dir / f"proc_{Path(key).name}"
                        img_array_resized = enhanced_resize(img_array, target_size, resize_quality)
                        cv2.imwrite(str(output_path), img_array_resized)
                        
                        # Delete original downloaded file
                        local_path.unlink(missing_ok=True)
                        
                        images_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing {key}: {e}")
                        # Clean up in case of error
                        local_path.unlink(missing_ok=True)
                        continue
            
            if response.get('IsTruncated') and (muestra == 0 or images_processed < muestra):
                continuation_token = response['NextContinuationToken']
            else:
                break
                
    except Exception as e:
        print(f"Error in image processing pipeline: {e}")
    
    print(f"\nFinished processing {images_processed} images, found hair in {hairs_detected} images")

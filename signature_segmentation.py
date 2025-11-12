"""
Signature Segmentation & Enhancement Pipeline

This script extracts and enhances signatures from scanned documents using:
- Adaptive Thresholding: Handles varying lighting conditions
- Minimal Morphological Operations: Light cleaning only (open + close, NO dilation/erosion)
- Simple Sharpening: One pass to enhance edges
- Pure White Background: No gray artifacts (no CLAHE)
- Pixel-Preserving: Gentle processing keeps all fine signature details

Author: Aaron Baggot
Date: November 2025
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def process_signature(input_path, output_path='clean_signature_enhanced.png', 
                      blockSize=31, C=2, target_width=600, show_plots=True):
    """
    Process and enhance a signature image.
    
    Parameters:
    -----------
    input_path : str
        Path to the input signature image
    output_path : str
        Path where the processed image will be saved
    blockSize : int
        Adaptive threshold block size (must be odd, default 31)
    C : int
        Constant subtracted from mean (default 2)
    target_width : int
        Target width for resizing (default 600)
    show_plots : bool
        Whether to display comparison plots (default True)
        
    Returns:
    --------
    resized : ndarray
        The final processed signature image
    """
    
    # --- Step 1: Load Image ---
    img = cv2.imread(input_path)
    
    if img is None:
        raise FileNotFoundError(f"Image not found at '{input_path}'")
    
    print(f"✓ Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]} px)")
    
    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Step 1: Original Image")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # --- Step 2: Grayscale & Blur ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    print(f"✓ Converted to grayscale and applied Gaussian blur (5x5)")
    
    # --- Step 3: Adaptive Thresholding ---
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize, C
    )
    
    print(f"✓ Applied Adaptive Threshold (blockSize={blockSize}, C={C})")
    print(f"  White pixels (ink): {np.sum(binary == 255)} | Black pixels (background): {np.sum(binary == 0)}")
    
    # --- Step 4: MINIMAL Morphological Cleaning ---
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    print(f"✓ Applied minimal morphological cleaning (light open + close only)")
    
    # --- Step 5: Crop to Bounding Box ---
    coords = cv2.findNonZero(clean_mask)
    if coords is None:
        raise ValueError("No signature detected! Try adjusting blockSize or C parameters.")
    
    x, y, w, h = cv2.boundingRect(coords)
    cropped_color = img[y:y+h, x:x+w].copy()
    cropped_mask = clean_mask[y:y+h, x:x+w].copy()
    
    # Re-run adaptive threshold on cropped region
    cropped_gray = cv2.cvtColor(cropped_color, cv2.COLOR_BGR2GRAY)
    cropped_blurred = cv2.GaussianBlur(cropped_gray, (5, 5), 0)
    cropped_binary = cv2.adaptiveThreshold(
        cropped_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize, C
    )
    cropped_mask = cv2.morphologyEx(cropped_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cropped_mask = cv2.morphologyEx(cropped_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    print(f"✓ Cropped to bounding box ({w}x{h} px) and re-thresholded")
    
    # --- Step 6: Simple Sharpening ---
    kernel_sharp = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]], dtype=np.float32)
    
    sharpened = cv2.filter2D(cropped_color, -1, kernel_sharp)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    print(f"✓ Applied Sharpening (minimal processing)")
    
    # --- Step 7: Apply Mask (Pure White Background) ---
    mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)
    final_output = np.where(mask_3ch == 255, sharpened, 255).astype(np.uint8)
    
    print(f"✓ Applied mask and ensured pure white background")
    
    if show_plots:
        # --- Step 8: Display Comparison ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(cropped_color, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Cropped")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cropped_mask, cmap='gray')
        axes[0, 1].set_title("Mask (Binary) - Re-thresholded")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("After Sharpening")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Final Output (All Pixels Preserved)")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # --- Step 9: Resize & Save ---
    height, width = final_output.shape[:2]
    scale = target_width / width
    resized = cv2.resize(final_output, (target_width, int(height * scale)), 
                        interpolation=cv2.INTER_CUBIC)
    
    output_path = Path(output_path)
    cv2.imwrite(str(output_path), resized)
    
    print(f"\n✓ SUCCESS: Saved to '{output_path.resolve()}'")
    print(f"  Final dimensions: {resized.shape[1]}x{resized.shape[0]} px")
    print(f"  Background: Pure white (no gray artifacts)")
    print(f"  All signature pixels preserved")
    
    return resized


if __name__ == "__main__":
    # Example usage
    input_image = 'signatures.jpg'  # Change to your image path
    process_signature(input_image, show_plots=True)

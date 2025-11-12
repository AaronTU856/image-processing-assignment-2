# Image Processing Assignment 2: Signature Segmentation & Enhancement

A Python-based image processing pipeline for extracting and enhancing signatures from scanned documents.

## Overview

This project implements a signature segmentation and enhancement algorithm that:
- **Adaptively thresholds** signatures based on local lighting conditions
- **Preserves fine details** with minimal morphological operations
- **Enhances edges** through controlled sharpening
- **Removes background** while maintaining pure white output
- **Captures all pixels** from faint or colored ink signatures

## Features

✨ **Key Features:**
- Adaptive thresholding for variable lighting
- Minimal morphological cleaning (no aggressive erosion/dilation)
- Selective sharpening with color channel preservation
- Pure white background guaranteed
- Fine pixel preservation with tunable parameters
- Comprehensive visualization and diagnostics

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/AaronTU856/image-processing-assignment-2.git
cd image-processing-assignment-2

# Install dependencies
pip install -r requirements.txt
```

## Usage

### As a Python Module

```python
from signature_segmentation import process_signature

# Process a signature image
result = process_signature(
    input_path='signatures.jpg',
    output_path='output.png',
    blockSize=31,  # Adaptive threshold neighborhood (must be odd)
    C=2,           # Constant subtracted from mean
    target_width=600,
    show_plots=True
)
```

### Command Line

```bash
python signature_segmentation.py
```

Edit the `__main__` section to specify your image path.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | - | Path to input signature image |
| `output_path` | str | `'clean_signature_enhanced.png'` | Output file path |
| `blockSize` | int | 31 | Adaptive threshold neighborhood (must be odd) |
| `C` | int | 2 | Constant subtracted from threshold mean |
| `target_width` | int | 600 | Output image width in pixels |
| `show_plots` | bool | True | Display comparison visualization |

### Parameter Tuning Guide

| Issue | Solution |
|-------|----------|
| Signature too faint | Increase `C` (e.g., 4-6) |
| Background noise | Decrease `C` (e.g., 0-1) or increase `blockSize` (39-43) |
| Broken strokes | Decrease `blockSize` (e.g., 27-29) |
| Over-sharpening | Modify kernel value from 9 to 5-7 |
| Missing fine pixels | Decrease `C` or use smaller `blockSize` |

## Pipeline Steps

1. **Load Image** - Read input signature image
2. **Grayscale & Blur** - Convert to grayscale and apply Gaussian blur
3. **Adaptive Thresholding** - Convert to binary mask based on local neighborhood
4. **Morphological Cleaning** - Light open/close to remove noise
5. **Crop** - Extract bounding box region
6. **Re-threshold** - Re-apply threshold on cropped region for better local adaptation
7. **Sharpening** - Enhance edges with sharpening kernel
8. **Masking** - Apply mask with pure white background
9. **Resize & Save** - Downscale to target width and save

## Example Output

The pipeline produces a 4-panel comparison showing:
- **Top-left**: Original cropped signature
- **Top-right**: Binary mask after thresholding
- **Bottom-left**: After sharpening
- **Bottom-right**: Final output with white background

## Technical Details

### Adaptive Thresholding
Uses `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C` mode to compute threshold values based on mean of neighborhood areas, handling varying lighting conditions.

### Minimal Morphology
Only applies `morphologyEx` with `MORPH_OPEN` and `MORPH_CLOSE` operations (single iteration each) to preserve all fine signature details.

### Sharpening Kernel
```
[-1 -1 -1]
[-1  9 -1]
[-1 -1 -1]
```
Can be adjusted to `[-1  5 -1]` for gentler sharpening.

## Requirements

See `requirements.txt`:
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0

## Author

Aaron Baggot - Image Processing Assignment 2

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

**Q: "Image not found" error**
- Ensure the image path is correct and file exists in the working directory

**Q: All black or all white output**
- Try adjusting `blockSize` (e.g., 27-35) or `C` (e.g., 0-6)

**Q: Missing signature pixels**
- Decrease `C` value or reduce `blockSize`
- Check that image contrast is sufficient

**Q: Background noise in output**
- Increase `C` value to strengthen background rejection
- Try increasing `blockSize` for smoother threshold

## References

- OpenCV Documentation: https://docs.opencv.org/
- Adaptive Thresholding: Niblack's and Sauvola's methods
- Image Processing: Digital Image Processing, Gonzalez & Woods

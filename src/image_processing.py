"""
Image processing steps for visualising how a chest X-ray is prepared.
The model input is grayscale -> resize -> normalise; CLAHE and edges are shown for analysis.
"""

from typing import Dict

import cv2
import numpy as np
from PIL import Image


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def run_pipeline(image: Image.Image, image_size: int = IMAGE_SIZE) -> Dict:
    """
    Apply each image processing step to an X-ray and keep the intermediate results.

    Args:
        image (Image.Image): Input X-ray
        image_size (int): Target size used by the model

    Returns:
        Dict: Intermediate images (uint8 arrays) and summary statistics
    """
    original = np.array(image.convert('RGB'))

    # Grayscale: X-rays carry intensity only
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    # Resize with bilinear interpolation to the model's input size
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # CLAHE: local histogram equalisation with a clip limit to avoid amplifying noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)

    # Edges: Gaussian blur to suppress noise, then Canny
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Normalisation: scale to [0, 1], then (x - mean) / std with ImageNet stats (red channel shown)
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    normalized_vis = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normalized_vis = cv2.cvtColor(
        cv2.applyColorMap(normalized_vis, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB
    )

    return {
        'original': original,
        'gray': gray,
        'resized': resized,
        'clahe': clahe,
        'edges': edges,
        'normalized': normalized_vis,
        'stats': {
            'original_size': f"{original.shape[1]}×{original.shape[0]}",
            'mean_intensity': float(resized.mean()),
            'contrast_before': float(resized.std()),
            'contrast_after': float(clahe.std()),
            'edge_density': float((edges > 0).mean()),
            'normalized_min': float(normalized.min()),
            'normalized_max': float(normalized.max()),
        },
    }


def colorize_heatmap(cam: np.ndarray) -> np.ndarray:
    """
    Convert a [0, 1] Grad-CAM map into an RGB JET heatmap.

    Args:
        cam (np.ndarray): Heatmap with values in [0, 1]

    Returns:
        np.ndarray: RGB heatmap (uint8)
    """
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

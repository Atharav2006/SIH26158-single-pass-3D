"""
SIH26158 Depth Fusion - Depth Quality Validation & Confidence Generation

This module validates relative monocular depth maps, computes measurable multi-cue
confidence maps, and extracts rigorous quality statistics.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

def compute_depth_confidence(
    rgb: np.ndarray,
    inv_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_inv_depth: float = 1e-4,
    max_inv_depth: float = 2000.0,
    border_margin: int = 20
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes a normalized confidence map in [0, 1] using physical/measurable signals:
    1. Image texture gradient (strong texture = high confidence, textureless = low confidence)
    2. Depth edge sharpness (sharp depth discontinuities = lower confidence due to boundary bleed)
    3. Distance from sensor image borders (sensor boundary attenuation)
    4. Valid depth mask
    
    Returns:
        confidence_map: [H, W] float32 in [0, 1]
        valid_mask: [H, W] bool mask
        stats: dict of quality diagnostics
    """
    H, W = inv_depth.shape
    
    # 1. Base Validity Mask
    finite_mask = np.isfinite(inv_depth)
    positive_mask = (inv_depth >= min_inv_depth) & (inv_depth <= max_inv_depth)
    base_mask = finite_mask & positive_mask
    if valid_mask is not None:
        base_mask &= valid_mask

    # 2. Border Attenuation (smooth falloff near sensor edge)
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    dist_border_x = np.minimum(x_coords, W - 1 - x_coords)
    dist_border_y = np.minimum(y_coords, H - 1 - y_coords)
    dist_border = np.minimum(dist_border_x, dist_border_y).astype(np.float32)
    border_weight = np.clip(dist_border / float(max(1, border_margin)), 0.0, 1.0)

    # 3. Image Texture Cue (Normalized Sobel Gradient)
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb.copy()
    
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_norm = np.clip(grad_mag / 64.0, 0.1, 1.0)  # Moderate texture gives baseline confidence

    # 4. Depth Discontinuity Attenuation (Avoid blurry depth edges)
    d_sobel_x = cv2.Sobel(inv_depth, cv2.CV_32F, 1, 0, ksize=3)
    d_sobel_y = cv2.Sobel(inv_depth, cv2.CV_32F, 0, 1, ksize=3)
    d_grad = np.sqrt(d_sobel_x**2 + d_sobel_y**2)
    # Relative depth gradient
    rel_d_grad = d_grad / np.maximum(inv_depth, 1.0)
    edge_attenuation = np.exp(-np.clip(rel_d_grad * 2.0, 0.0, 5.0))

    # Combine cues multiplicatively
    confidence = grad_norm * edge_attenuation * border_weight
    confidence[~base_mask] = 0.0
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)

    # Diagnostics
    valid_pixels = np.count_nonzero(base_mask)
    total_pixels = H * W
    valid_ratio = float(valid_pixels / total_pixels)

    valid_d_inv = inv_depth[base_mask]
    valid_conf = confidence[base_mask]

    stats = {
        "total_pixels": int(total_pixels),
        "valid_pixels": int(valid_pixels),
        "valid_ratio": valid_ratio,
        "invalid_ratio": float(1.0 - valid_ratio),
        "d_inv_percentiles": {
            "p1": float(np.percentile(valid_d_inv, 1)) if valid_pixels > 0 else 0.0,
            "p5": float(np.percentile(valid_d_inv, 5)) if valid_pixels > 0 else 0.0,
            "p25": float(np.percentile(valid_d_inv, 25)) if valid_pixels > 0 else 0.0,
            "p50": float(np.percentile(valid_d_inv, 50)) if valid_pixels > 0 else 0.0,
            "p75": float(np.percentile(valid_d_inv, 75)) if valid_pixels > 0 else 0.0,
            "p95": float(np.percentile(valid_d_inv, 95)) if valid_pixels > 0 else 0.0,
            "p99": float(np.percentile(valid_d_inv, 99)) if valid_pixels > 0 else 0.0
        },
        "confidence_percentiles": {
            "p1": float(np.percentile(valid_conf, 1)) if valid_pixels > 0 else 0.0,
            "p5": float(np.percentile(valid_conf, 5)) if valid_pixels > 0 else 0.0,
            "p25": float(np.percentile(valid_conf, 25)) if valid_pixels > 0 else 0.0,
            "p50": float(np.percentile(valid_conf, 50)) if valid_pixels > 0 else 0.0,
            "p75": float(np.percentile(valid_conf, 75)) if valid_pixels > 0 else 0.0,
            "p95": float(np.percentile(valid_conf, 95)) if valid_pixels > 0 else 0.0,
            "p99": float(np.percentile(valid_conf, 99)) if valid_pixels > 0 else 0.0
        }
    }

    return confidence, base_mask, stats

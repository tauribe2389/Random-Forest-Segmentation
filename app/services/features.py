"""Feature extraction utilities in a scikit-image style pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage import color, filters, img_as_float
from skimage.feature import local_binary_pattern

from .schemas import FeatureConfig


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert input image arrays to 3-channel RGB arrays."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported image shape: {image.shape}")


def feature_names(config: FeatureConfig) -> list[str]:
    """Return stable feature names for a config."""
    names: list[str] = []
    if config.use_rgb:
        names.extend(["rgb_r", "rgb_g", "rgb_b"])
    if config.use_hsv:
        names.extend(["hsv_h", "hsv_s", "hsv_v"])
    if config.use_lab:
        names.extend(["lab_l", "lab_a", "lab_b"])
    for sigma in config.gaussian_sigmas:
        names.extend(
            [
                f"gauss_{sigma:.3g}_r",
                f"gauss_{sigma:.3g}_g",
                f"gauss_{sigma:.3g}_b",
            ]
        )
    if config.use_sobel:
        names.append("sobel_gray")
    if config.use_lbp:
        names.append(f"lbp_p{config.lbp_points}_r{config.lbp_radius}")
    return names


def extract_feature_stack(
    image: np.ndarray,
    config: FeatureConfig,
) -> tuple[np.ndarray, list[str]]:
    """Extract per-pixel features and return (H, W, F) stack."""
    rgb = ensure_rgb(image)
    rgb_float = img_as_float(rgb)

    channels: list[np.ndarray] = []
    names: list[str] = []

    if config.use_rgb:
        channels.extend([rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]])
        names.extend(["rgb_r", "rgb_g", "rgb_b"])

    if config.use_hsv:
        hsv = color.rgb2hsv(rgb_float)
        channels.extend([hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]])
        names.extend(["hsv_h", "hsv_s", "hsv_v"])

    if config.use_lab:
        lab = color.rgb2lab(rgb_float)
        # Keep LAB values in original range for richer color separation.
        channels.extend([lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]])
        names.extend(["lab_l", "lab_a", "lab_b"])

    for sigma in config.gaussian_sigmas:
        blurred = gaussian_rgb(rgb_float, sigma=float(sigma))
        channels.extend([blurred[:, :, 0], blurred[:, :, 1], blurred[:, :, 2]])
        names.extend(
            [
                f"gauss_{sigma:.3g}_r",
                f"gauss_{sigma:.3g}_g",
                f"gauss_{sigma:.3g}_b",
            ]
        )

    gray = color.rgb2gray(rgb_float)
    if config.use_sobel:
        sobel = filters.sobel(gray)
        channels.append(sobel)
        names.append("sobel_gray")

    if config.use_lbp:
        lbp = local_binary_pattern(
            gray,
            P=config.lbp_points,
            R=config.lbp_radius,
            method="uniform",
        )
        lbp = lbp.astype(np.float32)
        max_lbp = float(lbp.max()) if lbp.size else 1.0
        if max_lbp > 0:
            lbp = lbp / max_lbp
        channels.append(lbp)
        names.append(f"lbp_p{config.lbp_points}_r{config.lbp_radius}")

    if not channels:
        raise ValueError(
            "Feature configuration produced no channels. Enable at least one feature."
        )

    stack = np.stack(channels, axis=-1).astype(np.float32)
    return stack, names


def gaussian_rgb(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with compatibility for older scikit-image versions."""
    try:
        return filters.gaussian(image, sigma=sigma, channel_axis=-1)
    except TypeError:
        # Older versions use multichannel=True.
        return filters.gaussian(image, sigma=sigma, multichannel=True)


def flatten_features(stack: np.ndarray) -> np.ndarray:
    """Flatten (H, W, F) feature stack into (N, F)."""
    return stack.reshape((-1, stack.shape[-1]))


def features_from_coordinates(
    feature_stack: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Gather feature vectors at (row, col) coordinates."""
    return feature_stack[rows, cols, :].astype(np.float32)

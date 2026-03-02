from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries, slic

from .image_io import ensure_dir, load_image_rgb, load_image_size


def cache_file_paths(cache_dir: Path, image_stem: str) -> tuple[Path, Path]:
    segments_path = cache_dir / f"{image_stem}__slic.npz"
    boundary_path = cache_dir / f"{image_stem}__boundaries.png"
    return segments_path, boundary_path


def clear_slic_cache(cache_dir: Path, image_stem: str) -> None:
    segments_path, boundary_path = cache_file_paths(cache_dir, image_stem)
    if segments_path.exists():
        segments_path.unlink()
    if boundary_path.exists():
        boundary_path.unlink()


def _compute_segments(
    image_rgb: np.ndarray,
    n_segments: int,
    compactness: float,
    sigma: float,
    colorspace: str,
) -> np.ndarray:
    use_lab = str(colorspace or "lab").strip().lower() == "lab"
    try:
        segments = slic(
            image_rgb,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            convert2lab=use_lab,
            start_label=0,
            channel_axis=-1,
        )
    except TypeError:
        # Compatibility fallback for older scikit-image releases.
        try:
            segments = slic(
                image_rgb,
                n_segments=n_segments,
                compactness=compactness,
                sigma=sigma,
                convert2lab=use_lab,
                multichannel=True,
            )
        except TypeError:
            segments = slic(
                image_rgb,
                n_segments=n_segments,
                compactness=compactness,
                sigma=sigma,
            )

    min_label = int(np.min(segments))
    if min_label != 0:
        segments = segments - min_label

    return segments.astype(np.int32, copy=False)


def _boundary_overlay(segments: np.ndarray, color: tuple[int, int, int, int]) -> np.ndarray:
    boundaries = find_boundaries(segments, mode="outer")
    h, w = segments.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[boundaries] = color
    return rgba


def load_or_create_slic_cache(
    image_path: Path,
    cache_dir: Path,
    n_segments: int,
    compactness: float,
    sigma: float,
    colorspace: str = "lab",
    boundary_color: tuple[int, int, int, int] = (255, 255, 255, 170),
) -> tuple[np.ndarray, Path]:
    ensure_dir(cache_dir)
    image_stem = image_path.stem
    segments_path, boundary_path = cache_file_paths(cache_dir, image_stem)
    expected_shape = load_image_size(image_path)
    setting_signature = (
        f"n={int(n_segments)}|c={float(compactness):.6f}|s={float(sigma):.6f}|cs={str(colorspace).lower()}"
    )

    segments = None
    if segments_path.exists():
        with np.load(segments_path) as loaded:
            cached_segments = loaded["segments"]
            cached_signature = ""
            if "setting_signature" in loaded:
                raw_signature = loaded["setting_signature"]
                try:
                    cached_signature = str(raw_signature.item())
                except Exception:
                    cached_signature = str(raw_signature)
        if tuple(cached_segments.shape) == tuple(expected_shape) and cached_signature == setting_signature:
            segments = cached_segments.astype(np.int32, copy=False)

    if segments is None:
        image_rgb = load_image_rgb(image_path)
        segments = _compute_segments(image_rgb, n_segments, compactness, sigma, colorspace)
        np.savez_compressed(
            segments_path,
            segments=segments,
            setting_signature=np.array(setting_signature),
        )
        if boundary_path.exists():
            boundary_path.unlink()

    if not boundary_path.exists():
        overlay = _boundary_overlay(segments, boundary_color)
        Image.fromarray(overlay, mode="RGBA").save(boundary_path)

    return segments, boundary_path

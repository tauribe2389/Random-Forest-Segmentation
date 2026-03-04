from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters
from skimage.color import rgb2gray, rgb2lab
from skimage.feature import local_binary_pattern
from skimage.segmentation import felzenszwalb, find_boundaries, quickshift, slic

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
    algorithm: str,
    n_segments: int,
    compactness: float,
    sigma: float,
    colorspace: str,
    quickshift_ratio: float,
    quickshift_kernel_size: int,
    quickshift_max_dist: float,
    quickshift_sigma: float,
    felzenszwalb_scale: float,
    felzenszwalb_sigma: float,
    felzenszwalb_min_size: int,
    texture_enabled: bool,
    texture_mode: str,
    texture_lbp_enabled: bool,
    texture_lbp_points: int,
    texture_lbp_radii: list[int],
    texture_lbp_method: str,
    texture_lbp_normalize: bool,
    texture_gabor_enabled: bool,
    texture_gabor_frequencies: list[float],
    texture_gabor_thetas: list[float],
    texture_gabor_bandwidth: float,
    texture_gabor_include_real: bool,
    texture_gabor_include_imag: bool,
    texture_gabor_include_magnitude: bool,
    texture_gabor_normalize: bool,
    texture_weight_color: float,
    texture_weight_lbp: float,
    texture_weight_gabor: float,
) -> np.ndarray:
    algorithm_key = str(algorithm or "slic").strip().lower()
    if algorithm_key not in {"slic", "slico", "quickshift", "felzenszwalb"}:
        algorithm_key = "slic"
    use_lab = str(colorspace or "lab").strip().lower() == "lab"

    if algorithm_key in {"slic", "slico"}:
        use_slico = algorithm_key == "slico"
        use_texture = bool(texture_enabled) and bool(texture_lbp_enabled or texture_gabor_enabled)
        slic_input = image_rgb
        slic_convert2lab = use_lab
        if use_texture and str(texture_mode or "append_to_color").strip().lower() == "append_to_color":
            source_color = rgb2lab(image_rgb) if use_lab else image_rgb.astype(np.float32, copy=False)

            def _normalize_channel(channel: np.ndarray) -> np.ndarray:
                arr = np.asarray(channel, dtype=np.float32)
                if arr.size <= 0:
                    return arr
                min_value = float(arr.min())
                max_value = float(arr.max())
                span = max_value - min_value
                if span <= 1e-8:
                    return np.zeros_like(arr, dtype=np.float32)
                return (arr - min_value) / span

            channels: list[np.ndarray] = []
            for index in range(source_color.shape[2]):
                channels.append(_normalize_channel(source_color[:, :, index]) * float(texture_weight_color))

            gray = rgb2gray(image_rgb).astype(np.float32, copy=False)
            if texture_lbp_enabled:
                for radius in texture_lbp_radii:
                    lbp = local_binary_pattern(
                        gray,
                        P=int(texture_lbp_points),
                        R=float(radius),
                        method=str(texture_lbp_method),
                    ).astype(np.float32, copy=False)
                    if texture_lbp_normalize:
                        lbp = _normalize_channel(lbp)
                    channels.append(lbp * float(texture_weight_lbp))
            if texture_gabor_enabled:
                for frequency in texture_gabor_frequencies:
                    freq = float(frequency)
                    for theta_value in texture_gabor_thetas:
                        theta = np.deg2rad(float(theta_value))
                        response_real, response_imag = filters.gabor(
                            gray,
                            frequency=freq,
                            theta=theta,
                            bandwidth=float(texture_gabor_bandwidth),
                        )
                        if texture_gabor_include_real:
                            real = np.asarray(response_real, dtype=np.float32)
                            if texture_gabor_normalize:
                                real = _normalize_channel(real)
                            channels.append(real * float(texture_weight_gabor))
                        if texture_gabor_include_imag:
                            imag = np.asarray(response_imag, dtype=np.float32)
                            if texture_gabor_normalize:
                                imag = _normalize_channel(imag)
                            channels.append(imag * float(texture_weight_gabor))
                        if texture_gabor_include_magnitude:
                            magnitude = np.hypot(response_real, response_imag).astype(np.float32)
                            if texture_gabor_normalize:
                                magnitude = _normalize_channel(magnitude)
                            channels.append(magnitude * float(texture_weight_gabor))
            if channels:
                slic_input = np.stack(channels, axis=-1).astype(np.float32, copy=False)
                slic_convert2lab = False
        try:
            segments = slic(
                slic_input,
                n_segments=n_segments,
                compactness=compactness,
                sigma=sigma,
                convert2lab=slic_convert2lab,
                start_label=0,
                channel_axis=-1,
                slic_zero=use_slico,
            )
        except TypeError:
            try:
                segments = slic(
                    slic_input,
                    n_segments=n_segments,
                    compactness=compactness,
                    sigma=sigma,
                    convert2lab=slic_convert2lab,
                    multichannel=True,
                    slic_zero=use_slico,
                )
            except TypeError:
                try:
                    segments = slic(
                        slic_input,
                        n_segments=n_segments,
                        compactness=compactness,
                        sigma=sigma,
                        slic_zero=use_slico,
                    )
                except TypeError:
                    segments = slic(
                        slic_input,
                        n_segments=n_segments,
                        compactness=compactness,
                        sigma=sigma,
                    )
    elif algorithm_key == "quickshift":
        try:
            segments = quickshift(
                image_rgb,
                ratio=quickshift_ratio,
                kernel_size=quickshift_kernel_size,
                max_dist=quickshift_max_dist,
                sigma=quickshift_sigma,
                convert2lab=use_lab,
                channel_axis=-1,
            )
        except TypeError:
            try:
                segments = quickshift(
                    image_rgb,
                    ratio=quickshift_ratio,
                    kernel_size=quickshift_kernel_size,
                    max_dist=quickshift_max_dist,
                    sigma=quickshift_sigma,
                    convert2lab=use_lab,
                    multichannel=True,
                )
            except TypeError:
                try:
                    segments = quickshift(
                        image_rgb,
                        ratio=quickshift_ratio,
                        kernel_size=quickshift_kernel_size,
                        max_dist=quickshift_max_dist,
                        sigma=quickshift_sigma,
                        convert2lab=use_lab,
                    )
                except TypeError:
                    segments = quickshift(
                        image_rgb,
                        ratio=quickshift_ratio,
                        kernel_size=quickshift_kernel_size,
                        max_dist=quickshift_max_dist,
                        sigma=quickshift_sigma,
                    )
    else:
        image_for_felzenszwalb = rgb2lab(image_rgb) if use_lab else image_rgb
        try:
            segments = felzenszwalb(
                image_for_felzenszwalb,
                scale=felzenszwalb_scale,
                sigma=felzenszwalb_sigma,
                min_size=felzenszwalb_min_size,
                channel_axis=-1,
            )
        except TypeError:
            try:
                segments = felzenszwalb(
                    image_for_felzenszwalb,
                    scale=felzenszwalb_scale,
                    sigma=felzenszwalb_sigma,
                    min_size=felzenszwalb_min_size,
                    multichannel=True,
                )
            except TypeError:
                segments = felzenszwalb(
                    image_for_felzenszwalb,
                    scale=felzenszwalb_scale,
                    sigma=felzenszwalb_sigma,
                    min_size=felzenszwalb_min_size,
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
    algorithm: str = "slic",
    quickshift_ratio: float = 1.0,
    quickshift_kernel_size: int = 5,
    quickshift_max_dist: float = 10.0,
    quickshift_sigma: float = 0.0,
    felzenszwalb_scale: float = 100.0,
    felzenszwalb_sigma: float = 0.8,
    felzenszwalb_min_size: int = 50,
    texture_enabled: bool = False,
    texture_mode: str = "append_to_color",
    texture_lbp_enabled: bool = False,
    texture_lbp_points: int = 8,
    texture_lbp_radii: list[int] | tuple[int, ...] = (1,),
    texture_lbp_method: str = "uniform",
    texture_lbp_normalize: bool = True,
    texture_gabor_enabled: bool = False,
    texture_gabor_frequencies: list[float] | tuple[float, ...] = (0.1, 0.2),
    texture_gabor_thetas: list[float] | tuple[float, ...] = (0.0, 45.0, 90.0, 135.0),
    texture_gabor_bandwidth: float = 1.0,
    texture_gabor_include_real: bool = False,
    texture_gabor_include_imag: bool = False,
    texture_gabor_include_magnitude: bool = True,
    texture_gabor_normalize: bool = True,
    texture_weight_color: float = 1.0,
    texture_weight_lbp: float = 0.25,
    texture_weight_gabor: float = 0.25,
    boundary_color: tuple[int, int, int, int] = (255, 255, 255, 170),
) -> tuple[np.ndarray, Path]:
    ensure_dir(cache_dir)
    algorithm_key = str(algorithm or "slic").strip().lower()
    if algorithm_key not in {"slic", "slico", "quickshift", "felzenszwalb"}:
        algorithm_key = "slic"
    normalized_texture_mode = str(texture_mode or "append_to_color").strip().lower() or "append_to_color"
    if normalized_texture_mode not in {"append_to_color"}:
        normalized_texture_mode = "append_to_color"
    normalized_texture_lbp_method = str(texture_lbp_method or "uniform").strip().lower() or "uniform"
    if normalized_texture_lbp_method not in {"uniform", "ror", "default"}:
        normalized_texture_lbp_method = "uniform"
    normalized_texture_lbp_radii: list[int] = []
    for value in texture_lbp_radii:
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            normalized_texture_lbp_radii.append(parsed)
    if not normalized_texture_lbp_radii:
        normalized_texture_lbp_radii = [1]
    normalized_texture_gabor_frequencies: list[float] = []
    for value in texture_gabor_frequencies:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            normalized_texture_gabor_frequencies.append(parsed)
    if not normalized_texture_gabor_frequencies:
        normalized_texture_gabor_frequencies = [0.1, 0.2]
    normalized_texture_gabor_thetas: list[float] = []
    for value in texture_gabor_thetas:
        try:
            normalized_texture_gabor_thetas.append(float(value))
        except (TypeError, ValueError):
            continue
    if not normalized_texture_gabor_thetas:
        normalized_texture_gabor_thetas = [0.0, 45.0, 90.0, 135.0]
    normalized_texture_enabled = bool(texture_enabled)
    if algorithm_key not in {"slic", "slico"}:
        normalized_texture_enabled = False
    if normalized_texture_enabled and not (bool(texture_lbp_enabled) or bool(texture_gabor_enabled)):
        normalized_texture_enabled = False
    if bool(texture_gabor_enabled) and not (
        bool(texture_gabor_include_real)
        or bool(texture_gabor_include_imag)
        or bool(texture_gabor_include_magnitude)
    ):
        texture_gabor_include_magnitude = True
    image_stem = image_path.stem
    segments_path, boundary_path = cache_file_paths(cache_dir, image_stem)
    expected_shape = load_image_size(image_path)
    texture_signature = {
        "enabled": normalized_texture_enabled,
        "mode": normalized_texture_mode,
        "lbp": {
            "enabled": bool(texture_lbp_enabled),
            "points": int(texture_lbp_points),
            "radii": normalized_texture_lbp_radii,
            "method": normalized_texture_lbp_method,
            "normalize": bool(texture_lbp_normalize),
        },
        "gabor": {
            "enabled": bool(texture_gabor_enabled),
            "frequencies": normalized_texture_gabor_frequencies,
            "thetas": normalized_texture_gabor_thetas,
            "bandwidth": float(texture_gabor_bandwidth),
            "include_real": bool(texture_gabor_include_real),
            "include_imag": bool(texture_gabor_include_imag),
            "include_magnitude": bool(texture_gabor_include_magnitude),
            "normalize": bool(texture_gabor_normalize),
        },
        "weights": {
            "color": float(texture_weight_color),
            "lbp": float(texture_weight_lbp),
            "gabor": float(texture_weight_gabor),
        },
    }
    setting_signature = (
        f"a={algorithm_key}|n={int(n_segments)}|c={float(compactness):.6f}|s={float(sigma):.6f}|"
        f"cs={str(colorspace).lower()}|"
        f"qr={float(quickshift_ratio):.6f}|qk={int(quickshift_kernel_size)}|"
        f"qmd={float(quickshift_max_dist):.6f}|qs={float(quickshift_sigma):.6f}|"
        f"fs={float(felzenszwalb_scale):.6f}|fss={float(felzenszwalb_sigma):.6f}|"
        f"fms={int(felzenszwalb_min_size)}|tx={json.dumps(texture_signature, sort_keys=True, separators=(',', ':'))}"
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
        segments = _compute_segments(
            image_rgb,
            algorithm_key,
            n_segments,
            compactness,
            sigma,
            colorspace,
            quickshift_ratio,
            quickshift_kernel_size,
            quickshift_max_dist,
            quickshift_sigma,
            felzenszwalb_scale,
            felzenszwalb_sigma,
            felzenszwalb_min_size,
            normalized_texture_enabled,
            normalized_texture_mode,
            bool(texture_lbp_enabled),
            int(texture_lbp_points),
            normalized_texture_lbp_radii,
            normalized_texture_lbp_method,
            bool(texture_lbp_normalize),
            bool(texture_gabor_enabled),
            normalized_texture_gabor_frequencies,
            normalized_texture_gabor_thetas,
            float(texture_gabor_bandwidth),
            bool(texture_gabor_include_real),
            bool(texture_gabor_include_imag),
            bool(texture_gabor_include_magnitude),
            bool(texture_gabor_normalize),
            float(texture_weight_color),
            float(texture_weight_lbp),
            float(texture_weight_gabor),
        )
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

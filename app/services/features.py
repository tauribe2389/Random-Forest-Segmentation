"""Feature extraction utilities in a scikit-image style pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from skimage import color, filters, img_as_float
from skimage.feature import local_binary_pattern, structure_tensor

try:
    from skimage.feature import structure_tensor_eigenvalues as _structure_tensor_eigenvalues
except ImportError:
    from skimage.feature import structure_tensor_eigvals as _structure_tensor_eigenvalues

from .schemas import FeatureConfig

LAWS_1D_KERNELS: dict[str, np.ndarray] = {
    "L5": np.asarray([1, 4, 6, 4, 1], dtype=np.float32),
    "E5": np.asarray([-1, -2, 0, 2, 1], dtype=np.float32),
    "S5": np.asarray([-1, 0, 2, 0, -1], dtype=np.float32),
    "R5": np.asarray([1, -4, 6, -4, 1], dtype=np.float32),
    "W5": np.asarray([-1, 2, 0, -2, 1], dtype=np.float32),
}


def _normalized_lbp_radii(config: FeatureConfig) -> list[int]:
    radii = [int(value) for value in getattr(config, "lbp_radii", []) if int(value) > 0]
    legacy_radius = int(getattr(config, "lbp_radius", 1))
    if not radii:
        return [max(1, legacy_radius)]
    if len(radii) == 1 and radii[0] == 1 and legacy_radius > 1:
        return [legacy_radius]
    seen: set[int] = set()
    ordered: list[int] = []
    for radius in radii:
        if radius in seen:
            continue
        seen.add(radius)
        ordered.append(radius)
    return ordered


def _format_num(value: float | int) -> str:
    return f"{float(value):.3g}"


def _iter_laws_pairs(config: FeatureConfig) -> list[tuple[str, str]]:
    vectors = [str(value).upper().strip() for value in getattr(config, "laws_vectors", [])]
    vectors = [value for value in vectors if value in LAWS_1D_KERNELS]
    if not vectors:
        vectors = ["L5", "E5", "S5", "R5", "W5"]
    include_rotations = bool(getattr(config, "laws_include_rotations", False))
    include_l5l5 = bool(getattr(config, "laws_include_l5l5", False))
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(vectors):
        start_j = 0 if include_rotations else i
        for j in range(start_j, len(vectors)):
            right = vectors[j]
            if not include_l5l5 and left == "L5" and right == "L5":
                continue
            pairs.append((left, right))
            if include_rotations and i != j:
                # Include directional counterpart when rotations are enabled.
                pairs.append((right, left))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def _local_window_from_sigma(sigma: float) -> int:
    radius = max(1, int(round(float(sigma) * 3.0)))
    return radius * 2 + 1


def _structure_eigenvalues(Axx: np.ndarray, Axy: np.ndarray, Ayy: np.ndarray) -> np.ndarray:
    try:
        return _structure_tensor_eigenvalues((Axx, Axy, Ayy))
    except TypeError:
        return _structure_tensor_eigenvalues(Axx, Axy, Ayy)


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
        for radius in _normalized_lbp_radii(config):
            names.append(f"lbp_p{config.lbp_points}_r{radius}")
    if config.use_gabor:
        if not (config.gabor_include_real or config.gabor_include_imag or config.gabor_include_magnitude):
            raise ValueError("Gabor enabled but no response components selected.")
        for frequency in config.gabor_frequencies:
            for theta in config.gabor_thetas:
                prefix = f"gabor_f{_format_num(frequency)}_t{_format_num(theta)}"
                if config.gabor_include_real:
                    names.append(f"{prefix}_real")
                if config.gabor_include_imag:
                    names.append(f"{prefix}_imag")
                if config.gabor_include_magnitude:
                    names.append(f"{prefix}_mag")
    if config.use_laws:
        window = int(config.laws_energy_window)
        for left, right in _iter_laws_pairs(config):
            names.append(f"laws_{left}{right}_w{window}")
    if config.use_structure_tensor:
        if config.structure_tensor_include_eigenvalues:
            names.extend(["structure_tensor_lambda1", "structure_tensor_lambda2"])
        if config.structure_tensor_include_coherence:
            names.append("structure_tensor_coherence")
        if config.structure_tensor_include_orientation:
            names.append("structure_tensor_orientation")
    if config.use_multiscale_local_stats:
        for sigma in config.local_stats_sigmas:
            sigma_label = _format_num(sigma)
            if config.local_stats_include_mean:
                names.append(f"local_stats_sigma{sigma_label}_mean")
            if config.local_stats_include_std:
                names.append(f"local_stats_sigma{sigma_label}_std")
            if config.local_stats_include_min:
                names.append(f"local_stats_sigma{sigma_label}_min")
            if config.local_stats_include_max:
                names.append(f"local_stats_sigma{sigma_label}_max")
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
        for radius in _normalized_lbp_radii(config):
            lbp = local_binary_pattern(
                gray,
                P=config.lbp_points,
                R=radius,
                method="uniform",
            )
            lbp = lbp.astype(np.float32)
            max_lbp = float(lbp.max()) if lbp.size else 1.0
            if max_lbp > 0:
                lbp = lbp / max_lbp
            channels.append(lbp)
            names.append(f"lbp_p{config.lbp_points}_r{radius}")

    if config.use_gabor:
        for frequency in config.gabor_frequencies:
            freq = float(frequency)
            for theta_value in config.gabor_thetas:
                theta = np.deg2rad(float(theta_value))
                response_real, response_imag = filters.gabor(
                    gray,
                    frequency=freq,
                    theta=theta,
                    bandwidth=float(config.gabor_bandwidth),
                )
                prefix = f"gabor_f{_format_num(freq)}_t{_format_num(theta_value)}"
                if config.gabor_include_real:
                    channels.append(np.asarray(response_real, dtype=np.float32))
                    names.append(f"{prefix}_real")
                if config.gabor_include_imag:
                    channels.append(np.asarray(response_imag, dtype=np.float32))
                    names.append(f"{prefix}_imag")
                if config.gabor_include_magnitude:
                    magnitude = np.hypot(response_real, response_imag).astype(np.float32)
                    channels.append(magnitude)
                    names.append(f"{prefix}_mag")

    if config.use_laws:
        window = max(1, int(config.laws_energy_window))
        use_abs = bool(config.laws_use_abs)
        for left, right in _iter_laws_pairs(config):
            kernel = np.outer(LAWS_1D_KERNELS[left], LAWS_1D_KERNELS[right]).astype(np.float32)
            response = ndimage.convolve(gray.astype(np.float32), kernel, mode="reflect")
            energy_input = np.abs(response) if use_abs else np.square(response)
            energy = ndimage.uniform_filter(energy_input, size=window, mode="reflect").astype(np.float32)
            channels.append(energy)
            names.append(f"laws_{left}{right}_w{window}")

    if config.use_structure_tensor:
        if not (
            config.structure_tensor_include_eigenvalues
            or config.structure_tensor_include_coherence
            or config.structure_tensor_include_orientation
        ):
            raise ValueError("Structure tensor enabled but no outputs selected.")
        Axx, Axy, Ayy = structure_tensor(
            gray,
            sigma=float(config.structure_tensor_sigma),
            mode="reflect",
        )
        rho = float(config.structure_tensor_rho)
        if rho > 0:
            Axx = filters.gaussian(Axx, sigma=rho, preserve_range=True)
            Axy = filters.gaussian(Axy, sigma=rho, preserve_range=True)
            Ayy = filters.gaussian(Ayy, sigma=rho, preserve_range=True)
        eigvals = _structure_eigenvalues(Axx, Axy, Ayy)
        lambda1 = np.asarray(eigvals[0], dtype=np.float32)
        lambda2 = np.asarray(eigvals[1], dtype=np.float32)
        if config.structure_tensor_include_eigenvalues:
            channels.extend([lambda1, lambda2])
            names.extend(["structure_tensor_lambda1", "structure_tensor_lambda2"])
        if config.structure_tensor_include_coherence:
            coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)
            channels.append(np.asarray(coherence, dtype=np.float32))
            names.append("structure_tensor_coherence")
        if config.structure_tensor_include_orientation:
            orientation = 0.5 * np.arctan2(2.0 * Axy, Axx - Ayy)
            # Normalize [-pi/2, pi/2] to [0, 1] for model stability.
            orientation = ((orientation + (np.pi / 2.0)) / np.pi).astype(np.float32)
            channels.append(orientation)
            names.append("structure_tensor_orientation")

    if config.use_multiscale_local_stats:
        if not (
            config.local_stats_include_mean
            or config.local_stats_include_std
            or config.local_stats_include_min
            or config.local_stats_include_max
        ):
            raise ValueError("Multi-scale local stats enabled but no outputs selected.")
        for sigma in config.local_stats_sigmas:
            sigma_value = float(sigma)
            sigma_label = _format_num(sigma_value)
            local_mean = filters.gaussian(gray, sigma=sigma_value, preserve_range=True).astype(np.float32)
            if config.local_stats_include_mean:
                channels.append(local_mean)
                names.append(f"local_stats_sigma{sigma_label}_mean")
            if config.local_stats_include_std:
                local_sq_mean = filters.gaussian(gray * gray, sigma=sigma_value, preserve_range=True).astype(np.float32)
                local_var = np.maximum(local_sq_mean - np.square(local_mean), 0.0)
                local_std = np.sqrt(local_var).astype(np.float32)
                channels.append(local_std)
                names.append(f"local_stats_sigma{sigma_label}_std")
            if config.local_stats_include_min or config.local_stats_include_max:
                window = _local_window_from_sigma(sigma_value)
                if config.local_stats_include_min:
                    local_min = ndimage.minimum_filter(gray, size=window, mode="reflect").astype(np.float32)
                    channels.append(local_min)
                    names.append(f"local_stats_sigma{sigma_label}_min")
                if config.local_stats_include_max:
                    local_max = ndimage.maximum_filter(gray, size=window, mode="reflect").astype(np.float32)
                    channels.append(local_max)
                    names.append(f"local_stats_sigma{sigma_label}_max")

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

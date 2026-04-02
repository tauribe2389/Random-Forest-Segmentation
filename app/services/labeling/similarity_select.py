from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from skimage.color import rgb2gray, rgb2lab
from skimage.feature import local_binary_pattern

from .image_io import ensure_dir, load_image_rgb


@dataclass(frozen=True)
class SimilarityFeatureConfig:
    lbp_points: int = 8
    lbp_radius: int = 1
    lbp_method: str = "uniform"

    def signature_payload(self) -> dict[str, Any]:
        return {
            "lbp_points": int(self.lbp_points),
            "lbp_radius": int(self.lbp_radius),
            "lbp_method": str(self.lbp_method),
        }


@dataclass
class SimilarityFeatureCache:
    lab_means: np.ndarray
    lbp_hist: np.ndarray
    pixel_counts: np.ndarray
    cache_key: str
    cache_path: Path
    cache_hit: bool


def normalize_feature_config(payload: Any) -> SimilarityFeatureConfig:
    values = payload if isinstance(payload, dict) else {}
    try:
        lbp_points = int(float(values.get("lbp_points", 8)))
    except (TypeError, ValueError):
        lbp_points = 8
    lbp_points = max(4, min(64, lbp_points))

    try:
        lbp_radius = int(float(values.get("lbp_radius", 1)))
    except (TypeError, ValueError):
        lbp_radius = 1
    lbp_radius = max(1, min(16, lbp_radius))

    lbp_method = str(values.get("lbp_method", "uniform")).strip().lower() or "uniform"
    if lbp_method not in {"uniform", "ror", "default"}:
        lbp_method = "uniform"

    return SimilarityFeatureConfig(
        lbp_points=lbp_points,
        lbp_radius=lbp_radius,
        lbp_method=lbp_method,
    )


@dataclass(frozen=True)
class SimilarityQueryConfig:
    color_enabled: bool = True
    texture_enabled: bool = True
    color_threshold: float = 18.0
    texture_threshold: float = 0.35


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    candidate = str(value).strip().lower()
    if candidate in {"1", "true", "yes", "on", "y"}:
        return True
    if candidate in {"0", "false", "no", "off", "n"}:
        return False
    return default


def normalize_query_config(payload: Any) -> SimilarityQueryConfig:
    values = payload if isinstance(payload, dict) else {}
    color_enabled = _as_bool(values.get("color_enabled", True), default=True)
    texture_enabled = _as_bool(values.get("texture_enabled", True), default=True)
    if not color_enabled and not texture_enabled:
        color_enabled = True

    try:
        color_threshold = float(values.get("color_threshold", 18.0))
    except (TypeError, ValueError):
        color_threshold = 18.0
    color_threshold = max(0.0, min(200.0, color_threshold))

    try:
        texture_threshold = float(values.get("texture_threshold", 0.35))
    except (TypeError, ValueError):
        texture_threshold = 0.35
    texture_threshold = max(0.0, min(10.0, texture_threshold))
    return SimilarityQueryConfig(
        color_enabled=color_enabled,
        texture_enabled=texture_enabled,
        color_threshold=color_threshold,
        texture_threshold=texture_threshold,
    )


def _signature_hash(parts: dict[str, Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def feature_cache_key(
    *,
    image_stem: str,
    superpixel_signature: str,
    feature_config: SimilarityFeatureConfig,
) -> str:
    parts = {
        "image_stem": str(image_stem),
        "superpixel_signature": str(superpixel_signature),
        "feature": feature_config.signature_payload(),
    }
    return _signature_hash(parts)


def feature_cache_path(
    *,
    cache_dir: Path,
    image_stem: str,
    cache_key: str,
) -> Path:
    feature_dir = (cache_dir / "simselect").resolve()
    ensure_dir(feature_dir)
    return feature_dir / f"{image_stem}__simfeat__{cache_key}.npz"


def superpixel_signature_from_settings(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _compute_lab_means(image_rgb: np.ndarray, segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = rgb2lab(image_rgb).astype(np.float32, copy=False)
    labels = segments.astype(np.int64, copy=False).ravel()
    max_label = int(labels.max(initial=-1))
    segment_count = max_label + 1
    if segment_count <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    pixel_counts = np.bincount(labels, minlength=segment_count).astype(np.int32, copy=False)
    safe_counts = np.maximum(pixel_counts.astype(np.float32), 1.0)
    lab_means = np.zeros((segment_count, 3), dtype=np.float32)
    for channel_index in range(3):
        channel_values = lab[:, :, channel_index].ravel()
        sums = np.bincount(labels, weights=channel_values, minlength=segment_count).astype(np.float32, copy=False)
        lab_means[:, channel_index] = sums / safe_counts
    return lab_means, pixel_counts


def _lbp_bin_count(points: int, method: str, lbp_codes: np.ndarray) -> int:
    if method == "uniform":
        return int(points + 2)
    if lbp_codes.size <= 0:
        return 1
    return max(1, int(lbp_codes.max(initial=0)) + 1)


def _compute_lbp_histograms(
    image_rgb: np.ndarray,
    segments: np.ndarray,
    *,
    lbp_points: int,
    lbp_radius: int,
    lbp_method: str,
    segment_count: int,
) -> np.ndarray:
    if segment_count <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    gray = rgb2gray(image_rgb).astype(np.float32, copy=False)
    gray_u8 = np.clip(np.rint(gray * 255.0), 0.0, 255.0).astype(np.uint8, copy=False)
    lbp_raw = local_binary_pattern(
        gray_u8,
        P=int(lbp_points),
        R=float(lbp_radius),
        method=str(lbp_method),
    )
    lbp_codes = np.rint(lbp_raw).astype(np.int64, copy=False)
    bin_count = _lbp_bin_count(int(lbp_points), str(lbp_method), lbp_codes)
    lbp_codes = np.clip(lbp_codes, 0, max(0, bin_count - 1))

    labels = segments.astype(np.int64, copy=False).ravel()
    code_values = lbp_codes.ravel()
    hist = np.zeros((segment_count, bin_count), dtype=np.float32)
    for segment_id in range(segment_count):
        segment_mask = labels == segment_id
        if not np.any(segment_mask):
            continue
        codes = code_values[segment_mask]
        counts = np.bincount(codes, minlength=bin_count).astype(np.float32, copy=False)
        total = float(counts.sum())
        if total > 0:
            hist[segment_id] = counts / total
    return hist


def load_or_create_feature_cache(
    *,
    image_path: Path,
    cache_dir: Path,
    segments: np.ndarray,
    superpixel_signature: str,
    feature_config: SimilarityFeatureConfig,
) -> SimilarityFeatureCache:
    image_stem = str(image_path.stem)
    cache_key = feature_cache_key(
        image_stem=image_stem,
        superpixel_signature=superpixel_signature,
        feature_config=feature_config,
    )
    cache_path = feature_cache_path(cache_dir=cache_dir, image_stem=image_stem, cache_key=cache_key)

    expected_shape = tuple(segments.shape)
    expected_segment_count = int(segments.max(initial=-1)) + 1
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as loaded:
                cached_shape = tuple(np.array(loaded["segments_shape"]).astype(np.int32).tolist())
                cached_signature = str(loaded["superpixel_signature"].item())
                cached_feature_signature = str(loaded["feature_signature"].item())
                cached_labs = loaded["lab_means"].astype(np.float32, copy=False)
                cached_lbp = loaded["lbp_hist"].astype(np.float32, copy=False)
                cached_counts = loaded["pixel_counts"].astype(np.int32, copy=False)
            if (
                cached_shape == expected_shape
                and int(cached_counts.shape[0]) == expected_segment_count
                and cached_labs.shape[0] == expected_segment_count
                and cached_lbp.shape[0] == expected_segment_count
                and cached_signature == str(superpixel_signature)
                and cached_feature_signature
                == json.dumps(feature_config.signature_payload(), sort_keys=True, separators=(",", ":"))
            ):
                return SimilarityFeatureCache(
                    lab_means=cached_labs,
                    lbp_hist=cached_lbp,
                    pixel_counts=cached_counts,
                    cache_key=cache_key,
                    cache_path=cache_path,
                    cache_hit=True,
                )
        except Exception:
            pass

    image_rgb = load_image_rgb(image_path)
    lab_means, pixel_counts = _compute_lab_means(image_rgb, segments)
    lbp_hist = _compute_lbp_histograms(
        image_rgb,
        segments,
        lbp_points=int(feature_config.lbp_points),
        lbp_radius=int(feature_config.lbp_radius),
        lbp_method=str(feature_config.lbp_method),
        segment_count=int(pixel_counts.shape[0]),
    )
    feature_signature = json.dumps(feature_config.signature_payload(), sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        cache_path,
        segments_shape=np.array(expected_shape, dtype=np.int32),
        superpixel_signature=np.array(str(superpixel_signature)),
        feature_signature=np.array(feature_signature),
        lab_means=lab_means.astype(np.float32, copy=False),
        lbp_hist=lbp_hist.astype(np.float32, copy=False),
        pixel_counts=pixel_counts.astype(np.int32, copy=False),
    )
    return SimilarityFeatureCache(
        lab_means=lab_means,
        lbp_hist=lbp_hist,
        pixel_counts=pixel_counts,
        cache_key=cache_key,
        cache_path=cache_path,
        cache_hit=False,
    )


def chi_square_distance(seed_hist: np.ndarray, hist_matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    seed = np.asarray(seed_hist, dtype=np.float32)
    matrix = np.asarray(hist_matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("hist_matrix must be 2D")
    if seed.ndim != 1:
        raise ValueError("seed_hist must be 1D")
    if matrix.shape[1] != seed.shape[0]:
        raise ValueError("histogram dimensions do not match")
    numerator = (matrix - seed[np.newaxis, :]) ** 2
    denominator = matrix + seed[np.newaxis, :] + float(eps)
    return 0.5 * np.sum(numerator / denominator, axis=1)


def candidate_segments_from_roi(segments: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    segment_array = np.asarray(segments)
    roi_array = np.asarray(roi_mask, dtype=bool)
    if segment_array.shape != roi_array.shape:
        raise ValueError("segments and roi_mask must have matching shape")
    if roi_array.size <= 0 or not np.any(roi_array):
        return np.zeros((0,), dtype=np.int32)
    ids = np.unique(segment_array[roi_array])
    return ids.astype(np.int32, copy=False)


def select_matching_superpixels(
    *,
    candidate_segment_ids: np.ndarray,
    seed_segment_id: int,
    lab_means: np.ndarray,
    lbp_hist: np.ndarray,
    query_config: SimilarityQueryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(candidate_segment_ids, dtype=np.int64).reshape(-1)
    lab_matrix = np.asarray(lab_means, dtype=np.float32)
    lbp_matrix = np.asarray(lbp_hist, dtype=np.float32)
    if lab_matrix.ndim != 2 or lab_matrix.shape[1] != 3:
        raise ValueError("lab_means must be shape [segments, 3]")
    if lbp_matrix.ndim != 2:
        raise ValueError("lbp_hist must be shape [segments, bins]")
    if lab_matrix.shape[0] != lbp_matrix.shape[0]:
        raise ValueError("lab_means and lbp_hist segment counts must match")

    segment_count = int(lab_matrix.shape[0])
    seed_id = int(seed_segment_id)
    if seed_id < 0 or seed_id >= segment_count:
        raise ValueError("seed_segment_id is out of range")
    if candidates.size <= 0:
        empty = np.zeros((0,), dtype=np.float32)
        return np.zeros((0,), dtype=np.int32), empty, empty

    valid = (candidates >= 0) & (candidates < segment_count)
    if not np.any(valid):
        empty = np.zeros((0,), dtype=np.float32)
        return np.zeros((0,), dtype=np.int32), empty, empty
    candidates = candidates[valid].astype(np.int32, copy=False)

    seed_lab = lab_matrix[seed_id]
    color_distances = np.linalg.norm(lab_matrix[candidates] - seed_lab[np.newaxis, :], axis=1).astype(
        np.float32,
        copy=False,
    )
    if lbp_matrix.shape[1] > 0:
        texture_distances = chi_square_distance(lbp_matrix[seed_id], lbp_matrix[candidates]).astype(
            np.float32,
            copy=False,
        )
    else:
        texture_distances = np.zeros((candidates.shape[0],), dtype=np.float32)

    matched = np.ones((candidates.shape[0],), dtype=bool)
    if bool(query_config.color_enabled):
        matched &= color_distances <= float(query_config.color_threshold)
    if bool(query_config.texture_enabled):
        matched &= texture_distances <= float(query_config.texture_threshold)
    return candidates[matched], color_distances, texture_distances

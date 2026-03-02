"""Mask thresholding and cleanup utilities for analysis-to-draft bootstrapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.measure import label as connected_components
from skimage.morphology import binary_dilation, disk, remove_small_holes


@dataclass(frozen=True)
class BootstrapCleanupConfig:
    remove_small_blobs: bool = False
    min_blob_px: int = 200
    fill_small_holes: bool = False
    max_hole_px: int = 200
    boundary_ignore: bool = False
    boundary_px: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "remove_small_blobs": bool(self.remove_small_blobs),
            "min_blob_px": int(max(0, self.min_blob_px)),
            "fill_small_holes": bool(self.fill_small_holes),
            "max_hole_px": int(max(0, self.max_hole_px)),
            "boundary_ignore": bool(self.boundary_ignore),
            "boundary_px": int(max(0, self.boundary_px)),
        }


def threshold_label_map(
    label_map: np.ndarray,
    conf_u8: np.ndarray | None,
    *,
    threshold_pct: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply confidence thresholding to a class-index map.

    Pixels with confidence below the threshold become class 0.
    """
    normalized_pct = int(max(0, min(100, int(threshold_pct))))
    threshold_u8 = int(round((normalized_pct / 100.0) * 255.0))
    output = np.asarray(label_map).copy()
    stats: dict[str, Any] = {
        "threshold_pct": normalized_pct,
        "threshold_u8": threshold_u8,
        "applied": False,
        "cleared_pixels": 0,
    }
    if conf_u8 is None:
        return output, stats
    conf_arr = np.asarray(conf_u8)
    if conf_arr.shape != output.shape:
        raise ValueError(
            f"Confidence shape mismatch: conf={conf_arr.shape}, label_map={output.shape}"
        )
    low_conf = conf_arr < threshold_u8
    if np.any(low_conf):
        output[low_conf] = 0
    stats["applied"] = True
    stats["cleared_pixels"] = int(low_conf.sum())
    return output, stats


def remove_small_blobs_by_class(label_map: np.ndarray, min_blob_px: int) -> tuple[np.ndarray, int]:
    """Set connected components smaller than ``min_blob_px`` to class 0."""
    min_size = int(max(0, int(min_blob_px)))
    if min_size <= 0:
        return np.asarray(label_map).copy(), 0

    output = np.asarray(label_map).copy()
    removed = 0
    class_values = np.unique(output)
    class_values = class_values[class_values > 0]
    for class_id in class_values.tolist():
        class_mask = output == int(class_id)
        if not np.any(class_mask):
            continue
        cc_map = connected_components(class_mask, connectivity=1)
        if int(cc_map.max()) <= 0:
            continue
        counts = np.bincount(cc_map.ravel())
        small_component_ids = np.where((counts < min_size) & (np.arange(counts.size) > 0))[0]
        if small_component_ids.size == 0:
            continue
        remove_mask = np.isin(cc_map, small_component_ids)
        removed += int(remove_mask.sum())
        output[remove_mask] = 0
    return output, removed


def fill_small_holes_by_class(label_map: np.ndarray, max_hole_px: int) -> tuple[np.ndarray, int]:
    """Fill small holes per class; only unlabeled pixels are filled."""
    max_hole = int(max(0, int(max_hole_px)))
    if max_hole <= 0:
        return np.asarray(label_map).copy(), 0

    output = np.asarray(label_map).copy()
    filled_pixels = 0
    class_values = np.unique(output)
    class_values = class_values[class_values > 0]
    for class_id in class_values.tolist():
        class_mask = output == int(class_id)
        if not np.any(class_mask):
            continue
        filled = remove_small_holes(class_mask, area_threshold=max_hole, connectivity=1)
        write_mask = filled & (output == 0)
        if not np.any(write_mask):
            continue
        output[write_mask] = int(class_id)
        filled_pixels += int(write_mask.sum())
    return output, filled_pixels


def boundary_ignore_band(label_map: np.ndarray, boundary_px: int) -> tuple[np.ndarray, int]:
    """Set a dilated label-boundary band to class 0."""
    width = int(max(0, int(boundary_px)))
    output = np.asarray(label_map).copy()
    if width <= 0:
        return output, 0

    edges = np.zeros(output.shape, dtype=bool)
    if output.shape[1] > 1:
        diff_h = output[:, 1:] != output[:, :-1]
        edges[:, 1:] |= diff_h
        edges[:, :-1] |= diff_h
    if output.shape[0] > 1:
        diff_v = output[1:, :] != output[:-1, :]
        edges[1:, :] |= diff_v
        edges[:-1, :] |= diff_v
    if not np.any(edges):
        return output, 0

    try:
        band = binary_dilation(edges, footprint=disk(width))
    except TypeError:
        # skimage<0.19 uses `selem` instead of `footprint`.
        band = binary_dilation(edges, selem=disk(width))
    ignored = int(np.count_nonzero(output[band] != 0))
    output[band] = 0
    return output, ignored


def cleanup_label_map(
    label_map: np.ndarray,
    config: BootstrapCleanupConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply deterministic cleanup operations in fixed order."""
    output = np.asarray(label_map).copy()
    stats = {
        "remove_small_blobs_applied": bool(config.remove_small_blobs),
        "remove_small_blobs_removed_pixels": 0,
        "fill_small_holes_applied": bool(config.fill_small_holes),
        "fill_small_holes_filled_pixels": 0,
        "boundary_ignore_applied": bool(config.boundary_ignore),
        "boundary_ignore_cleared_pixels": 0,
    }

    if config.remove_small_blobs:
        output, removed = remove_small_blobs_by_class(output, config.min_blob_px)
        stats["remove_small_blobs_removed_pixels"] = int(removed)

    if config.fill_small_holes:
        output, filled = fill_small_holes_by_class(output, config.max_hole_px)
        stats["fill_small_holes_filled_pixels"] = int(filled)

    if config.boundary_ignore:
        output, ignored = boundary_ignore_band(output, config.boundary_px)
        stats["boundary_ignore_cleared_pixels"] = int(ignored)

    return output, stats

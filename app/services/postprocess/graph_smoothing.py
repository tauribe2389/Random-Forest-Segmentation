"""Deterministic graph-energy smoothing on inference-time SLIC superpixels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from skimage import color, filters, measure, morphology
from skimage.segmentation import slic


EPS = 1e-8


@dataclass
class GraphSmoothConfig:
    """Configuration for graph-energy smoothing."""

    enabled: bool = False
    slic_target_superpixel_area_px: int = 900
    slic_n_segments: int | None = None
    slic_compactness: float = 10.0
    slic_sigma: float = 0.0
    lambda_smooth: float = 0.7
    edge_awareness: float = 0.5
    iterations: int = 8
    temperature: float = 1.0
    min_region_area_px: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GraphSmoothConfig":
        source = payload or {}
        return cls(
            enabled=bool(source.get("enabled", False)),
            slic_target_superpixel_area_px=int(source.get("slic_target_superpixel_area_px", 900)),
            slic_n_segments=(
                int(source["slic_n_segments"])
                if source.get("slic_n_segments") not in {None, ""}
                else None
            ),
            slic_compactness=float(source.get("slic_compactness", 10.0)),
            slic_sigma=float(source.get("slic_sigma", 0.0)),
            lambda_smooth=float(source.get("lambda_smooth", 0.7)),
            edge_awareness=float(source.get("edge_awareness", 0.5)),
            iterations=int(source.get("iterations", 8)),
            temperature=float(source.get("temperature", 1.0)),
            min_region_area_px=int(source.get("min_region_area_px", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_float_rgb(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.dtype == np.uint8:
        return image_rgb.astype(np.float32) / 255.0
    rgb = image_rgb.astype(np.float32)
    max_value = float(np.max(rgb)) if rgb.size else 1.0
    if max_value > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _resolve_slic_n_segments(height: int, width: int, config: GraphSmoothConfig) -> int:
    if config.slic_n_segments is not None and int(config.slic_n_segments) > 0:
        return int(np.clip(int(config.slic_n_segments), 100, 5000))
    total_pixels = max(1, int(height) * int(width))
    area = max(1, int(config.slic_target_superpixel_area_px))
    estimated = int(round(total_pixels / float(area)))
    return int(np.clip(estimated, 100, 5000))


def _relabel_contiguous(segments: np.ndarray) -> tuple[np.ndarray, int]:
    labels, inverse = np.unique(segments.astype(np.int64), return_inverse=True)
    relabeled = inverse.reshape(segments.shape).astype(np.int32)
    return relabeled, int(labels.size)


def _build_superpixels(image_rgb: np.ndarray, config: GraphSmoothConfig) -> tuple[np.ndarray, int]:
    height, width = image_rgb.shape[:2]
    n_segments = _resolve_slic_n_segments(height, width, config)
    slic_kwargs = {
        "n_segments": n_segments,
        "compactness": max(0.01, float(config.slic_compactness)),
        "sigma": max(0.0, float(config.slic_sigma)),
        "enforce_connectivity": True,
    }
    try:
        segments = slic(
            image_rgb,
            start_label=0,
            channel_axis=-1,
            **slic_kwargs,
        )
    except TypeError:
        # skimage<=0.18 compatibility
        segments = slic(
            image_rgb,
            multichannel=True,
            **slic_kwargs,
        )
    return _relabel_contiguous(segments)


def _region_means(values: np.ndarray, segments: np.ndarray, n_segments: int) -> np.ndarray:
    flat_segments = segments.reshape(-1)
    flat_values = values.reshape(-1, values.shape[-1]).astype(np.float64)
    counts = np.bincount(flat_segments, minlength=n_segments).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    means = np.zeros((n_segments, flat_values.shape[1]), dtype=np.float64)
    for channel in range(flat_values.shape[1]):
        sums = np.bincount(flat_segments, weights=flat_values[:, channel], minlength=n_segments)
        means[:, channel] = sums / counts
    return means


def _region_probabilities(proba: np.ndarray, segments: np.ndarray, n_segments: int) -> np.ndarray:
    flat_segments = segments.reshape(-1)
    flat_proba = proba.reshape(-1, proba.shape[-1]).astype(np.float64)
    counts = np.bincount(flat_segments, minlength=n_segments).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    region_proba = np.zeros((n_segments, flat_proba.shape[1]), dtype=np.float64)
    for class_idx in range(flat_proba.shape[1]):
        sums = np.bincount(flat_segments, weights=flat_proba[:, class_idx], minlength=n_segments)
        region_proba[:, class_idx] = sums / counts
    region_proba = np.clip(region_proba, EPS, 1.0)
    region_proba /= np.maximum(region_proba.sum(axis=1, keepdims=True), EPS)
    return region_proba.astype(np.float32)


def _collect_boundary_edges(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges_h_a = segments[:, :-1]
    edges_h_b = segments[:, 1:]
    mask_h = edges_h_a != edges_h_b

    edges_v_a = segments[:-1, :]
    edges_v_b = segments[1:, :]
    mask_v = edges_v_a != edges_v_b

    a_values = np.concatenate([edges_h_a[mask_h].reshape(-1), edges_v_a[mask_v].reshape(-1)])
    b_values = np.concatenate([edges_h_b[mask_h].reshape(-1), edges_v_b[mask_v].reshape(-1)])
    if a_values.size == 0:
        return np.array([], dtype=np.int64), mask_h.astype(bool), mask_v.astype(bool)

    low = np.minimum(a_values, b_values).astype(np.int64)
    high = np.maximum(a_values, b_values).astype(np.int64)
    max_id = int(np.max(segments)) + 1
    codes = low * max_id + high
    return codes.astype(np.int64), mask_h.astype(bool), mask_v.astype(bool)


def _boundary_gradient_by_edge(
    segments: np.ndarray,
    luminance_grad: np.ndarray,
    edge_codes: np.ndarray,
    mask_h: np.ndarray,
    mask_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if edge_codes.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    grad_h = 0.5 * (luminance_grad[:, :-1] + luminance_grad[:, 1:])
    grad_v = 0.5 * (luminance_grad[:-1, :] + luminance_grad[1:, :])
    grad_values = np.concatenate([grad_h[mask_h].reshape(-1), grad_v[mask_v].reshape(-1)]).astype(np.float64)

    unique_codes, inverse = np.unique(edge_codes, return_inverse=True)
    grad_sum = np.bincount(inverse, weights=grad_values, minlength=unique_codes.size)
    grad_count = np.bincount(inverse, minlength=unique_codes.size).astype(np.float64)
    grad_mean = grad_sum / np.maximum(grad_count, 1.0)

    max_id = int(np.max(segments)) + 1
    src = (unique_codes // max_id).astype(np.int32)
    dst = (unique_codes % max_id).astype(np.int32)
    return src, dst, grad_mean


def _edge_weights(
    image_rgb: np.ndarray,
    segments: np.ndarray,
    n_segments: int,
    edge_awareness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_codes, mask_h, mask_v = _collect_boundary_edges(segments)
    if edge_codes.size == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )

    image_lab = color.rgb2lab(image_rgb)
    region_lab = _region_means(image_lab, segments, n_segments)
    luminance = image_lab[:, :, 0].astype(np.float32)
    luminance_grad = filters.sobel(luminance).astype(np.float32)

    src, dst, grad_mean = _boundary_gradient_by_edge(
        segments=segments,
        luminance_grad=luminance_grad,
        edge_codes=edge_codes,
        mask_h=mask_h,
        mask_v=mask_v,
    )
    if src.size == 0:
        return src, dst, np.array([], dtype=np.float64)

    color_diff = np.linalg.norm(region_lab[src] - region_lab[dst], axis=1)
    color_scale = float(np.median(color_diff[color_diff > 0])) if np.any(color_diff > 0) else 1.0
    grad_scale = float(np.median(grad_mean[grad_mean > 0])) if np.any(grad_mean > 0) else 1.0
    color_scale = max(color_scale, EPS)
    grad_scale = max(grad_scale, EPS)

    base_similarity = np.exp(-color_diff / color_scale)
    boundary_term = np.exp(-grad_mean / grad_scale)
    awareness = float(np.clip(edge_awareness, 0.0, 1.0))
    weights = (1.0 - awareness) * base_similarity + awareness * (base_similarity * boundary_term)
    weights = np.clip(weights, 0.0, 1.0).astype(np.float64)
    return src, dst, weights


def _build_neighbor_lists(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    n_segments: int,
) -> list[list[tuple[int, float]]]:
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_segments)]
    for idx in range(src.size):
        i = int(src[idx])
        j = int(dst[idx])
        w = float(weights[idx])
        neighbors[i].append((j, w))
        neighbors[j].append((i, w))
    return neighbors


def _compute_energy(
    unary_energy: np.ndarray,
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    lambda_smooth: float,
) -> float:
    unary = float(np.sum(unary_energy[np.arange(labels.size), labels]))
    if src.size == 0:
        return unary
    pairwise = float(np.sum(weights * (labels[src] != labels[dst])))
    return unary + float(lambda_smooth) * pairwise


def _icm_optimize(
    unary_energy: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    lambda_smooth: float,
    iterations: int,
) -> tuple[np.ndarray, list[float], list[int]]:
    n_segments, n_classes = unary_energy.shape
    labels = np.argmin(unary_energy, axis=1).astype(np.int32)
    neighbors = _build_neighbor_lists(src, dst, weights, n_segments)

    energy_trace = [
        _compute_energy(unary_energy, labels, src, dst, weights, lambda_smooth)
    ]
    change_trace: list[int] = []

    for _ in range(max(1, int(iterations))):
        changes = 0
        for node in range(n_segments):
            linked = neighbors[node]
            if linked:
                neighbor_idx = np.array([n for n, _ in linked], dtype=np.int32)
                neighbor_w = np.array([w for _, w in linked], dtype=np.float64)
                label_mass = np.bincount(
                    labels[neighbor_idx],
                    weights=neighbor_w,
                    minlength=n_classes,
                )
                penalty = np.sum(neighbor_w) - label_mass
            else:
                penalty = np.zeros(n_classes, dtype=np.float64)
            node_energy = unary_energy[node] + float(lambda_smooth) * penalty
            next_label = int(np.argmin(node_energy))
            if next_label != int(labels[node]):
                labels[node] = next_label
                changes += 1
        change_trace.append(int(changes))
        energy_trace.append(
            _compute_energy(unary_energy, labels, src, dst, weights, lambda_smooth)
        )
        if changes == 0:
            break
    return labels, energy_trace, change_trace


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(logits)
    return exps / denom


def _refined_region_probabilities(
    unary_energy: np.ndarray,
    labels: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    lambda_smooth: float,
    temperature: float,
) -> np.ndarray:
    n_segments, n_classes = unary_energy.shape
    neighbors = _build_neighbor_lists(src, dst, weights, n_segments)
    temp = max(float(temperature), EPS)
    refined = np.zeros((n_segments, n_classes), dtype=np.float32)
    for node in range(n_segments):
        linked = neighbors[node]
        if linked:
            neighbor_idx = np.array([n for n, _ in linked], dtype=np.int32)
            neighbor_w = np.array([w for _, w in linked], dtype=np.float64)
            label_mass = np.bincount(
                labels[neighbor_idx],
                weights=neighbor_w,
                minlength=n_classes,
            )
            smooth_penalty = np.sum(neighbor_w) - label_mass
        else:
            smooth_penalty = np.zeros(n_classes, dtype=np.float64)
        logits = (-unary_energy[node] / temp) - (float(lambda_smooth) * smooth_penalty)
        refined[node] = _softmax(logits).astype(np.float32)
    row_sum = np.maximum(refined.sum(axis=1, keepdims=True), EPS)
    refined /= row_sum
    return refined


def _cleanup_tiny_regions(label_map: np.ndarray, min_region_area_px: int) -> np.ndarray:
    min_area = int(min_region_area_px)
    if min_area <= 0:
        return label_map

    output = label_map.copy()
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for class_idx in np.unique(label_map):
        mask = output == class_idx
        if not np.any(mask):
            continue
        cc = measure.label(mask, connectivity=1)
        for component_id in range(1, int(cc.max()) + 1):
            component = cc == component_id
            area = int(np.sum(component))
            if area >= min_area:
                continue
            dilated = morphology.binary_dilation(component, structure)
            border = dilated & (~component)
            neighbor_labels = output[border]
            if neighbor_labels.size == 0:
                continue
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            dominant = int(unique[np.argmax(counts)])
            output[component] = dominant
    return output


def graph_energy_smooth(
    image_rgb: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    config: GraphSmoothConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply deterministic graph-energy smoothing to class probabilities."""
    if proba.ndim != 3:
        raise ValueError("proba must be shaped (H, W, K).")
    height, width, n_classes = proba.shape
    if n_classes != len(classes):
        raise ValueError("classes length must match proba channel count.")

    image_float = _to_float_rgb(image_rgb)
    segments, n_segments = _build_superpixels(image_float, config)
    region_proba = _region_probabilities(proba, segments, n_segments)
    unary_energy = -np.log(np.clip(region_proba, EPS, 1.0)).astype(np.float64)

    src, dst, weights = _edge_weights(
        image_rgb=image_float,
        segments=segments,
        n_segments=n_segments,
        edge_awareness=config.edge_awareness,
    )

    labels_region, energy_trace, change_trace = _icm_optimize(
        unary_energy=unary_energy,
        src=src,
        dst=dst,
        weights=weights,
        lambda_smooth=float(config.lambda_smooth),
        iterations=int(config.iterations),
    )

    region_refined_proba = _refined_region_probabilities(
        unary_energy=unary_energy,
        labels=labels_region,
        src=src,
        dst=dst,
        weights=weights,
        lambda_smooth=float(config.lambda_smooth),
        temperature=float(config.temperature),
    )
    if not np.all(np.isfinite(region_refined_proba)):
        region_refined_proba = np.eye(n_classes, dtype=np.float32)[labels_region]

    proba_refined = region_refined_proba[segments]
    label_refined = labels_region[segments].astype(np.int32)
    label_refined = _cleanup_tiny_regions(label_refined, int(config.min_region_area_px))
    if int(config.min_region_area_px) > 0:
        proba_refined = np.eye(n_classes, dtype=np.float32)[label_refined]

    debug = {
        "n_superpixels": int(n_segments),
        "n_edges": int(src.size),
        "iterations_requested": int(config.iterations),
        "iterations_run": int(len(change_trace)),
        "energy_trace": [float(value) for value in energy_trace],
        "changes_per_iteration": [int(value) for value in change_trace],
        "converged": bool(change_trace and change_trace[-1] == 0),
    }
    return proba_refined.astype(np.float32), label_refined, debug


def compute_flip_stats(
    label_raw: np.ndarray,
    label_refined: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Compute global and per-class flip statistics."""
    if label_raw.shape != label_refined.shape:
        raise ValueError("label_raw and label_refined must have matching shapes.")
    total_pixels = int(label_raw.size)
    changed = label_raw != label_refined
    flip_count = int(np.sum(changed))
    flip_rate = (float(flip_count) / float(total_pixels)) if total_pixels else 0.0

    by_class: dict[str, dict[str, float | int]] = {}
    for class_idx, class_name in enumerate(classes):
        raw_mask = label_raw == class_idx
        refined_mask = label_refined == class_idx
        raw_count = int(np.sum(raw_mask))
        refined_count = int(np.sum(refined_mask))
        out_count = int(np.sum(raw_mask & (label_refined != class_idx)))
        in_count = int(np.sum((label_raw != class_idx) & refined_mask))
        by_class[class_name] = {
            "raw_pixels": raw_count,
            "refined_pixels": refined_count,
            "flipped_out_pixels": out_count,
            "flipped_in_pixels": in_count,
            "flipped_out_rate": (float(out_count) / float(raw_count)) if raw_count else 0.0,
            "flipped_in_rate": (float(in_count) / float(refined_count)) if refined_count else 0.0,
        }
    return {
        "total_pixels": total_pixels,
        "flip_pixels": flip_count,
        "flip_rate": flip_rate,
        "by_class": by_class,
    }


def compute_area_percentages(label: np.ndarray, classes: list[str]) -> dict[str, dict[str, float | int]]:
    """Compute per-class area percentages from a label map."""
    total_pixels = int(label.size)
    per_class: dict[str, dict[str, float | int]] = {}
    for class_idx, class_name in enumerate(classes):
        pixels = int(np.sum(label == class_idx))
        per_class[class_name] = {
            "pixels": pixels,
            "percent": (float(pixels) / float(total_pixels) * 100.0) if total_pixels else 0.0,
        }
    return {
        "total_pixels": total_pixels,
        "by_class": per_class,
    }


def compute_area_change(
    area_raw: dict[str, Any],
    area_refined: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    """Compute per-class area deltas between raw and refined masks."""
    raw_by_class = area_raw.get("by_class", {}) if isinstance(area_raw, dict) else {}
    refined_by_class = area_refined.get("by_class", {}) if isinstance(area_refined, dict) else {}
    class_names = sorted(set(raw_by_class.keys()) | set(refined_by_class.keys()))

    delta_by_class: dict[str, dict[str, float | int]] = {}
    for class_name in class_names:
        raw_stats = raw_by_class.get(class_name, {}) if isinstance(raw_by_class.get(class_name), dict) else {}
        refined_stats = (
            refined_by_class.get(class_name, {})
            if isinstance(refined_by_class.get(class_name), dict)
            else {}
        )
        raw_pixels = int(raw_stats.get("pixels", 0))
        refined_pixels = int(refined_stats.get("pixels", 0))
        raw_percent = float(raw_stats.get("percent", 0.0))
        refined_percent = float(refined_stats.get("percent", 0.0))
        delta_by_class[class_name] = {
            "raw_pixels": raw_pixels,
            "refined_pixels": refined_pixels,
            "delta_pixels": refined_pixels - raw_pixels,
            "raw_percent": raw_percent,
            "refined_percent": refined_percent,
            "delta_percent": refined_percent - raw_percent,
        }
    return {"by_class": delta_by_class}

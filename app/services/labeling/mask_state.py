from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .image_io import class_id_from_entry, load_binary_mask, mask_filename, mask_filename_for_class_id


@dataclass
class UndoAction:
    mode: str
    class_id: int
    sp_id: int
    previous_present: bool
    removed_from_class_ids: tuple[int, ...] = field(default_factory=tuple)
    frozen_removed: tuple[tuple[int, np.ndarray], ...] = field(default_factory=tuple)


@dataclass
class ImageMaskState:
    segments: np.ndarray
    selected: dict[int, set[int]]
    frozen_masks: dict[int, np.ndarray] = field(default_factory=dict)
    undo_stack: list[UndoAction] = field(default_factory=list)
    redo_stack: list[UndoAction] = field(default_factory=list)
    centroid_cache: dict[int, tuple[float, float]] | None = None

    def apply_click(self, class_id: int, sp_id: int, mode: str) -> bool:
        normalized_mode = str(mode).strip().lower()
        selected_set = self.selected.setdefault(int(class_id), set())
        previous_present = sp_id in selected_set
        removed_from_class_ids: list[int] = []
        frozen_removed: list[tuple[int, np.ndarray]] = []
        region = self.segments == int(sp_id)

        if normalized_mode == "add":
            selected_set.add(sp_id)
            for other_class_id, other_selected in self.selected.items():
                if int(other_class_id) == int(class_id):
                    continue
                if sp_id in other_selected:
                    other_selected.discard(sp_id)
                    removed_from_class_ids.append(int(other_class_id))
            for frozen_class_id, frozen_mask in self.frozen_masks.items():
                if tuple(frozen_mask.shape) != tuple(self.segments.shape):
                    continue
                removed_mask = region & frozen_mask
                if not np.any(removed_mask):
                    continue
                removed_indices = np.flatnonzero(removed_mask)
                frozen_mask[removed_mask] = False
                frozen_removed.append((int(frozen_class_id), removed_indices))
        elif normalized_mode == "remove":
            selected_set.discard(sp_id)
            frozen_mask = self.frozen_masks.get(int(class_id))
            if isinstance(frozen_mask, np.ndarray) and tuple(frozen_mask.shape) == tuple(self.segments.shape):
                removed_mask = region & frozen_mask
                if np.any(removed_mask):
                    removed_indices = np.flatnonzero(removed_mask)
                    frozen_mask[removed_mask] = False
                    frozen_removed.append((int(class_id), removed_indices))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        changed = (
            ((sp_id in selected_set) != previous_present)
            or bool(removed_from_class_ids)
            or bool(frozen_removed)
        )
        if changed:
            self.undo_stack.append(
                UndoAction(
                    mode=normalized_mode,
                    class_id=int(class_id),
                    sp_id=sp_id,
                    previous_present=previous_present,
                    removed_from_class_ids=tuple(sorted(removed_from_class_ids)),
                    frozen_removed=tuple(
                        (int(cid), indices.astype(np.int64, copy=True))
                        for cid, indices in frozen_removed
                    ),
                )
            )
            self.redo_stack.clear()
        return changed

    def undo(self) -> UndoAction | None:
        if not self.undo_stack:
            return None

        action = self.undo_stack.pop()
        selected_set = self.selected.setdefault(int(action.class_id), set())
        if action.previous_present:
            selected_set.add(action.sp_id)
        else:
            selected_set.discard(action.sp_id)
        if action.mode == "add":
            for removed_class_id in action.removed_from_class_ids:
                self.selected.setdefault(int(removed_class_id), set()).add(action.sp_id)
        elif action.mode != "remove":
            raise ValueError(f"Unknown undo/redo mode: {action.mode}")
        for removed_class_id, removed_indices in action.frozen_removed:
            frozen_mask = self.frozen_masks.get(int(removed_class_id))
            if isinstance(frozen_mask, np.ndarray):
                frozen_mask.flat[removed_indices] = True
        self.redo_stack.append(action)
        return action

    def redo(self) -> UndoAction | None:
        if not self.redo_stack:
            return None

        action = self.redo_stack.pop()
        selected_set = self.selected.setdefault(int(action.class_id), set())
        if action.mode == "add":
            selected_set.add(action.sp_id)
            for removed_class_id in action.removed_from_class_ids:
                self.selected.setdefault(int(removed_class_id), set()).discard(action.sp_id)
        elif action.mode == "remove":
            selected_set.discard(action.sp_id)
        else:
            raise ValueError(f"Unknown undo/redo mode: {action.mode}")
        for removed_class_id, removed_indices in action.frozen_removed:
            frozen_mask = self.frozen_masks.get(int(removed_class_id))
            if isinstance(frozen_mask, np.ndarray):
                frozen_mask.flat[removed_indices] = False
        self.undo_stack.append(action)
        return action

    def class_mask(self, class_id: int) -> np.ndarray:
        selected_set = self.selected.get(int(class_id), set())
        mask = selected_superpixels_to_mask(self.segments, selected_set)
        frozen_mask = self.frozen_masks.get(int(class_id))
        if isinstance(frozen_mask, np.ndarray) and tuple(frozen_mask.shape) == tuple(mask.shape):
            mask = mask | frozen_mask
        return mask

    def segment_centroids(self) -> dict[int, tuple[float, float]]:
        if self.centroid_cache is not None:
            return self.centroid_cache

        segments_int = self.segments.astype(np.int64, copy=False)
        h, w = segments_int.shape
        flat = segments_int.ravel()
        max_id = int(flat.max(initial=0))
        if max_id < 0:
            self.centroid_cache = {}
            return self.centroid_cache

        ys = np.repeat(np.arange(h, dtype=np.float32), w)
        xs = np.tile(np.arange(w, dtype=np.float32), h)

        counts = np.bincount(flat, minlength=max_id + 1)
        sum_x = np.bincount(flat, weights=xs, minlength=max_id + 1)
        sum_y = np.bincount(flat, weights=ys, minlength=max_id + 1)

        centroids: dict[int, tuple[float, float]] = {}
        for sp_id, count in enumerate(counts):
            if count <= 0:
                continue
            centroids[sp_id] = (float(sum_x[sp_id] / count), float(sum_y[sp_id] / count))

        self.centroid_cache = centroids
        return centroids


def selected_superpixels_to_mask(segments: np.ndarray, selected_set: set[int]) -> np.ndarray:
    if not selected_set:
        return np.zeros_like(segments, dtype=bool)
    selected_ids = np.fromiter(selected_set, dtype=segments.dtype)
    return np.isin(segments, selected_ids)


def selected_from_saved_masks(
    image_name: str,
    classes: list[dict[str, Any]],
    segments: np.ndarray,
    masks_dir: Path,
) -> dict[int, set[int]]:
    stem = Path(image_name).stem
    selected: dict[int, set[int]] = {}
    shape = tuple(segments.shape)

    for class_entry in classes:
        class_id = class_id_from_entry(class_entry)
        if class_id is None:
            continue
        mask_path = masks_dir / mask_filename_for_class_id(stem, class_id)
        if not mask_path.exists():
            legacy_name = str(class_entry.get("name", "")).strip()
            if legacy_name:
                mask_path = masks_dir / mask_filename(stem, legacy_name)
        if not mask_path.exists():
            selected[int(class_id)] = set()
            continue
        try:
            mask = load_binary_mask(mask_path, expected_shape=shape)
        except ValueError:
            selected[int(class_id)] = set()
            continue
        if not mask.any():
            selected[int(class_id)] = set()
            continue
        ids = np.unique(segments[mask])
        selected[int(class_id)] = set(int(v) for v in ids.tolist())

    return selected

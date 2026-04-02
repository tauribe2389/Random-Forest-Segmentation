"""Enum definitions for Model Foundry V2."""

from __future__ import annotations

from enum import Enum


class ArtifactKind(str, Enum):
    CANONICAL_LABEL_MAP = "canonical_label_map"
    PROVENANCE_MAP = "provenance_map"
    PROTECTION_MAP = "protection_map"
    SUPERPIXEL_CACHE = "superpixel_cache"
    DATASET_BUILD = "dataset_build"
    MODEL_ARTIFACT = "model_artifact"
    PREDICTION_RAW = "prediction_raw"
    PREDICTION_REFINED = "prediction_refined"
    PREDICTION_EXPORT = "prediction_export"
    DATASET_EXPORT = "dataset_export"


class ArtifactOwnerType(str, Enum):
    ANNOTATION_REVISION = "annotation_revision"
    REGISTERED_SNAPSHOT = "registered_snapshot"
    TRAINING_RUN = "training_run"
    PREDICTION_RUN = "prediction_run"

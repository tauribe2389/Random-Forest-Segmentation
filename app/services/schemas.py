"""Dataclasses for datasets, features, and training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    return text in {"1", "true", "on", "yes", "y"}


def parse_sigmas(raw_value: str) -> list[float]:
    """Parse a comma-separated sigma string into positive floats."""
    if not raw_value:
        return [1.0, 3.0]
    sigmas: list[float] = []
    for token in raw_value.split(","):
        text = token.strip()
        if not text:
            continue
        sigma = float(text)
        if sigma <= 0:
            raise ValueError("Gaussian sigmas must be greater than zero.")
        sigmas.append(sigma)
    if not sigmas:
        raise ValueError("At least one Gaussian sigma value is required.")
    return sigmas


@dataclass
class DatasetSpec:
    """Dataset registration form values."""

    name: str
    dataset_path: str
    image_root: str
    coco_json_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureConfig:
    """Feature extraction options for image segmentation."""

    use_rgb: bool = True
    use_hsv: bool = True
    use_lab: bool = True
    gaussian_sigmas: list[float] | tuple[float, ...] = (1.0, 3.0)
    use_sobel: bool = True
    use_lbp: bool = False
    lbp_points: int = 8
    lbp_radius: int = 1

    @classmethod
    def from_form(cls, form: dict[str, Any]) -> "FeatureConfig":
        sigmas = parse_sigmas(str(form.get("gaussian_sigmas", "1.0,3.0")))
        lbp_points = int(form.get("lbp_points", 8))
        lbp_radius = int(form.get("lbp_radius", 1))
        if lbp_points <= 0 or lbp_radius <= 0:
            raise ValueError("LBP points and radius must be positive integers.")

        return cls(
            use_rgb=_parse_bool(form.get("use_rgb"), default=False),
            use_hsv=_parse_bool(form.get("use_hsv"), default=False),
            use_lab=_parse_bool(form.get("use_lab"), default=False),
            gaussian_sigmas=sigmas,
            use_sobel=_parse_bool(form.get("use_sobel"), default=False),
            use_lbp=_parse_bool(form.get("use_lbp"), default=False),
            lbp_points=lbp_points,
            lbp_radius=lbp_radius,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureConfig":
        sigmas = data.get("gaussian_sigmas", [1.0, 3.0])
        return cls(
            use_rgb=bool(data.get("use_rgb", True)),
            use_hsv=bool(data.get("use_hsv", True)),
            use_lab=bool(data.get("use_lab", True)),
            gaussian_sigmas=list(sigmas),
            use_sobel=bool(data.get("use_sobel", True)),
            use_lbp=bool(data.get("use_lbp", False)),
            lbp_points=int(data.get("lbp_points", 8)),
            lbp_radius=int(data.get("lbp_radius", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gaussian_sigmas"] = list(self.gaussian_sigmas)
        return payload


@dataclass
class TrainConfig:
    """Random Forest and sampling configuration."""

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: str = "balanced_subsample"
    max_samples_per_class: int = 5000
    validation_split: float = 0.2
    overlap_policy: str = "higher_area_wins"
    use_rle: bool = True
    random_state: int = 42

    @classmethod
    def from_form(cls, form: dict[str, Any], default_seed: int = 42) -> "TrainConfig":
        max_depth_raw = str(form.get("max_depth", "")).strip()
        max_depth = int(max_depth_raw) if max_depth_raw else None
        class_weight_raw = str(form.get("class_weight", "balanced_subsample")).strip()
        class_weight = "balanced_subsample" if not class_weight_raw else class_weight_raw

        overlap_policy = str(form.get("overlap_policy", "higher_area_wins")).strip()
        if overlap_policy not in {"higher_area_wins", "last_annotation_wins"}:
            raise ValueError("Invalid overlap policy.")

        validation_split = float(form.get("validation_split", 0.2))
        if not 0 <= validation_split < 1:
            raise ValueError("Validation split must be in [0, 1).")

        n_estimators = int(form.get("n_estimators", 200))
        min_samples_split = int(form.get("min_samples_split", 2))
        min_samples_leaf = int(form.get("min_samples_leaf", 1))
        max_samples_per_class = int(form.get("max_samples_per_class", 5000))
        use_rle = _parse_bool(form.get("use_rle"), default=True)
        random_state = int(form.get("random_state", default_seed))

        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")
        if min_samples_split <= 0 or min_samples_leaf <= 0:
            raise ValueError("min_samples values must be greater than zero.")
        if max_samples_per_class <= 0:
            raise ValueError("max_samples_per_class must be greater than zero.")

        return cls(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            max_samples_per_class=max_samples_per_class,
            validation_split=validation_split,
            overlap_policy=overlap_policy,
            use_rle=use_rle,
            random_state=random_state,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return cls(
            n_estimators=int(data.get("n_estimators", 200)),
            max_depth=data.get("max_depth"),
            min_samples_split=int(data.get("min_samples_split", 2)),
            min_samples_leaf=int(data.get("min_samples_leaf", 1)),
            class_weight=str(data.get("class_weight", "balanced_subsample")),
            max_samples_per_class=int(data.get("max_samples_per_class", 5000)),
            validation_split=float(data.get("validation_split", 0.2)),
            overlap_policy=str(data.get("overlap_policy", "higher_area_wins")),
            use_rle=bool(data.get("use_rle", True)),
            random_state=int(data.get("random_state", 42)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


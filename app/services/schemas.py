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


def _parse_csv_tokens(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        tokens = [str(item).strip() for item in raw_value]
    else:
        tokens = [token.strip() for token in str(raw_value).split(",")]
    return [token for token in tokens if token]


def _parse_float_list(
    raw_value: Any,
    *,
    default: list[float],
    field_label: str,
    positive_only: bool = True,
) -> list[float]:
    tokens = _parse_csv_tokens(raw_value)
    if not tokens:
        return list(default)
    values: list[float] = []
    for token in tokens:
        number = float(token)
        if positive_only and number <= 0:
            raise ValueError(f"{field_label} values must be greater than zero.")
        values.append(float(number))
    return values


def _parse_positive_int_list(
    raw_value: Any,
    *,
    default: list[int],
    field_label: str,
) -> list[int]:
    tokens = _parse_csv_tokens(raw_value)
    if not tokens:
        return list(default)
    values: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        value = int(token)
        if value <= 0:
            raise ValueError(f"{field_label} values must be positive integers.")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _parse_laws_vectors(raw_value: Any) -> list[str]:
    allowed = {"L5", "E5", "S5", "R5", "W5"}
    tokens = _parse_csv_tokens(raw_value)
    if not tokens:
        return ["L5", "E5", "S5", "R5", "W5"]
    values: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.upper()
        if key not in allowed:
            raise ValueError("laws_vectors must only contain: L5, E5, S5, R5, W5.")
        if key in seen:
            continue
        seen.add(key)
        values.append(key)
    return values


def parse_sigmas(raw_value: str) -> list[float]:
    """Parse a comma-separated sigma string into positive floats."""
    return _parse_float_list(
        raw_value,
        default=[1.0, 3.0],
        field_label="Gaussian sigmas",
        positive_only=True,
    )


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
    lbp_radii: list[int] | tuple[int, ...] = (1,)
    use_gabor: bool = False
    gabor_frequencies: list[float] | tuple[float, ...] = (0.1, 0.2)
    gabor_thetas: list[float] | tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    gabor_bandwidth: float = 1.0
    gabor_include_real: bool = False
    gabor_include_imag: bool = False
    gabor_include_magnitude: bool = True
    use_laws: bool = False
    laws_vectors: list[str] | tuple[str, ...] = ("L5", "E5", "S5", "R5", "W5")
    laws_energy_window: int = 15
    laws_include_l5l5: bool = False
    laws_include_rotations: bool = False
    laws_use_abs: bool = True
    use_structure_tensor: bool = False
    structure_tensor_sigma: float = 1.0
    structure_tensor_rho: float = 3.0
    structure_tensor_include_eigenvalues: bool = True
    structure_tensor_include_orientation: bool = True
    structure_tensor_include_coherence: bool = True
    use_multiscale_local_stats: bool = False
    local_stats_sigmas: list[float] | tuple[float, ...] = (1.0, 3.0, 5.0)
    local_stats_include_mean: bool = True
    local_stats_include_std: bool = True
    local_stats_include_min: bool = False
    local_stats_include_max: bool = False

    @classmethod
    def from_form(cls, form: dict[str, Any]) -> "FeatureConfig":
        sigmas = parse_sigmas(str(form.get("gaussian_sigmas", "1.0,3.0")))
        lbp_points = int(form.get("lbp_points", 8))
        lbp_radius_fallback = int(form.get("lbp_radius", 1))
        lbp_radii = _parse_positive_int_list(
            form.get("lbp_radii", str(lbp_radius_fallback)),
            default=[lbp_radius_fallback],
            field_label="LBP radii",
        )
        if lbp_points <= 0:
            raise ValueError("LBP points must be a positive integer.")

        use_gabor = _parse_bool(form.get("use_gabor"), default=False)
        gabor_frequencies = _parse_float_list(
            form.get("gabor_frequencies", "0.1,0.2"),
            default=[0.1, 0.2],
            field_label="Gabor frequencies",
            positive_only=True,
        )
        gabor_thetas = _parse_float_list(
            form.get("gabor_thetas", "0,45,90,135"),
            default=[0.0, 45.0, 90.0, 135.0],
            field_label="Gabor thetas",
            positive_only=False,
        )
        gabor_bandwidth = float(form.get("gabor_bandwidth", 1.0))
        if gabor_bandwidth <= 0:
            raise ValueError("Gabor bandwidth must be greater than zero.")
        gabor_include_real = _parse_bool(form.get("gabor_include_real"), default=False)
        gabor_include_imag = _parse_bool(form.get("gabor_include_imag"), default=False)
        gabor_include_magnitude = _parse_bool(form.get("gabor_include_magnitude"), default=False)
        if use_gabor and not (gabor_include_real or gabor_include_imag or gabor_include_magnitude):
            raise ValueError("Enable at least one Gabor response component.")

        use_laws = _parse_bool(form.get("use_laws"), default=False)
        laws_vectors = _parse_laws_vectors(form.get("laws_vectors", "L5,E5,S5,R5,W5"))
        laws_energy_window = int(form.get("laws_energy_window", 15))
        if laws_energy_window <= 0:
            raise ValueError("Laws energy window must be a positive integer.")

        use_structure_tensor = _parse_bool(form.get("use_structure_tensor"), default=False)
        structure_tensor_sigma = float(form.get("structure_tensor_sigma", 1.0))
        structure_tensor_rho = float(form.get("structure_tensor_rho", 3.0))
        if structure_tensor_sigma <= 0:
            raise ValueError("Structure tensor sigma must be greater than zero.")
        if structure_tensor_rho < 0:
            raise ValueError("Structure tensor rho must be zero or greater.")
        structure_tensor_include_eigenvalues = _parse_bool(
            form.get("structure_tensor_include_eigenvalues"),
            default=False,
        )
        structure_tensor_include_orientation = _parse_bool(
            form.get("structure_tensor_include_orientation"),
            default=False,
        )
        structure_tensor_include_coherence = _parse_bool(
            form.get("structure_tensor_include_coherence"),
            default=False,
        )
        if use_structure_tensor and not (
            structure_tensor_include_eigenvalues
            or structure_tensor_include_orientation
            or structure_tensor_include_coherence
        ):
            raise ValueError("Enable at least one structure tensor output.")

        use_multiscale_local_stats = _parse_bool(form.get("use_multiscale_local_stats"), default=False)
        local_stats_sigmas = _parse_float_list(
            form.get("local_stats_sigmas", "1.0,3.0,5.0"),
            default=[1.0, 3.0, 5.0],
            field_label="Local stats sigmas",
            positive_only=True,
        )
        local_stats_include_mean = _parse_bool(form.get("local_stats_include_mean"), default=False)
        local_stats_include_std = _parse_bool(form.get("local_stats_include_std"), default=False)
        local_stats_include_min = _parse_bool(form.get("local_stats_include_min"), default=False)
        local_stats_include_max = _parse_bool(form.get("local_stats_include_max"), default=False)
        if use_multiscale_local_stats and not (
            local_stats_include_mean
            or local_stats_include_std
            or local_stats_include_min
            or local_stats_include_max
        ):
            raise ValueError("Enable at least one local stats output.")

        return cls(
            use_rgb=_parse_bool(form.get("use_rgb"), default=False),
            use_hsv=_parse_bool(form.get("use_hsv"), default=False),
            use_lab=_parse_bool(form.get("use_lab"), default=False),
            gaussian_sigmas=sigmas,
            use_sobel=_parse_bool(form.get("use_sobel"), default=False),
            use_lbp=_parse_bool(form.get("use_lbp"), default=False),
            lbp_points=lbp_points,
            lbp_radius=int(lbp_radii[0]),
            lbp_radii=lbp_radii,
            use_gabor=use_gabor,
            gabor_frequencies=gabor_frequencies,
            gabor_thetas=gabor_thetas,
            gabor_bandwidth=gabor_bandwidth,
            gabor_include_real=gabor_include_real,
            gabor_include_imag=gabor_include_imag,
            gabor_include_magnitude=gabor_include_magnitude,
            use_laws=use_laws,
            laws_vectors=laws_vectors,
            laws_energy_window=laws_energy_window,
            laws_include_l5l5=_parse_bool(form.get("laws_include_l5l5"), default=False),
            laws_include_rotations=_parse_bool(form.get("laws_include_rotations"), default=False),
            laws_use_abs=_parse_bool(form.get("laws_use_abs"), default=False),
            use_structure_tensor=use_structure_tensor,
            structure_tensor_sigma=structure_tensor_sigma,
            structure_tensor_rho=structure_tensor_rho,
            structure_tensor_include_eigenvalues=structure_tensor_include_eigenvalues,
            structure_tensor_include_orientation=structure_tensor_include_orientation,
            structure_tensor_include_coherence=structure_tensor_include_coherence,
            use_multiscale_local_stats=use_multiscale_local_stats,
            local_stats_sigmas=local_stats_sigmas,
            local_stats_include_mean=local_stats_include_mean,
            local_stats_include_std=local_stats_include_std,
            local_stats_include_min=local_stats_include_min,
            local_stats_include_max=local_stats_include_max,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureConfig":
        sigmas = _parse_float_list(
            data.get("gaussian_sigmas", [1.0, 3.0]),
            default=[1.0, 3.0],
            field_label="Gaussian sigmas",
            positive_only=True,
        )
        lbp_radius = int(data.get("lbp_radius", 1))
        lbp_radii = _parse_positive_int_list(
            data.get("lbp_radii", [lbp_radius]),
            default=[lbp_radius],
            field_label="LBP radii",
        )
        gabor_frequencies = _parse_float_list(
            data.get("gabor_frequencies", [0.1, 0.2]),
            default=[0.1, 0.2],
            field_label="Gabor frequencies",
            positive_only=True,
        )
        gabor_thetas = _parse_float_list(
            data.get("gabor_thetas", [0.0, 45.0, 90.0, 135.0]),
            default=[0.0, 45.0, 90.0, 135.0],
            field_label="Gabor thetas",
            positive_only=False,
        )
        local_stats_sigmas = _parse_float_list(
            data.get("local_stats_sigmas", [1.0, 3.0, 5.0]),
            default=[1.0, 3.0, 5.0],
            field_label="Local stats sigmas",
            positive_only=True,
        )
        return cls(
            use_rgb=bool(data.get("use_rgb", True)),
            use_hsv=bool(data.get("use_hsv", True)),
            use_lab=bool(data.get("use_lab", True)),
            gaussian_sigmas=sigmas,
            use_sobel=bool(data.get("use_sobel", True)),
            use_lbp=bool(data.get("use_lbp", False)),
            lbp_points=int(data.get("lbp_points", 8)),
            lbp_radius=int(lbp_radii[0]),
            lbp_radii=lbp_radii,
            use_gabor=bool(data.get("use_gabor", False)),
            gabor_frequencies=gabor_frequencies,
            gabor_thetas=gabor_thetas,
            gabor_bandwidth=float(data.get("gabor_bandwidth", 1.0)),
            gabor_include_real=bool(data.get("gabor_include_real", False)),
            gabor_include_imag=bool(data.get("gabor_include_imag", False)),
            gabor_include_magnitude=bool(data.get("gabor_include_magnitude", True)),
            use_laws=bool(data.get("use_laws", False)),
            laws_vectors=_parse_laws_vectors(data.get("laws_vectors", ["L5", "E5", "S5", "R5", "W5"])),
            laws_energy_window=int(data.get("laws_energy_window", 15)),
            laws_include_l5l5=bool(data.get("laws_include_l5l5", False)),
            laws_include_rotations=bool(data.get("laws_include_rotations", False)),
            laws_use_abs=bool(data.get("laws_use_abs", True)),
            use_structure_tensor=bool(data.get("use_structure_tensor", False)),
            structure_tensor_sigma=float(data.get("structure_tensor_sigma", 1.0)),
            structure_tensor_rho=float(data.get("structure_tensor_rho", 3.0)),
            structure_tensor_include_eigenvalues=bool(data.get("structure_tensor_include_eigenvalues", True)),
            structure_tensor_include_orientation=bool(data.get("structure_tensor_include_orientation", True)),
            structure_tensor_include_coherence=bool(data.get("structure_tensor_include_coherence", True)),
            use_multiscale_local_stats=bool(data.get("use_multiscale_local_stats", False)),
            local_stats_sigmas=local_stats_sigmas,
            local_stats_include_mean=bool(data.get("local_stats_include_mean", True)),
            local_stats_include_std=bool(data.get("local_stats_include_std", True)),
            local_stats_include_min=bool(data.get("local_stats_include_min", False)),
            local_stats_include_max=bool(data.get("local_stats_include_max", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gaussian_sigmas"] = list(self.gaussian_sigmas)
        lbp_radii = [int(item) for item in self.lbp_radii] if self.lbp_radii else [int(self.lbp_radius)]
        payload["lbp_radii"] = lbp_radii
        payload["lbp_radius"] = int(lbp_radii[0])
        payload["gabor_frequencies"] = list(self.gabor_frequencies)
        payload["gabor_thetas"] = list(self.gabor_thetas)
        payload["laws_vectors"] = [str(item) for item in self.laws_vectors]
        payload["local_stats_sigmas"] = list(self.local_stats_sigmas)
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


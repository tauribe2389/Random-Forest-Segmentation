# Superpixel Config Schema (v1)

This document defines the canonical configuration schema for labeler superpixel computation.
It is designed to support phased rollout without future breaking migrations.

## Goals

- Keep one canonical persisted shape for all algorithms.
- Allow phased enablement by validation gates, not by changing stored schema.
- Preserve existing SLIC defaults/behavior as a backward-compatible baseline.
- Keep per-image overrides compatible with dataset-default semantics.

## Canonical Config Object

All superpixel settings should normalize to this object before compute/caching:

```json
{
  "schema_version": 1,
  "algorithm": "slic",
  "preset_mode": "dataset_default",
  "preset_name": "dataset_default",
  "detail_level": "medium",
  "colorspace": "lab",
  "params": {
    "slic": {
      "n_segments": 1200,
      "compactness": 10.0,
      "sigma": 1.0,
      "enforce_connectivity": true,
      "min_size_factor": 0.5,
      "max_size_factor": 3.0
    },
    "slico": {
      "n_segments": 1200,
      "sigma": 1.0,
      "enforce_connectivity": true,
      "min_size_factor": 0.5,
      "max_size_factor": 3.0
    },
    "quickshift": {
      "ratio": 1.0,
      "kernel_size": 5,
      "max_dist": 10,
      "sigma": 0.0
    },
    "felzenszwalb": {
      "scale": 100.0,
      "sigma": 0.8,
      "min_size": 50
    },
    "watershed": {
      "markers": 1200,
      "compactness": 0.0,
      "gradient_sigma": 1.0,
      "connectivity": 1,
      "watershed_line": false
    }
  },
  "texture_channels": {
    "enabled": false,
    "mode": "append_to_color",
    "lbp": {
      "enabled": false,
      "points": 8,
      "radii": [1],
      "method": "uniform",
      "normalize": true
    },
    "gabor": {
      "enabled": false,
      "frequencies": [0.1, 0.2],
      "thetas_deg": [0.0, 45.0, 90.0, 135.0],
      "bandwidth": 1.0,
      "include_real": false,
      "include_imag": false,
      "include_magnitude": true,
      "normalize": true
    },
    "weights": {
      "color": 1.0,
      "lbp": 0.25,
      "gabor": 0.25
    }
  }
}
```

## Field Constraints

### Root

- `schema_version`: integer, currently `1` (required).
- `algorithm`: one of:
  - `slic`
  - `slico`
  - `quickshift`
  - `felzenszwalb`
  - `watershed`
- `preset_mode`: one of `dataset_default`, `detail`, `custom`.
- `preset_name`: non-empty string.
- `detail_level`: one of `low`, `medium`, `high`.
- `colorspace`: one of `lab`, `rgb`, `gray`.

### `params.slic`

- `n_segments`: integer `[50, 40000]`
- `compactness`: float `[0.01, 200.0]`
- `sigma`: float `[0.0, 10.0]`
- `enforce_connectivity`: bool
- `min_size_factor`: float `(0.0, 10.0]`
- `max_size_factor`: float `[1.0, 20.0]`

### `params.slico`

- Same as `slic`, except no user `compactness` term (SLICO behavior).
- `n_segments`: integer `[50, 40000]`
- `sigma`: float `[0.0, 10.0]`
- `enforce_connectivity`: bool
- `min_size_factor`: float `(0.0, 10.0]`
- `max_size_factor`: float `[1.0, 20.0]`

### `params.quickshift`

- `ratio`: float `(0.0, 20.0]`
- `kernel_size`: integer `[1, 100]`
- `max_dist`: float `(0.0, 200.0]`
- `sigma`: float `[0.0, 10.0]`

### `params.felzenszwalb`

- `scale`: float `(0.0, 10000.0]`
- `sigma`: float `[0.0, 10.0]`
- `min_size`: integer `[2, 50000]`

### `params.watershed`

- `markers`: integer `[16, 100000]`
- `compactness`: float `[0.0, 200.0]`
- `gradient_sigma`: float `[0.0, 10.0]`
- `connectivity`: integer in `{1, 2}`
- `watershed_line`: bool

### `texture_channels`

- `enabled`: bool
- `mode`: currently `append_to_color`
- `lbp.enabled`: bool
- `lbp.points`: integer `[4, 128]`
- `lbp.radii`: non-empty array of unique integers `[1, 64]`
- `lbp.method`: one of `uniform`, `ror`, `default`
- `lbp.normalize`: bool
- `gabor.enabled`: bool
- `gabor.frequencies`: non-empty array of floats `(0.0, 1.0]`
- `gabor.thetas_deg`: non-empty array of floats `[0.0, 180.0)`
- `gabor.bandwidth`: float `(0.0, 10.0]`
- `gabor.include_real`: bool
- `gabor.include_imag`: bool
- `gabor.include_magnitude`: bool
- At least one gabor include flag must be true when `gabor.enabled=true`.
- `gabor.normalize`: bool
- `weights.color`: float `(0.0, 10.0]`
- `weights.lbp`: float `[0.0, 10.0]`
- `weights.gabor`: float `[0.0, 10.0]`
- If `texture_channels.enabled=true`, then at least one of `lbp.enabled` or `gabor.enabled` must be true.

## Phased Enablement Rules

Schema is stable in all phases. Validation gates decide what is selectable.

- Phase 1:
  - Allowed algorithms: `slic`, `slico`
  - `texture_channels.enabled` must be `false`
- Phase 2:
  - Allowed algorithms: `slic`, `slico`, `quickshift`, `felzenszwalb`
  - `texture_channels.enabled` must be `false`
- Phase 3:
  - Allowed algorithms: all above plus `watershed` if desired
  - `texture_channels.enabled` may be `true` only for `slic` and `slico`

## Storage Shape

### New columns (recommended)

Add JSON columns while keeping legacy columns during transition:

- `labeler_projects.superpixel_config_json TEXT`
- `labeler_image_slic_overrides.superpixel_config_json TEXT`

Keep current `slic_*` columns for backward compatibility until all call sites are migrated.

### Canonical persistence rules

- Dataset defaults: store full canonical config in `labeler_projects.superpixel_config_json`.
- Per-image override:
  - If mode is `dataset_default`: delete override row.
  - Else: store full canonical config in override row `superpixel_config_json`.

## Migration Mapping (Legacy -> v1)

Map existing fields to canonical object:

- `algorithm`: `slic_algorithm` (fallback `slic`)
- `preset_name`: `slic_preset_name` (fallback `medium`)
- `detail_level`: `slic_detail_level` (fallback `medium`)
- `colorspace`: `slic_colorspace` (fallback `lab`)
- `params.slic.n_segments`: `slic_n_segments`
- `params.slic.compactness`: `slic_compactness`
- `params.slic.sigma`: `slic_sigma`
- `params.slico.n_segments`: `slic_n_segments`
- `params.slico.sigma`: `slic_sigma`
- Other algorithm param groups: use defaults above.
- `texture_channels`: defaults (`enabled=false`).

## API Payload Contract (`recompute_superpixels`)

### Request

```json
{
  "image_name": "img_001.png",
  "apply_remaining": false,
  "force_overwrite": false,
  "superpixel_config": { "...canonical object above..." }
}
```

Backward-compatible behavior:

- If `superpixel_config` is absent, map legacy top-level SLIC fields into canonical object.

### Response (new fields)

- `current_superpixel_config`: canonical object after normalization.
- Existing response fields remain.

## Caching Contract

Cache signature must hash normalized canonical config, not just SLIC subset.

Recommended signature source:

- Canonical JSON dump with sorted keys and compact separators.
- Include:
  - `schema_version`
  - `algorithm`
  - `colorspace`
  - selected algorithm params
  - `texture_channels` (full normalized block)

This ensures cache invalidation on algorithm or texture changes.

## Notes for Implementation

- `slic_cache.py` should become a generic `superpixel_cache.py` dispatcher.
- Keep `load_or_create_slic_cache(...)` shim as temporary compatibility wrapper.
- Warmup job (`slic_warmup`) should transition to generic superpixel warmup while keeping job type name stable initially, then optionally rename later.

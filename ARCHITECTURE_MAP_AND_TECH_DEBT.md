# Architecture Map And Tech Debt

## 1. Repo / Module Map

Top-level structure relevant to the observed workflow:

- `run.py`
  - local Flask entry point
- `app/__init__.py`
  - app factory and route registration
- `app/routes.py`
  - workspace, datasets, models, analysis, promotion, jobs, settings
- `app/labeler_routes.py`
  - labeler UI routes and labeler API endpoints
- `app/services/storage.py`
  - SQLite schema and filesystem path conventions
- `app/services/training.py`
  - random-forest training pipeline
- `app/services/inference.py`
  - random-forest inference / refinement pipeline
- `app/services/coco.py`
  - COCO loading/helpers
- `app/services/augmentation.py`
  - augmentation job generation and mask seeding behavior
- `app/services/labeling/mask_state.py`
  - in-memory labeler state, click/apply/frozen behavior
- `app/services/labeling/coco_export.py`
  - export from draft labeling project to COCO
- `app/services/schemas.py`
  - shared schema objects for routes/services
- `app/static/labeler.js`
  - labeler frontend state and API calls
- `tests/`
  - focused behavior tests around mask state, bootstrap masks, augmentation seeding, jobs, graph smoothing

## 2. Workflow-To-Code Mapping

### Workspace And Navigation

Primary implementation:

- `app/routes.py`

Responsibility:

- render workspace pages, datasets index, models pages, analysis pages, jobs, and settings

### Draft Dataset Creation And Browsing

Primary implementation:

- `app/routes.py`
- `app/services/storage.py`

Observed architectural pattern:

- a draft dataset is represented using a `labeler_project`-style storage structure plus dataset metadata rows
- storage and routing are tightly coupled through filesystem conventions

### Labeler Workflow

Primary implementation:

- `app/labeler_routes.py`
- `app/services/labeling/mask_state.py`
- `app/static/labeler.js`

Responsibilities:

- route to the labeler page for a dataset image
- load SLIC/superpixel state
- reconstruct editable masks from saved files
- process click, bulk-apply, save, and recompute actions
- maintain client-side dirty state and tool behavior

### Registration

Primary implementation:

- `app/routes.py`
- `app/services/labeling/coco_export.py`

Observed pattern:

- “register dataset” means export draft annotations into COCO artifacts and create a registered dataset row
- the registered snapshot boundary is therefore heavily tied to one export format

### Training

Primary implementation:

- `app/routes.py`
- `app/services/training.py`

Observed pattern:

- job creation is routed generically
- actual training path is hardcoded to RF semantic segmentation

### Analysis / Inference

Primary implementation:

- `app/routes.py`
- `app/services/inference.py`

Observed pattern:

- analysis run pages are generic product objects
- inference implementation is RF segmentation specific

### Promotion

Primary implementation:

- `app/routes.py`

Observed pattern:

- prediction promotion writes masks, labels, and bootstrap metadata into a new draft dataset structure
- good workflow idea, but still routed back into the same mask-storage conventions as ordinary drafts

## 3. Data Model / Persistence Map

### SQLite Metadata

Primary implementation:

- `app/services/storage.py`

Observed tables of interest:

- `datasets`
- `models`
- `analysis_runs`
- `analysis_run_items`
- `jobs`
- `labeler_projects`
- `labeler_image_slic_overrides`

Important observation:

- SQLite stores project metadata and workflow objects
- it does not store a first-class canonical annotation representation

### Filesystem-Backed Artifacts

The repo uses filesystem storage for large or structured artifacts. Important categories include:

- draft dataset images
- per-class mask PNGs
- optional `__frozen` mask PNGs
- COCO exports
- model artifacts
- analysis output files
- promotion/bootstrap JSON sidecars

This is a sensible direction for large artifacts, but the semantics of those files are not cleanly normalized in metadata.

### Overloaded Draft Object

Important design scar:

- `labeler_projects` effectively acts as the mutable-dataset backing store
- the same underlying storage abstraction is doing too much conceptual work

## 4. Annotation And Mask Lifecycle Map

### Current Save Path

Relevant code:

- `app/labeler_routes.py`
- `app/services/labeling/mask_state.py`

Observed flow:

1. Client edits selection state.
2. Server-side `ImageMaskState` merges selected superpixels and frozen raster state.
3. Save writes per-class raster masks to disk.

Key detail:

- saved files are per-class mask PNGs, not a single canonical class-index label map artifact

### Current Load Path

Relevant code:

- `_editable_selected_from_saved_masks()` in `app/labeler_routes.py`

Critical behavior:

- saved raster masks are turned back into editable selected-superpixel sets by taking all segment ids touching labeled pixels
- this is a lossy raster-to-superpixel reinterpretation step

Risk:

- any mask that is not aligned to current superpixel boundaries can silently broaden on reload

This is the central annotation-integrity problem in the current architecture.

### Frozen Mask Handling

Relevant code:

- `app/services/labeling/mask_state.py`
- recompute helpers in `app/labeler_routes.py`

Observed behavior:

- frozen pixels are stored as separate mask files per class
- `class_mask()` returns editable selected pixels union frozen pixels
- `apply_click()` can remove overlapping frozen pixels during edits, though undo/redo can restore them

Interpretation:

- freeze is implemented as a procedural editing behavior, not as a formal immutable protection contract

### Recompute Path

Relevant code:

- `_resolve_recompute_freeze_masks()`
- `_api_project_recompute_superpixels()`

Observed behavior:

- recompute can preserve masks by freezing them first
- if freeze is off and overwrite is confirmed, saved masks may be removed
- bootstrap datasets default toward freeze/preserve behavior

Interpretation:

- recompute is not safely “derived-only”
- it still participates in truth survival logic

### Promotion Path

Relevant code:

- `promote_analysis_run()` in `app/routes.py`

Observed behavior:

- creates new draft data from analysis output
- writes `labels/`, `masks/`, and `bootstrap/` provenance JSON

Risk:

- promoted annotations are still fed back into the same ordinary draft mask conventions
- later labeler load reinterprets them through superpixels

## 5. Current ML / Training Pipeline Map

### Training

Relevant code:

- `app/services/training.py`

Observed pipeline:

1. load registered COCO dataset
2. rasterize annotations into class-index targets
3. compute handcrafted per-pixel features
4. sample pixels
5. fit `RandomForestClassifier`
6. write model artifact with joblib

Implication:

- current “model training” is specifically RF semantic segmentation using engineered features

### Inference

Relevant code:

- `app/services/inference.py`

Observed pipeline:

1. load trained RF model artifact
2. compute same feature stack for new images
3. predict class probabilities / labels
4. optionally apply graph smoothing or refinement
5. write analysis outputs for review and promotion

### Dataset Export

Relevant code:

- `app/services/labeling/coco_export.py`

Observed behavior:

- draft annotations are exported to COCO for registered datasets
- export is based on base masks and does not encode richer provenance or protection semantics

## 6. Key Growth Scars / Technical Debt Hotspots

### No Formal Canonical Annotation Object

- truth lives in file conventions, not a first-class persisted entity
- this makes provenance, protection, revision history, and exact raster preservation difficult

### Superpixel Editing Aid Still Owns Too Much

- superpixels are not only an editor convenience
- they influence reconstruction of saved truth

### Freeze Is Not Semantically Strong Enough

- freeze sounds like protection
- implementation is closer to auxiliary raster preservation during editing/recompute

### Draft Dataset Detail Is An Application “God Screen”

- curation, schema, warmup, branching, registration, and progress live together
- this mirrors backend coupling

### Registered Dataset Equals COCO Snapshot

- there is no cleaner “registered snapshot” abstraction independent of one export format
- this blocks family-specific dataset builders

### ML Layer Is Generic In The Product, Specific In The Code

- routes and UI imply model extensibility
- service layer is still a single RF family

### Promotion Has Good Product Intent But Weak Internal Typing

- prediction-to-new-draft is the right workflow
- promoted artifacts do not become first-class provenance-bearing annotation revisions

### Terminology Drift

Examples:

- draft dataset vs labeler project
- registered dataset vs exported COCO snapshot
- frozen mask vs preserved mask vs bootstrap mask
- raw output vs refined output vs promoted annotation seed

## 7. CNN-Readiness Assessment

### What Already Helps

- draft vs registered lifecycle exists
- models and analysis are already first-class user-visible concepts
- filesystem artifact storage is compatible with larger ML assets
- promotion loop back into annotation is already in product form

### What Blocks Clean CNN Integration

- no canonical segmentation mask artifact independent of superpixel state
- no model-family abstraction
- no dataset-builder abstraction
- registered dataset identity is coupled to COCO export
- current training/inference config shape is RF-specific
- mask preservation is not trustworthy enough for CNN supervision

### Keras / U-Net Specific Assessment

Keras U-Net is a plausible first CNN family, but only if:

- canonical label masks are formalized first
- training datasets can emit exact image/mask pairs
- model-family-specific artifacts and configs are declared through a stable interface
- TensorFlow/Keras runtime is isolated as optional family-specific infrastructure rather than leaking into the entire app core

## 8. Specific Risks Around Label Preservation, Recalculation, And Mask Integrity

### Risk 1: Silent Mask Expansion On Reload

Location:

- `_editable_selected_from_saved_masks()` in `app/labeler_routes.py`

Why it matters:

- user-authored or promoted raster truth can expand to full touched superpixels when reloaded

### Risk 2: Recompute Is Not Purely Derived

Location:

- recompute helpers and API in `app/labeler_routes.py`

Why it matters:

- recomputing an editing aid still participates in whether saved masks survive and how later editable state is rebuilt

### Risk 3: Freeze Does Not Mean Strong Protection

Location:

- `app/services/labeling/mask_state.py`

Why it matters:

- users can infer immutable preservation guarantees that the system does not formally provide

### Risk 4: Promoted Predictions Lose Rich Lineage

Location:

- `promote_analysis_run()` in `app/routes.py`

Why it matters:

- predictions become ordinary draft masks plus sidecar metadata, instead of first-class provenance-bearing annotation revisions

### Risk 5: COCO Snapshot Is Doing Two Jobs

Location:

- registration/export path

Why it matters:

- immutable dataset identity and one export representation are conflated

### Risk 6: RF Assumptions Leak Across The Product

Location:

- `app/services/training.py`
- `app/services/inference.py`

Why it matters:

- future CNN support would be forced through seams that do not yet exist

## Conclusion

The current application has a valid product loop, but its internals are still shaped by superpixel-era editing assumptions and a single RF segmentation pipeline. The most urgent architecture issue is not “modularity” in the abstract. It is that canonical annotation truth, derived editing aids, and prediction artifacts are not yet separated strongly enough to preserve labels with confidence or to support CNN-ready training data cleanly.

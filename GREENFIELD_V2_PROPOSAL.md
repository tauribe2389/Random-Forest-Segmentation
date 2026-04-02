# Greenfield V2 Proposal

## 1. Executive Summary

Model Foundry v2 should preserve the current product loop while rebuilding the internal model around explicit truth, explicit lineage, and explicit model-family seams.

The workflow to preserve is:

`workspace library -> draft dataset -> human annotation -> registered snapshot -> model family training -> prediction review -> promotion into new draft`

The core redesign is:

- make user-authored labels the canonical truth
- treat superpixels as a derived editing aid only
- store predictions separately from annotations
- introduce immutable registered snapshots that are independent of any single export format
- introduce dataset builders and model-family adapters so RF and CNN segmentation can coexist cleanly

The highest-priority prerequisite for trustworthy CNN support is not a new training button. It is exact canonical mask storage with provenance and protection semantics that survive recompute, reload, promotion, and export without silent mutation.

## 2. Core Design Principles

1. User-authored labels are canonical truth.
2. Derived masks are not truth.
3. Predictions are not truth.
4. Mask lineage must be understandable.
5. CNN support must be first-class, not bolted onto RF assumptions.
6. Superpixels are an editing aid, not the owner of truth.

## 3. Proposed Domain / Data Architecture

### Core Entities

#### Workspace

Owns:

- image assets
- draft datasets
- registered snapshots
- models
- prediction runs
- settings

#### ImageAsset

Immutable record for a source image:

- `image_id`
- workspace ownership
- path
- checksum
- width / height

#### DraftDataset

Mutable annotation workspace:

- references a set of `ImageAsset` ids
- owns current class schema
- owns current annotation heads per image
- used for human editing and branching

#### RegisteredSnapshot

Immutable training and evaluation contract:

- frozen image membership
- frozen class schema
- frozen annotation revision ids
- snapshot metadata and creation lineage

Important rule:

- this is the dataset identity for training
- COCO export becomes one derived artifact of the snapshot, not the snapshot definition itself

#### AnnotationRevision

First-class persisted annotation truth for one image:

- `annotation_revision_id`
- `image_id`
- `dataset_id` or branch head reference
- `parent_revision_id`
- native-resolution `label_map`
- native-resolution `provenance_map`
- native-resolution `protection_map`
- manifest metadata
- author, timestamp, and operation summary

This is the core missing entity in v1.

#### DerivedArtifact

Non-canonical outputs attached to an image, dataset, snapshot, model, or prediction run:

- superpixel cache
- preview overlay
- COCO export
- RF feature cache
- CNN tile manifest
- evaluation summary

#### PredictionRun

Execution of a model against a registered snapshot:

- `prediction_run_id`
- `model_id`
- `registered_snapshot_id`
- config
- artifact manifest
- status

#### PredictionArtifact

Per-image or run-level prediction outputs:

- raw logits or probabilities
- raw class map
- refined class map
- confidence summaries
- provenance of refinement steps

#### PredictionExportArtifact

Machine-readable externalized outputs derived from a `PredictionRun` for downstream tooling:

- native-resolution class-index raster masks
- export manifest
- per-image metadata records
- optional per-class binary masks
- optional confidence or probability maps
- optional compressed numeric bundles such as `npz`

### Storage Strategy

- keep SQLite for metadata and manifests
- keep large arrays and masks on the filesystem
- use versioned manifest records rather than implicit path conventions

## 4. Proposed Annotation And Mask Lifecycle

### A. Canonical Annotation Representation

For semantic segmentation, the canonical truth should be:

- a native-resolution raster `label_map` where each pixel stores a class id

Optional supporting artifacts:

- imported polygons
- brush or edit operation logs
- vector overlays

Those are not the authoritative training truth. The canonical truth for segmentation is the raster label map because:

- CNN training needs exact pixel-aligned masks
- promotion and import flows naturally produce raster data
- it avoids reinterpreting truth through superpixels

Provenance should be tracked through:

- `provenance_map` at pixel level where needed
- revision-level metadata describing operation type, source model, threshold, cleanup steps, and parent revision

Suggested provenance values:

- `authored`
- `promoted`
- `imported`
- `derived_seed`

### B. Mask Preservation

Superpixel recompute must never mutate `AnnotationRevision.label_map`.

Formal semantics:

- `label_map`: canonical truth
- `provenance_map`: where the label came from
- `protection_map`: whether batch or destructive tools may overwrite the pixel without explicit override
- `superpixel_cache`: derived editing aid only

Replace v1 “freeze” with explicit protection states visible in the UI:

- `Authored`
- `Promoted`
- `Imported`
- `Protected`
- `Derived Aid`

Formal rule:

- recompute, smoothing previews, or editing-aid refreshes may invalidate and rebuild derived artifacts
- they may not modify canonical label data

### E. Prediction Vs Truth

Predictions remain separate from annotations:

- raw prediction artifact
- refined prediction artifact
- export artifact
- promoted annotation revision created only by explicit user action

Promotion rules:

- default path is promotion into a new draft dataset branch
- promotion creates a new `AnnotationRevision`
- revision metadata records source model, confidence threshold, cleanup settings, and source prediction artifact ids

### Prediction Export Contract

Analysis runs must produce exportable, machine-readable raster artifacts suitable for downstream external post-processing.

Required export per image:

- native-resolution class-index mask aligned to the source image grid

Required metadata per image or run:

- `image_id`
- `registered_snapshot_id`
- `prediction_run_id`
- source artifact type such as `raw` or `refined`
- class schema version
- class-id to class-name mapping
- image width and height
- model family
- threshold, refinement, and cleanup settings
- artifact checksum

Optional exports:

- per-class binary masks
- confidence maps
- probability maps
- logits where supported by the model family
- compressed numeric bundles for Python workflows such as `npz`

Required run-level manifest:

- lists every exported file
- records file type and semantic meaning
- records whether the export came from raw prediction or refined prediction
- records the exact class schema and model provenance used to generate the export

Formal rule:

- exported prediction rasters are derived artifacts of a `PredictionRun`
- they are never treated as canonical annotation truth unless explicitly promoted into a new annotation revision

### Lifecycle Summary

1. User edits create or update an `AnnotationRevision`.
2. Superpixels are generated as a cache over the image, not the truth.
3. Registered snapshots point to immutable annotation revision ids.
4. Model families train from builders generated off the snapshot.
5. Prediction runs emit prediction artifacts.
6. User may review and explicitly promote prediction artifacts into new annotation revisions, typically in a new draft branch.

## 5. Proposed ML / Model-Family Architecture

### C. Multi-Family Support

Introduce `ModelFamilyAdapter` as the stable seam.

Each family declares:

- `family_id`
- `task_type`
- config schema
- required dataset builder type
- train artifact manifest shape
- inference output contract
- evaluation hooks

Initial family ids:

- `rf.semantic_segmentation`
- `keras.unet.semantic_segmentation`
- later `deeplab.semantic_segmentation`
- future instance-oriented families

### D. DatasetBuilder Abstraction

Registered snapshots feed family-specific builders.

#### RF Semantic Segmentation Builder

Outputs:

- sampled feature tensors
- label vectors
- sampling manifest
- feature configuration manifest

#### CNN Semantic Segmentation Builder

Outputs:

- image paths
- canonical mask paths
- split manifest
- class schema manifest
- optional tile manifest
- augmentation config manifest

#### Future Instance Segmentation Builder

Outputs:

- instance masks
- bounding boxes
- categories
- image manifests

This is how one canonical dataset supports multiple training targets.

### Train / Infer Contracts

Each family adapter should expose equivalent lifecycle methods conceptually:

- `validate_snapshot(snapshot_id, config)`
- `build_training_artifacts(snapshot_id, config)`
- `train(training_manifest, config)`
- `predict(model_id, snapshot_id, config)`
- `export_predictions(prediction_run_id, export_config)`
- `evaluate(prediction_run_id, config)`

Routes should call these contracts through family metadata, not by branching on RF-specific assumptions.

## 6. CNN / Keras / U-Net Integration Plan

### G. How Keras And U-Net Fit

Keras U-Net should be introduced as the first CNN semantic segmentation family because:

- it matches the app’s current semantic-segmentation focus
- it can consume exact image/mask pairs directly
- it creates a clean forcing function for mask correctness and family abstraction

But it should be added only after three prerequisites:

1. canonical annotation revision storage exists
2. registered snapshots are independent of COCO export
3. dataset builder and model family seams exist

### Architecture Changes Required Before CNN Is Trustworthy

- exact native-resolution class-index label maps
- no raster truth reconstruction through superpixels
- family-aware model metadata
- builder manifests for image/mask pair training data
- prediction artifacts stored separately from truth

### Keras Runtime Isolation

Keep Keras optional:

- isolate TensorFlow/Keras dependencies to the CNN family runtime path
- avoid making core Flask app startup depend on heavy CNN packages
- store family-specific environment or capability metadata with the adapter

### Model Selection UX

Model creation should become family-aware:

1. choose task type
2. choose family
3. present family-specific config form generated from schema
4. validate compatible registered snapshot and builder outputs

This is much stronger than adding a single U-Net dropdown to the current RF form.

## 7. UI / UX Improvements Focused On Preservation And Clarity

### F. Labeler UX

The labeler should visibly distinguish:

- `Truth`
- `Protected`
- `Prediction`
- `Promoted`
- `Derived Aid`

Concrete changes:

- show overlay badges and legend for provenance and protection
- show destructive-action warnings that explicitly state whether canonical labels will change
- label recompute actions as: `Regenerate editing aid only. Saved labels unchanged.`
- separate “apply to truth” actions from “preview / derived / assist” actions

### Draft Dataset UX

Split today’s overloaded draft detail concepts into clearer sections:

- image membership and curation
- class schema
- annotation progress
- editing-aid maintenance
- branching and registration

### Prediction Review UX

Show three separate concepts:

- model prediction
- refined prediction
- exported machine-readable artifact package
- promoted annotation branch

Promotion should always state:

- whether a new draft will be created
- what threshold and refinement choices are embedded in the promoted revision metadata

Export should always state:

- whether the export was generated from raw or refined prediction artifacts
- what raster formats and numeric bundles were emitted
- what class schema and model provenance are attached to the export manifest

## 8. Migration Strategy

### H. Incremental Migration Plan

#### Phase 1: Documentation And Vocabulary

- document v1 exactly as-is
- introduce shared v2 terms in docs and code comments:
  - `draft dataset`
  - `registered snapshot`
  - `annotation revision`
  - `prediction artifact`
  - `derived artifact`
  - `protection`

#### Phase 2: Annotation Service Boundary

- add a thin annotation service boundary around current draft-mask storage
- remove direct route and UI ownership of mask semantics

#### Phase 3: Registered Snapshot And Builder Seams

- make registration create an immutable snapshot record independent of COCO export
- make COCO one builder or export output

#### Phase 4: Model Family Metadata

- add `model_family`, `task_type`, and artifact manifests while RF remains the only implemented family

#### Phase 5: Canonical Annotation Revisions

- introduce exact label-map persistence plus provenance and protection maps
- stop reconstructing truth via superpixels

#### Phase 6: CNN Semantic Segmentation

- implement `keras.unet.semantic_segmentation`
- add CNN builder and prediction artifact storage

#### Phase 7: Promotion And Export Migration

- convert promotion, augmentation seeding, and downstream exports to operate on annotation revisions and prediction artifacts

### What Can Remain Compatible

- workspace shell and navigation
- jobs infrastructure
- local artifact browsing
- draft-to-registered-to-analysis-to-promoted-draft workflow
- existing RF family as one adapter implementation

## 9. Immediate High-Value Refactors Before Full Rewrite

1. Stop reconstructing saved raster masks through current superpixels.
2. Make promoted and bootstrap outputs first-class provenance-bearing annotations.
3. Add `model_family` and artifact-manifest metadata immediately, even while RF is the only family.
4. Separate registered snapshot identity from COCO export artifact generation.
5. Clarify UI copy now around workspace library vs draft dataset, prediction vs truth, and recompute aid vs saved label.

## 10. Open Questions / Tradeoffs

- Should protection be pixel-granular only, or also support region-level locked groups for UX convenience?
- Should edit-operation logs be retained long-term, or only revision snapshots plus summary provenance?
- For very large images, should canonical label maps support chunked or tiled storage rather than one full raster file?
- Should COCO remain the primary export for interoperability even after registered snapshots become the internal source of truth?
- Should promoted predictions default to `protected=false` but `provenance=promoted`, or should some promotion modes create protected seeds intentionally?

## Explicit Answers To The Redesign Questions

### A. Canonical Annotation Representation

Use both raster truth and optional auxiliary vectors, but make the native-resolution raster class-index `label_map` the source of truth. Polygons and edit logs are supporting provenance, not authoritative segmentation truth.

### B. Mask Preservation

Recomputing superpixels only regenerates derived editing aids. It never changes canonical truth. Replace vague freeze behavior with explicit provenance plus explicit protection semantics. Protection means batch tools and destructive transforms cannot overwrite those pixels without an explicit override action.

### C. ML Architecture Support

Support families through `ModelFamilyAdapter` plus builder contracts. RF, U-Net, DeepLab-like models, and future instance models all plug into the same snapshot/build/train/predict lifecycle, but each declares its own config schema and output manifest.

### D. Dataset Generation

One registered snapshot feeds multiple deterministic builders. RF builder emits engineered features and sampled labels. CNN builder emits image/mask pairs and split or tile manifests. Future instance builders emit instance-level targets. Builders are manifest-backed and never mutate the source annotations.

### E. Prediction Vs Truth

Store raw predictions, refined predictions, exported prediction artifacts, and truth as different entities. Promotion is explicit, creates a new annotation revision, and records source artifacts and thresholds. Default promotion path creates or updates a new draft branch rather than overwriting source truth.

Analysis runs must also support downstream export of machine-readable raster outputs. At minimum this includes class-index masks plus metadata; optional exports include per-class binaries, confidence maps, and compressed numeric bundles such as `npz`.

### F. Labeler UX

The labeler must show clear visual states for authored truth, protected areas, prediction overlays, promoted seeds, and derived editing aids. Any action that can change truth must say so explicitly. Recompute must clearly say it only refreshes editing aids.

### G. CNN Integration

Introduce Keras/U-Net only after exact canonical masks and family-neutral builder seams exist. The largest blockers today are superpixel-tied truth reconstruction, RF-specific training assumptions, and the lack of a canonical training mask artifact.

### H. Migration Strategy

Improve incrementally. First fix annotation truth and snapshot/build contracts. Then add family metadata. Then introduce CNN support as a new adapter. Keep v1 RF training functional during the transition and allow v1/v2 data structures to coexist where practical.

# V2 Implementation Plan

## Summary

`model_foundry_v2` will be built as a separate Flask app inside the current repo, with its own SQLite database, filesystem artifact store, tests, and run entrypoint. V1 stays untouched during development. The first implementation phases focus on exact canonical label persistence, immutable snapshot identity, explicit artifact lineage, and service contracts that will later support multiple model families cleanly.

## Phase Sequence

### Phase 0: Planning and scaffolding

- create the `model_foundry_v2` package and runtime layout
- add config, entrypoint, templates, and a health route
- define typed domain entities, enums, repository interfaces, and service boundaries
- initialize an isolated v2 database and artifact directory structure

### Phase 1: Core data model and persistence

- implement metadata tables for workspaces, images, draft datasets, classes, annotation revisions, heads, snapshots, artifacts, models, training runs, and prediction runs
- implement canonical label, provenance, and protection map save/load using exact `.npy` storage
- add revision checksums and revision locking so snapshot references remain immutable
- expose minimal workspace/dataset setup UI and JSON annotation endpoints

### Phase 2: Minimal real labeling workflow

- add a thin annotation surface for one image
- save and reload canonical label maps without drift
- keep superpixels optional and derived-only

### Later phases

- snapshot builders and export builders
- analysis runs and machine-readable raster exports
- model-family registry and adapter execution
- Keras U-Net training and inference
- UX refinement and optional v1 migration helpers

## Target Architecture

Core entities:

- `Workspace`
- `ImageAsset`
- `DraftDataset`
- `RegisteredSnapshot`
- `AnnotationRevision`
- `ArtifactRecord`
- `PredictionRun`
- `ModelDefinition`
- `TrainingRun`

Core architectural rules:

- canonical truth is a native-resolution class-index raster label map
- predictions, exports, and promoted results are separate artifacts
- superpixels are derived editing aids only
- COCO is a derived export, not the canonical dataset identity
- artifact metadata lives in SQLite, while arrays and large files live on disk

## What Will Be Built First

- isolated Flask scaffold at `model_foundry_v2`
- database bootstrap and repository layer
- `AnnotationService`, `ArtifactService`, `SnapshotService`, and model-family registry stubs
- minimal HTML workflow for creating workspaces, images, datasets, and classes
- exact save/load JSON endpoints for annotation heads

## Deferred Intentionally

- polished annotation UX
- COCO export implementation
- prediction execution
- analysis review UI
- real model training and Keras runtime integration
- v1 data import or migration helpers

## Risks And Dependencies

- snapshot immutability requires annotation revisions to lock once referenced by a registered snapshot
- future CNN support depends on exact canonical mask persistence landing correctly now
- derived artifact contracts need to be explicit early so analysis export can fit without rewiring the app

## V1 And V2 Coexistence

- v1 remains at the repo root and continues to run on port `5000`
- v2 lives in `model_foundry_v2/` and defaults to port `5001`
- v2 uses separate `instance/` and `data/` directories and must not read or write v1 storage

# Current Workflow Spec

## Basis And Scope

This document describes the current application as observed on March 24, 2026 in the local app at `http://127.0.0.1:5000`, using the existing `demo` workspace and its already-populated artifacts. The observations are grounded in actual UI navigation plus code inspection. Where I did not execute a destructive action, I state that explicitly.

The dominant observed path is:

`workspace -> draft dataset -> labeler -> registered snapshot -> model -> analysis -> promoted draft`

## What The App Is Today

The application is a local workflow tool for:

- managing a workspace image library
- assembling mutable draft datasets from workspace images
- labeling images with superpixel-assisted segmentation tools
- registering draft datasets into immutable training snapshots
- training segmentation models
- running analysis/inference against registered datasets
- reviewing outputs and promoting predictions into a new draft dataset

The current implementation is strongest as a random-forest segmentation workbench with a draft-to-registered lifecycle and a promotion loop back into annotation.

## Key Screens And Their Purpose

### Home / Workspace Landing

Observed pages:

- `/`
- `/workspace/5`

Purpose:

- choose a workspace
- see workspace-level navigation
- jump to images, datasets, models, analysis, jobs, and settings

### Workspace Images

Observed page:

- `/workspace/5/images`

Purpose:

- browse the workspace’s raw image library
- inspect what images exist before adding them into a draft dataset

Observed behavior:

- image browsing is library-oriented, not labeling-oriented
- clicking into labeling from this context redirects with the message: `Open the labeler from a dataset page. Workspace images are inspect-only.`

Interpretation:

- this is an important product seam: images belong to the workspace library, but annotation belongs to a dataset-scoped draft project
- the current route naming makes that easy to miss

### Datasets Index

Observed page:

- `/workspace/5/datasets`

Purpose:

- show both draft datasets and registered datasets
- create a draft dataset from selected workspace images
- browse registered outputs

Observed behavior:

- the page mixes mutable draft work and immutable registered artifacts in one place
- draft datasets are the entry point for labeling
- registered datasets appear as frozen/exported outputs of draft work

### Draft Dataset Detail

Observed page:

- `/workspace/5/datasets/15`

Purpose:

- operational hub for a mutable labeling project

Observed capabilities on this page:

- class creation and editing
- image membership and progress
- labeling entry points
- SLIC warmup / superpixel readiness
- dataset branching
- registration into immutable snapshot form

Interpretation:

- this page carries too many responsibilities at once
- it is where the product’s historical growth is most visible

### Labeler

Observed page:

- `/workspace/5/datasets/15/images/download%20(1).jpg/label`

Purpose:

- apply labels to a dataset-scoped image using superpixel-assisted tools

Observed editing model:

- single click applies a label immediately and marks the page dirty
- brush, marquee, and similarity tools stage a selection first, then require explicit apply
- undo/redo is available
- there is an advanced superpixels area including recompute and freeze-related controls

Observed state behavior:

- one click produced `Unsaved changes`
- I then used undo and did not save, so discovery did not intentionally mutate stored annotations

Interpretation:

- the interaction model is mixed: some tools are direct-write, others are staged
- the labeler’s mental model is superpixel-first rather than annotation-first

### Models

Observed pages:

- `/workspace/5/models`
- `/workspace/5/models/6`

Purpose:

- browse trained models
- inspect a specific model’s metadata and outputs
- launch training jobs

Observed behavior:

- UI framing is generic enough to imply broader model support
- actual controls, artifacts, and code path are centered on random-forest segmentation

### Analysis

Observed pages:

- `/workspace/5/analysis`
- `/workspace/5/analysis/new`
- `/workspace/5/analysis/4`

Purpose:

- run a trained model against a registered dataset
- review raw and refined outputs
- inspect analysis artifacts and per-image results

Observed behavior:

- analysis is a first-class object in the product, not just an ephemeral prediction
- the review page distinguishes raw vs refined outputs

### Promotion

Observed page:

- `/workspace/5/analysis/4/promote`

Purpose:

- turn analysis output into a new draft dataset for continued human labeling

Observed behavior:

- promotion defaults toward creating a new draft dataset rather than overwriting source data
- the page exposes confidence-threshold and cleanup options
- refined output is used when available

Interpretation:

- this is one of the strongest existing workflow concepts in the app
- it already implies a good v2 principle: predictions should branch into a new human-editable draft, not silently become truth

### Jobs And Settings

Observed pages:

- `/workspace/5/jobs`
- `/workspace/5/settings`

Purpose:

- browse background activity and job status
- manage workspace-level settings

## Major Workflows

### 1. Workspace Selection And Orientation

1. Open the app landing page.
2. Select a workspace.
3. Use workspace navigation to reach images, datasets, models, analysis, jobs, or settings.

### 2. Draft Dataset Creation

1. Open workspace images or datasets.
2. Select workspace images for inclusion.
3. Create a draft dataset.
4. Open the draft dataset detail page.

Key state transition:

- raw workspace images become members of a dataset-scoped mutable labeling project

### 3. Class Schema Setup

1. Open a draft dataset detail page.
2. Create classes.
3. Edit class names/colors as needed.
4. Return to the labeler using this class schema.

### 4. Labeling An Image

1. Open a draft dataset image in the labeler.
2. Select a target class.
3. Use click, brush, marquee, or similarity selection.
4. Apply changes where required.
5. Save.
6. Move to the next image.

Observed state transitions:

- user interaction mutates an in-browser working state
- save writes class masks back to disk
- unsaved/dirty state blocks casual navigation

### 5. Recomputing Superpixels

1. Open the advanced superpixels area.
2. Adjust recompute options.
3. Choose whether masks should be frozen/preserved.
4. Recompute superpixels.

Observed behavior:

- the feature is presented as an editing aid operation
- in implementation, it also participates in how saved masks are interpreted when state is reconstructed

### 6. Registering A Dataset

1. Open a draft dataset detail page.
2. Trigger dataset registration.
3. The app exports a COCO-style snapshot and records a registered dataset version.

Observed state transition:

- mutable draft project becomes immutable registered dataset artifact

### 7. Training A Model

1. Open models.
2. Choose a registered dataset.
3. Configure training options.
4. Launch a job.
5. Inspect model detail when complete.

Observed behavior:

- the flow is productized as a general model-training loop
- the implementation is specifically random-forest semantic segmentation

### 8. Running Analysis

1. Open analysis creation.
2. Choose a model and registered dataset.
3. Launch inference.
4. Review outputs on the analysis detail page.

### 9. Reviewing Outputs

1. Open an analysis run.
2. Browse image-level predictions.
3. Compare raw outputs and refined outputs.
4. Decide whether the results are promotion-worthy.

### 10. Promoting Predictions Into A New Draft

1. Open analysis promotion.
2. Choose threshold and cleanup settings.
3. Create a new draft dataset from predictions.
4. Resume human editing in that new draft.

## Observed State Model

The user-facing lifecycle can be described as:

1. `Workspace image library`
2. `Draft dataset`
3. `Draft image annotation state`
4. `Registered dataset snapshot`
5. `Model`
6. `Analysis / prediction artifacts`
7. `Promoted draft dataset`

That is a decent product skeleton. The problem is that the implementation does not give each of these stages equally clear storage and semantic boundaries.

## Confusing Or Fragile Behaviors

### Workspace Images vs Dataset Images

- the product has the right conceptual split, but route names and page framing do not make it obvious enough
- users can easily assume workspace images are directly labelable

### Draft Dataset Detail Is Overloaded

- one screen owns curation, schema, annotation progress, SLIC readiness, branching, and registration
- this makes the app feel historically accumulated rather than intentionally modeled

### Mixed Edit Semantics In The Labeler

- single click writes immediately
- other tools stage then apply
- the user must internalize different mutation semantics for different tools on the same page

### Freeze Is Ambiguous

- the UI implies preservation
- the system does not present a precise model of what is protected, what is derived, and what can still be changed by later operations

### “Saved Labels” And “Superpixel State” Are Too Entangled

- the user mental model is likely “I painted labels”
- the system mental model is closer to “the current saved raster is converted back into selected superpixels and re-rendered”

### Generic ML Framing Hides RF-Specific Assumptions

- the UI suggests a general modeling platform
- the underlying path is a handcrafted-feature random-forest segmentation pipeline

## Specific Notes On Label Preservation And Mask Mutation Risk

- current persistence stores class mask PNGs rather than a first-class canonical annotation object
- saved raster masks are turned back into editable selected-superpixel sets, so any mask that is not aligned with current superpixel boundaries can expand on reload
- promoted or imported masks are especially vulnerable because they are more likely to be raster-true rather than superpixel-native
- frozen content exists, but the system does not communicate it as a rigorous annotation state with precise guarantees
- predictions branch into a new draft, which is good, but after promotion they re-enter the same ordinary storage pathway without a rich truth-vs-prediction lineage model

## Screens / Page References Used During Discovery

The following pages were directly inspected through browser automation:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/workspace/5`
- `http://127.0.0.1:5000/workspace/5/images`
- `http://127.0.0.1:5000/workspace/5/datasets`
- `http://127.0.0.1:5000/workspace/5/datasets/15`
- `http://127.0.0.1:5000/workspace/5/datasets/15/images/download%20(1).jpg/label`
- `http://127.0.0.1:5000/workspace/5/models`
- `http://127.0.0.1:5000/workspace/5/models/6`
- `http://127.0.0.1:5000/workspace/5/analysis`
- `http://127.0.0.1:5000/workspace/5/analysis/new`
- `http://127.0.0.1:5000/workspace/5/analysis/4`
- `http://127.0.0.1:5000/workspace/5/analysis/4/promote`
- `http://127.0.0.1:5000/workspace/5/jobs`
- `http://127.0.0.1:5000/workspace/5/settings`

## Discovery Limits

I intentionally avoided destructive mutations during discovery.

- I entered the labeler and verified dirty-state behavior with a single click plus undo.
- I did not intentionally save altered annotations.
- I did not intentionally re-run registration, full model training, or promotion on top of the existing demo data.

Accordingly, the workflow description above is grounded in real navigation and in the visible app state, while some destructive transitions were confirmed through code inspection rather than by re-executing them against the demo workspace.

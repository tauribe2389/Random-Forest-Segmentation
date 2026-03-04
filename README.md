# Model Foundry (Local-Only)

Model Foundry is a local-first Flask app that combines:

- Superpixel image annotation + COCO export
- Random-forest segmentation training
- Batch analysis/inference runs
- Local model/run registry in SQLite

## Key Updates in This Version

- New integrated superpixel labeler at `/labeler`
- Dataset-driven labeler projects with in-app class management
- New augmentation workflow at `/augment/new`
- COCO RLE support in training pipeline (enabled by default)
- SME-aligned HTML/CSS shell and branding
- Superpixel config schema draft for phased multi-algorithm rollout: `docs/superpixel_config_schema.md`

## Project Layout

```text
model_foundry/
├── app/
│   ├── routes.py
│   ├── services/
│   │   ├── coco.py
│   │   ├── training.py
│   │   ├── inference.py
│   │   ├── schemas.py
│   │   ├── storage.py
│   │   └── labeling/
│   ├── templates/
│   └── static/
├── scripts/
├── instance/
├── models/
├── runs/
├── requirements.txt
└── run.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Open: `http://127.0.0.1:5000`

## Workflow

1. Open `/labeler` and create/select a labeler project
2. For dataset-linked projects, sync classes from dataset or edit classes in-app
3. Annotate via superpixels and export COCO (RLE)
4. Register dataset in `/datasets/new` (or use `/augment/new` to generate + register)
5. Train model from dashboard
6. Run analysis in `/analysis/new`

## RLE Behavior

- Training now decodes COCO RLE segmentations (compressed/uncompressed)
- Polygons are still supported
- Train setting `use_rle` defaults to `true`

## Optional Environment Variables

- `SEG_APP_BASE_DIR`
- `SEG_APP_INSTANCE_DIR`
- `SEG_APP_MODELS_DIR`
- `SEG_APP_RUNS_DIR`
- `SEG_APP_DB_PATH`
- `SEG_APP_CODE_VERSION`
- `SEG_APP_RANDOM_SEED`
- `FLASK_SECRET_KEY`
- `SEG_APP_LABELER_DIR`
- `SEG_APP_LABELER_IMAGES_DIR`
- `SEG_APP_LABELER_MASKS_DIR`
- `SEG_APP_LABELER_COCO_DIR`
- `SEG_APP_LABELER_CACHE_DIR`
- `SEG_APP_LABELER_CATEGORIES`
- `SEG_APP_LABELER_SLIC_N_SEGMENTS`
- `SEG_APP_LABELER_SLIC_COMPACTNESS`
- `SEG_APP_LABELER_SLIC_SIGMA`
- `SEG_APP_LABELER_MIN_COCO_AREA`
- `SEG_APP_JOB_WORKER_COUNT`
- `SEG_APP_JOB_QUEUE_POLL_INTERVAL`
- `SEG_APP_JOB_HEARTBEAT_INTERVAL`
- `SEG_APP_JOB_STALE_HEARTBEAT_SECONDS`

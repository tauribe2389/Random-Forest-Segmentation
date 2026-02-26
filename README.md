# Local Random-Forest Image Segmentation App (Flask + Jinja2)

This project is a local-only web app for:

- Registering COCO datasets from local folders
- Training Random-Forest segmentation models
- Running inference/analysis on selected images
- Tracking model registry and analysis run history in SQLite

No external APIs, no telemetry, no cloud services.

## Project Tree

```text
c:\Users\Antonio\FM
├── app
│   ├── __init__.py
│   ├── routes.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── coco.py
│   │   ├── features.py
│   │   ├── inference.py
│   │   ├── schemas.py
│   │   ├── storage.py
│   │   └── training.py
│   ├── static
│   │   └── style.css
│   └── templates
│       ├── analysis_detail.html
│       ├── analysis_new.html
│       ├── base.html
│       ├── dashboard.html
│       ├── dataset_new.html
│       ├── model_detail.html
│       └── train_status.html
├── instance
├── models
├── runs
├── scripts
│   └── generate_demo_dataset.py
├── requirements.txt
├── run.py
└── README.md
```

## Requirements

- Python 3.10+ recommended
- Windows/macOS/Linux (local filesystem access required)

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
python run.py
```

4. Open:

```text
http://127.0.0.1:5000
```

## Optional: Generate a Demo Dataset

```powershell
python scripts\generate_demo_dataset.py --output-dir demo_dataset --num-images 10 --image-size 256 --seed 42
```

Then register dataset:

- Dataset folder path: `<project>\demo_dataset`
- Image root: `images`
- COCO JSON path: `annotations.json`

## Workflow

1. Register a dataset (`/datasets/new`)
2. On dashboard (`/`), train a model:
   - choose dataset
   - set RF hyperparameters
   - set feature config
   - click Train
3. View model details (`/models/<id>`)
4. Create analysis run (`/analysis/new`):
   - select model
   - input image paths or folder+glob
5. Review run outputs (`/analysis/<run_id>`)

## Data/Model Behavior Notes

- Class labels are never hardcoded.
- Class mapping is dynamic per dataset:
  - `background` is class index `0`
  - COCO categories map to class indices `1..K` in dataset category order.
- Overlap handling is configurable (`higher_area_wins` or `last_annotation_wins`), persisted in model metadata.
- MVP supports polygon segmentations.
  - RLE annotations are skipped with warnings, and count is recorded in metrics.
- Trained artifacts:
  - `model.joblib`
  - `metadata.json` (dataset info, checksum, class mapping, feature/train config, metrics, timestamp, code version)

## Persistence

SQLite DB file is created automatically at:

```text
instance/segmentation.sqlite3
```

Tables are auto-created on startup (`CREATE TABLE IF NOT EXISTS`):

- `datasets`
- `models`
- `analysis_runs`
- `analysis_run_items`

## Environment Variables (Optional)

- `SEG_APP_BASE_DIR`
- `SEG_APP_INSTANCE_DIR`
- `SEG_APP_MODELS_DIR`
- `SEG_APP_RUNS_DIR`
- `SEG_APP_DB_PATH`
- `SEG_APP_CODE_VERSION`
- `SEG_APP_RANDOM_SEED`
- `FLASK_SECRET_KEY`

By default, all paths are local subfolders under the project root.


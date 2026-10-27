"""
notebook_path_config.py
=======================
Configuration cell to prepend at the top of sic-project.ipynb for local
execution.  In Kaggle, set USE_KAGGLE = True (or leave the auto-detection
logic in place).

Run this in Jupyter with:
    %run notebook_path_config.py
Or copy the code into the first cell of the notebook.

IMPORTANT: This file does NOT contain any trained model weights.  It only
configures paths for data loading and artifact export.
"""

import os
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────
# Auto-detect Kaggle environment
ON_KAGGLE = Path("/kaggle/input").exists()

if ON_KAGGLE:
    # Kaggle: dataset mounted at /kaggle/input/datasets/esraameltohamy/sic-proj/
    DATA_CSV = Path("/kaggle/input/datasets/esraameltohamy/sic-proj/data.csv")
    # Kaggle output directory for artifacts
    ARTIFACT_DIR = Path("/kaggle/working/artifacts")
else:
    # Local: data.csv is in the project root
    # Resolve relative to this file's location to be CWD-independent
    _PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_CSV = _PROJECT_ROOT / "data.csv"
    # Local artifact directory = the API artifact directory
    ARTIFACT_DIR = _PROJECT_ROOT / "electricity-theft-api" / "artifacts"

# Override with environment variable if set
_env_data = os.environ.get("SIC_DATA_CSV")
_env_artifacts = os.environ.get("SIC_ARTIFACT_DIR")
if _env_data:
    DATA_CSV = Path(_env_data)
if _env_artifacts:
    ARTIFACT_DIR = Path(_env_artifacts)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_CSV.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_CSV}\n"
        "Options:\n"
        "  1. Place data.csv in the project root directory.\n"
        "  2. Set the SIC_DATA_CSV environment variable to the full path.\n"
        "  3. Run on Kaggle where the dataset is mounted automatically.\n"
    )

print(f"[config] ON_KAGGLE   : {ON_KAGGLE}")
print(f"[config] DATA_CSV    : {DATA_CSV}")
print(f"[config] ARTIFACT_DIR: {ARTIFACT_DIR}")
print(f"[config] data.csv exists: {DATA_CSV.exists()} ({DATA_CSV.stat().st_size / 1e6:.1f} MB)")

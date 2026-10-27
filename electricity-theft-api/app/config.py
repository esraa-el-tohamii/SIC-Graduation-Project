"""
config.py — Application configuration.

All tuneable constants live here so that main.py, model_loader.py,
preprocessing.py, and inference.py import from a single source of truth.
"""
from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(
    os.environ.get("MODEL_ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))
).resolve()
MODELS_DIR = ARTIFACTS_DIR / "models"
EXPLAINABILITY_DIR = ARTIFACTS_DIR / "explainability"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

# Required model and metadata files.
STACK_MODEL_FILE = MODELS_DIR / "final_stack_model.joblib"
CNN_ENSEMBLE_FILE = MODELS_DIR / "cnn_ens_state_dicts.pt"
FEATURE_NAMES_FILE = METADATA_DIR / "stack_feature_names.joblib"
EXPLAINER_FILE = EXPLAINABILITY_DIR / "explainer.joblib"
NUM_CNN_MODELS        = 5

# ── Inference constants ────────────────────────────────────────────────────
EXPECTED_READINGS: int = 1034          # number of daily readings required
NUM_WEEKS: int = 148                   # ceil(1034 / 7)
PAD_DAYS: int = NUM_WEEKS * 7 - EXPECTED_READINGS   # = 2

# CNN architecture constants (must match training)
CNN_WIDE_IN: int = EXPECTED_READINGS + 1   # 1035 (readings + missing_pct)
CNN_IN_CH:   int = 2                       # channel 0: clean, channel 1: mask

# Decision threshold (not calibrated — see README)
DEFAULT_THRESHOLD: float = 0.5

# Target label meaning
LABEL_NAMES = {0: "Normal", 1: "Theft flag"}

# ── Runtime ───────────────────────────────────────────────────────────────
APP_VERSION: str = "1.0.0"
MODEL_VERSION: str = "WideDeepV2-5seed + XGBoost-stack"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

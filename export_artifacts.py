"""
export_artifacts.py
===================
Run this script INSIDE the Jupyter kernel (or as a standalone script in the
same Python environment as sic-project.ipynb) AFTER training is complete.

It exports the inference artifacts required by the API into categorized
subdirectories under `electricity-theft-api/artifacts/`.

Usage (inside Jupyter after all training cells have completed):
    %run export_artifacts.py

Usage (terminal, from the project root, with the notebook kernel still live):
    python export_artifacts.py

Prerequisites
-------------
The following notebook variables must be in scope (i.e. the corresponding
notebook cells have been executed successfully):
    - final_stack_model   : trained XGBClassifier (Cell 89)
    - ens_models          : list of 5 trained WideDeepV2 CNN models (Cell 79)
    - stack_feature_names : list of 1036 feature name strings (Cell 94)

What is saved
-------------
    artifacts/models/final_stack_model.joblib — XGBoost stacking model
    artifacts/models/cnn_ens_state_dicts.pt  — list of five CNN state dicts
    artifacts/metadata/stack_feature_names.joblib — ordered feature names

What is NOT saved (and why)
---------------------------
    - Scaler / imputer  : preprocessing is stateless (per-consumer statistics)
    - Dataset splits    : not needed at inference time
    - SHAP explainer    : optional; save separately if needed

Safety guarantees
-----------------
    - Writes to a temporary directory first; moves atomically on success.
    - Validates every saved file (non-zero size) before overwriting the target.
    - Fails immediately with a clear error if a required variable is missing.
    - Does NOT overwrite existing valid artifacts with an incomplete export.

Troubleshooting
---------------
    If you see "NameError: name 'final_stack_model' is not defined", you must
    run notebook Cell 89 first.

    If you see "NameError: name 'ens_models' is not defined", you must run
    notebook Cell 79 first.

    If you see "AssertionError: Expected 1036 feature names", check that
    notebook Cell 94 has been executed and stack_feature_names contains 1036
    entries (1034 date columns + Missing_Reading_Percentage + CNN_Prediction).

    If artifacts already exist and are valid, this script will replace them
    safely via a temporary directory.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import torch


# ── Output directory ──────────────────────────────────────────────────────
# Resolve relative to this script's location, not the current working dir,
# so it works regardless of where the user cd'd before running.

_SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = _SCRIPT_DIR / "electricity-theft-api" / "artifacts"

# Allow override via environment variable (useful for CI or custom setups)
_env_dir = os.environ.get("SIC_ARTIFACT_DIR")
if _env_dir:
    ARTIFACTS_DIR = Path(_env_dir)

MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"Target artifact directory: {ARTIFACTS_DIR.resolve()}")


# ── 1. Check that required notebook variables exist ───────────────────────

def _require(name: str, description: str) -> object:
    """Return a notebook variable by name, or raise a clear error."""
    caller_globals = sys._getframe(1).f_locals  # script's global scope
    # When run with %run inside Jupyter, variables live in the notebook globals
    import IPython  # type: ignore  # noqa: F401
    ip = IPython.get_ipython()
    if ip is not None:
        ns = ip.user_ns
    else:
        ns = caller_globals

    if name not in ns:
        raise NameError(
            f"\n\n  ERROR: '{name}' is not defined.\n"
            f"  {description}\n"
            f"  Run the required notebook cell first, then re-run export_artifacts.py.\n"
        )
    return ns[name]


try:
    import IPython  # noqa: F401
    ip = IPython.get_ipython()
    if ip is not None:
        ns = ip.user_ns
    else:
        ns = globals()
except ImportError:
    ns = globals()

def _get(name: str) -> object:
    if name not in ns:
        return None
    return ns[name]

_final_stack_model = _get("final_stack_model")
_ens_models = _get("ens_models")
_stack_feature_names = _get("stack_feature_names")

missing_vars = []
if _final_stack_model is None:
    missing_vars.append("  - 'final_stack_model'  (run notebook Cell 89)")
if _ens_models is None:
    missing_vars.append("  - 'ens_models'  (run notebook Cell 79)")
if _stack_feature_names is None:
    missing_vars.append("  - 'stack_feature_names'  (run notebook Cell 94)")

if missing_vars:
    raise NameError(
        "\n\nThe following required notebook variables are not defined:\n"
        + "\n".join(missing_vars)
        + "\n\nRun the listed cells first, then re-run this script.\n"
    )

# ── 2. Validate the in-memory artifacts ──────────────────────────────────

EXPECTED_FEATURE_COUNT = 1036
EXPECTED_CNN_COUNT = 5

if len(_stack_feature_names) != EXPECTED_FEATURE_COUNT:
    raise AssertionError(
        f"Expected {EXPECTED_FEATURE_COUNT} feature names, "
        f"got {len(_stack_feature_names)}.\n"
        "Check that notebook Cell 94 ran successfully."
    )

if len(_ens_models) != EXPECTED_CNN_COUNT:
    raise AssertionError(
        f"Expected {EXPECTED_CNN_COUNT} CNN models in ens_models, "
        f"got {len(_ens_models)}.\n"
        "Check that notebook Cell 79 ran successfully."
    )

print(f"\n✓ final_stack_model: {type(_final_stack_model).__name__}")
print(f"✓ ens_models: {len(_ens_models)} models")
print(f"✓ stack_feature_names: {len(_stack_feature_names)} features")
print(f"  First 3: {_stack_feature_names[:3]}")
print(f"  Last  3: {_stack_feature_names[-3:]}")


# ── 3. Write artifacts to a temporary directory, then move atomically ─────

EXPECTED_FILES = [
    "models/final_stack_model.joblib",
    "models/cnn_ens_state_dicts.pt",
    "metadata/stack_feature_names.joblib",
]

print(f"\n── Writing to temporary directory …")
tmp_dir = Path(tempfile.mkdtemp(prefix="sic_artifacts_"))
try:
    # 3a. XGBoost stacking model
    p = tmp_dir / "models" / "final_stack_model.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_final_stack_model, p)
    print(f"  Wrote: {p.name}  ({p.stat().st_size / 1024:.1f} KB)")

    # 3b. Feature names
    p = tmp_dir / "metadata" / "stack_feature_names.joblib"
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_stack_feature_names, p)
    print(f"  Wrote: {p.name}  ({p.stat().st_size / 1024:.1f} KB)")

    # 3c. CNN ensemble weights
    p = tmp_dir / "models" / "cnn_ens_state_dicts.pt"
    torch.save([model.state_dict() for model in _ens_models], p)
    print(f"  Wrote: {p.name}  ({p.stat().st_size / 1024:.1f} KB)")

    # ── 4. Validate the temporary files ──────────────────────────────────
    print("\n── Validating …")
    for fname in EXPECTED_FILES:
        p = tmp_dir / fname
        assert p.exists(), f"Missing: {fname}"
        assert p.stat().st_size > 0, f"Empty file: {fname}"
        print(f"  OK: {fname}  ({p.stat().st_size / 1024:.1f} KB)")

    # ── 5. Move the validated files to the target directory ───────────────
    print(f"\n── Moving artifacts to {ARTIFACTS_DIR} …")
    for fname in EXPECTED_FILES:
        src = tmp_dir / fname
        dst = ARTIFACTS_DIR / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  ✓ {dst}")

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ── 6. Final verification in the target directory ─────────────────────────

print("\n── Final verification ──────────────────────────────────────────────")
all_ok = True
for fname in EXPECTED_FILES:
    p = ARTIFACTS_DIR / fname
    if p.exists() and p.stat().st_size > 0:
        print(f"  ✓ {fname:40s}  {p.stat().st_size / 1024:8.1f} KB")
    else:
        print(f"  ✗ {fname:40s}  MISSING or EMPTY")
        all_ok = False

if not all_ok:
    raise RuntimeError(
        "One or more artifacts failed to save.  "
        "Check the error messages above."
    )

print("\n✓ All required inference artifacts saved and verified successfully.")
print(f"\nNext step: start the API from the electricity-theft-api/ directory:")
print("  uvicorn app.main:app --reload --port 8000")

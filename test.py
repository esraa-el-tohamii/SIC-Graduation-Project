import sys
import importlib.metadata as metadata

print("Python:", sys.version)

packages = [
    "torch",
    "tensorflow",
    "keras",
    "scikit-learn",
    "xgboost",
    "fastapi",
    "uvicorn",
    "joblib",
    "numpy",
    "pandas",
]

for package in packages:
    try:
        version = metadata.version(package)
        print(f"{package}: {version}")
    except metadata.PackageNotFoundError:
        print(f"{package}: Not installed")
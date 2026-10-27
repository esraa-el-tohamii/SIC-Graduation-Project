# conftest.py — shared pytest configuration
# Ensures the project root is on sys.path so "app.*" imports work correctly.
import sys
from pathlib import Path

# Add the electricity-theft-api directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

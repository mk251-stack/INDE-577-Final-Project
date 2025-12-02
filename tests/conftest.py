import sys
from pathlib import Path

# Ensure the src directory is on the path so tests can import the package
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))
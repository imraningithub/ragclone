import os
import sys
from pathlib import Path

os.environ.setdefault("PROVIDER", "fake")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

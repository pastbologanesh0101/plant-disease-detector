import os
import sys

# Ensure the repo root (containing detector.py / synthetic.py / detect.py) is
# importable regardless of how pytest is invoked.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

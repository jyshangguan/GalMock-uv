#!/usr/bin/env python
"""Run galmockuv demo — Layer A: build intrinsic cube.

Usage (from repo root, in alma/DysmalPy conda env):
    python demo/run_layer_a.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galmockuv import run_pipeline, load_metadata

config_path = str(ROOT / 'demo' / 'config.toml')

print("=" * 60)
print("galmockuv demo — Layer A")
print("=" * 60)

results = run_pipeline(config_path, layers='A')

print("\nDone.")

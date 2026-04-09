#!/usr/bin/env python
"""galmockuv entry point: run the mock ALMA CO observation pipeline.

Usage:
    # DysmalPy environment — build intrinsic cube only:
    python galmockuv.py config.toml --layers A

    # CASA environment — simulate + measure:
    casa --nologger --nogui -c "exec(open('galmockuv.py').read())" config.toml --layers B+C
"""

import sys
import argparse
from pathlib import Path

# Ensure mocks/ is on sys.path so ``import galmockuv`` works
MOCKS_DIR = str(Path(__file__).resolve().parent)
if MOCKS_DIR not in sys.path:
    sys.path.insert(0, MOCKS_DIR)


def main():
    parser = argparse.ArgumentParser(
        description='galmockuv: mock ALMA CO observation pipeline'
    )
    parser.add_argument('config', help='Path to TOML or YAML config file')
    parser.add_argument(
        '--layers', default='A+B+C',
        choices=['A', 'B+C', 'A+B+C'],
        help='Pipeline layers to run (default: A+B+C)',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Override output directory',
    )
    args = parser.parse_args()

    from galmockuv import run_pipeline

    run_pipeline(args.config, layers=args.layers, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

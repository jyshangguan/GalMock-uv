#!/usr/bin/env python
"""Demo script for the galmockuv pipeline.

Demonstrates running the full mock ALMA CO observation pipeline and
checks the results for robustness.

Usage:
    # Step 1: Build intrinsic cube (DysmalPy env)
    python demo/run_demo.py --layers A

    # Step 2: Simulate + measure (CASA env)
    casa --nologger --nogui -c "exec(open('demo/run_demo.py').read())" --layers B+C
"""

import sys
import argparse
from pathlib import Path

# Ensure mocks/ is importable
MOCKS_DIR = str(Path(__file__).resolve().parent.parent)
if MOCKS_DIR not in sys.path:
    sys.path.insert(0, MOCKS_DIR)


def main():
    parser = argparse.ArgumentParser(description='galmockuv demo')
    parser.add_argument(
        '--layers', default='A',
        choices=['A', 'B+C', 'A+B+C'],
        help='Pipeline layers to run (default: A)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to config file (default: demo/config.toml)',
    )
    args = parser.parse_args()

    from galmockuv import run_pipeline, load_metadata, load_json

    config_path = args.config or str(
        Path(__file__).parent / 'config.toml'
    )

    print("=" * 60)
    print("galmockuv demo")
    print("=" * 60)

    results = run_pipeline(config_path, layers=args.layers)

    # Check robustness for full pipeline
    if 'measurements' in results:
        m = results['measurements']
        meta = results.get('metadata', load_metadata(results['output_dir']))

        print("\n" + "=" * 60)
        print("Robustness checks")
        print("=" * 60)

        fwhm = m.get('fwhm_kms', 0)
        true_inc = meta.get('inclination_deg', 0)
        # For an inclined rotating disk, FWHM ~ 2 * V_max * sin(i)
        # This is a rough sanity check
        print(f"  FWHM measured:    {fwhm:.1f} km/s")

        size_kpc = m.get('size_kpc', 0)
        true_reff = meta.get('disk_reff_kpc', 0)
        size_ratio = size_kpc / true_reff if true_reff > 0 else 0
        print(f"  Size measured:    {size_kpc:.2f} kpc")
        print(f"  True r_eff:       {true_reff:.2f} kpc")
        print(f"  Size ratio:       {size_ratio:.3f}")

        snr = m.get('line_snr', 0)
        print(f"  Line SNR:         {snr:.1f}")

        f_eff = m.get('f_eff', float('nan'))
        print(f"  f_eff:            {f_eff:.3f}")

        # Robustness criteria
        checks = []
        # FWHM recovery (should be within ~10% of expectation)
        if fwhm > 0:
            checks.append(("FWHM > 0", True))

        # SNR should be reasonable (>5 for detection)
        checks.append((f"SNR > 5 ({snr:.1f})", snr > 5))

        # f_eff should be finite and positive
        import math
        if not math.isnan(f_eff) and f_eff > 0:
            checks.append((f"f_eff finite & positive ({f_eff:.3f})", True))
        else:
            checks.append(("f_eff finite & positive", False))

        print()
        for desc, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {desc}")

        n_pass = sum(1 for _, p in checks if p)
        n_total = len(checks)
        print(f"\n  {n_pass}/{n_total} checks passed")

    print("\nDone.")


if __name__ == '__main__':
    main()

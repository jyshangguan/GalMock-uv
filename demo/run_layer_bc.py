"""Run galmockuv demo — Layers B+C: simulate + measure.

Usage (from repo root, in CASA):
    casa --nologger --nogui -c "exec(open('demo/run_layer_bc.py').read())"
"""
import sys
import os

# Ensure package is importable: script lives in demo/, so repo root is one level up.
# Use CWD (where 'casa' was invoked) rather than __file__ (unavailable in exec).
root = os.path.abspath('.')
# If CWD is demo/, go up one level
if os.path.basename(root) == 'demo':
    root = os.path.dirname(root)
if root not in sys.path:
    sys.path.insert(0, root)

from galmockuv import run_pipeline, load_metadata

config_path = os.path.join(root, 'demo', 'config.toml')

print("=" * 60)
print("galmockuv demo — Layers B+C")
print("=" * 60)

results = run_pipeline(config_path, layers='B+C')

# Robustness checks
if 'measurements' in results:
    m = results['measurements']
    meta = results.get('metadata', load_metadata(results['output_dir']))

    print("\n" + "=" * 60)
    print("Robustness checks")
    print("=" * 60)

    fwhm = m.get('fwhm_kms', 0)
    size_kpc = m.get('size_kpc', 0)
    true_reff = meta.get('disk_reff_kpc', 0)
    snr = m.get('line_snr', 0)
    f_eff = m.get('f_eff', float('nan'))

    print(f"  FWHM measured:    {fwhm:.1f} km/s")
    print(f"  Size measured:    {size_kpc:.2f} kpc")
    print(f"  True r_eff:       {true_reff:.2f} kpc")
    print(f"  Size ratio:       {size_kpc / true_reff:.3f}" if true_reff > 0 else "  Size ratio:       N/A")
    print(f"  Line SNR:         {snr:.1f}")
    print(f"  f_eff:            {f_eff:.3f}")

    import math
    checks = []
    if fwhm > 0:
        checks.append(("FWHM > 0", True))
    checks.append((f"SNR > 5 ({snr:.1f})", snr > 5))
    if not math.isnan(f_eff) and f_eff > 0:
        checks.append((f"f_eff finite & positive ({f_eff:.3f})", True))
    else:
        checks.append(("f_eff finite & positive", False))

    print()
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
    print(f"\n  {sum(1 for _, p in checks if p)}/{len(checks)} checks passed")

print("\nDone.")

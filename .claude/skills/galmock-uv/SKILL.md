---
name: galmock-uv
description: >
  This skill should be used when the user asks to "run a mock ALMA observation",
  "simulate CO line data", "create a mock galaxy cube", "run galmockuv",
  "measure dynamical mass from CO", "compute f_eff", "run the mock pipeline",
  "build a DysmalPy cube", "simulate ALMA with CASA", or "measure FWHM and size
  from a mock observation".
---

# galmockuv Workflow

Build an intrinsic CO line cube from a galaxy model, simulate ALMA
interferometric observations, image the data, and measure CO line FWHM,
source size, and dynamical mass proxy with efficiency factor *f*_eff.

## Prerequisites

- **Layer A** requires a DysmalPy environment (e.g. `alma` conda env with
  `dysmalpy`, `spectral-cube`).
- **Layers B+C** require CASA (`casatools`, `casatasks`).
- The `galmockuv` package must be importable.  The entry point
  `galmockuv.py` inserts its parent directory into `sys.path`.
- A TOML config file describing the galaxy and ALMA observation.

## Workflow

### Step 1 — Write a config file

Create a TOML config.  All keys are flat (no nested tables).  Minimal example:

```toml
source_id = "my_galaxy"
redshift = 2.5
co_restfreq_ghz = 345.796          # CO(3-2)
log10_baryonic_mass_msun = 10.62
disk_reff_kpc = 8.0
disk_invq = 5.0
n_disk = 1.0                       # exponential
bt = 0.0                           # no bulge
log10_halo_mass_msun = 12.0
halo_concentration = 5.0
inclination_deg = 45.0
pa_deg = 30.0
intrinsic_sigma_kms = 30.0
pressure_support = true
pixscale_arcsec = 0.15
npix_x = 201
npix_y = 201
channel_width_kms = 10.0
velocity_start_kms = -1000.0
nchan = 201
flux_scale = 4.3e-6                # normalize to ~mJy
antennalist = "alma.cycle4.2"
totaltime = "600s"
thermalnoise = "tsys-atm"
user_pwv = 1.5
mapsize = "35arcsec"
seed = 42
```

A full example is in `demo/config.toml`.

### Step 2 — Run Layer A (DysmalPy environment)

Build the intrinsic CO cube.  `conda run` does not work on this system;
use the shell hook to activate the environment:

```bash
eval "$(/home/shangguan/Softwares/miniconda3/bin/conda shell.bash hook)" && conda activate alma && python galmockuv.py config.toml --layers A
```

Or programmatically:

```python
from galmockuv import run_pipeline
results = run_pipeline("config.toml", layers="A")
print(results["cube_path"])   # path to intrinsic_cube.fits
```

Output: `intrinsic_cube.fits` (FITS cube with frequency axis),
`metadata.yaml` (true parameters).

### Step 3 — Run Layers B+C (CASA environment)

Simulate ALMA observation and measure.  The naive `exec(open(...).read())`
fails because `__file__` and `argparse` are not defined in the exec scope.
Pass an explicit globals dict:

```bash
casa --nologger --nogui -c "
import sys, argparse, pathlib
sys.argv = ['galmockuv.py', 'config.toml', '--layers', 'B+C']
galmockuv_globals = {'__name__': '__main__', '__file__': 'galmockuv.py', 'sys': sys, 'argparse': argparse, 'pathlib': pathlib, '__builtins__': __builtins__}
exec(open('galmockuv.py').read(), galmockuv_globals)
"
```

Or programmatically (inside CASA):

```python
from galmockuv import run_pipeline
results = run_pipeline("config.toml", layers="B+C")
```

Output: `{source_id}.ms` (MeasurementSet), cleaned image directory,
`measurements.json`, `measurement_details.yaml`.

### Step 4 — Read results

```python
from galmockuv import load_metadata, load_json

meta = load_metadata("output/my_galaxy")
meas = load_json("output/my_galaxy", "measurements.json")

print(f"FWHM       = {meas['fwhm_kms']:.1f} +/- {meas['fwhm_err_kms']:.1f} km/s")
print(f"Size       = {meas['size_kpc']:.1f} kpc")
print(f"SNR        = {meas['line_snr']:.1f}")
print(f"Proxy mass = {meas['proxy_mass_msun']:.2e} Msun")
print(f"True mass  = {meas['true_mass_msun']:.2e} Msun")
print(f"f_eff      = {meas['f_eff']:.3f}")
```

### Step 5 — Generate report plots (optional)

Requires CASA environment (uses `casatools.ms` for visibility data and
`galmockuv.casa_utils` for UV fitting):

```bash
casa --nologger --nogui -c "
import sys, pathlib
exec(open('demo/generate_report_plots.py').read(), {'__name__': '__main__', '__file__': 'demo/generate_report_plots.py', '__builtins__': __builtins__, 'sys': sys, 'pathlib': pathlib})
"
```

Plots are saved to `demo/figs/` (git-tracked).  Pipeline data (JSON, CL)
stays in the output directory (git-ignored).

Produces 4 figures: `intrinsic_summary.png`, `visibility_data.png`,
`cleaned_summary.png`, `uv_amplitude_fit.png`.

## Config Parameters

### Galaxy model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source_id` | (required) | Output directory name |
| `redshift` | (required) | Redshift |
| `co_restfreq_ghz` | (required) | CO line rest frequency |
| `log10_baryonic_mass_msun` | 10.0 | log10(M_star + M_gas) |
| `disk_reff_kpc` | 5.0 | Effective radius |
| `disk_invq` | 5.0 | Axis ratio (thickness) |
| `n_disk` | 1.0 | Sersic index (1 = exponential) |
| `bt` | 0.0 | Bulge-to-total ratio |
| `log10_halo_mass_msun` | 12.0 | NFW virial mass |
| `halo_concentration` | 5.0 | NFW concentration |
| `inclination_deg` | 45.0 | Disk inclination |
| `pa_deg` | 0.0 | Position angle |
| `intrinsic_sigma_kms` | 30.0 | Velocity dispersion |
| `pressure_support` | true | Asymmetric drift correction |
| `flux_scale` | 1.0 | Normalize to mJy (tuning param) |

### ALMA simulation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `antennalist` | `alma.cycle7.7` | ALMA config |
| `totaltime` | `"600s"` | On-source time |
| `integration_s` | 10.0 | Correlator dump time |
| `thermalnoise` | `"tsys-atm"` | Noise model |
| `user_pwv` | 1.5 | PWV (mm) |
| `mapsize` | `"5arcsec"` | FOV |
| `seed` | 42 | Random seed |

## Interpreting Results

### f_eff

The key output is *f*_eff = *M*_baryonic / *M*_proxy, where
*M*_proxy = FWHM^2 * D / G.  This factor encodes pressure support, beam
smearing, inclination effects, and disk-to-halo mass ratio.

| f_eff range | Interpretation |
|-------------|---------------|
| ~0.1--0.2 | Typical for high-z star-forming galaxies |
| ~0.3--0.5 | Massive, rotation-dominated disks |
| < 0.1 | Strong pressure support or beam dilution |
| > 0.5 | Check config — may be face-on or unusually massive halo |

### FWHM recovery

The measured FWHM (from Gaussian fit to cleaned spectrum) should recover the
intrinsic FWHM at ~95--100% for SNR > 10.  Lower SNR or aggressive cleaning
can bias FWHM low.

### Size measurement

The UV-fit size comes from `uvmodelfit` on channel-averaged visibilities.
For marginally resolved sources (theta_source < beam), the size uncertainty
is large and the measurement may be a lower limit.

## Output Files

### Pipeline data (in output directory, git-ignored)

| File | Description |
|------|-------------|
| `intrinsic_cube.fits` | Intrinsic CO cube (nchan, ny, nx) |
| `metadata.yaml` | True input parameters |
| `{source_id}.ms` | Simulated MeasurementSet |
| `casa_sim_params.yaml` | CASA simulation parameters used |
| `measurements.json` | FWHM, size, mass, f_eff |
| `measurement_details.yaml` | Full measurement details |
| `report_data.json` | Key numbers extracted by plotting script |
| `uvfit_report.cl` | CASA component list from UV model fit |

### Report figures (in `demo/figs/`, git-tracked)

| File | Description |
|------|-------------|
| `intrinsic_summary.png` | Moment 0/1/2 + spectrum (intrinsic) |
| `visibility_data.png` | Visibility amplitudes + uv-coverage |
| `cleaned_summary.png` | Moment 0/1/2 + spectrum (cleaned) |
| `uv_amplitude_fit.png` | Binned visibilities + model fit |

## Demo

A complete working demo is in `demo/`:

```bash
# Layer A (DysmalPy env)
eval "$(/home/shangguan/Softwares/miniconda3/bin/conda shell.bash hook)" && conda activate alma && python galmockuv.py demo/config.toml --layers A

# Layers B+C (CASA env)
casa --nologger --nogui -c "
import sys, argparse, pathlib
sys.argv = ['galmockuv.py', 'demo/config.toml', '--layers', 'B+C']
galmockuv_globals = {'__name__': '__main__', '__file__': 'galmockuv.py', 'sys': sys, 'argparse': argparse, 'pathlib': pathlib, '__builtins__': __builtins__}
exec(open('galmockuv.py').read(), galmockuv_globals)
"

# Report plots (CASA env)
casa --nologger --nogui -c "
import sys, pathlib
exec(open('demo/generate_report_plots.py').read(), {'__name__': '__main__', '__file__': 'demo/generate_report_plots.py', '__builtins__': __builtins__, 'sys': sys, 'pathlib': pathlib})
"
```

Demo config: z=2.5 galaxy, Mbar=10^10.62, Mhalo=10^12, 3 kpc disk,
45 deg inclination, ALMA C43-2, 600s integration.  Expected results:
FWHM ~302 km/s, SNR ~12, size ~6.6 kpc, f_eff ~0.30.

## Troubleshooting

### `conda run` fails with `__conda_exe` not found

Use the shell hook instead:

```bash
eval "$(/home/shangguan/Softwares/miniconda3/bin/conda shell.bash hook)" && conda activate alma
```

### `exec(open('galmockuv.py').read())` fails in CASA

`exec()` does not define `__file__`, `argparse`, or `sys` in the exec'd
code's namespace.  Pass them explicitly in a globals dict (see Step 3 above).

### DysmalPy segfaults on import (erfa/pyerfa conflict)

If `import dysmalpy` segfaults in `erfa.core.epj2jd`, the `pyerfa` package
may need reinstalling:

```bash
pip install --force-reinstall --no-deps pyerfa
```

### scipy `_spropack` import error

If `from scipy.sparse.linalg._propack import _spropack` fails, the scipy
install may be corrupted (e.g. from a numpy major version mismatch during
`pip install --force-reinstall scipy`).  Fix by reinstalling both in the
correct order:

```bash
pip install 'numpy<2.0.0' 'scipy<1.14'
```

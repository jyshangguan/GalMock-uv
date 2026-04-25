# galmockuv -- Mock ALMA CO Observations of High-z Galaxies

`galmockuv` simulates spatially-resolved ALMA CO line observations of high-redshift galaxies. Given a set of physical parameters (mass, size, inclination, kinematics), it produces a simulated MeasurementSet and cleaned image cube, then measures the CO line FWHM, source size, and dynamical mass proxy. The package is designed to quantify the biases and scatter in CO-based dynamical mass estimates.

## Pipeline Overview

The pipeline has three layers, each with a different environment requirement:

| Layer | Function | Environment | Output |
|-------|----------|-------------|--------|
| **A** | Build intrinsic CO cube from galaxy model | DysmalPy (Python) | `intrinsic_cube.fits` |
| **B** | Simulate ALMA interferometric observation | CASA | `{source_id}.ms` |
| **C** | Image, fit line, measure size & mass | CASA | `measurements.json` |

Layer A requires a Python environment with DysmalPy and SpectralCube. Layers B and C require CASA (they use `casatasks` and `casatools`).

## Quick Start

### 1. Write a config file

Create a TOML file (or YAML for backward compatibility) describing the galaxy and ALMA observation. A full example is at `demo/config.toml`:

```toml
# Source identification
source_id = "z25_demo"

# Cosmology & redshift
redshift = 2.5

# CO line
co_restfreq_ghz = 345.796   # CO(3-2) rest frequency in GHz

# Baryonic component (exponential disk)
log10_baryonic_mass_msun = 10.62
disk_reff_kpc = 8.0
disk_invq = 5.0              # axis ratio (thickness)
n_disk = 1.0                 # Sersic index (1 = exponential)
bt = 0.0                     # bulge-to-total ratio
noord_flat = true

# Dark matter halo (NFW)
log10_halo_mass_msun = 12.0
halo_concentration = 5.0

# Geometry
inclination_deg = 45.0
pa_deg = 30.0

# Gas kinematics
intrinsic_sigma_kms = 30.0    # constant velocity dispersion
sigmaz_kpc = 0.9              # vertical scale height
pressure_support = true       # asymmetric drift correction

# Cube sampling
pixscale_arcsec = 0.15
npix_x = 201
npix_y = 201
channel_width_kms = 10.0
velocity_start_kms = -1000.0
nchan = 201
flux_scale = 4.3e-6           # normalize peak to ~mJy level

# ALMA simulation
alma_mode = "simobserve"
totaltime = "600s"
integration_s = 10.0
antennalist = "alma.cycle4.2"
thermalnoise = "tsys-atm"
user_pwv = 1.5
mapsize = "35arcsec"
seed = 42

# Measurement
tclean_niter = 1000
size_fit_model = "G"          # Gaussian UV fit
```

### 2. Run the pipeline

Layer A must be run in a DysmalPy environment. Layers B+C must be run inside CASA.

```bash
# Layer A (DysmalPy environment):
python galmockuv.py demo/config.toml --layers A

# Layers B+C (CASA environment):
casa --nologger --nogui -c "exec(open('galmockuv.py').read())" demo/config.toml --layers B+C
```

The entry point `galmockuv.py` is a thin argparse wrapper. It accepts a config path and `--layers` (`A`, `B+C`, or `A+B+C`). An optional `--output-dir` overrides the output directory from the config.

### 3. Read the results

Measurements are saved as JSON in the output directory:

```python
from galmockuv import load_json

meas = load_json("outputs/demo/z25_demo", "measurements.json")
print(f"FWHM = {meas['fwhm_kms']:.1f} km/s")
print(f"Size = {meas['size_kpc']:.1f} kpc")
print(f"Proxy mass = {meas['proxy_mass_msun']:.2e} Msun")
print(f"f_eff = {meas['f_eff']:.3f}")
```

Key output fields:

| Field | Description |
|-------|-------------|
| `fwhm_kms` | CO line FWHM via non-parametric half-max method |
| `fwhm_err_kms` | FWHM uncertainty (0 for non-parametric) |
| `line_snr` | Peak line flux / rms of line-free channels |
| `size_arcsec` | Source angular size from `uvmodelfit` or MCMC |
| `size_kpc` | Physical size (arcsec * kpc/arcsec) |
| `proxy_mass_msun` | FWHM^2 * D / G |
| `true_mass_msun` | Input baryonic mass |
| `f_eff` | M_baryonic / M_proxy |

## Config Reference

All parameters are optional and have defaults. Here are the most commonly adjusted ones:

### Galaxy model (Layer A)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source_id` | (required) | Output directory name |
| `redshift` | (required) | Redshift |
| `co_restfreq_ghz` | (required) | CO line rest frequency |
| `log10_baryonic_mass_msun` | 10.0 | log10 total baryonic mass (stars + gas) |
| `disk_reff_kpc` | 5.0 | Disk effective radius |
| `disk_invq` | 5.0 | Axis ratio (sets disk thickness) |
| `n_disk` | 1.0 | Sersic index (1 = exponential disk) |
| `bt` | 0.0 | Bulge-to-total ratio |
| `log10_halo_mass_msun` | 12.0 | NFW virial mass |
| `halo_concentration` | 5.0 | NFW concentration parameter |
| `inclination_deg` | 45.0 | Disk inclination |
| `pa_deg` | 0.0 | Position angle (E of N) |
| `intrinsic_sigma_kms` | 30.0 | Constant isotropic velocity dispersion |
| `pressure_support` | true | Include asymmetric drift correction |

### ALMA simulation (Layer B)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `antennalist` | `alma.cycle7.7` | ALMA antenna configuration |
| `totaltime` | `"600s"` | Total on-source integration time |
| `integration_s` | 10.0 | Correlator integration time |
| `thermalnoise` | `"tsys-atm"` | Noise model (`tsys-atm` or `tsys-sim`) |
| `user_pwv` | 1.5 | Precipitable water vapor (mm) |
| `mapsize` | `"5arcsec"` | FOV for simulation |
| `seed` | 42 | Random seed for reproducibility |

### Measurement (Layer C)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tclean_niter` | 1000 | `tclean` minor cycle iterations |
| `size_fit_model` | `"G"` | UV model component type (`G` = Gaussian) |

## Programmatic API

For batch processing or custom workflows, use the Python API directly:

```python
from galmockuv import load_config, run_pipeline

config = load_config("my_galaxy.toml")

# Run all layers (requires both environments)
results = run_pipeline("my_galaxy.toml", layers="A+B+C")

# Or run layers individually
results_a = run_pipeline("my_galaxy.toml", layers="A")
# ... switch to CASA environment ...
results_bc = run_pipeline("my_galaxy.toml", layers="B+C")
```

`run_pipeline()` returns a dict with keys depending on which layers ran:

- **Layer A**: `config_path`, `output_dir`, `cube_path`, `metadata`
- **Layer B**: `ms_path`, `sim_metadata`
- **Layer C**: `measurements`, `measurement_details`

## Package Structure

```
galmockuv/
  __init__.py       # Version, lazy imports
  env.py            # Environment detection (is_casa_env, is_dysmalpy_env)
  io.py             # Config loading (TOML/YAML), metadata/JSON I/O
  build_cube.py     # Layer A: DysmalPy galaxy model -> intrinsic cube
  simulate.py       # Layer B: CASA simobserve -> MeasurementSet
  measure.py        # Layer C: tclean + uvmodelfit -> measurements
  pipeline.py       # Orchestration (run_pipeline)
  plotting.py       # Diagnostic plots (moment maps, spectra, UV)
  casa_utils.py     # Vendored UV analysis utilities
```

## Demo

A complete working demo is in `demo/`:

- `demo/config.toml` -- config for a z=2.5 galaxy
- `demo/run_layer_a.py` -- Layer A script (DysmalPy env)
- `demo/run_layer_bc.py` -- Layers B+C script (CASA env)
- `demo/run_report_plots.py` -- generates publication-quality figures (any Python env)
- `demo/demo_report.md` -- validation report with all figures
- `demo/output/z25_demo/` -- demo output files

To reproduce:

```bash
# 1. Layer A (DysmalPy env)
python demo/run_layer_a.py

# 2. Layers B+C (CASA env)
casa --nologger --nogui -c "exec(open('demo/run_layer_bc.py').read())"

# 3. Generate report plots (any Python env)
python demo/run_report_plots.py
```

The demo produces:

| File | Description |
|------|-------------|
| `intrinsic_summary.png` | Moment 0, 1, 2 + spectrum of the intrinsic cube |
| `visibility_data.png` | Visibility amplitudes and uv-coverage |
| `cleaned_summary.png` | Moment 0, 1, 2 + spectrum of the cleaned cube |
| `uv_amplitude_fit.png` | Binned visibilities with model fit |
| `corner_plot.png` | MCMC posterior corner plot |
| `measurements.json` | Quantitative measurements (FWHM, size, mass, f_eff) |

## How f_eff is Defined

The key output of this package is *f*_eff, the efficiency factor relating the true baryonic mass to the dynamical mass proxy:

*f*_eff = *M*_baryonic / *M*_proxy

where *M*_proxy = FWHM^2 * D / G, with D being the UV-fit source size. This factor encodes the combined effect of pressure support, beam smearing, inclination uncertainty, and disk-to-halo mass ratio. Values near 1 indicate the proxy mass is a good estimate of the baryonic mass; values << 1 indicate significant overestimation. For typical high-z star-forming galaxies, *f*_eff ~ 0.1--0.2.

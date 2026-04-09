# galmockuv — Mock ALMA CO Observations of High-z Galaxies

## Overview

galmockuv simulates spatially-resolved ALMA CO line observations of high-redshift
galaxies.  Given physical parameters (mass, size, inclination, kinematics), it
builds an intrinsic CO cube with DysmalPy, simulates ALMA interferometric
observations with CASA `simobserve`, images and measures the cleaned cube, and
outputs FWHM, source size, and dynamical mass proxy with efficiency factor
*f*_eff = *M*_baryonic / *M*_proxy.

## Module Map

| Module | Purpose |
|--------|---------|
| `galmockuv/__init__.py` | Version (`0.1.0`), lazy imports of core functions |
| `galmockuv/env.py` | Environment detection (`is_casa_env`, `is_dysmalpy_env`) |
| `galmockuv/io.py` | Config loading (TOML/YAML), metadata/JSON I/O |
| `galmockuv/build_cube.py` | Layer A: DysmalPy galaxy model -> intrinsic FITS cube |
| `galmockuv/simulate.py` | Layer B: CASA `simobserve` -> MeasurementSet |
| `galmockuv/measure.py` | Layer C: `tclean` + `uvmodelfit` -> measurements |
| `galmockuv/pipeline.py` | Orchestration (`run_pipeline`) |
| `galmockuv/plotting.py` | Diagnostic plots (moment maps, spectra, UV) |
| `galmockuv/casa_utils.py` | Vendored UV analysis utilities (`plot_uvbins`, `fit_uv_model`) |

## Environment

Two environments are needed (no single env has both DysmalPy and CASA):

| Environment | Provides | Layers |
|-------------|----------|--------|
| DysmalPy (e.g. `alma` conda env) | `dysmalpy`, `spectral-cube` | A |
| CASA (`casatools`, `casatasks`) | CASA tools/tasks | B+C |

Key deps (both envs): `numpy`, `scipy`, `matplotlib`, `astropy`, `yaml`.
Layer A also needs: `dysmalpy`, `spectral-cube`.
Layer C also needs: `casatools`, `casatools.ms`, `casatools.table`.

The entry point `galmockuv.py` inserts its parent directory into `sys.path` so
`import galmockuv` works without pip installation.

## Pipeline Layers

| Layer | Function | Env | Input | Output |
|-------|----------|-----|-------|--------|
| **A** | `build_cube(config, output_dir)` | DysmalPy | TOML config | `intrinsic_cube.fits`, `metadata.yaml` |
| **B** | `simulate_alma(cube_path, config, output_dir)` | CASA | FITS cube | `{source_id}.ms`, `casa_sim_params.yaml` |
| **C** | `measure_from_ms(ms_path, config, output_dir, metadata)` | CASA | MS | `measurements.json`, `measurement_details.yaml` |

## Config Format

TOML (preferred) or YAML.  All keys are flat (no nested tables):

```toml
source_id = "z25_demo"
redshift = 2.5
co_restfreq_ghz = 345.796
log10_baryonic_mass_msun = 10.62
disk_reff_kpc = 8.0
inclination_deg = 45.0
totaltime = "600s"
antennalist = "alma.cycle4.2"
```

See `demo/config.toml` for all parameters.  The `io.py` module has a
`_flatten_config()` fallback for nested TOML tables.

## Critical Gotchas

1. **`SpectralCube.moment()` is unreliable for CASA FITS cubes**.  It applies
   unit conversions that depend on the spectral axis units (Hz vs km/s) and
   beam handling.  For frequency-axis cubes it returns values inflated by the
   channel width in Hz (~3.3e6).  Always compute moments manually from raw
   FITS data using `numpy.nansum`.

2. **FITS axis ordering**: CASA stores NAXIS as (RA, Dec, Freq, Stokes).  In
   numpy this becomes shape (Stokes, Freq, Dec, RA).  After `np.squeeze()` the
   Stokes axis is removed, giving (nchan, ny, nx).  Do not transpose.

3. **Velocity axis for cleaned cubes**: CASA `tclean` outputs observed-frame
   frequencies.  Use `v = c * (f_center - freqs) / f_center` (not `RESTFRQ` as
   denominator) to get consistent km/s/channel matching the intrinsic cube.

4. **Moment map masking**: A 2D spatial mask from moment-0 includes ALL channels
   for masked pixels, including noisy line-free channels.  This corrupts the
   intensity-weighted velocity average.  Use a 3D channel+spatial mask that
   excludes channels where integrated spectrum < 2-sigma of line-free RMS.

5. **`flux_scale` parameter**: DysmalPy produces arbitrary normalization.  The
   `flux_scale` factor converts to physical mJy-level fluxes.  Typical values
   are 1e-6 to 1e-5.  This is not a standard astrophysical quantity — it is a
   tuning parameter set by the user.

6. **Environment gates**: `run_pipeline()` checks `is_casa_env()` and
   `is_dysmalpy_env()` before running each layer.  Running Layer A in CASA or
   Layer B+C without CASA will raise a clear error.

## Quick Reference

| Function | Description |
|----------|-------------|
| `run_pipeline(config, layers='A', output_dir=None)` | Run specified layers, return results dict |
| `load_config(path)` | Load TOML or YAML config file |
| `save_metadata(metadata, output_dir)` | Save metadata as YAML |
| `load_metadata(output_dir)` | Load metadata YAML |
| `load_json(output_dir, filename)` | Load JSON file |
| `save_json(data, output_dir, filename)` | Save dict as JSON |
| `setup_output_dir(base_dir, source_id)` | Create output directory |
| `build_cube(config, output_dir)` | Layer A: build intrinsic cube |
| `simulate_alma(cube_path, config, output_dir)` | Layer B: simulate ALMA |
| `measure_from_ms(ms_path, config, output_dir, metadata)` | Layer C: measure from MS |
| `measure_from_imaged_cube(cube_path, config, output_dir, metadata)` | Layer C alt: measure from FITS |
| `plot_uvbins(vis, ...)` | Binned UV amplitude plot with fit overlay |
| `fit_uv_model(vis, ...)` | Gaussian UV model fit via `uvmodelfit` |

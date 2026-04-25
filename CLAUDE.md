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
| `galmockuv/measure.py` | Layer C: `tclean` + line fitting + UV size measurement |
| `galmockuv/pipeline.py` | Orchestration (`run_pipeline`) |
| `galmockuv/plotting.py` | Diagnostic plots (FITS-based masked moments, spectra, UV) |
| `galmockuv/casa_utils.py` | Vendored UV analysis utilities (`plot_uvbins`, `fit_uv_model`) |

## Environment

Two environments are needed (no single env has both DysmalPy and CASA):

| Environment | Provides | Layers |
|-------------|----------|--------|
| DysmalPy (e.g. `alma` conda env) | `dysmalpy`, `spectral-cube`, `jax` | A |
| CASA (`casatools`, `casatasks`) | CASA tools/tasks | B+C |

Key deps (both envs): `numpy`, `scipy`, `matplotlib`, `astropy`, `yaml`.
Layer A also needs: `dysmalpy` (JAX-accelerated fork), `spectral-cube`, `jax`, `jaxlib`.
Layer C also needs: `casatools`, `casatools.ms`, `casatools.table`.

**JAX note**: The DysmalPy installation at
`/home/shangguan/Softwares/my_modules/dysmalpy/` is a JAX-accelerated fork that
replaces the Cython `cutils` backend with `jax.numpy` vectorised operations.
`zcalc_truncate=True` is enabled by default in `build_cube` to use sparse
propagation (only active z-slices), avoiding OOM on large cubes (201^3+).

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
channel_width_kms = 10.0
line_window_kms = 500.0       # expected max observed line FWHM
spectral_n_linefree = 10      # extra line-free channels per side for RMS
oversample = 3                # DysmalPy spatial oversampling factor
line_fit_model = "doublepeak" # "gaussian", "doublepeak", or "doublepeak_asymmetric"
uv_fit_method = "uvmodelfit"  # "uvmodelfit" (CASA) or "mcmc" (galfit_uv, needs emcee)
uv_fit_model = "gaussian"     # UV model for mcmc: "gaussian", "sersic", or "point"
```

The spectral range (`nchan`, `velocity_start_kms`) is auto-computed.  When
`line_window_kms` is set (recommended), the range is
`line_window_kms/2 + spectral_n_linefree * channel_width_kms` per side.
Otherwise it falls back to
`spectral_n_sigma * intrinsic_sigma_kms + spectral_n_linefree * channel_width_kms`.
Explicit `nchan` and `velocity_start_kms` can be provided to override either.

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

7. **galfit_uv import order**: `import galfit_uv` must appear before `import numpy`
   (the package sets `OMP_NUM_THREADS=1`).  For line profile fitting only
   (`line_fit_model != 'gaussian'`), measure.py bypasses `galfit_uv.__init__`
   via `importlib.util` to avoid the `emcee` dependency in CASA.  For MCMC UV
   fitting (`uv_fit_method='mcmc'`), `emcee` must be installed.

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
| `plot_summary(fits_path, output_dir, source_id, apply_mask=False, ...)` | 2x2 summary (m0, m1, m2, spectrum+FWHM) |
| `fwhm_half_max(vel, flux)` | Non-parametric FWHM via half-maximum width |
| `plot_moment_maps(fits_path, output_dir, source_id, apply_mask=False, ...)` | Individual moment map figures |
| `read_fits_cube(path)` | Read FITS cube → (data_3d, header, pixscale, extent) |
| `get_velocity_axis(header, restfreq_ghz=None)` | Build km/s velocity axis from FITS header |
| `make_signal_mask(m0, n_sigma=3.0, dilate=2)` | Spatial mask from sigma_clipped_stats + dilation |
| `make_3d_mask(data_3d, vel, spatial_mask, channel_sigma=2.0)` | 3D channel+spatial mask |
| `compute_moment0/1/2(data_3d, vel, mask=None)` | Manual moment computation with optional 3D mask |
| `plot_uvbins(vis, ...)` | Binned UV amplitude plot with fit overlay |
| `fit_uv_model(vis, ...)` | Gaussian UV model fit via `uvmodelfit` |

## Development strategy

The `dev/` folder can be used as a workspace for developing specific features in the future. Use it to keep maintaining the following notes. Update them in or after every run of development.
- **plan.md** Write down the small plan for each tasks and the check list. For each run, check if the taskes have been finished.
- **problem.md** Note the problems we meet multiple times so one can pay attention and avoid them.
- **develop_log.md** The log of all the changes. No need to be in detail but note what has been changed and fixed.
---
name: galmock-plot
description: >
  Use this skill when the user asks to "make plots", "plot moment maps", "generate
  report figures", "plot the spectrum", "plot UV data", "visualize the cube", or
  "create diagnostic plots".
---

# Plotting: Moment Maps, Spectra, and UV

Generate diagnostic plots from FITS cubes and MeasurementSets.  All moment
maps are computed manually from raw FITS data (not `SpectralCube.moment()`,
which is broken for CASA frequency-axis cubes).

## Prerequisites

- The `galmockuv` package must be importable (DysmalPy or base env — no CASA
  needed for FITS-based plots).
- UV visibility plots require CASA (`casatools.ms`).

## Quick Start

```python
from galmockuv.plotting import plot_summary, plot_moment_maps

# 2x2 summary: moment-0, moment-1, moment-2, spectrum + FWHM
plot_summary("output/my_galaxy/intrinsic_cube.fits",
             "output/my_galaxy", "my_galaxy", apply_mask=False,
             restfreq_ghz=345.796)

# Individual moment maps (with 3D masking for cleaned data)
plot_moment_maps("output/my_galaxy/my_galaxy.cube.image.fits",
                 "output/my_galaxy", "my_galaxy", apply_mask=True,
                 restfreq_ghz=345.796)
```

## Masking

**For intrinsic cubes** (`apply_mask=False`): No masking needed — the cube is
noise-free.

**For cleaned cubes** (`apply_mask=True`): A 3D mask is applied to moment-1
and moment-2:
1. **Spatial mask**: 3σ threshold on moment-0 via `sigma_clipped_stats`, then
   2-pixel binary dilation
2. **Channel mask**: 2σ threshold on integrated spectrum, with RMS estimated
   from the outer 20% of the velocity range (line-free channels)
3. **Minimum flux**: Voxels below 5% of the 99th flux percentile are excluded
   from moment-1/2 to avoid division artifacts

Without masking, moment-1 includes noisy line-free channels and produces garbage
velocities at edge pixels.

## Key Functions

### FWHM measurement

| Function | Description |
|----------|-------------|
| `fwhm_half_max(vel, flux)` | Non-parametric FWHM via half-maximum width (in `measure.py`) |

### FITS I/O and masking

| Function | Description |
|----------|-------------|
| `read_fits_cube(path)` | Read FITS → (data_3d, header, pixscale, extent) |
| `get_velocity_axis(header, restfreq_ghz=None)` | km/s velocity axis from header |
| `make_signal_mask(m0, n_sigma=3.0, dilate=2)` | Spatial mask (sigma_clipped + dilation) |
| `make_3d_mask(data_3d, vel, spatial_mask, channel_sigma=2.0)` | 3D channel+spatial mask |
| `compute_moment0(data_3d)` | Integrated intensity |
| `compute_moment1(data_3d, vel, mask=None)` | Intensity-weighted velocity |
| `compute_moment2(data_3d, vel, mask=None)` | Velocity dispersion |
| `compute_spectrum(data_3d, vel)` | Integrated spectrum sorted by velocity |
| `fit_gaussian_spectrum(vel, spec)` | Gaussian fit → (fwhm, fwhm_err, amp, cen, sigma) |

### High-level plotting

| Function | Input | Output |
|----------|-------|--------|
| `plot_summary(fits_path, output_dir, source_id, ...)` | FITS path | `summary.png` (2x2 panel) |
| `plot_moment_maps(fits_path, output_dir, source_id, ...)` | FITS path | `moment0/1/2.png` |
| `plot_integrated_spectrum(vel_axis, flux, output_dir, source_id)` | arrays | `integrated_spectrum.png` |
| `plot_spectrum_fit(vel_axis, flux, fit_result, ...)` | arrays | `spectrum_fit.png` |
| `plot_uv_amplitude(uv_dist, amp, amp_err, fit_result, ...)` | arrays | `uv_amplitude.png` |
| `plot_batch_summary(catalog, output_dir)` | astropy Table | Scatter plots |

### UV plotting (requires CASA)

| Function | Description |
|----------|-------------|
| `plot_uvbins(vis, ...)` | Binned UV amplitude plot with fit overlay |
| `fit_uv_model(vis, ...)` | Gaussian UV model fit via `uvmodelfit` |

## Report Plots (demo)

The demo report script generates 5 publication-quality figures from the full
pipeline output.  Runs with regular Python — CASA-only figures (visibility data,
UV fit) are skipped gracefully when `casatools` is unavailable:

```bash
python demo/run_report_plots.py
```

Output (in `demo/figs/`): `intrinsic_summary.png`, `visibility_data.png`,
`cleaned_summary.png`, `uv_amplitude_fit.png`, `corner_plot.png`.

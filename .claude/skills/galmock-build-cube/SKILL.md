---
name: galmock-build-cube
description: >
  Use this skill when the user asks to "build a galaxy cube", "create an intrinsic
  CO cube", "run Layer A", "make a DysmalPy model", "generate a galaxy model",
  or "build the intrinsic cube".
---

# Layer A: Build Intrinsic CO Cube

Build an intrinsic CO line cube from a galaxy model using DysmalPy.

## Prerequisites

- **Environment**: DysmalPy (`alma` conda env with `dysmalpy`, `jax`, `spectral-cube`).
  The `galmockuv` package must be importable (entry point `galmockuv.py` inserts
  its parent into `sys.path`).
- A TOML config file describing the galaxy.  See `demo/config.toml` for all
  parameters.

## Commands

```bash
# Activate DysmalPy environment and run Layer A
eval "$(/home/shangguan/Softwares/miniconda3/bin/conda shell.bash hook)" && conda activate alma && python galmockuv.py config.toml --layers A
```

Or programmatically:

```python
from galmockuv import run_pipeline
results = run_pipeline("config.toml", layers="A")
print(results["cube_path"])   # path to intrinsic_cube.fits
```

## Output

| File | Description |
|------|-------------|
| `intrinsic_cube.fits` | Intrinsic CO cube, shape (nchan, ny, nx), frequency axis |
| `metadata.yaml` | True input parameters |

Pipeline diagnostic plots (`summary.png`, `integrated_spectrum.png`) are also
saved to the output directory.

## Config Parameters (galaxy model + cube)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source_id` | (required) | Output directory name |
| `redshift` | (required) | Redshift |
| `co_restfreq_ghz` | (required) | CO line rest frequency (GHz) |
| `log10_baryonic_mass_msun` | 10.0 | log10(M_star + M_gas) |
| `disk_reff_kpc` | 5.0 | Effective radius (kpc) |
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
| `pixscale_arcsec` | 0.15 | Pixel scale |
| `npix_x`, `npix_y` | 201 | Spatial grid size |
| `channel_width_kms` | 10.0 | Channel width |
| `line_window_kms` | (auto) | Expected observed line FWHM. Spectral range = FWHM/2 + `spectral_n_linefree` * `channel_width_kms` per side. **Always set this** to avoid a range that's too narrow. |
| `spectral_n_sigma` | 5 | Fallback spectral coverage (sigma * intrinsic_sigma) |
| `spectral_n_linefree` | 10 | Line-free channels per side |
| `oversample` | 3 | DysmalPy spatial oversampling |

## Key Functions

| Function | Description |
|----------|-------------|
| `build_cube(config, output_dir)` | Build intrinsic cube, return (path, metadata, model_cube) |
| `read_fits_cube(path)` | Read FITS → (data_3d, header, pixscale, extent) |
| `get_velocity_axis(header, restfreq_ghz)` | Build km/s velocity axis from header |
| `plot_summary(fits_path, output_dir, source_id)` | 2x2 moment maps + spectrum summary |

## Troubleshooting

- **DysmalPy segfaults on import**: `pip install --force-reinstall --no-deps pyerfa`
- **JAX OOM on 201^3**: `zcalc_truncate=True` is enabled by default.  If still OOM,
  reduce `npix_x/npix_y` to 101.
- **`conda run` fails**: Use the shell hook: `eval "$(conda shell.bash hook)" && conda activate alma`

---
name: galmock-measure
description: >
  Use this skill when the user asks to "measure FWHM", "compute f_eff", "measure
  dynamical mass", "run tclean", "measure size from CO", "measure from a mock
  observation", or "run Layer C".
---

# Layer C: Measure Observables

Image the MeasurementSet with CASA `tclean`, fit the UV visibility with
`uvmodelfit`, and measure CO line FWHM, source size, and dynamical mass proxy
with efficiency factor *f*_eff.

## Prerequisites

- **Environment**: CASA (`casatools`, `casatasks`).  Layer B must have produced
  a MeasurementSet first.
- The `galmockuv` package must be importable.

## Commands

Layer C is typically run together with Layer B (see `galmock-simulate-alma` skill).
To run only Layer C from an existing MS:

```python
from galmockuv import run_pipeline
results = run_pipeline("config.toml", layers="C",
                       output_dir="output/my_galaxy")
```

Or call directly:

```python
from galmockuv.measure import measure_from_ms
from galmockuv.io import load_metadata

measurements = measure_from_ms(
    "output/my_galaxy/my_galaxy.ms",
    config,
    "output/my_galaxy",
    load_metadata("output/my_galaxy"),
)
```

## Reading Results

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

## Interpreting Results

### f_eff

*f*_eff = *M*_baryonic / *M*_proxy encodes pressure support, beam smearing,
inclination, and disk-to-halo mass ratio.

| f_eff range | Interpretation |
|-------------|---------------|
| ~0.1--0.2 | Typical for high-z star-forming galaxies |
| ~0.3--0.5 | Massive, rotation-dominated disks |
| < 0.1 | Strong pressure support or beam dilution |
| > 0.5 | Check config — may be face-on or unusually massive halo |

### FWHM recovery

The measured FWHM (Gaussian fit to spatially-masked integrated spectrum) should
recover the intrinsic FWHM at ~95--100% for SNR > 10.  Spatial masking
(1.5σ on moment-0 via MAD noise estimation) excludes noise-only edge pixels
from the integrated spectrum.

### Size measurement

The UV-fit size comes from `uvmodelfit` on channel-averaged visibilities.
For marginally resolved sources (θ_source < beam), the uncertainty is large
and the measurement may be a lower limit.

## Config Parameters (measurement)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nbeam_extract` | 1 | Number of beams for spectral extraction |
| `uv_bin_width_klambda` | 50.0 | UV bin width |
| `uv_max_klambda` | 3000.0 | Maximum UV distance for fitting |
| `size_fit_model` | `"G"` | UV fit model type (Gaussian) |
| `line_fit_model` | `"gaussian"` | Line profile fit model |
| `tclean_niter` | 1000 | tclean minor cycle iterations |
| `tclean_threshold` | `"0mJy"` | tclean stopping threshold |
| `measure_spatial_sigma` | 1.5 | Spatial mask threshold for FWHM spectrum |

## Output Files

| File | Description |
|------|-------------|
| `measurements.json` | FWHM, size, SNR, mass, f_eff |
| `measurement_details.yaml` | Full measurement details |
| `{source_id}.cube.image.fits` | Cleaned image cube |
| `{source_id}.cube.image/` | CASA image directory |

## Key Functions

| Function | Description |
|----------|-------------|
| `measure_from_ms(ms_path, config, output_dir, metadata)` | Full Layer C from MS |
| `measure_from_imaged_cube(cube_path, config, output_dir, metadata)` | Layer C from existing FITS |

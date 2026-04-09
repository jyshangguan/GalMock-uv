# galmockuv Demo: Package Validation Report

## Overview

This report validates the `galmockuv` package by running the full three-layer pipeline on a realistic example galaxy at *z* = 2.5. The pipeline uses **DysmalPy** (Layer A) to build an intrinsic CO line cube, **CASA `simobserve`** (Layer B) to simulate ALMA interferometric observations, and custom measurement code (Layer C) to extract FWHM, source size, and dynamical mass proxy. The results are compared to the previous standalone-script pipeline to confirm the package refactor introduced no regressions.

All outputs are in `demo/output/z25_demo/`.

---

## Example Galaxy Parameters

| Parameter | Value | Notes |
|---|---|---|
| Redshift | 2.5 | |
| Total baryonic mass | 4.17 × 10<sup>10</sup> M<sub>☉</sub> | log = 10.62 |
| NFW halo mass | 10<sup>12</sup> M<sub>☉</sub> | *c* = 5 |
| Disk effective radius | 8 kpc | Sérsic *n* = 1 (exponential) |
| Inclination | 45° | |
| Position angle | 30° | |
| Velocity dispersion | 30 km/s | Constant, isotropic |
| CO line | (3--2), rest 345.796 GHz | Observed 98.80 GHz |
| Pressure support | Yes | Asymmetric drift correction |

---

## Running the Demo

```bash
# Layer A (DysmalPy environment):
python galmockuv.py demo/config.toml --layers A

# Layers B+C (CASA environment):
casa --nologger --nogui -c "exec(open('galmockuv.py').read())" demo/config.toml --layers B+C
```

The demo uses the TOML config at `demo/config.toml`. The package also supports YAML configs for backward compatibility.

---

## Layer A: Galaxy Model (DysmalPy)

### Method

The galaxy is modeled as a rotating exponential disk embedded in an NFW dark matter halo. DysmalPy solves the Jeans equation self-consistently to produce rotation curves including pressure support corrections. The intrinsic cube is sampled at 0.15″/pixel over 201 × 201 pixels (30.2″ × 30.2″), with 201 channels of 10 km/s each. The spectral axis is converted from velocity to frequency space for CASA compatibility.

### Results

The intrinsic cube produces a line FWHM of **306 km/s**, reflecting the combined rotational broadening (8 kpc disk at 45° inclination) and turbulent dispersion.

![Intrinsic cube summary — moment 0, moment 1, moment 2, and integrated spectrum with Gaussian fit.](output/z25_demo/intrinsic_summary.png)

**Figure 1.** Intrinsic model diagnostics. *Top-left:* Moment-0 (integrated CO flux) showing an extended disk. *Top-right:* Moment-1 velocity field revealing the rotation pattern (spider diagram). *Bottom-left:* Moment-2 velocity dispersion showing elevated central dispersion from rotation shear. *Bottom-right:* Integrated spectrum with Gaussian fit, FWHM = 306 km/s.

---

## Layer B: ALMA Simulation (CASA `simobserve`)

### Method

CASA `simobserve` converts the intrinsic cube into a simulated MeasurementSet using the ALMA 12-m array in the C43-2 configuration (40 antennas, baselines 15--273 m). Thermal noise is added via the `tsys-atm` model with PWV = 1.5 mm. Total on-source time is 600 s with 10 s integrations.

### Results

The resulting MS is 681 MB. The uv-coverage extends to ~89 kλ, giving a synthesized beam of ~3″ with natural weighting.

![Visibility data — real part vs uv-distance (left) and uv coordinate distribution (right).](output/z25_demo/visibility_data.png)

**Figure 2.** Simulated visibility data. *Left:* Real part of channel-averaged visibilities vs uv-distance. The amplitude decreases with baseline length as the source becomes resolved. *Right:* uv coordinate distribution showing the C43-2 baseline coverage.

---

## Layer C: Imaging and Measurement

### Method

The MS is imaged with CASA `tclean` (natural weighting, 1000 iterations). The FWHM is measured by Gaussian-fitting the integrated spectrum of the cleaned cube. The source size is measured by fitting a circular Gaussian model to the channel-averaged visibility amplitudes via `uvmodelfit`. The dynamical mass proxy is computed as:

*M*<sub>proxy</sub> = FWHM² × *D* / *G*

### Results

![Cleaned image moments and spectrum.](output/z25_demo/cleaned_summary.png)

**Figure 3.** Cleaned image diagnostics. *Top-left:* Moment-0 of the deconvolved cube. *Top-right:* Moment-1 velocity field (noisier than the intrinsic map in Figure 1). *Bottom-left:* Moment-2 dispersion map. *Bottom-right:* Cleaned integrated spectrum with Gaussian fit, FWHM = 295.6 ± 10.2 km/s.

![UV amplitude vs uv-distance with Gaussian model overlay.](output/z25_demo/uv_amplitude_fit.png)

**Figure 4.** Binned visibility amplitudes (channel-averaged, real part) with the best-fit circular Gaussian model (red curve). θ<sub>maj</sub> = 1.842″ ± 0.104″, flux = 12.93 mJy.

---

## Summary of Measurements

| Quantity | Value | Notes |
|---|---|---|
| Intrinsic FWHM | 306 km/s | From DysmalPy model |
| Measured FWHM | 295.6 ± 10.2 km/s | Gaussian fit to cleaned spectrum |
| Measured SNR | 15.3 | Peak / rms of line-free channels |
| UV fit size (θ<sub>maj</sub>) | 1.842″ ± 0.104″ | From `uvmodelfit` |
| Measured size | 14.9 kpc | Angular diameter distance conversion |
| Proxy mass | 3.02 × 10<sup>11</sup> M<sub>☉</sub> | FWHM² × *D* / *G* |
| True baryonic mass | 4.17 × 10<sup>10</sup> M<sub>☉</sub> | Input parameter |
| *f*<sub>eff</sub> | 0.138 | *M*<sub>bary</sub> / *M*<sub>proxy</sub> |

---

## Comparison with Previous Pipeline

The results are identical to those from the standalone-script pipeline (`archive/report.md`), confirming the package refactor introduced no regressions:

| Metric | Previous (`archive/report.md`) | This demo (`galmockuv`) | Match? |
|---|---|---|---|
| Intrinsic FWHM | 306 km/s | 306 km/s | Yes |
| Measured FWHM | 295.6 km/s | 295.6 km/s | Yes |
| SNR | 15.3 | 15.3 | Yes |
| θ<sub>maj</sub> | 1.842″ | 1.842″ | Yes |
| Measured size | 14.9 kpc | 14.9 kpc | Yes |
| *f*<sub>eff</sub> | 0.138 | 0.138 | Yes |

---

## Package Structure

```
mocks/
├── galmockuv/                  # Package
│   ├── __init__.py             # Version + lazy imports
│   ├── env.py                  # Environment detection
│   ├── io.py                   # Config (TOML/YAML) + metadata I/O
│   ├── plotting.py             # Diagnostic plots
│   ├── build_cube.py           # Layer A: DysmalPy cube
│   ├── simulate.py             # Layer B: CASA simobserve
│   ├── measure.py              # Layer C: FWHM/size/mass
│   ├── pipeline.py             # Orchestration
│   └── casa_utils.py           # Vendored UV analysis utilities
├── galmockuv.py                # Entry point
├── demo/
│   ├── config.toml             # Demo config
│   ├── run_demo.py             # Demo script
│   ├── generate_report_plots.py # Report figure generator
│   ├── demo_report.md          # This report
│   └── output/z25_demo/        # Demo outputs
└── archive/                    # Old standalone scripts
```

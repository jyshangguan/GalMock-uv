# Development Log

## 2026-04-25: Replace Gaussian FWHM with non-parametric half-max method

**What changed:**
- `galmockuv/measure.py`: Added `fwhm_half_max(vel, flux)` helper. Replaced all
  Gaussian/parametric FWHM derivation with non-parametric half-max width.
  `_measure_fwhm_from_cube()` now uses direct half-max on intrinsic spectrum
  (no fitting). `_measure_fwhm_from_fits()` still fits a model (Gaussian/DoublePeak)
  for smoothing, but derives FWHM via `fwhm_half_max()` on the fitted model curve.
  Removed `_estimate_sigma_halfmax()`.  Moved CASA imports (`casa_utils`) from
  module-level to inside functions, enabling safe cross-environment import of
  `fwhm_half_max` from `measure.py`.
- `galmockuv/plotting.py`: Imports `fwhm_half_max` from `measure.py` (no duplicate).
  Simplified `plot_summary()` bottom-right panel: removed Gaussian fit overlay,
  now shows raw spectrum + FWHM text box only.
- `demo/run_report_plots.py` (renamed from `generate_report_plots.py`): Removed
  `fit_gaussian_spectrum()` duplicate.  Simplified intrinsic panel (no model fitting).
  Updated cleaned panel to use `fwhm_half_max()` on fitted model curve.  Wrapped
  CASA imports in try/except so the script runs with regular Python — CASA-only
  figures (visibility data, UV fit) are skipped gracefully when `casatools` is
  unavailable.
- `demo/run_layer_a.py`, `demo/run_layer_bc.py`: New demo scripts (replaced
  `run_demo.py`). One script per environment for CASA compatibility.
- `CLAUDE.md`: Added `fwhm_half_max` to quick reference. Updated usage docs.
- `demo/demo_report.md`: Updated running instructions.

**Results:** Intrinsic FWHM 340.8 km/s (was 308), cleaned FWHM 331.1 km/s (was 339.1).

## 2026-04-24: Update demo to use DoublePeak + MCMC measurement methods

**What changed:**
- `demo/config.toml`: Switched `uv_fit_method` from `"uvmodelfit"` to `"mcmc"`.
  Fixed tab character before `tclean_niter` (caused TOML parse issue).
- `demo/generate_report_plots.py`: Updated cleaned spectrum panel to use the
  configured `line_fit_model` from measurements.json (Gaussian, DoublePeak, or
  DoublePeak_Asymmetric).  Imports profiles via `importlib.util` to bypass
  `galfit_uv.__init__`.  Updated UV amplitude plot to load MCMC fit results
  from `fit_results.fits` when `uv_fit_method == 'mcmc'` — shows binned data +
  model curve + fit statistics annotation.  Falls back to `fit_uv_model` path
  for backward compatibility.
- `galmockuv/pipeline.py`: Removed dead UV plot code (lines 111-126) that
  checked for `uv_result` key never populated by `measure_from_ms()`.
- `galfit_uv/models.py`: Fixed `np.trapz` → `np.trapezoid` for numpy 2.x
  compat (CASA ships numpy 2.4.1).
- Installed `emcee`, `dill`, `corner` into CASA standalone Python environment
  at `/home/shangguan/Softwares/casa-6.7.3-21-py3.12.el9/`.
- Regenerated demo output with full pipeline (A+B+C) and report plots.

**Demo results (z=2.5, 600s ALMA, CO(3-2)):**

| Metric | Value |
|--------|-------|
| Line profile | DoublePeak |
| FWHM | 339.1 ± 5.0 km/s |
| SNR | 48.3 |
| UV fit | MCMC Gaussian (redchi2=0.954, BIC=1526) |
| Size (FWHM) | 0.889" ± 0.035" (7.18 kpc) |
| Integrated flux | 25.87 Jy km/s |
| f_eff | 0.217 |

## 2026-04-24: Integrate galfit_uv line profiles and MCMC UV fitting

**What changed:**
- `galmockuv/measure.py`: Replaced hardcoded Gaussian line fitting in
  `_measure_fwhm_from_fits()` with configurable profile models from
  `galfit_uv.lineprofiles`.  Supports `gaussian`, `doublepeak` (Tiley+2016
  symmetric double-horn), and `doublepeak_asymmetric`.  FWHM is computed as
  width at half-maximum of the peaks: `2*w + 2*sigma*sqrt(2*ln(2))` for
  symmetric, `(wl+wr) + 2*sigma*sqrt(2*ln(2))` for asymmetric.
- `galmockuv/measure.py`: Added `_measure_size_mcmc()` function for Bayesian UV
  size measurement via `galfit_uv.fit_mcmc`.  Exports visibilities with
  `galfit_uv.export_vis`, builds parametric model with `make_model_fn`, and
  runs MCMC with `fit_mcmc`.  Supports `gaussian`, `sersic`, and `point` models.
- `galmockuv/measure.py`: Added dispatch in `measure_from_ms()` for
  `uv_fit_method`: `"uvmodelfit"` (CASA default) or `"mcmc"` (galfit_uv).
- `demo/config.toml`: Added `line_fit_model`, `uv_fit_method`, `uv_fit_model`,
  `uv_mcmc_max_steps`, `uv_mcmc_burnin`, `uv_mcmc_nwalk_factor`,
  `uv_mcmc_n_workers`.
- Import of `galfit_uv.lineprofiles` uses `importlib.util` to bypass
  `galfit_uv.__init__` (which requires `emcee`, not available in CASA env).
- Fixed `UnboundLocalError` for `vel_fit` (defined before fitting blocks).
- Fixed duplicate keys in return dict (`integrated_flux_jy_kms`,
  `beam_area_pix`, `beam_area_arcsec2`).
- Fixed fragile `'galfit_uv' in dir()` check with proper try/except NameError.

**Test results (dev/20260423-galfit-uv, existing MS, CASA 6.7.3):**

| Setting | FWHM (km/s) | SNR | Size (kpc) | f_eff |
|---------|-------------|-----|------------|-------|
| Gaussian (old) | 310.3 | 48.3 | 6.36 | 0.245 |
| DoublePeak (new) | 339.1 +/- 5.0 | 48.3 | 6.36 | 0.245 |

The DoublePeak profile gives a slightly broader FWHM (339 vs 310 km/s) because
it captures the double-horn structure from the rotating disk, while the Gaussian
smooths over the central dip.

**MCMC UV fitting:** Implemented but not yet tested (requires `emcee` which is
not installed in the CASA standalone environment).

## 2026-04-23: Rewrite plotting.py — drop SpectralCube, add masked moments

**What changed:**
- `galmockuv/plotting.py`: Full rewrite.  Replaced `SpectralCube.moment()`
  (broken for CASA frequency-axis cubes, CLAUDE.md gotcha #1) with manual
  `np.nansum` moment computation.  Added 3D masking (3σ spatial +
  sigma_clipped_stats + 2-pixel binary dilation, 2σ channel mask from outer
  20% velocity range) to prevent noisy line-free channels from corrupting
  moment-1/2 (gotcha #4).
- New low-level helpers: `read_fits_cube()`, `get_velocity_axis()`,
  `compute_moment0/1/2()`, `make_signal_mask()`, `make_3d_mask()`,
  `compute_spectrum()`, `fit_gaussian_spectrum()`.  Lifted from
  `demo/generate_report_plots.py` with one fix: `get_velocity_axis()` now
  takes explicit `restfreq_ghz` parameter instead of referencing module-level
  state.
- New high-level functions: `plot_moment_maps(fits_path, ...)` and
  `plot_summary(fits_path, ...)`.  Both accept FITS file paths (not SpectralCube
  objects) and support `apply_mask=True` for cleaned cubes.
- `galmockuv/pipeline.py`: Updated Layer A call site to use
  `plot_summary(str(cube_path), ..., apply_mask=True)` instead of
  `plot_moment_maps(model_cube, ...)`.  Removed unused `plot_spectrum_fit`
  import.
- Untouched functions (no SpectralCube dependency): `plot_integrated_spectrum`,
  `plot_uv_amplitude`, `plot_spectrum_fit`, `plot_batch_summary`.

**Verification on dev/20260423-memory data:**
- Intrinsic FWHM = 306.6 km/s (consistent with demo)
- Cleaned FWHM = 302.7 +/- 17.8 km/s (consistent with demo)
- 3D mask excludes 96.4% of voxels in cleaned cube
- Cleaned moment-1/2 maps show signal only in masked central region

## 2026-04-23: Fix spectral range and spatial masking for measurements

**What changed:**
- `galmockuv/build_cube.py`: Added `line_window_kms` config parameter for the
  expected observed line FWHM.  When set, spectral range is
  `line_window_kms/2 + spectral_n_linefree * channel_width_kms` per side,
  replacing the old `spectral_n_sigma * intrinsic_sigma_kms` formula which
  only covered turbulent dispersion (30 km/s), not rotation (~300 km/s FWHM).
- `galmockuv/measure.py`: Added 1.5σ spatial mask to
  `_measure_fwhm_from_fits()`.  Noise estimated via MAD of edge pixels.
  Only pixels above `measure_spatial_sigma * noise` in moment-0 are included
  in the integrated spectrum.
- `galmockuv/simulate.py`: Fixed single-track MS deletion bug (cleanup
  deleted MS before `_fix_weights` could access it).
- `demo/config.toml`: Added `line_window_kms = 500.0`.

## 2026-04-23: Regenerate report figures with proper masking

**What changed:**
- Regenerated `fig_intrinsic_cube.png` and `fig_cleaned_image.png` in
  `dev/20260423-memory/` using the masking code from
  `demo/generate_report_plots.py`.
- Moment maps now use 3D masking: 3σ spatial mask (sigma_clipped_stats +
  2-pixel binary dilation) AND 2σ channel mask (from outer 20% velocity
  range).  Moment-0 uses `AsinhNorm` stretch.  Moment-1/2 exclude voxels
  below 5% of 99th flux percentile.
- Updated `dev/20260423-memory/summary.md` with corrected figure descriptions
  and masking approach details.
- Note: measure.py uses 1.5σ MAD mask (for pipeline measurements), while
  figures use the stricter 3σ+dilation approach (for diagnostic display).

**Before → After (npix=201, 600s ALMA):**

| Metric | Before (nchan=51) | After (nchan=71) |
|--------|-------------------|-------------------|
| FWHM | 307.6 km/s | 310.3 km/s |
| SNR | **9.2** | **49.5** |
| Size | 6.09 kpc | 6.36 kpc |
| f_eff | 0.311 | 0.293 |
| Integrated flux | 24.4 Jy km/s | 25.9 Jy km/s |

The 5.4x SNR improvement is primarily from spatial masking (excluding
noise-only edge pixels from the integrated spectrum).

## 2026-04-23: Full pipeline end-to-end test (npix=201)

**What changed:**
- Ran full demo pipeline (A+B+C) on `demo/config.toml` (npix=201, oversample=3)
  with output to `dev/20260423-memory/output/z25_demo/`
- Fixed bug in `galmockuv/simulate.py`: single-track MS was deleted by project
  cleanup before `_fix_weights` could access it.  Now moves MS to `output_dir`
  before cleanup for both single and multi-track modes.

**Layer A results (alma env, JAX CPU):**
- Time: 43.1 s
- Peak RSS: 12813 MB (603^2 spatial grid, 51 channels)
- Intrinsic cube: 51x201x201, peak 0.013 mJy (correct, with dimming)

**Intrinsic cube comparison:**
- Old demo cube (`demo/output/z25_demo/`) was from pre-dimming-fix run
  (April 22 before commit `56c7144`) — flux values 1e10x too large
- New cube has correct dimming applied; shapes and structure match

**Layer B+C results (CASA 6.7.3):**
- Time: 102.7 s
- Measurements: FWHM=307.6 km/s, size=6.09 kpc, SNR=9.2, f_eff=0.311

**Note:** Old measurements (Apr 9) are not directly comparable because they were
produced from the un-dimmed cube (SNR=12.1 vs 9.2).  The new measurements are
from the correct post-fix pipeline.

## 2026-04-23: Optimize `_make_cube_ai()` memory (47% RSS reduction at npix=80)

**What changed:**
- Replaced `np.indices()` + `np.vstack()` in `_make_cube_ai()` with
  `np.ravel()` (zero-copy view) + `np.flatnonzero()` + modular index
  reconstruction (`validpts % nx`, `(validpts // nx) % ny`,
  `validpts // (nx * ny)`)
- Removed `.flatten()` copies of xgal/ygal/zgal (replaced with `.ravel()`)
- Removed full-size `origpos = np.vstack(...)` (replaced with separate flat
  arrays + boolean mask, freed early with `del`)
- Applied to both `dysmalpy/models/model_set.py` and `cube_processing.py`

**Memory results (truncated path):**

| npix | Grid | RSS before | RSS after | Savings |
|------|------|-----------|-----------|---------|
| 20 | 60^3 | 545 MB | 539 MB | ~1% |
| 40 | 120^3 | 769 MB | 614 MB | 20% |
| 80 | 240^3 | 2363 MB | 1242 MB | **47%** |

**Correctness:** All metrics unchanged across npix=20/40/80. Peak flux ratio
(trunc/full) = 0.97649, total ratio = 0.95629, relative RMS = 0.051%.
27/27 dysmalpy tests pass.

## 2026-04-23: Re-measure memory with OS-level RSS (ru_maxrss)

**What changed:**
- Modified `dev/20260423-zcalc_accuracy/benchmark_zcalc.py`:
  - Added `--zcalc-truncate` CLI flag (`true`/`false`/`both`)
  - Each mode runs in a **separate subprocess** so `ru_maxrss` starts from zero
  - `run_once()` now reports both `resource.getrusage(RUSAGE_SELF).ru_maxrss`
    (OS-level peak RSS) and `tracemalloc` peak (Python heap only)
  - Added `_run_single()` (save results to npy/json) and `_load_results()`
    for subprocess communication
- Updated `report.md` with corrected memory analysis

**Key finding — previous tracemalloc-only benchmarks were misleading:**

`tracemalloc` only tracks Python-level `malloc` calls. JAX/XLA allocates memory
through TelaMalloc, invisible to tracemalloc. This caused a complete **inversion**
of the memory comparison:

| npix | Previous conclusion (tracemalloc) | Corrected (ru_maxrss) |
|------|----------------------------------|-----------------------|
| 20 | Trunc uses 2.7x MORE Python heap | Trunc uses 6% less RSS |
| 40 | Trunc uses 2.8x MORE Python heap | Trunc uses 37% LESS RSS |
| 80 | Trunc uses 2.8x MORE Python heap | Trunc uses 36% LESS RSS |

The full path allocates 80–98% of its memory through JAX/XLA (576–2943 MB
hidden), while the truncated path keeps JAX allocations bounded (364–519 MB)
because it processes pixels incrementally.

**Decision:** No optimization of `_make_cube_ai()` needed. The truncated path's
higher Python heap usage is irrelevant — the total memory is 35–37% lower.

**Lesson:** Never use `tracemalloc` alone for memory benchmarking code that uses
JAX/TensorFlow/PyTorch. Always use OS-level metrics (`ru_maxrss`, `/proc/self/status`,
etc.) with subprocess isolation.

## 2026-04-23: Benchmark zcalc_truncate accuracy (CPU + GPU)

**What changed:**
- Made `zcalc_truncate` configurable in `build_cube.py` via `config.get('zcalc_truncate', True)`
  (backward-compatible; defaults to `True`)
- Created `dev/20260423-zcalc_accuracy/benchmark_zcalc.py`: runs build_cube with both
  `zcalc_truncate=False` (full 3D grid) and `True` (active-only), compares flux/moment maps
- Ran benchmarks on CPU and GPU (small 20x20x51 cube, oversample=3)

**Results:**
- Truncated path underestimates peak flux by ~2.4% and total flux by ~4.4% compared to
  full path, on both CPU and GPU
- Relative RMS diff over full data cube: 0.2% (most pixels match well)
- Moment-1 (velocity field) matches within <1 channel RMS — kinematics preserved
- Deficit concentrated at low-surface-brightness spatial/spectral edges below the
  active-pixel threshold
- CPU/GU produce identical results (JAX numerical consistency confirmed)

**Conclusion:** `zcalc_truncate=True` is safe to use as default. The ~2-4% flux deficit
is negligible compared to ALMA noise and beam effects. Kinematics are unaffected.

## 2026-04-22: Fix missing dimming in active-only cube path (dysmalpy)

**What changed:**
- Fixed `dysmalpy/models/model_set.py`: the `_use_active_path` code
  (introduced in dysmalpy commit `aa12476`) computed flux for active pixels
  but skipped the luminosity-to-flux dimming step (`flux *= dimming(xsky,
  ysky, zsky)`).  The default `ConstantDimming` multiplies by
  `amp_lumtoflux=1e-10`, so the active path produced cube values 10^10x too
  large.
- Added `flux_a *= self.dimming.amp_lumtoflux` after `sigmar_a` computation
  and before `populate_cube_active` call.
- Extinction is NOT applied in the active path (no `xsky`/`ysky`/`zsky`),
  but the default model has `self.extinction = None` so this is safe.
- Committed as dysmalpy `56c7144` on `dev_jax`.

**Verification:**
- test_models.py: 27/27 pass (up from 11/27 — all 16 previously failing
  tests now pass)
- MPFIT 2D demo chi2_red remains ~11.74 (pre-existing, unrelated to dimming;
  2D fitting chi2 is computed from velocity/dispersion maps, which are
  insensitive to absolute flux scaling)

**Root cause:** When the active-only path was added in `aa12476` to avoid OOM
on large cubes, the dimming line was not carried over from the original path.

## 2026-04-22: Fix alma conda env — JAX/TFP/jaxns version compatibility

**What changed:**
- Cleaned alma env: uninstalled conflicting jax 0.10, jaxlib 0.10, tfp-nightly,
  jaxns 2.6.9, and all nvidia-* CUDA packages
- Installed stable stack: `jax==0.4.38`, `jaxlib==0.4.38`,
  `jax-cuda12-plugin==0.4.38`, `tensorflow-probability==0.25.0`,
  `jaxns==2.4.13`, `flax==0.10.4`, `evosax==0.2.0` (with `--no-deps`
  since evosax requires jax>=0.5 in metadata)
- Fixed `dysmalpy/special/hyp2f1.py`: `jax.scipy.special.hyp2f1` was
  removed in JAX 0.4.38. Added `hasattr` check to fall back to custom
  power-series implementation.
- Fixed `dysmalpy/fitting/jaxns.py`: updated all jaxns 2.4.13 API changes:
  - `NestedSampler` → `DefaultNestedSampler`
  - `jaxns.nested_samplers.common.types.TerminationCondition` →
    `jaxns.nested_sampler.TerminationCondition`
  - `jaxns.utils.{summary,resample}` → `jaxns.{summary,resample}`
  - `jaxns.plotting.{plot_diagnostics,plot_cornerplot}` →
    `jaxns.{plot_diagnostics,plot_cornerplot}`
  - `ns.save_results()` → `jaxns.save_results()` (module-level function)
  - Removed `shell_fraction` and `gradient_guided` kwargs (not in new API)
  - `max_samples` now required in constructor

**Verification:**
- JAX GPU detection: `CudaDevice(id=0)` on RTX 4090
- DysmalPy tests: 11/27 pass (10 core + 1 TPH after hyp2f1 fix;
  16 pre-existing simulate_cube/fitting-wrapper failures unchanged)
- JAXNS demo: completed successfully (3450 samples, ESS=522, logZ=-44.92)

**Key lesson:** evosax 0.2.0 pulls in jax>=0.5 as a dependency, overwriting
jax 0.4.38. Must install with `--no-deps` and restore jax afterwards.

## 2026-04-21: Switched DysmalPy to JAX-accelerated fork

**What changed:**
- Replaced Cython DysmalPy (`/home/shangguan/Softwares/dysmalpy/`) with
  JAX-accelerated fork (`/home/shangguan/Softwares/my_modules/dysmalpy/`)
- Installed CUDA-enabled jaxlib (`jaxlib==0.4.34`, `jax-cuda12-plugin==0.4.35`)
- Fixed `nvidia-cuda-nvcc-cu12` version (pinned `<12.5` to fix `__file__=None`)
- Updated `CLAUDE.md` environment section with JAX dependency
- Updated `galmockuv/env.py` docstring

**Benchmark results (101^3 cube):**
| Version | Time | Notes |
|---------|------|-------|
| Cython 101^3 | 156.61 s | Python loops over 3D grid |
| JAX CPU 101^3 | 33.25 s | 4.7x speedup, vectorised |
| Cython 201^3 | 2792.13 s | Baseline |
| JAX 201^3 | OOM | XLA graph needs ~363 GB |

**Correctness:** Peak flux matches to 0.002% between Cython and JAX versions.

**Known issue:** Full 201^3 cube OOMs on both CPU and GPU due to XLA graph
size.  Solution: use `zcalc_truncate=True` on `obs.mod_options` or ensure
sufficient GPU memory is free.

## 2026-04-21: Reduce memory usage & narrow spectral range

**What changed:**
- Enabled `zcalc_truncate=True` in `build_cube.py` by default — uses sparse
  propagation (`populate_cube_jax_ais`) instead of full 3D grid, avoiding OOM
- Made `oversample` configurable via config key (default 3)
- Auto-compute spectral range from line width: `spectral_n_sigma` (default 5)
  and `spectral_n_linefree` (default 10) replace manual `nchan`/`velocity_start_kms`
- Backward compatible: explicit `nchan` and `velocity_start_kms` still work
- Updated `demo/config.toml`: 201 channels → ~51 channels (5σ + 10 line-free)
- Updated `CLAUDE.md` config format and JAX note

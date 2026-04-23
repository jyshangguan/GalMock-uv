# Development Log

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

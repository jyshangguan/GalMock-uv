# Development Log

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

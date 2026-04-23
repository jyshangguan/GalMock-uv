# Recurring Problems

## Spectral range formula ignores rotation broadening

The auto-computed spectral range (`nchan`, `velocity_start_kms`) uses
`spectral_n_sigma * intrinsic_sigma_kms` which only covers the turbulent
velocity dispersion (e.g., 5×30 = 150 km/s).  The observed line width is
dominated by rotation (~300 km/s FWHM), so the computed range is far too
narrow and produces no line-free channels.  Always set `line_window_kms`
to the expected observed line FWHM when running the pipeline.

## FWHM measurement without spatial masking

`_measure_fwhm_from_fits()` sums over ALL spatial pixels to compute the
integrated spectrum, including thousands of noise-only edge pixels.  This
degrades SNR and biases the FWHM.  Always use spatial masking (set
`measure_spatial_sigma` in config, default 1.5) to exclude low-S/N pixels.

## JAX + numpy version conflicts

The JAX DysmalPy fork requires `numpy>=2.0` and `astropy>=6.0`, but the
Cython DysmalPy requires `numpy<2.0` and `astropy<6.0`.  Switching between
the two requires reinstalling numpy, astropy, and pyerfa.  After installing
the JAX fork, run:

```bash
pip install --ignore-installed --no-deps pyerfa
```

to fix the `erfa._ARRAY_API not found` error caused by stale pyerfa compiled
against numpy 1.x.

## JAX CUDA setup: nvidia-cuda-nvcc-cu12

`nvidia-cuda-nvcc-cu12>=12.5` sets `__file__ = None`, which causes JAX to
crash at import with `TypeError: argument should be a str`.  Pin to a
working version:

```bash
pip install 'nvidia-cuda-nvcc-cu12>=12.1,<12.5'
```

## JAX XLA OOM for large cubes

JAX's XLA compiler fuses the entire `vmap` computation into one graph.
For `populate_cube_jax` with 201 spectral channels and 201^2 spatial pixels,
XLA attempts to allocate ~363 GB of intermediates.  This exceeds both GPU
memory (24 GB) and sometimes system RAM.  Mitigations:

- Use `zcalc_truncate=True` on `obs.mod_options` to use the sparse AIS variant
- Reduce cube size (101^3 works fine on CPU)
- Free GPU memory for GPU execution

## evosax overwrites JAX version

`evosax==0.2.0` requires `jax>=0.5` in its pip metadata.  Installing it with
`pip install evosax` will pull in jax 0.10 and jaxlib 0.10, silently replacing
jax 0.4.38.  Always install evosax with `--no-deps` and verify jax version
afterwards:

```bash
pip install --no-deps evosax
python -c "import jax; print(jax.__version__)"  # must be 0.4.38
```

## jax.scipy.special.hyp2f1 removed in JAX 0.4.38

JAX removed `jax.scipy.special.hyp2f1` in version 0.4.38.  DysmalPy's
`hyp2f1.py` now checks `hasattr(jax.scipy.special, 'hyp2f1')` and falls back
to the custom power-series implementation when the builtin is unavailable.

## `np.indices()` + `np.vstack()` memory trap

`np.indices(shape)` creates `ndim` full-size arrays (3 x N^3 at 240^3 = 995 MB).
Combined with `np.vstack()` and `.flatten()` copies, this can dominate memory
in helper functions like `_make_cube_ai()`.  Always prefer `np.ravel()` (a view,
zero-copy) + `np.flatnonzero()` + modular index reconstruction
(`idx % nx`, `(idx // nx) % ny`, `idx // (nx * ny)`) to recover C-order
multi-dimensional indices from flat indices without allocating full grids.

## Dimming not applied in active-only cube path

The `_use_active_path` in `model_set.py` evaluates light profiles for active
pixels only (to save memory), but when it was introduced (commit `aa12476`)
the `flux *= dimming(xsky, ysky, zsky)` line was not carried over from the
original path.  This produced cube values 10^10x too large.  Fixed in commit
`56c7144`.  When adding new code paths that mirror existing ones, always check
that all transformations (dimming, extinction, etc.) are applied consistently.

# Recurring Problems

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

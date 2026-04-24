"""Pipeline orchestration: run layers A -> B -> C."""

import sys
from pathlib import Path

from .io import load_config, setup_output_dir, save_metadata, load_metadata
from .env import is_casa_env, is_dysmalpy_env, get_env_type


def run_pipeline(config_path, layers='A', output_dir=None):
    """Run the galmockuv pipeline.

    Parameters
    ----------
    config_path : str or Path
        Path to a TOML or YAML config file.
    layers : str
        Which pipeline layers to run. Options:
        - ``'A'``: build intrinsic cube (requires DysmalPy)
        - ``'B+C'``: simulate + measure (requires CASA)
        - ``'A+B+C'``: full pipeline (requires both)
    output_dir : str or Path, optional
        Override the output directory.

    Returns
    -------
    results : dict
        Dictionary with keys per completed layer ('cube_path', 'ms_path',
        'metadata', 'measurements', etc.).
    """
    config = load_config(config_path)
    source_id = config['source_id']

    if output_dir is None:
        base_dir = config.get('output_base', 'outputs')
        output_dir = setup_output_dir(base_dir, source_id)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"galmockuv pipeline")
    print(f"  source: {source_id}")
    print(f"  env:    {get_env_type()}")
    print(f"  layers: {layers}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}")

    results = {'config_path': str(config_path), 'output_dir': str(output_dir)}

    # ---- Layer A: Build intrinsic cube ----
    if 'A' in layers:
        if not is_dysmalpy_env():
            raise RuntimeError(
                "Layer A requires DysmalPy but it is not available in the "
                "current environment. Activate the alma conda environment."
            )
        from .build_cube import build_cube
        from .plotting import plot_summary, plot_integrated_spectrum

        cube_path, metadata, model_cube = build_cube(config, output_dir)
        results['cube_path'] = str(cube_path)
        results['metadata'] = metadata

        plot_summary(str(cube_path), output_dir, source_id, apply_mask=True,
                     restfreq_ghz=config.get('co_restfreq_ghz'))

        vel_axis = model_cube.spectral_axis.to_value('km/s')
        spec = model_cube.sum(axis=(1, 2)).value
        plot_integrated_spectrum(vel_axis, spec, output_dir, source_id)

        print(f"\n[Layer A complete] Intrinsic cube: {cube_path}")

    # ---- Layer B: Simulate ALMA observation ----
    if 'B' in layers:
        if not is_casa_env():
            raise RuntimeError(
                "Layer B requires CASA (casatasks) but it is not available in "
                "the current environment. Run inside CASA."
            )
        from .simulate import simulate_alma

        cube_path = results.get('cube_path') or str(
            Path(output_dir) / 'intrinsic_cube.fits'
        )

        ms_path, sim_params = simulate_alma(cube_path, config, output_dir)
        results['ms_path'] = str(ms_path)
        results['sim_params'] = sim_params

        print(f"\n[Layer B complete] MeasurementSet: {ms_path}")

    # ---- Layer C: Measure observables ----
    if 'C' in layers:
        if not is_casa_env():
            raise RuntimeError(
                "Layer C requires CASA (casatasks) but it is not available in "
                "the current environment. Run inside CASA."
            )

        # Try the MS path first; fall back to intrinsic cube
        ms_path = results.get('ms_path')
        metadata = results.get('metadata') or load_metadata(output_dir)

        from .measure import measure_from_ms

        measurements = measure_from_ms(ms_path, config, output_dir, metadata)
        results['measurements'] = measurements

        print(f"\n[Layer C complete] Measurements saved to {output_dir}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Pipeline complete. Summary:")
    if 'cube_path' in results:
        print(f"  Intrinsic cube: {results['cube_path']}")
    if 'ms_path' in results:
        print(f"  MeasurementSet: {results['ms_path']}")
    if 'measurements' in results:
        m = results['measurements']
        print(f"  FWHM:  {m.get('fwhm_kms', 'N/A'):.1f} km/s")
        print(f"  Size:  {m.get('size_kpc', 'N/A'):.2f} kpc")
        print(f"  SNR:   {m.get('line_snr', 'N/A'):.1f}")
        print(f"  f_eff: {m.get('f_eff', 'N/A'):.3f}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    return results

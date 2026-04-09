"""Layer B: Simulate an ALMA observation from an intrinsic FITS cube.

Uses CASA ``simobserve`` or ``simalma`` to convert the intrinsic 3D cube
into a simulated MeasurementSet (MS).

This script is designed to be called from within CASA:
    casa -c "import simulate_alma; simulate_alma.run(...)"
"""

import numpy as np
import os
from pathlib import Path

from galmockuv.io import save_metadata, load_config


def _compute_freq_params(config):
    """Compute observed frequency and channel width in Hz.

    CASA ``simobserve`` requires ``incenter`` in GHz and ``inwidth`` in Hz.

    Parameters
    ----------
    config : dict
        Must contain: redshift, co_restfreq_ghz, channel_width_kms.

    Returns
    -------
    obs_freq_ghz : float
    channel_width_hz : float
    """
    z = float(config['redshift'])
    freq_rest = float(config['co_restfreq_ghz'])
    dv_kms = float(config['channel_width_kms'])

    obs_freq_ghz = freq_rest / (1.0 + z)
    # Doppler: dv/c = dnu/nu  =>  dnu = nu * dv / c
    c_kms = 2.99792458e5  # km/s
    channel_width_hz = obs_freq_ghz * 1e9 * dv_kms / c_kms

    return obs_freq_ghz, channel_width_hz


def simulate_alma(cube_path, config, output_dir):
    """Run CASA simobserve on an intrinsic FITS cube.

    Parameters
    ----------
    cube_path : str or Path
        Path to the intrinsic FITS cube.
    config : dict
        Configuration with ALMA observing parameters.
    output_dir : str or Path
        Directory to save outputs.

    Returns
    -------
    ms_path : Path
        Path to the noisy MeasurementSet.
    sim_metadata : dict
        Simulation parameters used.
    """
    output_dir = Path(output_dir)
    cube_path = Path(cube_path)

    if not cube_path.exists():
        raise FileNotFoundError(f"Intrinsic cube not found: {cube_path}")

    # ---- CASA imports (must run inside CASA) ----
    from casatasks import simobserve, simalma

    # ---- Parse config ----
    source_id = config['source_id']
    mode = config.get('alma_mode', 'simobserve')
    pixscale = float(config['pixscale_arcsec'])

    obs_freq_ghz, chan_width_hz = _compute_freq_params(config)

    totaltime = config['totaltime']
    if isinstance(totaltime, (int, float)):
        totaltime_str = f"{totaltime}s"
    else:
        totaltime_str = str(totaltime)

    integration = float(config.get('integration_s', 10.0))
    antennalist = config.get('antennalist', 'alma.cycle7.7')
    # Resolve antennalist path: CASA may not find it automatically.
    # Check in known CASA data directories.
    # The actual files have a .cfg extension (e.g. alma.cycle7.7.cfg).
    if not os.path.exists(antennalist):
        home = os.environ.get('HOME', '/home/shangguan')
        casa_data_dir = os.path.join(home, '.casa', 'data', 'alma', 'simmos')
        if os.path.isdir(casa_data_dir):
            candidate = os.path.join(casa_data_dir, os.path.basename(antennalist))
            if not candidate.endswith('.cfg'):
                candidate += '.cfg'
            if os.path.exists(candidate):
                antennalist = candidate
    print(f"  antennalist resolved to: {antennalist}")
    print(f"  antennalist exists = {os.path.exists(antennalist)}")
    thermalnoise = config.get('thermalnoise', 'tsys-atm')
    pwv = float(config.get('user_pwv', 1.5))
    mapsize = config.get('mapsize', '5arcsec')
    seed = int(config.get('seed', 42))

    # CASA simobserve creates a project directory at <project>/ relative to
    # cwd. Inside it stores <project>.<antennalist>.skymodel etc.
    # If we pass an absolute path as project, CASA treats the entire path as
    # the directory name and then nests it inside itself.
    # Solution: pass a unique simple name and let CASA create it in cwd.
    # After simulation, move outputs to the intended output_dir.
    import uuid
    project_uid = f"sim_{source_id}_{uuid.uuid4().hex[:8]}"
    project = project_uid

    # ---- Log all parameters ----
    sim_params = {
        'mode': mode,
        'project': project,
        'skymodel': str(cube_path),
        'incenter': f"{obs_freq_ghz:.6f}GHz",
        'inwidth': f"{chan_width_hz:.1f}Hz",
        'incell': f"{pixscale}arcsec",
        'totaltime': totaltime_str,
        'integration': f"{integration}s",
        'antennalist': antennalist,
        'thermalnoise': thermalnoise,
        'user_pwv': pwv,
        'mapsize': mapsize,
        'seed': seed,
    }

    print(f"[simulate_alma] Running CASA {mode} for {source_id}...")
    print(f"  obs_freq = {obs_freq_ghz:.4f} GHz")
    print(f"  chan_width = {chan_width_hz:.1f} Hz")
    print(f"  project = {project}")
    print(f"  antennalist = {antennalist}")

    # ---- Run simulation ----
    if mode == 'simalma':
        simalma(
            project=project,
            skymodel=str(cube_path),
            incenter=sim_params['incenter'],
            inwidth=sim_params['inwidth'],
            incell=sim_params['incell'],
            setpointings=True,
            direction="",
            mapsize=mapsize,
            integration=f"{integration}s",
            totaltime=totaltime_str,
            antennalist_12m=antennalist,
            thermalnoise=thermalnoise,
            user_pwv=pwv,
            seed=seed,
            overwrite=True,
        )
        # simalma output MS name
        antenna_tag = '12m'
    else:
        simobserve(
            project=project,
            skymodel=str(cube_path),
            incenter=sim_params['incenter'],
            inwidth=sim_params['inwidth'],
            incell=sim_params['incell'],
            setpointings=True,
            direction="",
            mapsize=mapsize,
            integration=f"{integration}s",
            totaltime=totaltime_str,
            antennalist=antennalist,
            thermalnoise=thermalnoise,
            user_pwv=pwv,
            seed=seed,
            overwrite=True,
        )
        # The antennalist basename (without .cfg) appears in the MS name
        antenna_tag = Path(antennalist).stem  # e.g. 'alma.cycle7.7' from 'alma.cycle7.7.cfg'

    # CASA puts the MS inside the project directory.
    # Output naming: <project>/<project>.<antenna_tag>.noisy.ms
    project_dir = Path(project)
    # Prefer the noisy MS (with thermal noise); fall back to noiseless
    noisy_ms = project_dir / f"{project}.{antenna_tag}.noisy.ms"
    noiseless_ms = project_dir / f"{project}.{antenna_tag}.ms"
    if noisy_ms.exists():
        ms_path = noisy_ms
    elif noiseless_ms.exists():
        ms_path = noiseless_ms
    else:
        raise RuntimeError(
            f"CASA simulation did not produce MS. Checked:\n"
            f"  {noisy_ms}\n"
            f"  {noiseless_ms}\n"
            f"Project dir contents: {list(project_dir.iterdir()) if project_dir.exists() else 'NOT FOUND'}"
        )

    print(f"[simulate_alma] MS created: {ms_path}")
    # Compute total size of the MS directory
    import shutil
    total_size = sum(f.stat().st_size for f in ms_path.rglob('*') if f.is_file())
    print(f"[simulate_alma] MS size: {total_size / 1e6:.1f} MB")

    # Save simulation metadata
    save_metadata(sim_params, output_dir, 'casa_sim_params.yaml')

    # Move MS to final location
    final_ms_path = output_dir / f"{source_id}.ms"
    _move_ms(ms_path, final_ms_path)
    ms_path = final_ms_path

    # Clean up the CASA project directory
    sim_project_dir = Path(project)
    if sim_project_dir.exists() and sim_project_dir.is_dir():
        shutil.rmtree(str(sim_project_dir))

    return ms_path, sim_params


def _move_ms(src, dst):
    """Move a CASA MeasurementSet directory.

    Parameters
    ----------
    src : Path
    dst : Path
    """
    import shutil
    if dst.exists():
        shutil.rmtree(str(dst))
    shutil.move(str(src), str(dst))

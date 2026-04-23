"""Layer B: Simulate an ALMA observation from an intrinsic FITS cube.

Uses CASA ``simobserve`` or ``simalma`` to convert the intrinsic 3D cube
into a simulated MeasurementSet (MS).

This script is designed to be called from within CASA:
    casa -c "import simulate_alma; simulate_alma.run(...)"
"""

import numpy as np
import os
import shutil
import uuid
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


def _parse_totaltime_seconds(totaltime):
    """Parse the totaltime config value into seconds.

    Parameters
    ----------
    totaltime : str, int, or float
        Integration time.  Accepts ``'18400s'``, ``18400``, or ``18400.0``.

    Returns
    -------
    totaltime_s : float
        Integration time in seconds.
    totaltime_str : str
        Normalised string representation (e.g. ``'18400s'``).
    """
    if isinstance(totaltime, (int, float)):
        return float(totaltime), f"{totaltime}s"
    s = str(totaltime).strip()
    if s.endswith('s'):
        return float(s[:-1]), s
    return float(s), f"{float(s)}s"


def _resolve_antennalist(antennalist):
    """Resolve an antennalist name to a full file path.

    CASA may not find the antenna configuration file automatically.
    This checks in ``$HOME/.casa/data/alma/simmos/`` and appends
    ``.cfg`` if needed.

    Parameters
    ----------
    antennalist : str
        Antenna configuration name or path.

    Returns
    -------
    antennalist : str
        Resolved path to the ``.cfg`` file.
    """
    if os.path.exists(antennalist):
        return antennalist
    home = os.environ.get('HOME', '/home/shangguan')
    casa_data_dir = os.path.join(home, '.casa', 'data', 'alma', 'simmos')
    if not os.path.isdir(casa_data_dir):
        return antennalist
    candidate = os.path.join(casa_data_dir, os.path.basename(antennalist))
    if not candidate.endswith('.cfg'):
        candidate += '.cfg'
    if os.path.exists(candidate):
        return candidate
    return antennalist


def _resolve_hourangle(config):
    """Resolve the hour angle for simobserve.

    Without a constraint, simobserve with ``totaltime`` > source visibility
    window offsets the track away from transit, adding high-noise
    low-elevation visibilities that prevent image noise from scaling as
    1/sqrt(t) when using ``thermalnoise='tsys-atm'``.

    Config key ``hourangle`` overrides the default behaviour.  Accepted
    CASA formats include ``'transit'``, ``'0h'``, ``'-3:00:00'``,
    ``'5h'``, etc.  Set it to an empty string ``''`` or ``'auto'`` to
    restore the unconstrained default.

    When ``hourangle`` is not set (or ``None``), the default is ``'transit'``
    which centres the observation track on the source transit.  This keeps
    the highest-elevation (lowest-Tsys) data at the centre of the track and
    gives the best noise scaling.

    Parameters
    ----------
    config : dict
        Simulation configuration.  May contain ``hourangle``.

    Returns
    -------
    hourangle : str
        CASA hour-angle string for ``simobserve`` / ``simalma``.
    """
    ha = config.get('hourangle', None)
    if ha is not None:
        return str(ha)
    return "transit"


def _run_single_track(cube_path, source_id, config, totaltime_str,
                       seed, hourangle):
    """Run a single simobserve/simalma track and return the MS path.

    This is the low-level helper that wraps one CASA simobserve call.
    It does **not** fix weights or move the MS — the caller handles that.

    Parameters
    ----------
    cube_path : Path
        Path to the intrinsic FITS cube.
    source_id : str
        Source identifier (used in the project name).
    config : dict
        Full simulation configuration.
    totaltime_str : str
        Integration time for this track (e.g. ``'18400s'``).
    seed : int
        Random seed for this track.
    hourangle : str
        Hour angle for this track.

    Returns
    -------
    ms_path : Path
        Path to the newly created MeasurementSet (in CASA project dir).
    sim_params : dict
        Simulation parameters used for this track.
    project : str
        CASA project directory name (caller should clean up).
    """
    from casatasks import simobserve, simalma

    mode = config.get('alma_mode', 'simobserve')
    pixscale = float(config['pixscale_arcsec'])
    obs_freq_ghz, chan_width_hz = _compute_freq_params(config)
    integration = float(config.get('integration_s', 10.0))
    antennalist = _resolve_antennalist(config.get('antennalist', 'alma.cycle7.7'))
    thermalnoise = config.get('thermalnoise', 'tsys-atm')
    pwv = float(config.get('user_pwv', 1.5))
    mapsize = config.get('mapsize', '5arcsec')

    project = f"sim_{source_id}_{uuid.uuid4().hex[:8]}"

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
        'hourangle': hourangle,
        'seed': seed,
    }

    print(f"[simulate_alma] Running CASA {mode} for {source_id}...")
    print(f"  obs_freq = {obs_freq_ghz:.4f} GHz")
    print(f"  chan_width = {chan_width_hz:.1f} Hz")
    print(f"  project = {project}")
    print(f"  antennalist = {antennalist}")
    print(f"  totaltime = {totaltime_str}, seed = {seed}, hourangle = {hourangle}")

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
            hourangle=hourangle,
            seed=seed,
            overwrite=True,
        )
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
            hourangle=hourangle,
            seed=seed,
            overwrite=True,
        )
        antenna_tag = Path(antennalist).stem

    # Locate the MS inside the CASA project directory.
    project_dir = Path(project)
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

    total_size = sum(f.stat().st_size for f in ms_path.rglob('*') if f.is_file())
    print(f"[simulate_alma] MS created: {ms_path} ({total_size / 1e6:.1f} MB)")

    return ms_path, sim_params, project


def _move_ms(src, dst):
    """Move a CASA MeasurementSet directory.

    Parameters
    ----------
    src : Path
    dst : Path
    """
    if dst.exists():
        shutil.rmtree(str(dst))
    shutil.move(str(src), str(dst))


def simulate_alma(cube_path, config, output_dir):
    """Run CASA simobserve on an intrinsic FITS cube.

    When ``totaltime`` exceeds ``max_track_hours`` (default 8 h), the
    integration is automatically split into multiple short tracks centred
    on transit.  Each track is simulated independently with a unique seed,
    then the resulting MeasurementSets are concatenated with CASA
    ``concat``.  This mimics how real ALMA observations are scheduled and
    ensures image noise scales as 1/sqrt(t).

    Parameters
    ----------
    cube_path : str or Path
        Path to the intrinsic FITS cube.
    config : dict
        Configuration with ALMA observing parameters.  May contain
        ``max_track_hours`` (float, default 8) to control the maximum
        single-track duration.
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

    source_id = config['source_id']
    thermalnoise = config.get('thermalnoise', 'tsys-atm')
    seed_base = int(config.get('seed', 42))
    hourangle = _resolve_hourangle(config)

    totaltime_s, totaltime_str = _parse_totaltime_seconds(config['totaltime'])
    max_track_hours = float(config.get('max_track_hours', 8))
    max_track_s = max_track_hours * 3600.0

    n_tracks = max(1, int(np.ceil(totaltime_s / max_track_s)))
    track_s = totaltime_s / n_tracks  # distribute evenly

    multi_track = n_tracks > 1
    if multi_track:
        print(f"[simulate_alma] Multi-track scheduling: "
              f"{totaltime_s:.0f}s total = {n_tracks} tracks x {track_s:.0f}s "
              f"(max {max_track_hours}h per track)")

    # ------------------------------------------------------------------
    # Run single track or multiple tracks
    # ------------------------------------------------------------------
    track_ms_paths = []
    sim_params_first = None

    for i in range(n_tracks):
        track_seed = seed_base + i * 1000
        track_time_str = f"{track_s}s"

        if multi_track:
            # For multi-track, always force transit centring regardless of
            # user hourangle setting so each track is at good elevation.
            track_ha = "transit"
        else:
            track_ha = hourangle

        ms_path, sim_params, project_dir = _run_single_track(
            cube_path, source_id, config, track_time_str, track_seed, track_ha
        )

        if sim_params_first is None:
            sim_params_first = sim_params

        if multi_track:
            # Move track MS to a temporary location under output_dir
            track_ms_name = f"{source_id}_track{i:02d}.ms"
            track_final = output_dir / track_ms_name
            _move_ms(ms_path, track_final)
            track_ms_paths.append(track_final)
        else:
            # Single track: move MS to output_dir before cleanup
            track_ms_name = f"{source_id}_track00.ms"
            track_final = output_dir / track_ms_name
            _move_ms(ms_path, track_final)
            track_ms_paths.append(track_final)

        # Clean up CASA project directory
        if Path(project_dir).exists() and Path(project_dir).is_dir():
            shutil.rmtree(str(project_dir))

    # ------------------------------------------------------------------
    # Concatenate tracks if needed
    # ------------------------------------------------------------------
    if multi_track:
        from casatasks import concat as casa_concat

        concat_ms = output_dir / f"{source_id}_concat.ms"
        if concat_ms.exists():
            shutil.rmtree(str(concat_ms))

        print(f"[simulate_alma] Concatenating {n_tracks} tracks -> {concat_ms}")
        casa_concat(
            vis=[str(p) for p in track_ms_paths],
            concatvis=str(concat_ms),
        )

        # Remove individual track MSs to save disk
        for p in track_ms_paths:
            if p.exists():
                shutil.rmtree(str(p))

        ms_path = concat_ms
    else:
        ms_path = track_ms_paths[0]

    # ------------------------------------------------------------------
    # Fix WEIGHT/SIGMA (on the final single or concatenated MS)
    # ------------------------------------------------------------------
    if thermalnoise == 'tsys-atm':
        _fix_weights(ms_path)

    # Save simulation metadata
    sim_params_first['n_tracks'] = n_tracks
    sim_params_first['track_time'] = f"{track_s}s" if multi_track else totaltime_str
    sim_params_first['totaltime_requested'] = totaltime_str
    if multi_track:
        sim_params_first['hourangle_effective'] = 'transit (multi-track)'
    save_metadata(sim_params_first, output_dir, 'casa_sim_params.yaml')

    # Move final MS to its permanent location
    final_ms_path = output_dir / f"{source_id}.ms"
    if ms_path != final_ms_path:
        _move_ms(ms_path, final_ms_path)
    ms_path = final_ms_path

    print(f"[simulate_alma] Final MS: {ms_path}")
    return ms_path, sim_params_first


def _fix_weights(ms_path):
    """Set WEIGHT/SIGMA from per-time-dump noise estimates.

    simobserve with ``thermalnoise='tsys-atm'`` adds elevation-dependent
    thermal noise to the DATA column but leaves WEIGHT and SIGMA at their
    default value of 1.0.  This means tclean weights all visibilities
    equally, including very noisy low-elevation data, which prevents image
    noise from scaling as 1/sqrt(t) for long integration times.

    This function estimates the per-visibility noise from line-free channels
    by computing the RMS across all baselines at each time dump, then sets
    WEIGHT = 1/sigma^2 and SIGMA = sigma.

    Parameters
    ----------
    ms_path : Path
        Path to the MeasurementSet (will be modified in place).
    """
    from casatools import table as casa_table

    chunk_size = 200_000
    n_corr = 2

    tb = casa_table()
    tb.open(str(ms_path), nomodify=False)
    nrow = tb.nrows()

    # Read TIME column to group visibilities by time dump
    times = tb.getcol("TIME")

    # Compute per-dump noise: for each unique time, compute RMS across all
    # baselines in line-free channels.  This gives a single noise estimate
    # per time dump.
    unique_times = np.unique(times)
    n_dumps = len(unique_times)
    dump_noise = np.zeros(n_dumps)

    line_free_ch = [0, 1, 2, 3]  # first 4 channels

    for i, ut in enumerate(unique_times):
        mask = times == ut
        rows = np.where(mask)[0]
        sq_sum = np.array([])
        for start in range(0, len(rows), chunk_size):
            end = min(start + chunk_size, len(rows))
            row_slice = rows[start:end]
            data = tb.getcol("DATA", row_slice[0], end - start)
            vals = []
            for ch in line_free_ch:
                vals.extend(data[0, ch, :].real.tolist())
                vals.extend(data[0, ch, :].imag.tolist())
            if len(sq_sum) == 0:
                sq_sum = np.array(vals) ** 2
            else:
                sq_sum = np.concatenate([sq_sum, np.array(vals) ** 2])

        dump_noise[i] = np.sqrt(np.mean(sq_sum)) if len(sq_sum) > 0 else 1.0

    tb.close()

    # Map dump noise back to per-visibility weights
    time_to_noise = dict(zip(unique_times.tolist(), dump_noise))

    tb = casa_table()
    tb.open(str(ms_path), nomodify=False)

    for start in range(0, nrow, chunk_size):
        end = min(start + chunk_size, nrow)
        chunk_times = times[start:end]
        n = end - start

        w_col = np.zeros((n_corr, n), dtype=np.float64)
        s_col = np.zeros((n_corr, n), dtype=np.float64)
        for j in range(n):
            sig = time_to_noise.get(chunk_times[j], 1.0)
            sig = max(sig, 1e-30)
            w_col[0, j] = 1.0 / sig ** 2
            w_col[1, j] = 1.0 / sig ** 2
            s_col[0, j] = sig
            s_col[1, j] = sig

        tb.putcol("WEIGHT", w_col, start, n)
        tb.putcol("SIGMA", s_col, start, n)

    tb.flush()
    tb.close()

    median_noise = np.median(dump_noise)
    print(f"[simulate_alma] Fixed WEIGHT/SIGMA: median per-dump noise = {median_noise:.4e} Jy, "
          f"range = [{dump_noise.min():.4e}, {dump_noise.max():.4e}]")

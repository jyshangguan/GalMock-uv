"""Layer A: Build an intrinsic 3D CO line cube with DysmalPy.

Generates a galaxy model (exponential disk + NFW halo + constant dispersion)
and produces a 3D spectral cube with known input parameters.

Requires the ``alma`` conda environment (or any env with dysmalpy installed).
CASA is NOT needed for this step.
"""

import numpy as np
import astropy.units as u
from pathlib import Path
from dysmalpy import galaxy, models, observation, instrument

from galmockuv.io import save_metadata


def build_cube(config, output_dir):
    """Build a DysmalPy galaxy model and generate an intrinsic 3D CO cube.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Expected keys:
        - source_id, redshift, co_restfreq_ghz
        - log10_baryonic_mass_msun, disk_reff_kpc, disk_invq, n_disk, bt,
          noord_flat
        - log10_halo_mass_msun, halo_concentration
        - inclination_deg, pa_deg
        - intrinsic_sigma_kms, sigmaz_kpc
        - pressure_support, adiabatic_contract
        - pixscale_arcsec, npix_x, npix_y
        - channel_width_kms, velocity_start_kms, nchan  (or auto-computed from
          spectral_n_sigma, spectral_n_linefree, channel_width_kms)
        - oversample (default 3)
        - intrinsic_beam_major_arcsec, intrinsic_lsf_sigma_kms
    output_dir : str or Path
        Directory to save all outputs.

    Returns
    -------
    cube_path : Path
        Path to the output FITS cube.
    metadata : dict
        Dictionary of true physical parameters for this mock.
    model_cube : spectral_cube.SpectralCube
        The generated spectral cube object.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Parse config ----
    source_id = config['source_id']
    z = float(config['redshift'])
    co_restfreq = float(config['co_restfreq_ghz'])

    log_mbar = float(config['log10_baryonic_mass_msun'])
    r_eff = float(config['disk_reff_kpc'])
    invq = float(config.get('disk_invq', 5.0))
    n_disk = float(config.get('n_disk', 1.0))
    bt = float(config.get('bt', 0.0))
    noord = config.get('noord_flat', True)

    log_mhalo = float(config['log10_halo_mass_msun'])
    conc = float(config['halo_concentration'])

    inc = float(config['inclination_deg'])
    pa = float(config['pa_deg'])

    sigma0 = float(config['intrinsic_sigma_kms'])
    sigmaz = float(config.get('sigmaz_kpc', 0.9))

    p_supp = config.get('pressure_support', True)
    a_contr = config.get('adiabatic_contract', False)

    pixscale = float(config['pixscale_arcsec'])
    npix_x = int(config['npix_x'])
    npix_y = int(config['npix_y'])
    dv = float(config['channel_width_kms'])

    # Auto-compute spectral range from line width, unless manually overridden
    if 'nchan' in config and 'velocity_start_kms' in config:
        nchan = int(config['nchan'])
        v_start = float(config['velocity_start_kms'])
    elif 'line_window_kms' in config:
        line_fwhm = float(config['line_window_kms'])
        n_linefree = int(config.get('spectral_n_linefree', 10))
        half_range = line_fwhm / 2.0 + n_linefree * dv
        nchan = int(np.ceil(2 * half_range / dv)) + 1
        if nchan % 2 == 0:
            nchan += 1
        v_start = -half_range
    else:
        n_sigma = float(config.get('spectral_n_sigma', 5))
        n_linefree = int(config.get('spectral_n_linefree', 10))
        half_range = n_sigma * sigma0 + n_linefree * dv
        nchan = int(np.ceil(2 * half_range / dv)) + 1
        if nchan % 2 == 0:
            nchan += 1
        v_start = -half_range

    beam_arcsec = float(config.get('intrinsic_beam_major_arcsec', 0.01))
    lsf_kms = float(config.get('intrinsic_lsf_sigma_kms', 0.1))

    # ---- Build galaxy model ----
    gal = galaxy.Galaxy(z=z, name=source_id)
    mod_set = models.ModelSet()

    # Baryonic: pure exponential disk (bt=0, n_disk=1)
    bary = models.DiskBulge(
        total_mass=log_mbar, bt=bt, r_eff_disk=r_eff, n_disk=n_disk,
        invq_disk=invq, noord_flat=noord,
        name='bary'
    )

    # Dark matter halo: NFW
    halo = models.NFW(mvirial=log_mhalo, conc=conc, z=z, name='halo')

    # Constant velocity dispersion (gas tracer)
    disp = models.DispersionConst(sigma0=sigma0, tracer='LINE', name='disp')

    # Vertical gas distribution
    zh = models.ZHeightGauss(sigmaz=sigmaz, name='zheight')

    # Geometry: must match observation name
    geom = models.Geometry(
        inc=inc, pa=pa, xshift=0.0, yshift=0.0,
        obs_name='OBS', name='geom'
    )

    # Add components
    mod_set.add_component(bary, light=True)
    mod_set.add_component(halo)
    mod_set.add_component(disp)
    mod_set.add_component(zh)
    mod_set.add_component(geom)

    # Kinematic options
    mod_set.kinematic_options.pressure_support = p_supp
    mod_set.kinematic_options.adiabatic_contract = a_contr

    # ---- Set up observation and instrument ----
    obs = observation.Observation(name='OBS', tracer='LINE')
    oversample = int(config.get('oversample', 3))
    obs.mod_options.oversample = oversample
    obs.mod_options.zcalc_truncate = config.get('zcalc_truncate', True)

    inst = instrument.Instrument()
    # Use beam/LSF much smaller than pixel/channel to avoid convolving the
    # intrinsic model.  The real telescope beam is applied by CASA simobserve.
    inst.beam = instrument.GaussianBeam(major=beam_arcsec * u.arcsec)
    inst.lsf = instrument.LSF(lsf_kms * u.km / u.s)
    inst.pixscale = pixscale * u.arcsec
    inst.fov = [npix_x, npix_y]
    inst.spec_type = 'velocity'
    inst.spec_step = dv * u.km / u.s
    inst.spec_start = v_start * u.km / u.s
    inst.nspec = nchan
    inst.ndim = 3
    inst.moment = False

    inst.set_beam_kernel()
    inst.set_lsf_kernel()

    obs.instrument = inst

    # ---- Wire and generate ----
    gal.model = mod_set
    gal.add_observation(obs)

    print(f"[build_cube] Generating cube for {source_id} "
          f"(z={z}, logMbar={log_mbar}, reff={r_eff} kpc, inc={inc} deg)...")
    gal.create_model_data()
    print("[build_cube] Cube generation complete.")

    # ---- Extract cube ----
    model_cube = gal.observations['OBS'].model_cube.data  # SpectralCube

    # ---- Apply flux scaling if configured ----
    # DysmalPy produces kinematic templates with arbitrary flux normalization.
    # Scale to realistic brightness for the simulated observation.
    flux_scale = float(config.get('flux_scale', 1.0))
    if flux_scale != 1.0:
        print(f"[build_cube] Applying flux scale factor: {flux_scale}")
        raw_data = model_cube.unmasked_data[:].value
        scaled_data = raw_data * flux_scale
        print(f"[build_cube] Peak before scale: {raw_data.max():.2e}, "
              f"after: {scaled_data.max():.2e}")
        # Write to temp file, scale, then use as final
        from astropy.io import fits
        temp_path = output_dir / 'intrinsic_cube_raw.fits'
        model_cube_freq_orig = model_cube.with_spectral_unit(
            u.GHz,
            velocity_convention='radio',
            rest_value=co_restfreq * 1e9 * u.Hz
        )
        model_cube_freq_orig.header['RESTFRQ'] = co_restfreq * 1e9
        model_cube_freq_orig.write(str(temp_path), overwrite=True)

        # Scale the data in the FITS file
        with fits.open(str(temp_path), mode='update') as hdul:
            hdul[0].data *= flux_scale
        model_cube_freq_path = temp_path
    else:
        model_cube_freq_path = None

    # ---- Convert to frequency space for CASA compatibility ----
    # CASA simobserve requires a frequency axis with RESTFRQ set.
    # The DysmalPy cube is in velocity space; convert it here.
    restfreq_hz = co_restfreq * 1e9  # GHz -> Hz
    cube_path = output_dir / 'intrinsic_cube.fits'

    if model_cube_freq_path is not None:
        # Already written and scaled by flux_scale
        import shutil
        shutil.move(str(model_cube_freq_path), str(cube_path))
    else:
        model_cube_freq = model_cube.with_spectral_unit(
            u.GHz,
            velocity_convention='radio',
            rest_value=restfreq_hz * u.Hz
        )
        # Explicitly set RESTFRQ in the header
        model_cube_freq.header['RESTFRQ'] = restfreq_hz
        model_cube_freq.write(str(cube_path), overwrite=True)

    print(f"[build_cube] Cube written to {cube_path}")

    # ---- Compute diagnostics ----
    # Integrated spectrum (sum over spatial axes)
    spec_data = model_cube.sum(axis=(1, 2))  # Quantity
    vel_axis = model_cube.spectral_axis.to(u.km / u.s).value  # km/s

    # ---- Save metadata ----
    metadata = {
        'source_id': source_id,
        'redshift': z,
        'co_restfreq_ghz': co_restfreq,
        'log10_baryonic_mass_msun': log_mbar,
        'baryonic_mass_msun': 10**log_mbar,
        'disk_reff_kpc': r_eff,
        'disk_invq': invq,
        'n_disk': n_disk,
        'bt': bt,
        'log10_halo_mass_msun': log_mhalo,
        'halo_mass_msun': 10**log_mhalo,
        'halo_concentration': conc,
        'inclination_deg': inc,
        'pa_deg': pa,
        'intrinsic_sigma_kms': sigma0,
        'sigmaz_kpc': sigmaz,
        'pressure_support': p_supp,
        'adiabatic_contract': a_contr,
        'pixscale_arcsec': pixscale,
        'npix_x': npix_x,
        'npix_y': npix_y,
        'channel_width_kms': dv,
        'nchan': nchan,
        'cube_shape': list(model_cube.shape),
    }
    save_metadata(metadata, output_dir, 'metadata.yaml')

    return cube_path, metadata, model_cube

"""Layer C: Measure observables from a simulated ALMA MeasurementSet.

This module handles:
1. FWHM measurement from the integrated spectrum (imaged cube or direct from vis)
2. Source size measurement from UV visibilities (using casa_utils)
3. Mass proxy computation: M_dyn ~ f * FWHM^2 * D / G

The UV size measurement requires CASA (casatasks, casatools).
The spectrum measurement can work with either CASA (tclean imaging) or
a pre-existing imaged cube.
"""

import numpy as np
from pathlib import Path

from galmockuv.io import save_json, load_metadata, save_metadata
from galmockuv.casa_utils import average_uvdata, fit_uv_model, get_beam_from_fits, compute_beam_area_pix


# Physical constant
G_PC = 4.301e-3    # pc (km/s)^2 / M_sun
G_KPC = G_PC / 1e3  # kpc (km/s)^2 / M_sun


def measure_from_imaged_cube(cube_path, config, output_dir, metadata):
    """Measure FWHM and size from an already-imaged cube.

    This is the simpler path: the cube has already been imaged from the MS
    (e.g. by tclean), and we measure the spectrum directly.

    For the mock pipeline, the cube produced by build_dysmalpy_cube is the
    *intrinsic* cube.  The observable cube should come from CASA imaging of
    the simulated MS.  However, for testing purposes we can also measure
    the intrinsic cube to verify internal consistency.

    Parameters
    ----------
    cube_path : str or Path
        Path to the FITS cube (intrinsic or imaged).
    config : dict
        Configuration dictionary.
    output_dir : str or Path
    metadata : dict
        True input parameters (from build_dysmalpy_cube).

    Returns
    -------
    measurements : dict
    """
    output_dir = Path(output_dir)
    from spectral_cube import SpectralCube

    cube = SpectralCube.read(str(cube_path))
    source_id = config.get('source_id', 'unknown')

    # ---- Measure FWHM from integrated spectrum ----
    fwhm_result = _measure_fwhm_from_cube(cube, config)

    # ---- Estimate angular size from the moment-0 map ----
    # For now, we use a simple second-moment measurement on the moment-0 map.
    # A proper UV-based size measurement requires the MS (see measure_from_ms).
    size_result = _measure_size_from_moment0(cube, config)

    # ---- Compute mass proxy ----
    mass_result = _compute_mass_proxy(
        fwhm_result['fwhm_kms'],
        size_result.get('size_kpc', metadata.get('disk_reff_kpc', 0)),
        metadata
    )

    measurements = {
        'source_id': source_id,
        'fwhm_kms': fwhm_result['fwhm_kms'],
        'fwhm_err_kms': fwhm_result.get('fwhm_err_kms', 0),
        'line_snr': fwhm_result.get('snr', 0),
        'size_arcsec': size_result.get('size_arcsec', 0),
        'size_kpc': size_result.get('size_kpc', 0),
        'size_method': size_result.get('method', 'moment0_second_moment'),
        'proxy_mass_msun': mass_result['proxy_mass'],
        'true_mass_msun': mass_result['true_mass'],
        'f_eff': mass_result['f_eff'],
    }

    save_json(measurements, output_dir, 'measurements.json')
    print(f"[measure] FWHM = {fwhm_result['fwhm_kms']:.1f} km/s")
    print(f"[measure] Size  = {size_result.get('size_kpc', 0):.2f} kpc")
    print(f"[measure] f_eff = {mass_result['f_eff']:.3f}")

    return measurements


def measure_from_ms(ms_path, config, output_dir, metadata):
    """Full measurement pipeline from a CASA MeasurementSet.

    Images the MS, extracts the spectrum, and fits the UV visibility
    amplitudes to measure the source size.

    This function requires CASA (casatasks, casatools).

    Parameters
    ----------
    ms_path : str or Path
        Path to the simulated MeasurementSet.
    config : dict
        Configuration dictionary.
    output_dir : str or Path
    metadata : dict
        True input parameters.

    Returns
    -------
    measurements : dict
    """
    output_dir = Path(output_dir)
    ms_path = Path(ms_path)
    source_id = config.get('source_id', 'unknown')

    # ---- CASA imports ----
    from casatasks import tclean, split, exportfits
    from casatools import ms as ms_tool

    # ---- Step 1: Image the MS ----
    imagename = str(output_dir / source_id)
    cube_imagename = imagename + '.cube'

    print(f"[measure_from_ms] Imaging {ms_path} ...")

    # Clean up any existing tclean output
    import shutil
    for suffix in ['.image', '.image.fits', '.residual', '.model', '.psf',
                   '.mask', '.pb', '.sumwt', '.weight']:
        p = Path(cube_imagename + suffix)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(str(p))
            else:
                p.unlink()

    # Use CASA tclean to produce a cube
    tclean(
        vis=str(ms_path),
        imagename=cube_imagename,
        outframe='LSRK',
        veltype='radio',
        restfreq=f"{config['co_restfreq_ghz']}GHz",
        specmode='cube',
        nchan=-1,  # all channels
        cell=f"{config['pixscale_arcsec']}arcsec",
        imsize=[config['npix_x'], config['npix_y']],
        weighting='natural',
        niter=int(config.get('tclean_niter', 1000)),
        threshold=config.get('tclean_threshold', '0mJy'),
        interactive=False,
    )

    # Export the CASA image to FITS for reading
    cube_image_dir = Path(cube_imagename + '.image')
    cube_fits = Path(cube_imagename + '.image.fits')
    if not cube_image_dir.exists():
        raise RuntimeError(f"tclean did not produce image: {cube_image_dir}")
    exportfits(imagename=str(cube_image_dir), fitsimage=str(cube_fits),
               overwrite=True)
    print(f"[measure_from_ms] FITS exported: {cube_fits}")

    print(f"[measure_from_ms] Cube imaged: {cube_fits}")

    # ---- Step 2: Measure FWHM from imaged cube ----
    # Use astropy.io.fits since spectral_cube may not be installed in CASA
    fwhm_result = _measure_fwhm_from_fits(str(cube_fits), config)

    # ---- Step 3: Measure size from UV visibilities ----
    # Average channels for continuum-like measurement of the source size
    # Split all channels averaged
    avg_ms = str(output_dir / 'avg.ms')
    # Clean up existing avg.ms
    avg_ms_path = Path(avg_ms)
    if avg_ms_path.exists():
        shutil.rmtree(str(avg_ms_path))
    split(vis=str(ms_path), outputvis=avg_ms,
          datacolumn='corrected', timebin='1e8', combine='scan',
          width=9999)

    uv_result = _measure_size_from_uv(avg_ms, config, output_dir)

    # ---- Step 4: Compute mass proxy ----
    size_kpc = uv_result.get('size_kpc', metadata.get('disk_reff_kpc', 0))
    mass_result = _compute_mass_proxy(fwhm_result['fwhm_kms'], size_kpc, metadata)

    measurements = {
        'source_id': source_id,
        'fwhm_kms': fwhm_result['fwhm_kms'],
        'fwhm_err_kms': fwhm_result.get('fwhm_err_kms', 0),
        'line_snr': fwhm_result.get('snr', 0),
        'size_arcsec': uv_result.get('size_arcsec', 0),
        'size_err_arcsec': uv_result.get('size_err_arcsec', 0),
        'size_kpc': uv_result.get('size_kpc', 0),
        'size_method': 'uv_gaussian_fit',
        'proxy_mass_msun': mass_result['proxy_mass'],
        'true_mass_msun': mass_result['true_mass'],
        'f_eff': mass_result['f_eff'],
        'integrated_flux_jy_kms': fwhm_result.get('integrated_flux_jy_kms', 0),
        'beam_major_arcsec': fwhm_result.get('beam_major_arcsec', 0),
        'beam_minor_arcsec': fwhm_result.get('beam_minor_arcsec', 0),
    }

    save_json(measurements, output_dir, 'measurements.json')

    # Save measurement metadata
    meas_meta = {
        'fwhm_result': {k: v for k, v in fwhm_result.items()
                        if isinstance(v, (int, float, str, bool))},
        'uv_result': {k: v for k, v in uv_result.items()
                      if isinstance(v, (int, float, str, bool, list))},
        'mass_result': {k: v for k, v in mass_result.items()
                        if isinstance(v, (int, float, str, bool))},
        'beam_info': {
            'beam_major_arcsec': fwhm_result.get('beam_major_arcsec', 0),
            'beam_minor_arcsec': fwhm_result.get('beam_minor_arcsec', 0),
            'beam_area_pix': fwhm_result.get('beam_area_pix', 0),
            'beam_area_arcsec2': fwhm_result.get('beam_area_arcsec2', 0),
        },
        'integrated_flux_jy_kms': fwhm_result.get('integrated_flux_jy_kms', 0),
    }
    save_metadata(meas_meta, output_dir, 'measurement_details.yaml')

    print(f"[measure_from_ms] FWHM = {fwhm_result['fwhm_kms']:.1f} km/s")
    print(f"[measure_from_ms] Size  = {uv_result.get('size_kpc', 0):.2f} kpc")
    print(f"[measure_from_ms] Flux  = {fwhm_result.get('integrated_flux_jy_kms', 0):.4f} Jy km/s")
    print(f"[measure_from_ms] f_eff = {mass_result['f_eff']:.3f}")

    return measurements


# ======================================================================
# Internal helper functions
# ======================================================================

def _measure_fwhm_from_cube(cube, config):
    """Measure FWHM from a SpectralCube.

    Parameters
    ----------
    cube : SpectralCube
    config : dict

    Returns
    -------
    dict with 'fwhm_kms', 'fwhm_err_kms', 'snr', 'centroid_kms'
    """
    # Sum over spatial axes to get integrated spectrum
    spec = cube.sum(axis=(1, 2)).value
    vel = cube.spectral_axis.to('km/s').value

    # Sort by velocity (ascending)
    idx = np.argsort(vel)
    vel = vel[idx]
    spec = spec[idx]

    # Fit Gaussian to the spectrum
    from scipy.optimize import curve_fit

    def gaussian(x, amp, cen, sigma):
        return amp * np.exp(-0.5 * ((x - cen) / sigma)**2)

    # Initial guesses
    ipeak = np.argmax(spec)
    amp0 = spec[ipeak]
    cen0 = vel[ipeak]
    sigma0 = 100.0  # km/s initial guess

    try:
        popt, pcov = curve_fit(gaussian, vel, spec,
                               p0=[amp0, cen0, sigma0],
                               maxfev=10000)
        amp, cen, sigma = popt
        perr = np.sqrt(np.diag(pcov))
        sigma_err = perr[2]
    except RuntimeError:
        # Fall back to simple half-max width
        sigma = _estimate_sigma_halfmax(vel, spec)
        sigma_err = 0.0
        amp = amp0
        cen = cen0

    fwhm_kms = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma  # FWHM = 2.355 * sigma
    fwhm_err_kms = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_err

    # SNR: peak / rms of line-free channels
    line_mask = np.abs(vel - cen) < 3.0 * fwhm_kms
    line_free = spec[~line_mask]
    if len(line_free) > 0:
        rms = np.std(line_free)
        snr = amp / rms if rms > 0 else 0
    else:
        rms = 0
        snr = 0

    # Smooth fit curve
    vel_fit = np.linspace(vel.min(), vel.max(), 500)
    spec_fit = gaussian(vel_fit, amp, cen, sigma)

    return {
        'fwhm_kms': float(fwhm_kms),
        'fwhm_err_kms': float(fwhm_err_kms),
        'centroid_kms': float(cen),
        'snr': float(snr),
        'rms': float(rms),
        'vel': vel,
        'spec': spec,
        'vel_fit': vel_fit,
        'spec_fit': spec_fit,
    }


def _estimate_sigma_halfmax(vel, spec):
    """Estimate sigma from half-maximum width."""
    ipeak = np.argmax(spec)
    half_max = 0.5 * spec[ipeak]
    above = spec >= half_max
    indices = np.where(above)[0]
    if len(indices) < 2:
        return 100.0
    fwhm = vel[indices[-1]] - vel[indices[0]]
    return fwhm / 2.355


def _measure_fwhm_from_fits(fits_path, config):
    """Measure FWHM from a FITS cube using astropy.io.fits.

    This is a CASA-compatible alternative to _measure_fwhm_from_cube that
    does not require spectral_cube.

    Parameters
    ----------
    fits_path : str
        Path to a FITS cube (e.g. from CASA tclean output).
    config : dict

    Returns
    -------
    dict with 'fwhm_kms', 'fwhm_err_kms', 'snr', 'centroid_kms'
    """
    from astropy.io import fits
    from scipy.optimize import curve_fit

    hdul = fits.open(fits_path)
    data = hdul[0].data.squeeze()  # shape: (nchan, ny, nx)
    header = hdul[0].header

    # Read beam info for Jy/beam -> Jy conversion
    try:
        beam_area_pix, beam_area_arcsec2, bmaj_arcsec, bmin_arcsec = \
            compute_beam_area_pix(hdul)
        has_beam = True
    except (ValueError, KeyError):
        beam_area_pix = 1.0
        beam_area_arcsec2 = 0.0
        bmaj_arcsec = 0.0
        bmin_arcsec = 0.0
        has_beam = False

    hdul.close()

    # Spatial mask: only include pixels with significant emission in moment-0
    if data.ndim == 3:
        mom0 = np.nansum(data, axis=0)
        # Estimate noise from edge pixels (MAD-based)
        n_edge = 5
        edge = np.concatenate([
            mom0[:n_edge, :].ravel(), mom0[-n_edge:, :].ravel(),
            mom0[:, :n_edge].ravel(), mom0[:, -n_edge:].ravel()
        ])
        noise_est = np.median(np.abs(edge)) * 1.4826  # MAD -> sigma
        mask_sigma = float(config.get('measure_spatial_sigma', 1.5))
        spatial_mask = mom0 > mask_sigma * noise_est
        n_masked = int(np.sum(spatial_mask))
        spec_raw = np.nansum(data[:, spatial_mask], axis=1)
    else:
        spatial_mask = None
        n_masked = 0
        spec_raw = data

    # spec_raw is in Jy/beam * pixels (scale-dependent on beam size).
    # Keep it for backward-compatible FWHM fitting (scale-invariant),
    # but also compute proper Jy spectrum for flux integration.
    spec = spec_raw  # alias for FWHM fitting (unchanged behavior)
    spec_jy = spec_raw / beam_area_pix if has_beam else spec_raw

    nchan = len(spec)

    # Build frequency axis from FITS header keywords directly.
    # CASA FITS has axis order: RA (1), Dec (2), FREQ (3), [STOKES (4)].
    # The numpy array after squeeze is (nchan, ny, nx) when STOKES=1.
    c_ms = 2.99792458e8  # m/s

    # Find the spectral axis (CTYPE with FREQ or VOPT)
    spectral_axis_num = None
    for ax in range(1, header.get('NAXIS', 4) + 1):
        ctype = header.get(f'CTYPE{ax}', '').upper()
        if 'FREQ' in ctype or 'VOPT' in ctype or 'VEL' in ctype:
            spectral_axis_num = ax
            break

    if spectral_axis_num is None:
        # Fallback: axis 3 (standard CASA output)
        spectral_axis_num = 3

    crval = header.get(f'CRVAL{spectral_axis_num}', 0)
    cdelt = header.get(f'CDELT{spectral_axis_num}', 0)
    crpix = header.get(f'CRPIX{spectral_axis_num}', 1)
    nax = header.get(f'NAXIS{spectral_axis_num}', nchan)
    # FITS uses 1-based pixel indexing
    freqs = crval + (np.arange(nax) - crpix + 1) * cdelt  # Hz

    # Convert to velocity: use observed line center as reference to get
    # velocity *offsets* from line center (not the cosmological velocity).
    restfreq = header.get('RESTFRQ', 0)
    if restfreq <= 0:
        restfreq = float(config['co_restfreq_ghz']) * 1e9

    ctype = header.get(f'CTYPE{spectral_axis_num}', '').upper()
    if 'FREQ' in ctype:
        f_center = freqs[nax // 2]  # observed line center frequency
        vel = c_ms * (f_center - freqs) / f_center / 1e3  # km/s offsets
    else:
        vel = freqs / 1e3  # already in velocity, convert to km/s

    # Sort by velocity (ascending)
    idx = np.argsort(vel)
    vel = vel[idx]
    spec = spec[idx]
    spec_jy = spec_jy[idx]

    # Channel width (km/s) — assume uniform spacing after sort
    dv = float(np.median(np.abs(np.diff(vel)))) if len(vel) > 1 else 0.0

    # Fit Gaussian
    def gaussian(x, amp, cen, sigma):
        return amp * np.exp(-0.5 * ((x - cen) / sigma)**2)

    ipeak = np.argmax(np.abs(spec))
    amp0 = spec[ipeak]
    cen0 = vel[ipeak]
    sigma0 = 100.0

    try:
        popt, pcov = curve_fit(gaussian, vel, spec,
                               p0=[amp0, cen0, sigma0],
                               maxfev=10000)
        amp, cen, sigma = popt
        perr = np.sqrt(np.diag(pcov))
        sigma_err = perr[2]
    except RuntimeError:
        sigma = _estimate_sigma_halfmax(vel, spec)
        sigma_err = 0.0
        amp = amp0
        cen = cen0

    fwhm_kms = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    fwhm_err_kms = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_err

    # SNR: peak / rms of line-free channels
    line_mask = np.abs(vel - cen) < 3.0 * fwhm_kms
    line_free = spec[~line_mask]
    if len(line_free) > 0 and len(line_free) < len(spec):
        rms = np.std(line_free)
    else:
        # Fall back: use edge channels (first/last 15%) as line-free estimate
        n_edge = max(1, int(0.15 * len(spec)))
        edge_channels = np.concatenate([spec[:n_edge], spec[-n_edge:]])
        rms = np.std(edge_channels)
    snr = abs(amp) / rms if rms > 0 else 0

    vel_fit = np.linspace(vel.min(), vel.max(), 500)
    spec_fit = gaussian(vel_fit, amp, cen, sigma)

    # Integrated flux: sum spec_jy over line channels * dv
    if has_beam and dv > 0:
        integrated_flux_jy_kms = float(np.sum(spec_jy[line_mask]) * dv)
    else:
        integrated_flux_jy_kms = 0.0

    return {
        'fwhm_kms': float(fwhm_kms),
        'fwhm_err_kms': float(fwhm_err_kms),
        'centroid_kms': float(cen),
        'snr': float(snr),
        'rms': float(rms),
        'vel': vel,
        'spec': spec,
        'vel_fit': vel_fit,
        'spec_fit': spec_fit,
        # New keys: integrated flux and beam info
        'integrated_flux_jy_kms': integrated_flux_jy_kms,
        'beam_area_pix': float(beam_area_pix) if has_beam else 0.0,
        'beam_area_arcsec2': float(beam_area_arcsec2) if has_beam else 0.0,
        'beam_major_arcsec': float(bmaj_arcsec) if has_beam else 0.0,
        'beam_minor_arcsec': float(bmin_arcsec) if has_beam else 0.0,
        'spec_jy': spec_jy,
    }


def _measure_size_from_moment0(cube, config):
    """Estimate source size from second moment of moment-0 map.

    Parameters
    ----------
    cube : SpectralCube
    config : dict

    Returns
    -------
    dict with 'size_arcsec', 'size_kpc'
    """
    m0 = cube.moment(order=0).value
    pixscale = np.abs(cube.header['CDELT1']) * 3600  # arcsec/pixel
    ny, nx = m0.shape

    # Centroid (flux-weighted)
    total = np.nansum(m0)
    if total <= 0:
        return {'size_arcsec': 0, 'size_kpc': 0, 'method': 'moment0_second_moment'}

    cx = np.nansum(np.arange(nx) * np.nansum(m0, axis=0)) / total
    cy = np.nansum(np.arange(ny) * np.nansum(m0, axis=1)) / total

    # Second moment (flux-weighted variance)
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    var_x = np.nansum((xx - cx)**2 * m0) / total
    var_y = np.nansum((yy - cy)**2 * m0) / total
    sigma_arcsec = np.sqrt(var_x + var_y) * pixscale  # 1-sigma radius

    # Convert to FWHM diameter
    fwhm_arcsec = 2.355 * sigma_arcsec  # FWHM = 2.355 * sigma for Gaussian

    # Physical size
    z = float(config.get('redshift', 0))
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dl_mpc = cosmo.angular_diameter_distance(z).value
    kpc_per_arcsec = dl_mpc * 1e3 * np.pi / (180.0 * 3600.0)
    size_kpc = fwhm_arcsec * kpc_per_arcsec

    return {
        'size_arcsec': float(fwhm_arcsec),
        'size_kpc': float(size_kpc),
        'method': 'moment0_second_moment',
    }


def _measure_size_from_uv(avg_ms, config, output_dir):
    """Measure source size from UV visibilities using casa_utils.

    Parameters
    ----------
    avg_ms : str
        Path to channel-averaged MeasurementSet.
    config : dict
    output_dir : Path

    Returns
    -------
    dict with 'size_arcsec', 'size_err_arcsec', 'size_kpc'
    """
    output_dir = Path(output_dir)

    # Set up UV bins
    uv_bin_width = float(config.get('uv_bin_width_klambda', 50.0))
    uv_max = float(config.get('uv_max_klambda', 3000.0))
    uv_bins = np.arange(0, uv_max + uv_bin_width, uv_bin_width)

    print(f"[measure_size_uv] Binning UV data: {len(uv_bins)-1} bins, "
          f"0-{uv_max} klambda, width={uv_bin_width}")

    # Average visibility amplitudes
    # Note: split() renames the selected column to 'data' in the output MS
    uv_dist, amp, err_lo, err_hi = average_uvdata(
        vis=avg_ms,
        datacolumn='data',
        uvbins=uv_bins,
        units='klambda',
        axis='real',
        skip_empty=True,
        verbose=False,
    )

    # Fit Gaussian size model via uvmodelfit
    comptype = config.get('size_fit_model', 'G')
    # Initial guess: circular Gaussian, 0.2 Jy, at center
    sourcepar = [0.2, 0., 0., 0.3, 1.0, 0.]
    # Vary flux (0) and bmaj (3); fix position and axrat/pa
    varpar = [0, 3]

    fit_result = fit_uv_model(
        vis=avg_ms,
        comptype=comptype,
        sourcepar=sourcepar,
        varpar=varpar,
        outfile=str(output_dir / 'uvfit.cl'),
        uvbin_params={'n_bins': len(uv_bins) - 1, 'binning_type': 'width'},
        datacolumn='data',
        avg_axis='real',
        verbose=False,
    )

    size_arcsec = 0
    size_err_arcsec = 0
    if fit_result['success'] and fit_result['size'] is not None:
        size_arcsec = fit_result['size']['bmaj']['value']
        size_err_arcsec = fit_result['size']['bmaj']['error']
        print(f"[measure_size_uv] Fit succeeded: bmaj = {size_arcsec:.4f} +/- "
              f"{size_err_arcsec:.4f} arcsec")
    else:
        print(f"[measure_size_uv] WARNING: UV fit did not converge!")

    # Convert to physical size
    z = float(config.get('redshift', 0))
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dl_mpc = cosmo.angular_diameter_distance(z).value
    kpc_per_arcsec = dl_mpc * 1e3 * np.pi / (180.0 * 3600.0)
    size_kpc = size_arcsec * kpc_per_arcsec

    return {
        'size_arcsec': float(size_arcsec),
        'size_err_arcsec': float(size_err_arcsec),
        'size_kpc': float(size_kpc),
        'uv_dist': uv_dist.tolist() if hasattr(uv_dist, 'tolist') else list(uv_dist),
        'uv_amp': amp.tolist() if hasattr(amp, 'tolist') else list(amp),
        'fit_result': fit_result,
    }


def _compute_mass_proxy(fwhm_kms, size_kpc, metadata):
    """Compute dynamical mass proxy and effective scaling factor.

    proxy_mass = FWHM^2 * D / G   (in M_sun)

    where:
      FWHM in km/s
      D in kpc (source diameter)
      G = 4.301 kpc (km/s)^2 / M_sun

    f_eff = M_true / proxy_mass

    Parameters
    ----------
    fwhm_kms : float
    size_kpc : float
    metadata : dict
        Must contain 'baryonic_mass_msun' (or 'log10_baryonic_mass_msun').

    Returns
    -------
    dict with 'proxy_mass', 'true_mass', 'f_eff'
    """
    true_mass = metadata.get('baryonic_mass_msun', 0)
    if true_mass == 0:
        log_m = metadata.get('log10_baryonic_mass_msun', 10)
        true_mass = 10**log_m

    if size_kpc > 0 and fwhm_kms > 0:
        proxy_mass = fwhm_kms**2 * size_kpc / G_KPC
        f_eff = true_mass / proxy_mass
    else:
        proxy_mass = 0
        f_eff = np.nan

    return {
        'proxy_mass': float(proxy_mass),
        'true_mass': float(true_mass),
        'f_eff': float(f_eff),
    }

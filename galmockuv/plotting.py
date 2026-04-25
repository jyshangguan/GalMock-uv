"""Diagnostic plotting utilities.

All moment-map functions read raw FITS data directly and compute moments
manually with ``np.nansum`` rather than using ``SpectralCube.moment()``.
SpectralCube applies unit conversions that depend on the spectral axis
units (Hz vs km/s) and beam handling, which produces incorrect moment
values for CASA-exported FITS cubes with a frequency axis.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from pathlib import Path
from scipy.optimize import curve_fit
from astropy.io import fits


# ---------------------------------------------------------------------------
# Low-level helpers: FITS I/O, velocity axis, masking, moment computation
# ---------------------------------------------------------------------------

def read_fits_cube(fits_path):
    """Read a FITS cube, returning (data_3d, header, pixscale_arcsec, extent).

    FITS stores data in reverse NAXIS order in numpy.  For standard radio cubes:
      NAXIS1=RA, NAXIS2=Dec, NAXIS3=FREQ, [NAXIS4=STOKES]
      => numpy shape = (STOKES, FREQ, DEC, RA) = e.g. (1, 201, 201, 201)
    After squeezing Stokes, we get (nchan, ny, nx).

    Parameters
    ----------
    fits_path : str or Path
        Path to a FITS cube.

    Returns
    -------
    data : ndarray, shape (nchan, ny, nx)
    header : astropy.io.fits.Header
    pixscale : float
        Arcsec per pixel.
    extent : list of 4 float
        [ra0, ra1, dec0, dec1] in arcsec for ``imshow``.
    """
    hdul = fits.open(fits_path)
    data = hdul[0].data
    header = hdul[0].header
    hdul.close()

    data = np.squeeze(data)
    assert data.ndim == 3, f"Expected 3D cube after squeeze, got {data.shape}"

    pixscale = np.abs(header['CDELT1']) * 3600  # NAXIS1 = RA
    nchan, ny, nx = data.shape
    ra0 = (-nx // 2 - 0.5) * pixscale
    ra1 = (nx // 2 + 0.5) * pixscale
    dec0 = (-ny // 2 - 0.5) * pixscale
    dec1 = (ny // 2 + 0.5) * pixscale
    extent = [ra0, ra1, dec0, dec1]

    return data, header, pixscale, extent


def get_velocity_axis(header, restfreq_ghz=None):
    """Build velocity axis (km/s offsets) from FITS header.

    For frequency-axis cubes, uses ``v = c * (f_center - freqs) / f_center``
    (observed-frame centred).  This matches the convention used by the
    intrinsic DysmalPy cube.

    Parameters
    ----------
    header : astropy.io.fits.Header
    restfreq_ghz : float, optional
        CO rest frequency in GHz.  Falls back to the ``RESTFRQ`` header
        keyword if not provided.

    Returns
    -------
    vel : ndarray, shape (nchan,)
        Velocity offset in km/s.
    """
    nax = header.get('NAXIS', 4)
    spectral_axis_num = None
    for ax_i in range(1, nax + 1):
        ctype = header.get('CTYPE' + str(ax_i), '').upper()
        if 'FREQ' in ctype or 'VOPT' in ctype or 'VEL' in ctype:
            spectral_axis_num = ax_i
            break
    if spectral_axis_num is None:
        spectral_axis_num = 3

    crval = header.get('CRVAL' + str(spectral_axis_num), 0)
    cdelt = header.get('CDELT' + str(spectral_axis_num), 0)
    crpix = header.get('CRPIX' + str(spectral_axis_num), 1)
    nch = header.get('NAXIS' + str(spectral_axis_num))
    freqs = crval + (np.arange(nch) - crpix + 1) * cdelt  # Hz

    restfreq = header.get('RESTFRQ', 0)
    if restfreq <= 0 and restfreq_ghz is not None:
        restfreq = restfreq_ghz * 1e9

    ctype = header.get('CTYPE' + str(spectral_axis_num), '').upper()
    if 'FREQ' in ctype:
        f_center = freqs[nch // 2]
        vel = 2.99792458e8 * (f_center - freqs) / f_center / 1e3  # km/s
    else:
        vel = freqs / 1e3
    return vel


def compute_moment0(data_3d):
    """Moment-0: integrated intensity (sum over spectral axis)."""
    return np.nansum(data_3d, axis=0)


def make_signal_mask(m0, n_sigma=3.0, dilate=2):
    """Spatial mask from moment-0 using sigma-clipped noise estimate.

    Parameters
    ----------
    m0 : ndarray, shape (ny, nx)
    n_sigma : float
        Detection threshold above RMS.
    dilate : int
        Number of binary dilation iterations.

    Returns
    -------
    mask : bool ndarray, shape (ny, nx)
    """
    from astropy.stats import sigma_clipped_stats
    _, _, std = sigma_clipped_stats(m0, sigma=3, maxiters=5)
    mask = m0 > n_sigma * std
    if dilate > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=dilate)
    return mask


def make_3d_mask(data_3d, vel, spatial_mask, channel_sigma=2.0):
    """3D (channel x spatial) mask for robust moment calculation.

    Combines a 2D spatial mask with a 1D channel mask.  Line-free RMS is
    estimated from the outer 20% of the velocity range.

    Parameters
    ----------
    data_3d : ndarray, shape (nchan, ny, nx)
    vel : ndarray, shape (nchan,)
    spatial_mask : bool ndarray, shape (ny, nx)
    channel_sigma : float
        Channel detection threshold above line-free RMS.

    Returns
    -------
    mask : bool ndarray, shape (nchan, ny, nx)
    """
    from astropy.stats import sigma_clipped_stats
    nch = data_3d.shape[0]

    spec = np.nansum(data_3d, axis=(1, 2))
    idx = np.argsort(vel)
    vel_s = vel[idx]
    spec_s = spec[idx]

    n_free = nch // 5
    line_free = np.concatenate([spec_s[:n_free], spec_s[-n_free:]])
    _, _, rms = sigma_clipped_stats(line_free, sigma=3, maxiters=5)

    channel_mask_sorted = spec_s > channel_sigma * rms
    channel_mask = np.zeros(nch, dtype=bool)
    channel_mask[idx] = channel_mask_sorted

    return channel_mask[:, None, None] & spatial_mask[None, :, :]


def compute_moment1(data_3d, vel, mask=None):
    """Moment-1: intensity-weighted mean velocity.

    Pixels with total flux below 5% of the 99th percentile are set to NaN.

    Parameters
    ----------
    data_3d : ndarray, shape (nchan, ny, nx)
    vel : ndarray, shape (nchan,)
    mask : None, 2D (ny, nx), or 3D (nchan, ny, nx) bool array

    Returns
    -------
    m1 : ndarray, shape (ny, nx)
        Intensity-weighted mean velocity in km/s.
    """
    idx = np.argsort(vel)
    d = data_3d[idx]
    v = vel[idx]

    if mask is not None:
        d = d.copy()
        if mask.ndim == 3:
            d[~mask[idx]] = 0
        else:
            d[:, ~mask] = 0

    total = np.nansum(d, axis=0)

    if np.any(total > 0):
        min_total = 0.05 * np.nanpercentile(total[total > 0], 99)
    else:
        min_total = 0

    with np.errstate(invalid='ignore', divide='ignore'):
        m1 = np.where(total > min_total,
                      np.nansum(v[:, None, None] * d, axis=0) / total,
                      np.nan)
    return m1


def compute_moment2(data_3d, vel, mask=None):
    """Moment-2: intensity-weighted velocity dispersion (sigma).

    Parameters
    ----------
    data_3d : ndarray, shape (nchan, ny, nx)
    vel : ndarray, shape (nchan,)
    mask : None, 2D (ny, nx), or 3D (nchan, ny, nx) bool array

    Returns
    -------
    m2 : ndarray, shape (ny, nx)
        Velocity dispersion in km/s.
    """
    idx = np.argsort(vel)
    d = data_3d[idx]
    v = vel[idx]

    if mask is not None:
        d = d.copy()
        if mask.ndim == 3:
            d[~mask[idx]] = 0
        else:
            d[:, ~mask] = 0

    m1 = compute_moment1(data_3d, vel, mask=mask)

    total = np.nansum(d, axis=0)
    if np.any(total > 0):
        min_total = 0.05 * np.nanpercentile(total[total > 0], 99)
    else:
        min_total = 0

    m1_safe = np.where(np.isfinite(m1), m1, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        m2_var = np.where(
            total > min_total,
            np.nansum(((v[:, None, None] - m1_safe[None, :, :]) ** 2) * d, axis=0) / total,
            np.nan,
        )
    m2_sigma = np.sqrt(np.maximum(m2_var, 0))
    return m2_sigma


def compute_spectrum(data_3d, vel):
    """Integrated spectrum (sum over spatial axes), sorted by velocity."""
    spec = np.nansum(data_3d, axis=(1, 2))
    idx = np.argsort(vel)
    return vel[idx], spec[idx]


from .measure import fwhm_half_max

def fit_gaussian_spectrum(vel, spec):
    """Fit a Gaussian to a spectrum.

    Returns
    -------
    fwhm, fwhm_err, amp, cen, sigma : float
    """
    ipeak = np.argmax(np.abs(spec))
    try:
        popt, pcov = curve_fit(_gaussian, vel, spec,
                               p0=[spec[ipeak], vel[ipeak], 100], maxfev=10000)
        amp, cen, sigma = popt
        sigma_err = np.sqrt(np.diag(pcov))[2]
        fwhm = 2.355 * sigma
        fwhm_err = 2.355 * sigma_err
        return fwhm, fwhm_err, amp, cen, sigma
    except RuntimeError:
        return 0, 0, spec[ipeak], vel[ipeak], 100


def _gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)


# ---------------------------------------------------------------------------
# High-level plotting functions
# ---------------------------------------------------------------------------

def plot_moment_maps(fits_path, output_dir, source_id, apply_mask=False,
                     restfreq_ghz=None):
    """Generate individual moment-0, moment-1, moment-2 map figures.

    Parameters
    ----------
    fits_path : str or Path
        Path to a FITS cube (intrinsic or cleaned).
    output_dir : str or Path
        Directory to save plots.
    source_id : str
        Source identifier for plot titles.
    apply_mask : bool
        If True, apply 3D masking to moment-1 and moment-2.
    restfreq_ghz : float, optional
        CO rest frequency in GHz.  Passed to ``get_velocity_axis``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, header, _, extent = read_fits_cube(fits_path)
    vel = get_velocity_axis(header, restfreq_ghz=restfreq_ghz)

    m0 = compute_moment0(data)

    if apply_mask:
        spatial_mask = make_signal_mask(m0, n_sigma=3.0, dilate=2)
        mask = make_3d_mask(data, vel, spatial_mask, channel_sigma=2.0)
    else:
        mask = None

    m1 = compute_moment1(data, vel, mask=mask)
    m2 = compute_moment2(data, vel, mask=mask)

    for order, label, cmap, mom_data in [
        (0, 'Moment 0', 'viridis', m0),
        (1, 'Moment 1', 'RdBu_r', m1),
        (2, 'Moment 2', 'magma', m2),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        if order == 0:
            vmax = np.nanpercentile(np.abs(mom_data[mom_data != 0]), 99) if np.any(mom_data != 0) else 1
            im = ax.imshow(mom_data, origin='lower', extent=extent,
                           cmap=cmap, norm=AsinhNorm(vmin=0, vmax=vmax),
                           aspect='equal')
            unit_label = 'Flux'
        elif order == 1:
            valid = mom_data[np.isfinite(mom_data)]
            vmax = np.nanpercentile(np.abs(valid), 99) if len(valid) > 0 else 1
            im = ax.imshow(mom_data, origin='lower', extent=extent,
                           cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
            unit_label = 'km/s'
        else:
            valid = mom_data[np.isfinite(mom_data) & (mom_data > 0)]
            vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 1
            im = ax.imshow(mom_data, origin='lower', extent=extent,
                           cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
            unit_label = 'km/s'

        plt.colorbar(im, ax=ax, label=unit_label)
        ax.set_xlabel(r'$\Delta$RA (")')
        ax.set_ylabel(r'$\Delta$Dec (")')
        ax.set_title(f'{label} - {source_id}')
        plt.tight_layout()
        fig.savefig(output_dir / f'moment{order}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_summary(fits_path, output_dir, source_id, apply_mask=False,
                 restfreq_ghz=None, extra_panel_fn=None):
    """Generate a 2x2 summary figure: moment-0, moment-1, moment-2, spectrum+fit.

    Parameters
    ----------
    fits_path : str or Path
        Path to a FITS cube (intrinsic or cleaned).
    output_dir : str or Path
        Directory to save the plot.
    source_id : str
    apply_mask : bool
        If True, apply 3D masking to moment-1 and moment-2.
    restfreq_ghz : float, optional
        CO rest frequency in GHz.
    extra_panel_fn : callable, optional
        ``extra_panel_fn(ax)`` draws custom content in the bottom-right panel.
        If None, an integrated spectrum with Gaussian fit is shown.
    """
    output_dir = Path(output_dir)
    data, header, _, extent = read_fits_cube(fits_path)
    vel = get_velocity_axis(header, restfreq_ghz=restfreq_ghz)

    m0 = compute_moment0(data)

    if apply_mask:
        spatial_mask = make_signal_mask(m0, n_sigma=3.0, dilate=2)
        mask = make_3d_mask(data, vel, spatial_mask, channel_sigma=2.0)
    else:
        mask = None

    m1 = compute_moment1(data, vel, mask=mask)
    m2 = compute_moment2(data, vel, mask=mask)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Moment 0
    ax = axes[0, 0]
    vmax = np.nanpercentile(np.abs(m0[m0 != 0]), 99) if np.any(m0 != 0) else 1
    im = ax.imshow(m0, origin='lower', extent=extent, cmap='viridis',
                   norm=AsinhNorm(vmin=0, vmax=vmax), aspect='equal')
    plt.colorbar(im, ax=ax)
    ax.set_title('Moment 0 (Integrated Flux)')
    ax.set_xlabel(r'$\Delta$RA (")')
    ax.set_ylabel(r'$\Delta$Dec (")')

    # Moment 1
    ax = axes[0, 1]
    valid = m1[np.isfinite(m1)]
    vmax1 = np.nanpercentile(np.abs(valid), 99) if len(valid) > 0 else 1
    im = ax.imshow(m1, origin='lower', extent=extent, cmap='RdBu_r',
                   vmin=-vmax1, vmax=vmax1, aspect='equal')
    plt.colorbar(im, ax=ax)
    ax.set_title('Moment 1 (Velocity Field)')
    ax.set_xlabel(r'$\Delta$RA (")')
    ax.set_ylabel(r'$\Delta$Dec (")')

    # Moment 2
    ax = axes[1, 0]
    valid2 = m2[np.isfinite(m2) & (m2 > 0)]
    vmax2 = np.nanpercentile(valid2, 99) if len(valid2) > 0 else 1
    im = ax.imshow(m2, origin='lower', extent=extent, cmap='magma',
                   vmin=0, vmax=vmax2, aspect='equal')
    plt.colorbar(im, ax=ax)
    ax.set_title('Moment 2 (Velocity Dispersion)')
    ax.set_xlabel(r'$\Delta$RA (")')
    ax.set_ylabel(r'$\Delta$Dec (")')

    # Bottom-right panel: spectrum + FWHM (or custom)
    ax = axes[1, 1]
    if extra_panel_fn is not None:
        extra_panel_fn(ax)
    else:
        vel_s, spec_s = compute_spectrum(data, vel)
        fwhm = fwhm_half_max(vel_s, spec_s)

        ax.plot(vel_s, spec_s, 'k-', lw=1)
        ax.text(0.05, 0.95, f"FWHM = {fwhm:.1f} km/s",
                transform=ax.transAxes, va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Flux')
        ax.set_title('Integrated Spectrum')
        ax.axhline(0, color='gray', ls='--', lw=0.5)

    fig.suptitle(f'Mock Summary - {source_id}', fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Array-based plotting (no FITS dependency, kept from original)
# ---------------------------------------------------------------------------

def plot_integrated_spectrum(vel_axis, flux, output_dir, source_id,
                             flux_unit='mJy', vel_unit='km/s'):
    """Plot the integrated spectrum.

    Parameters
    ----------
    vel_axis : array-like
        Velocity axis values.
    flux : array-like
        Flux values.
    output_dir : str or Path
    source_id : str
    flux_unit : str
    vel_unit : str
    """
    output_dir = Path(output_dir)
    vel = np.asarray(vel_axis)
    flx = np.asarray(flux)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vel, flx, 'k-', lw=1)
    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel(f'Velocity ({vel_unit})')
    ax.set_ylabel(f'Flux ({flux_unit})')
    ax.set_title(f'Integrated Spectrum - {source_id}')
    ax.minorticks_on()
    plt.tight_layout()
    fig.savefig(output_dir / 'integrated_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_uv_amplitude(uv_dist, amp, amp_err, fit_result, output_dir, source_id):
    """Plot binned UV amplitude vs uv-distance with best-fit model.

    Parameters
    ----------
    uv_dist : array-like
        UV distance in klambda.
    amp : array-like
        Visibility amplitude in mJy.
    amp_err : array-like
        Amplitude uncertainty in mJy.
    fit_result : dict
        Dictionary with 'uv_model' (array of model amplitudes) and 'best_fit_size'.
    output_dir : str or Path
    source_id : str
    """
    output_dir = Path(output_dir)
    uv = np.asarray(uv_dist)
    am = np.asarray(amp)
    ae = np.asarray(amp_err)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(uv, am, yerr=ae, fmt='ko', ms=4, capsize=2, label='Data')

    if fit_result is not None and 'uv_model' in fit_result:
        ax.plot(uv, fit_result['uv_model'], 'r-', lw=2,
                label=f"Fit: $\\theta$={fit_result.get('best_fit_size', 0):.3f}\"")

    ax.set_xlabel('uv-distance (k$\\lambda$)')
    ax.set_ylabel('Visibility Amplitude (mJy)')
    ax.set_title(f'UV Amplitude - {source_id}')
    ax.legend()
    ax.minorticks_on()
    plt.tight_layout()
    fig.savefig(output_dir / 'uv_amplitude.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_spectrum_fit(vel_axis, flux, fit_result, output_dir, source_id,
                      flux_unit='mJy', vel_unit='km/s'):
    """Plot spectrum with line fit overlay.

    Parameters
    ----------
    vel_axis : array-like
    flux : array-like
    fit_result : dict
        Must contain 'x_fit', 'y_fit', 'fwhm', 'centroid'.
    output_dir : str or Path
    source_id : str
    flux_unit : str
    vel_unit : str
    """
    output_dir = Path(output_dir)
    vel = np.asarray(vel_axis)
    flx = np.asarray(flux)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vel, flx, 'k-', lw=1, label='Data')

    if fit_result is not None and 'x_fit' in fit_result:
        ax.plot(fit_result['x_fit'], fit_result['y_fit'], 'r-', lw=2, label='Fit')
        ax.axvline(fit_result['centroid'], color='gray', ls=':', lw=1)
        txt = f"FWHM = {fit_result['fwhm']:.1f} km/s"
        if 'fwhm_err' in fit_result:
            txt += f" $\\pm$ {fit_result['fwhm_err']:.1f}"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.axhline(0, color='gray', ls='--')
    ax.set_xlabel(f'Velocity ({vel_unit})')
    ax.set_ylabel(f'Flux ({flux_unit})')
    ax.set_title(f'Spectrum Fit - {source_id}')
    ax.legend()
    ax.minorticks_on()
    plt.tight_layout()
    fig.savefig(output_dir / 'spectrum_fit.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_batch_summary(catalog, output_dir):
    """Generate calibration plots for a batch run.

    Parameters
    ----------
    catalog : astropy.table.Table
        Master catalog with measured and true parameters.
    output_dir : str or Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colnames = catalog.colnames
    plot_pairs = []
    if 'fwhm_kms' in colnames and 'log10_baryonic_mass_msun' in colnames:
        plot_pairs.append(('fwhm_kms', 'log10_baryonic_mass_msun',
                           'FWHM (km/s)', r'$\log_{10}\,M_{\rm bar}\,(M_\odot)$',
                           'fwhm_vs_mass'))
    if 'size_kpc' in colnames and 'disk_reff_kpc' in colnames:
        plot_pairs.append(('size_kpc', 'disk_reff_kpc',
                           'Measured D (kpc)', 'True $r_{\\rm eff}$ (kpc)',
                           'size_vs_true'))
    if 'f_eff' in colnames and 'inclination_deg' in colnames:
        plot_pairs.append(('f_eff', 'inclination_deg',
                           '$f_{\\rm eff}$', 'Inclination (deg)',
                           'feff_vs_inc'))
    if 'f_eff' in colnames and 'log10_baryonic_mass_msun' in colnames:
        plot_pairs.append(('f_eff', 'log10_baryonic_mass_msun',
                           '$f_{\\rm eff}$', r'$\log_{10}\,M_{\rm bar}\,(M_\odot)$',
                           'feff_vs_mass'))

    for xcol, ycol, xlabel, ylabel, fname in plot_pairs:
        x = np.array(catalog[xcol], dtype=float)
        y = np.array(catalog[ycol], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x[mask], y[mask], c='k', s=20, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.minorticks_on()
        plt.tight_layout()
        fig.savefig(output_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

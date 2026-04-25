"""Generate report-quality plots for the galmockuv demo.

NOTE: We compute moment maps manually from raw FITS data rather than using
SpectralCube.moment().  SpectralCube applies unit conversions that depend on
the spectral axis units (Hz vs km/s) and beam handling, which produces
incorrect moment values for CASA-exported FITS cubes with a frequency axis.
Manual computation gives consistent, correct results in all cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.optimize import curve_fit

from galmockuv.io import load_metadata, load_json

OUT = Path('demo/output/z25_demo')
FIGS = Path('demo/figs')
FIGS.mkdir(parents=True, exist_ok=True)
meta = load_metadata(OUT)
meas = load_json(OUT, 'measurements.json')

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
z = meta['redshift']

line_fit_model = meas.get('line_fit_model', 'gaussian')

# Load the configured profile function (bypass galfit_uv.__init__ to avoid emcee dep)
if line_fit_model != 'gaussian':
    import importlib.util as _ilu
    _lp_spec = _ilu.spec_from_file_location(
        'galfit_uv_lineprofiles',
        '/home/shangguan/Softwares/my_modules/Galfit-uv/galfit_uv/lineprofiles.py',
    )
    _lp = _ilu.module_from_spec(_lp_spec)
    _lp_spec.loader.exec_module(_lp)
    _profile_map = {
        'doublepeak': _lp.Gaussian_DoublePeak,
        'doublepeak_asymmetric': _lp.Gaussian_DoublePeak_Asymmetric,
    }
    _line_profile = _profile_map[line_fit_model]


def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma)**2)


from galmockuv.measure import fwhm_half_max


def read_fits_cube(fits_path):
    """Read a FITS cube, returning (data_3d, header, pixscale_arcsec, extent).

    FITS stores data in reverse NAXIS order in numpy.  For standard radio cubes:
      NAXIS1=RA, NAXIS2=Dec, NAXIS3=FREQ, [NAXIS4=STOKES]
      => numpy shape = (STOKES, FREQ, DEC, RA) = e.g. (1, 201, 201, 201)
    After squeezing Stokes, we get (nchan, ny, nx) which is what we need.

    For the DysmalPy intrinsic cube (3D):
      NAXIS1=RA, NAXIS2=Dec, NAXIS3=FREQ
      => numpy shape = (FREQ, DEC, RA) = (201, 201, 201) = (nchan, ny, nx)
    """
    hdul = fits.open(fits_path)
    data = hdul[0].data
    header = hdul[0].header
    hdul.close()

    # Squeeze any length-1 axes (typically Stokes)
    data = np.squeeze(data)

    # data should now be (nchan, ny, nx) in all standard cases
    # because FITS stores in reverse NAXIS order:
    #   NAXIS1(RA) -> last numpy axis
    #   NAXIS2(Dec) -> second-to-last
    #   NAXIS3(Freq) -> third-to-last = axis 0 after squeeze
    #   NAXIS4(Stokes) -> first = axis 0 before squeeze
    assert data.ndim == 3, f"Expected 3D cube after squeeze, got {data.shape}"

    pixscale = np.abs(header['CDELT1']) * 3600  # arcsec/pixel (NAXIS1 = RA)
    nchan, ny, nx = data.shape
    ra0 = (-nx // 2 - 0.5) * pixscale
    ra1 = (nx // 2 + 0.5) * pixscale
    dec0 = (-ny // 2 - 0.5) * pixscale
    dec1 = (ny // 2 + 0.5) * pixscale
    extent = [ra0, ra1, dec0, dec1]

    return data, header, pixscale, extent


def get_velocity_axis(header):
    """Build velocity axis (km/s offsets) from FITS header."""
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
    if restfreq <= 0:
        restfreq = meta['co_restfreq_ghz'] * 1e9

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
    """Create a spatial mask from moment-0 using sigma-clipped noise estimate.

    Pixels with m0 > n_sigma * rms are included, then the mask is dilated
    by ``dilate`` iterations to capture faint surrounding emission.

    Returns a boolean array with same shape as m0.
    """
    from astropy.stats import sigma_clipped_stats
    _, _, std = sigma_clipped_stats(m0, sigma=3, maxiters=5)
    rms = std
    mask = m0 > n_sigma * rms
    if dilate > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=dilate)
    return mask


def make_3d_mask(data_3d, vel, spatial_mask, channel_sigma=2.0):
    """Create a 3D (channel x spatial) mask for robust moment calculation.

    Combines a 2D spatial mask with a 1D channel mask derived from the
    integrated spectrum.  Channels where the integrated flux exceeds
    ``channel_sigma`` times the line-free RMS are kept.  This prevents
    noisy line-free channels from corrupting the intensity-weighted
    velocity and dispersion estimates.

    Returns a boolean 3D array with same shape as ``data_3d``.
    """
    from astropy.stats import sigma_clipped_stats
    nch = data_3d.shape[0]

    # Integrated spectrum (sorted by velocity)
    spec = np.nansum(data_3d, axis=(1, 2))
    idx = np.argsort(vel)
    vel_s = vel[idx]
    spec_s = spec[idx]

    # Line-free RMS from outer 20% of velocity range
    n_free = nch // 5
    line_free = np.concatenate([spec_s[:n_free], spec_s[-n_free:]])
    _, _, rms = sigma_clipped_stats(line_free, sigma=3, maxiters=5)

    # Channel mask
    channel_mask_sorted = spec_s > channel_sigma * rms
    channel_mask = np.zeros(nch, dtype=bool)
    channel_mask[idx] = channel_mask_sorted

    # Combine: channel AND spatial (broadcasting)
    return channel_mask[:, None, None] & spatial_mask[None, :, :]


def compute_moment1(data_3d, vel, mask=None):
    """Moment-1: intensity-weighted mean velocity.

    If ``mask`` is a 2D spatial mask, only masked pixels contribute.
    If ``mask`` is a 3D mask, only masked voxels contribute.
    Pixels with total flux below a minimum threshold are set to NaN
    to avoid division-by-near-zero artifacts.
    """
    idx = np.argsort(vel)
    d = data_3d[idx]
    v = vel[idx]

    if mask is not None:
        d = d.copy()
        d[~mask[idx]] = 0 if mask.ndim == 3 else 0
        if mask.ndim == 2:
            d[:, ~mask] = 0

    total = np.nansum(d, axis=0)

    # Minimum flux threshold: 5% of 99th percentile of positive totals
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

    If ``mask`` is a 2D spatial mask, only masked pixels contribute.
    If ``mask`` is a 3D mask, only masked voxels contribute.
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


def plot_moment_panels(data_3d, vel, extent, title_prefix, out_path,
                       extra_panel_fn=None, apply_mask=False):
    """Generate a 2x2 summary figure: moment0, moment1, moment2, + extra panel.

    If ``apply_mask`` is True, a 3D channel+spatial mask is derived and
    applied to moment-1 and moment-2 calculations.  The channel mask
    excludes noisy line-free channels; the spatial mask excludes empty
    pixels.  A minimum flux threshold per pixel avoids division artifacts.
    """
    m0 = compute_moment0(data_3d)

    if apply_mask:
        spatial_mask = make_signal_mask(m0, n_sigma=3.0, dilate=2)
        mask = make_3d_mask(data_3d, vel, spatial_mask, channel_sigma=2.0)
    else:
        mask = None

    m1 = compute_moment1(data_3d, vel, mask=mask)
    m2 = compute_moment2(data_3d, vel, mask=mask)

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

    # Extra panel (spectrum + fit)
    ax = axes[1, 1]
    if extra_panel_fn is not None:
        extra_panel_fn(ax)

    fig.suptitle(title_prefix, fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# Figure 1: Intrinsic cube summary (4-panel)
# =========================================================================
print("Generating intrinsic_summary.png...")
data_int, hdr_int, _, extent_int = read_fits_cube(str(OUT / 'intrinsic_cube.fits'))
vel_int = get_velocity_axis(hdr_int)

fwhm_int = [0]  # mutable container for closure

def intrinsic_spectrum_panel(ax):
    vel_s, spec_s = compute_spectrum(data_int, vel_int)
    fwhm = fwhm_half_max(vel_s, spec_s)
    fwhm_int[0] = fwhm

    ax.plot(vel_s, spec_s, 'k-', lw=1)
    ax.text(0.05, 0.95,
            'FWHM = {:.1f} km/s'.format(fwhm),
            transform=ax.transAxes, va='top',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('Flux')
    ax.set_title('Integrated Spectrum')

plot_moment_panels(data_int, vel_int, extent_int,
                    'Intrinsic Model \u2014 {}'.format(meta['source_id']),
                    FIGS / 'intrinsic_summary.png',
                    extra_panel_fn=intrinsic_spectrum_panel)
print("  Intrinsic FWHM = {:.1f} km/s".format(fwhm_int[0]))


# =========================================================================
# Figure 2: Visibility data (2-panel)
# =========================================================================
print("Generating visibility_data.png...")
try:
    from casatools import ms as ms_tool
    _has_casa = True
except ImportError:
    _has_casa = False

if _has_casa:
    ms_path = str(OUT / 'z25_demo.ms')
    mst = ms_tool()
    mst.open(ms_path)
    mst.selectinit(datadescid=0)
    chunk = mst.getdata(['u', 'v', 'data'])
    mst.close()

    uu = np.array(chunk['u']).flatten()
    vv = np.array(chunk['v']).flatten()
    data_uv = np.array(chunk['data']).flatten()

    freq_obs = meta['co_restfreq_ghz'] * 1e9 / (1 + z)
    lam = 2.99792458e8 / freq_obs
    uvdist = np.sqrt(uu**2 + vv**2) / lam / 1e3
    vis_real = np.real(data_uv)

    np.random.seed(42)
    n_max = 50000
    sel = np.random.choice(len(uvdist), min(n_max, len(uvdist)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(uvdist[sel], vis_real[sel], s=0.3, alpha=0.3, c='k', rasterized=True)
    ax.set_xlabel(r'uv-distance (k$\lambda$)')
    ax.set_ylabel('Re(V) (Jy)')
    ax.set_title('Visibility Amplitude')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlim(0, np.percentile(uvdist, 99))
    finite_vis = vis_real[np.isfinite(vis_real)]
    rng = np.nanpercentile(np.abs(finite_vis), [1, 99])
    ax.set_ylim(rng[0] * 1.2, rng[1] * 1.2)

    ax = axes[1]
    u_kl = uu / lam / 1e3
    v_kl = vv / lam / 1e3
    ax.scatter(u_kl[sel], v_kl[sel], s=0.3, alpha=0.3, c='k', rasterized=True)
    ax.set_xlabel(r'u (k$\lambda$)')
    ax.set_ylabel(r'v (k$\lambda$)')
    ax.set_title('uv Coverage')
    ax.set_aspect('equal')
    lim = np.percentile(np.abs(np.concatenate([u_kl, v_kl])), 99)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    plt.tight_layout()
    fig.savefig(FIGS / 'visibility_data.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Max uv-distance: {:.1f} klambda".format(uvdist.max()))
else:
    uvdist = np.array([])
    print("  Skipping visibility_data.png (CASA not available)")


# =========================================================================
# Figure 3: Cleaned image summary (4-panel)
# =========================================================================
print("Generating cleaned_summary.png...")
data_clean, hdr_clean, _, extent_clean = read_fits_cube(str(OUT / 'z25_demo.cube.image.fits'))
vel_clean = get_velocity_axis(hdr_clean)

fwhm_clean = [0]
fwhm_err_clean = [0]

def cleaned_spectrum_panel(ax):
    vel_s, spec_s = compute_spectrum(data_clean, vel_clean)
    vel_fit = np.linspace(vel_s.min(), vel_s.max(), 500)
    ipeak = np.argmax(np.abs(spec_s))
    amp0 = float(spec_s[ipeak])
    cen0 = float(vel_s[ipeak])

    spec_fit = None
    fit_label = ''

    if line_fit_model == 'gaussian':
        try:
            popt, _ = curve_fit(gaussian, vel_s, spec_s,
                                p0=[amp0, cen0, 50.0],
                                maxfev=20000)
            amp, cen, sigma = popt
            spec_fit = gaussian(vel_fit, amp, cen, sigma)
            fit_label = 'Gaussian'
        except (RuntimeError, ValueError):
            pass
    elif line_fit_model == 'doublepeak':
        w0 = max(50.0, abs(vel_s[ipeak] - vel_s[0]) * 0.15)
        try:
            popt, _ = curve_fit(_line_profile, vel_s, spec_s,
                                p0=[amp0, amp0*0.3, cen0, 80.0, w0],
                                maxfev=20000,
                                bounds=([0, 0, vel_s.min(), 1, 1],
                                        [np.inf, np.inf, vel_s.max(), 500, 500]))
            spec_fit = _line_profile(vel_fit, *popt)
            fit_label = 'DoublePeak'
        except (RuntimeError, ValueError):
            pass
    elif line_fit_model == 'doublepeak_asymmetric':
        w0 = max(50.0, abs(vel_s[ipeak] - vel_s[0]) * 0.15)
        try:
            popt, _ = curve_fit(_line_profile, vel_s, spec_s,
                                p0=[amp0, amp0*0.9, amp0*0.3, cen0,
                                    80.0, w0*0.9, w0*1.1],
                                maxfev=20000,
                                bounds=([0, 0, 0, vel_s.min(), 1, 0.1, 0.1],
                                        [np.inf, np.inf, np.inf, vel_s.max(), 500, 500, 500]))
            spec_fit = _line_profile(vel_fit, *popt)
            fit_label = 'DoublePeak (asym)'
        except (RuntimeError, ValueError):
            pass

    if spec_fit is not None:
        fwhm = fwhm_half_max(vel_fit, spec_fit)
        ax.plot(vel_s, spec_s, 'k-', lw=1, label='Data')
        ax.plot(vel_fit, spec_fit, 'r-', lw=2, label=fit_label)
        ax.legend()
    else:
        fwhm = fwhm_half_max(vel_s, spec_s)
        ax.plot(vel_s, spec_s, 'k-', lw=1)

    fwhm_clean[0] = fwhm
    fwhm_err_clean[0] = 0.0

    ax.axvline(cen0, color='gray', ls=':', lw=1)
    ax.text(0.05, 0.95,
            'FWHM = {:.1f} km/s'.format(fwhm),
            transform=ax.transAxes, va='top',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('Flux')
    ax.set_title('Cleaned Spectrum')
    ax.axhline(0, color='gray', ls='--', lw=0.5)

plot_moment_panels(data_clean, vel_clean, extent_clean,
                    'Cleaned Image \u2014 {}'.format(meta['source_id']),
                    FIGS / 'cleaned_summary.png',
                    extra_panel_fn=cleaned_spectrum_panel,
                    apply_mask=True)
print("  Cleaned FWHM = {:.1f} km/s".format(fwhm_clean[0]))


# =========================================================================
# Figure 4: UV amplitude fit
# =========================================================================
print("Generating uv_amplitude_fit.png...")

uv_fit_method = meas.get('uv_fit_method', 'uvmodelfit')
max_uvdist = float(uvdist.max()) if len(uvdist) > 0 else 0.0

if _has_casa:
    avg_ms = str(OUT / 'avg.ms')

    if uv_fit_method == 'mcmc':
        mcmc_fits_path = str(OUT / 'mcmc_fit' / 'fit_results.fits')
        mcmc_hdul = fits.open(mcmc_fits_path)
        mcmc_hdr = mcmc_hdul[0].header
        data_ext = mcmc_hdul[1].data
        model_ext = mcmc_hdul[2].data
        samples_ext = mcmc_hdul[4].data

        u_data = np.array(data_ext['U'])
        v_data = np.array(data_ext['V'])
        vis_re = np.array(data_ext['RE'])
        vis_im = np.array(data_ext['IM'])
        weight = np.array(data_ext['WEIGHT'])

        lam = meta['co_restfreq_ghz'] * 1e9 / (1 + z) / 2.99792458e8
        uvdist_data = np.sqrt(u_data**2 + v_data**2) / lam / 1e3
        amp_data = np.sqrt(vis_re**2 + vis_im**2) * 1e3

        model_re = np.array(model_ext['RE'])
        model_im = np.array(model_ext['IM'])
        uvdist_model = np.sqrt(u_data**2 + v_data**2) / lam / 1e3
        amp_model = np.sqrt(model_re**2 + model_im**2) * 1e3

        n_samples = min(100, len(samples_ext))
        idx_s = np.random.choice(len(samples_ext), size=n_samples, replace=False)
        amp_mean = amp_model.mean()

        mcmc_hdul.close()

        fig, ax = plt.subplots(figsize=(10, 6))

        np.random.seed(42)
        n_max = 30000
        sel = np.random.choice(len(uvdist_data), min(n_max, len(uvdist_data)), replace=False)
        ax.scatter(uvdist_data[sel], amp_data[sel], s=0.3, alpha=0.15, c='steelblue',
                   rasterized=True, label='Data')

        from scipy.stats import binned_statistic
        uv_bins = np.logspace(np.log10(uvdist_data.min() * 0.9),
                              np.log10(uvdist_data.max() * 1.1), 20)
        bin_means, _, _ = binned_statistic(uvdist_data, amp_data,
                                            statistic='mean', bins=uv_bins)
        bin_errs, _, _ = binned_statistic(uvdist_data, amp_data,
                                           statistic='std', bins=uv_bins)
        bin_centers = 0.5 * (uv_bins[:-1] + uv_bins[1:])
        mask = np.isfinite(bin_means) & np.isfinite(bin_errs)
        ax.errorbar(bin_centers[mask], bin_means[mask], yerr=bin_errs[mask],
                    fmt='o', color='k', ms=5, capsize=3, label='Binned data')

        model_means, _, _ = binned_statistic(uvdist_model, amp_model,
                                              statistic='mean', bins=uv_bins)
        ax.plot(bin_centers[mask], model_means[mask], 'r-', lw=2, label='MCMC model')

        ax.set_xscale('log')
        ax.set_xlabel(r'uv-distance (k$\lambda$)')
        ax.set_ylabel('Amplitude (mJy)')
        ax.set_title('UV Amplitude — {}'.format(meta['source_id']))
        ax.legend(fontsize=10)

        chi2 = mcmc_hdr.get('REDCHI2', 0)
        bic = mcmc_hdr.get('BIC', 0)
        size_arcsec = meas.get('size_arcsec', 0)
        size_err = meas.get('size_err_arcsec', 0)
        size_method = meas.get('size_method', 'mcmc')
        ax.text(0.97, 0.97,
                f'FWHM = {size_arcsec:.3f}" $\\pm$ {size_err:.3f}"\n'
                f'$\\chi^2_\\nu$ = {chi2:.3f}, BIC = {bic:.0f}\n'
                f'({size_method})',
                transform=ax.transAxes, va='top', ha='right', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        fig.savefig(FIGS / 'uv_amplitude_fit.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  MCMC size = {size_arcsec:.3f}\" ± {size_err:.3f}\" (FWHM)")

        report_data = {
            'intrinsic_fwhm_kms': float(fwhm_int[0]),
            'cleaned_fwhm_kms': float(fwhm_clean[0]),
            'cleaned_fwhm_err_kms': float(fwhm_err_clean[0]),
            'uv_bmaj_arcsec': float(size_arcsec),
            'uv_bmaj_err_arcsec': float(size_err),
            'uv_flux_mjy': 0.0,
            'max_uvdist_klambda': max_uvdist,
            'measured_snr': float(meas['line_snr']),
            'measured_size_kpc': float(meas['size_kpc']),
            'measured_f_eff': float(meas['f_eff']),
        }

    else:
        from galmockuv.casa_utils import plot_uvbins, fit_uv_model

        fit_result = plot_uvbins(
            vis=avg_ms,
            datacolumn='data',
            avg_axis='real',
            uvbin_params={'n_bins': 15, 'binning_type': 'log'},
            plotfile=str(FIGS / 'uv_amplitude_fit.png'),
            target_name=meta['source_id'],
            fit_results=None,
            verbose=False,
        )

        uv_info_result = fit_uv_model(
            vis=avg_ms,
            comptype='G',
            sourcepar=[0.2, 0., 0., 0.3, 1.0, 0.],
            varpar=[0, 3],
            outfile=str(OUT / 'uvfit_report.cl'),
            uvbin_params={'n_bins': 15, 'binning_type': 'log'},
            datacolumn='data',
            avg_axis='real',
            verbose=False,
        )

        fit_result_with_model = plot_uvbins(
            vis=avg_ms,
            datacolumn='data',
            avg_axis='real',
            uvbin_params={'n_bins': 15, 'binning_type': 'log'},
            plotfile=str(FIGS / 'uv_amplitude_fit.png'),
            target_name=meta['source_id'],
            fit_results=uv_info_result,
            verbose=False,
        )

        print("  UV fit bmaj = {:.3f}\"".format(uv_info_result['size']['bmaj']['value']))

        report_data = {
            'intrinsic_fwhm_kms': float(fwhm_int[0]),
            'cleaned_fwhm_kms': float(fwhm_clean[0]),
            'cleaned_fwhm_err_kms': float(fwhm_err_clean[0]),
            'uv_bmaj_arcsec': float(uv_info_result['size']['bmaj']['value']),
            'uv_bmaj_err_arcsec': float(uv_info_result['size']['bmaj']['error']),
            'uv_flux_mjy': float(uv_info_result['flux']['value'] * 1000),
            'max_uvdist_klambda': max_uvdist,
            'measured_snr': float(meas['line_snr']),
            'measured_size_kpc': float(meas['size_kpc']),
            'measured_f_eff': float(meas['f_eff']),
        }
else:
    print("  Skipping uv_amplitude_fit.png (CASA not available)")
    report_data = {
        'intrinsic_fwhm_kms': float(fwhm_int[0]),
        'cleaned_fwhm_kms': float(fwhm_clean[0]),
        'cleaned_fwhm_err_kms': float(fwhm_err_clean[0]),
        'max_uvdist_klambda': max_uvdist,
    }

from galmockuv.io import save_json
save_json(report_data, OUT, 'report_data.json')

print("\nAll report plots generated successfully.")
print("Key results:")
for k, v in report_data.items():
    print("  {}: {:.3f}".format(k, v))

"""Diagnostic plotting utilities."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path


def plot_moment_maps(spectral_cube, output_dir, source_id):
    """Generate moment-0, moment-1, moment-2 maps from a SpectralCube.

    Parameters
    ----------
    spectral_cube : spectral_cube.SpectralCube
        The spectral cube (must have .moment() method).
    output_dir : str or Path
        Directory to save plots.
    source_id : str
        Source identifier for plot titles.
    """
    output_dir = Path(output_dir)
    pixscale = np.abs(spectral_cube.header['CDELT1']) * 3600  # arcsec/pix
    nx, ny = spectral_cube.shape[1], spectral_cube.shape[2]

    # Compute extent in arcsec (relative to center)
    ra0 = (-nx // 2 - 0.5) * pixscale
    ra1 = (nx // 2 + 0.5) * pixscale
    dec0 = (-ny // 2 - 0.5) * pixscale
    dec1 = (ny // 2 + 0.5) * pixscale
    extent = [ra0, ra1, dec0, dec1]

    for order, label, cmap in [(0, 'Moment 0', 'viridis'),
                                (1, 'Moment 1', 'RdBu_r'),
                                (2, 'Moment 2', 'magma')]:
        mom = spectral_cube.moment(order=order)
        data = mom.value

        fig, ax = plt.subplots(figsize=(6, 5))
        if order == 0:
            vmax = np.nanpercentile(np.abs(data), 99)
            vmin = 0
            norm = matplotlib.colors.AsinhNorm(vmin=0, vmax=vmax)
            unit_label = str(mom.unit)
        elif order == 1:
            valid = data[np.isfinite(data)]
            vmax = np.nanpercentile(np.abs(valid), 99) if len(valid) > 0 else 1
            vmin = -vmax
            unit_label = 'km/s'
        else:
            # Moment 2 returns variance; convert to sigma (km/s)
            data = np.sqrt(np.maximum(data, 0))
            valid = data[np.isfinite(data) & (data > 0)]
            vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 1
            vmin = 0
            unit_label = 'km/s'

        im = ax.imshow(data, origin='lower', extent=extent,
                       cmap=cmap, norm=norm if order == 0 else None,
                       vmin=vmin if order != 0 else None,
                       vmax=vmax if order != 0 else None,
                       aspect='equal')

        plt.colorbar(im, ax=ax, label=unit_label)
        ax.set_xlabel(r'$\Delta$RA (")')
        ax.set_ylabel(r'$\Delta$Dec (")')
        ax.set_title(f'{label} - {source_id}')
        plt.tight_layout()
        fig.savefig(output_dir / f'moment{order}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    return output_dir


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


def plot_summary_single(cube, meas, output_dir, source_id):
    """Multi-panel summary figure for one source.

    Parameters
    ----------
    cube : spectral_cube.SpectralCube
    meas : dict
        Measurement results.
    output_dir : str or Path
    source_id : str
    """
    output_dir = Path(output_dir)
    pixscale = np.abs(cube.header['CDELT1']) * 3600
    nx, ny = cube.shape[1], cube.shape[2]
    ra0 = (-nx // 2 - 0.5) * pixscale
    ra1 = (nx // 2 + 0.5) * pixscale
    dec0 = (-ny // 2 - 0.5) * pixscale
    dec1 = (ny // 2 + 0.5) * pixscale
    extent = [ra0, ra1, dec0, dec1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: moment-0
    ax = axes[0, 0]
    m0 = cube.moment(order=0).value
    vmax = np.nanpercentile(m0, 99)
    im = ax.imshow(m0, origin='lower', extent=extent, cmap='viridis',
                   vmin=0, vmax=vmax, aspect='equal')
    plt.colorbar(im, ax=ax)
    ax.set_title('Moment 0')
    ax.set_xlabel(r'$\Delta$RA (")')
    ax.set_ylabel(r'$\Delta$Dec (")')

    # Panel 2: moment-1
    ax = axes[0, 1]
    m1 = cube.moment(order=1).value
    vmax = np.nanpercentile(np.abs(m1[m1 != 0]), 99) if np.any(m1 != 0) else 1
    im = ax.imshow(m1, origin='lower', extent=extent, cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, aspect='equal')
    plt.colorbar(im, ax=ax)
    ax.set_title('Moment 1')
    ax.set_xlabel(r'$\Delta$RA (")')
    ax.set_ylabel(r'$\Delta$Dec (")')

    # Panel 3: integrated spectrum
    ax = axes[1, 0]
    spec = cube.sum(axis=(1, 2)).value
    vel = np.arange(len(spec))  # channel index
    ax.plot(vel, spec, 'k-', lw=1)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Flux')
    ax.set_title('Integrated Spectrum')

    # Panel 4: summary text
    ax = axes[1, 1]
    ax.axis('off')
    txt_lines = ['Summary:', '']
    for k, v in meas.items():
        if isinstance(v, float):
            txt_lines.append(f'{k}: {v:.3f}')
        else:
            txt_lines.append(f'{k}: {v}')
    ax.text(0.1, 0.9, '\n'.join(txt_lines), transform=ax.transAxes,
            va='top', fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.suptitle(f'Mock Summary - {source_id}', fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(output_dir / 'summary.png', dpi=150, bbox_inches='tight')
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

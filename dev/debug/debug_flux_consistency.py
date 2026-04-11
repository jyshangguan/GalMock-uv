#!/usr/bin/env python
"""debug_flux_consistency.py -- Diagnose flux inconsistency across ALMA antenna configs.

Demonstrates and quantifies two causes of recovered flux variation:
  1. Real physics: interferometric missing flux (extended arrays resolve out emission)
  2. Measurement artifact: improper Jy/beam -> Jy conversion in flux extraction

Usage (alma conda env has both DysmalPy and CASA):
  /home/shangguan/Softwares/miniconda3/envs/alma/bin/python debug_flux_consistency.py
"""

import sys
import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GALMOCK_REPO = Path('/home/shangguan/Softwares/my_modules/GalMock-uv')
if str(GALMOCK_REPO) not in sys.path:
    sys.path.insert(0, str(GALMOCK_REPO))

DEBUG_DIR = Path('/home/shangguan/Softwares/my_modules/GalMock-uv/dev/debug')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_FLUX_JY_KMS = 1.0
TOTALTIME = '36800s'
ANTENNA_CONFIGS = ['alma.cycle7.5', 'alma.cycle7.3', 'alma.cycle7.1']
SOURCE_PREFIX = 'fluxdebug'

BASE_CONFIG = {
    'source_id': SOURCE_PREFIX,
    'redshift': 2.2,
    'co_restfreq_ghz': 461.041,
    'log10_baryonic_mass_msun': 11.5,
    'disk_reff_kpc': 5.0,
    'disk_invq': 5.0,
    'n_disk': 1.0,
    'bt': 0.0,
    'noord_flat': True,
    'log10_halo_mass_msun': 12.5,
    'halo_concentration': 5.0,
    'inclination_deg': 20.0,
    'pa_deg': 30.0,
    'intrinsic_sigma_kms': 30.0,
    'sigmaz_kpc': 0.9,
    'pressure_support': True,
    'adiabatic_contract': False,
    'pixscale_arcsec': 0.05,
    'npix_x': 128,
    'npix_y': 128,
    'channel_width_kms': 10.0,
    'velocity_start_kms': -1000.0,
    'nchan': 201,
    'intrinsic_beam_major_arcsec': 0.01,
    'intrinsic_lsf_sigma_kms': 0.1,
    'flux_scale': 1e-5,
    'alma_mode': 'simobserve',
    'integration_s': 10.0,
    'thermalnoise': 'tsys-atm',
    'user_pwv': 1.5,
    'mapsize': '15arcsec',
    'seed': 42,
    'nbeam_extract': 1,
    'uv_bin_width_klambda': 50.0,
    'uv_max_klambda': 3000.0,
    'size_fit_model': 'G',
    'line_fit_model': 'gaussian',
    'tclean_niter': 1000,
    'tclean_threshold': '0mJy',
    'output_base': str(DEBUG_DIR),
}


# ===================================================================
# Utility functions
# ===================================================================

def normalize_cube_flux(cube_path, channel_width_kms,
                        target_flux_jy_kms=TARGET_FLUX_JY_KMS):
    """Rescale FITS cube so total integrated line flux = target."""
    from astropy.io import fits

    hdul = fits.open(str(cube_path))
    data = hdul[0].data.copy()
    header = hdul[0].header
    hdul.close()

    current_flux = np.nansum(data) * channel_width_kms
    if current_flux <= 0:
        print(f"  [normalize] WARNING: current flux = {current_flux:.4e}, skip")
        return 1.0

    scale = target_flux_jy_kms / current_flux
    data *= scale
    header['BUNIT'] = 'Jy/pixel'
    fits.writeto(str(cube_path), data, header, overwrite=True)

    verify = np.nansum(data) * channel_width_kms
    print(f"  [normalize] {current_flux:.4e} -> {verify:.4f} Jy km/s  "
          f"(scale={scale:.4e})")
    return scale


def get_beam_from_fits(hdul):
    """Extract beam (BMAJ, BMIN in arcsec) from a CASA FITS file.

    CASA tclean stores beam info in a BEAMS binary table extension (HDU 1),
    with units of arcsec.  Older CASA versions used PRIMARY header BMAJ/BMIN
    keywords in degrees.  For per-channel beams, returns the median across
    channels.
    """
    from astropy.io import fits

    # Try PRIMARY header first (older CASA versions, units = degrees)
    header = hdul[0].header
    bmaj = header.get('BMAJ', None)
    bmin = header.get('BMIN', None)

    if bmaj is not None and bmin is not None and bmaj > 0 and bmin > 0:
        return float(bmaj) * 3600, float(bmin) * 3600  # deg -> arcsec

    # Try BEAMS binary table extension (units = arcsec)
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.name == 'BEAMS':
            colnames = hdu.columns.names if hdu.columns else []
            if 'BMAJ' in colnames and 'BMIN' in colnames:
                bmaj_vals = hdu.data['BMAJ']
                bmin_vals = hdu.data['BMIN']
                # Use median beam for per-channel beams
                bmaj = float(np.nanmedian(bmaj_vals))  # already arcsec
                bmin = float(np.nanmedian(bmin_vals))  # already arcsec
                return bmaj, bmin

    raise ValueError("No beam information found in PRIMARY header or BEAMS table")


def compute_beam_area_pix(hdul):
    """Beam area in pixels from CASA FITS file.

    Gaussian beam area: omega = pi * BMAJ * BMIN / (4 * ln(2))
    beam_area_pix = omega / pixscale^2

    Reads beam from BEAMS table extension when BMAJ/BMIN are missing from
    the PRIMARY header.
    """
    bmaj_arcsec, bmin_arcsec = get_beam_from_fits(hdul)
    header = hdul[0].header
    pixscale_arcsec = abs(header.get('CDELT1', 1)) * 3600

    beam_area_arcsec2 = (np.pi * bmaj_arcsec * bmin_arcsec
                         / (4.0 * np.log(2)))
    beam_area_pix = beam_area_arcsec2 / pixscale_arcsec**2
    return beam_area_pix, beam_area_arcsec2, bmaj_arcsec, bmin_arcsec


def build_velocity_axis(header):
    """Velocity axis (km/s) from CASA tclean FITS header.

    Uses f_center (middle channel) as reference, per CLAUDE.md Gotcha #3.
    """
    c_ms = 2.99792458e8

    spectral_axis_num = None
    for ax in range(1, header.get('NAXIS', 4) + 1):
        ctype = header.get(f'CTYPE{ax}', '').upper()
        if 'FREQ' in ctype:
            spectral_axis_num = ax
            break
    if spectral_axis_num is None:
        spectral_axis_num = 3

    crval = header.get(f'CRVAL{spectral_axis_num}', 0)
    cdelt = header.get(f'CDELT{spectral_axis_num}', 0)
    crpix = header.get(f'CRPIX{spectral_axis_num}', 1)
    nax = header.get(f'NAXIS{spectral_axis_num}', 0)
    freqs = crval + (np.arange(nax) - crpix + 1) * cdelt  # Hz

    f_center = freqs[nax // 2]
    vel = c_ms * (f_center - freqs) / f_center / 1e3  # km/s offsets
    dv = abs(np.median(np.diff(vel)))
    return vel, dv


def fit_gaussian_spectrum(vel, spec):
    """Fit Gaussian to spectrum, return line params and line mask."""
    def gaussian(x, amp, cen, sigma):
        return amp * np.exp(-0.5 * ((x - cen) / sigma)**2)

    # Replace NaN/Inf with 0 for fitting
    spec_clean = np.where(np.isfinite(spec), spec, 0.0)

    ipeak = np.argmax(np.abs(spec_clean))
    amp0, cen0, sigma0 = spec_clean[ipeak], vel[ipeak], 100.0
    try:
        popt, _ = curve_fit(gaussian, vel, spec_clean,
                            p0=[amp0, cen0, sigma0], maxfev=10000)
        amp, cen, sigma = popt
    except (RuntimeError, ValueError):
        amp, cen, sigma = amp0, cen0, 100.0

    fwhm = 2.355 * sigma
    line_mask = np.abs(vel - cen) < 3.0 * fwhm
    rms = np.std(spec_clean[~line_mask]) if np.sum(~line_mask) > 0 else 1e-10

    return {'amp': amp, 'cen': cen, 'sigma': sigma, 'fwhm': fwhm,
            'line_mask': line_mask, 'rms': rms}


def extract_spectra(data, hdul):
    """Extract spectra using Method A (raw) and Method B (corrected)."""
    beam_area_pix, beam_area_arcsec2, bmaj_arcsec, bmin_arcsec = compute_beam_area_pix(hdul)
    nchan = data.shape[0]

    spec_A = np.array([np.nansum(data[ch]) for ch in range(nchan)])
    spec_B = np.array([np.nansum(data[ch]) / beam_area_pix
                       for ch in range(nchan)])

    return {
        'method_A': spec_A,
        'method_B': spec_B,
        'beam_area_pix': beam_area_pix,
        'beam_area_arcsec2': beam_area_arcsec2,
        'bmaj_arcsec': bmaj_arcsec,
        'bmin_arcsec': bmin_arcsec,
    }


def run_tclean_dirty(ms_path, config, output_dir):
    """Run tclean with niter=0 for dirty image."""
    from casatasks import tclean, exportfits

    sid = config['source_id']
    dirty_name = str(output_dir / f'{sid}_dirty')

    # Cleanup
    for suffix in ['.image', '.image.fits', '.residual', '.model', '.psf',
                   '.mask', '.pb', '.sumwt', '.weight']:
        p = Path(dirty_name + suffix)
        if p.exists():
            (shutil.rmtree if p.is_dir() else os.unlink)(str(p))

    tclean(
        vis=str(ms_path), imagename=dirty_name,
        outframe='LSRK', veltype='radio',
        restfreq=f"{config['co_restfreq_ghz']}GHz",
        specmode='cube', nchan=-1,
        cell=f"{config['pixscale_arcsec']}arcsec",
        imsize=[config['npix_x'], config['npix_y']],
        weighting='natural', niter=0,
        threshold='0mJy', interactive=False,
    )

    dirty_fits = Path(dirty_name + '.image.fits')
    if not dirty_fits.exists():
        exportfits(imagename=str(Path(dirty_name + '.image')),
                   fitsimage=str(dirty_fits), overwrite=True)
    return dirty_fits


# ===================================================================
# Section 1: Generate mock data
# ===================================================================

def section1_generate_data():
    """Build cube (Layer A), normalize, simobserve (B), tclean (C)."""
    from galmockuv.build_cube import build_cube
    from galmockuv.simulate import simulate_alma
    from galmockuv.measure import measure_from_ms
    from galmockuv.io import load_metadata

    print('=' * 70)
    print(f'SECTION 1: Generate mock data (totaltime={TOTALTIME})')
    print('=' * 70)

    # Check existing data
    all_exist = True
    for ant in ANTENNA_CONFIGS:
        sid = f'{SOURCE_PREFIX}_{ant}_{TOTALTIME}'
        d = DEBUG_DIR / sid
        if not (d / 'intrinsic_cube.fits').exists() or \
           not (d / f'{sid}.ms').exists():
            all_exist = False
            break
    if all_exist:
        print(f'  All 3 configs exist. Skipping generation.')
        return

    # --- Layer A: build once ---
    first_sid = f'{SOURCE_PREFIX}_{ANTENNA_CONFIGS[0]}_{TOTALTIME}'
    cfg_a = BASE_CONFIG.copy()
    cfg_a['source_id'] = first_sid
    cfg_a['totaltime'] = TOTALTIME
    cfg_a['antennalist'] = ANTENNA_CONFIGS[0]
    out_a = DEBUG_DIR / first_sid
    out_a.mkdir(parents=True, exist_ok=True)

    print('\n--- Layer A: build intrinsic cube ---')
    cube_path, metadata, _ = build_cube(cfg_a, out_a)
    print(f'  Cube: {cube_path}')

    dv = float(cfg_a['channel_width_kms'])
    normalize_cube_flux(cube_path, dv)

    # Copy to other dirs
    for ant in ANTENNA_CONFIGS[1:]:
        sid = f'{SOURCE_PREFIX}_{ant}_{TOTALTIME}'
        out_d = DEBUG_DIR / sid
        out_d.mkdir(parents=True, exist_ok=True)
        for fname in ['intrinsic_cube.fits', 'metadata.yaml']:
            src = out_a / fname
            dst = out_d / fname
            if src.exists():
                shutil.copy2(str(src), str(dst))
        print(f'  Copied -> {sid}/')

    # --- Layer B + C for each config ---
    for ant in ANTENNA_CONFIGS:
        sid = f'{SOURCE_PREFIX}_{ant}_{TOTALTIME}'
        out_d = DEBUG_DIR / sid
        cfg = BASE_CONFIG.copy()
        cfg['source_id'] = sid
        cfg['totaltime'] = TOTALTIME
        cfg['antennalist'] = ant

        cube_file = out_d / 'intrinsic_cube.fits'
        ms_file = out_d / f'{sid}.ms'

        if not ms_file.exists():
            print(f'\n--- Layer B: simobserve {ant} ---')
            ms_path, _ = simulate_alma(cube_file, cfg, out_d)
        else:
            ms_path = ms_file
            print(f'\n--- Layer B: {ant} MS exists, skip ---')

        meas_file = out_d / 'measurements.json'
        if not meas_file.exists():
            meta = load_metadata(out_d)
            print(f'--- Layer C: tclean + measure {ant} ---')
            measure_from_ms(ms_path, cfg, out_d, meta)
        else:
            print(f'--- Layer C: {ant} measurements exist, skip ---')

    print('\n' + '=' * 70)
    print('SECTION 1 COMPLETE')
    print('=' * 70)


# ===================================================================
# Section 2: Verify intrinsic cube
# ===================================================================

def section2_verify_intrinsic():
    """Confirm the normalized intrinsic cube is 1.0 Jy km/s."""
    from astropy.io import fits

    print('\n' + '=' * 70)
    print('SECTION 2: Verify intrinsic cube')
    print('=' * 70)

    sid = f'{SOURCE_PREFIX}_{ANTENNA_CONFIGS[0]}_{TOTALTIME}'
    cube_path = DEBUG_DIR / sid / 'intrinsic_cube.fits'

    hdul = fits.open(str(cube_path))
    data = hdul[0].data.squeeze()
    header = hdul[0].header
    hdul.close()

    total_flux = np.nansum(data) * 10.0  # dv = 10 km/s

    print(f'  File:    {cube_path.name}')
    print(f'  Shape:   {data.shape}')
    print(f'  BUNIT:   {header.get("BUNIT", "NOT SET")}')
    print(f'  CDELT1:  {header.get("CDELT1", 0):.4e} deg/pix')
    print(f'  CDELT3:  {header.get("CDELT3", 0):.4e} Hz/ch')
    print(f'  RESTFRQ: {header.get("RESTFRQ", 0):.4e} Hz')
    print(f'  Peak:    {np.nanmax(data):.4e} Jy/pixel')
    print(f'  Flux:    {total_flux:.6f} Jy km/s  (target={TARGET_FLUX_JY_KMS})')
    print(f'  Match:   {"YES" if abs(total_flux - TARGET_FLUX_JY_KMS) < 0.01 else "NO"}')


# ===================================================================
# Section 3: Check tclean output headers
# ===================================================================

def section3_check_headers():
    """Print tclean image headers and beam areas."""
    from astropy.io import fits

    print('\n' + '=' * 70)
    print('SECTION 3: tclean output headers')
    print('=' * 70)

    results = {}
    for ant in ANTENNA_CONFIGS:
        sid = f'{SOURCE_PREFIX}_{ant}_{TOTALTIME}'
        fits_path = DEBUG_DIR / sid / f'{sid}.cube.image.fits'
        if not fits_path.exists():
            print(f'  WARNING: {fits_path} not found')
            continue

        hdul = fits.open(str(fits_path))
        header = hdul[0].header

        ba_pix, ba_arcsec2, bmaj_as, bmin_as = compute_beam_area_pix(hdul)
        hdul.close()

        print(f'\n  {ant}:')
        print(f'    BUNIT:      {header.get("BUNIT", "NOT SET")}')
        print(f'    BMAJ:       {bmaj_as:.4f} arcsec  (from BEAMS table)')
        print(f'    BMIN:       {bmin_as:.4f} arcsec  (from BEAMS table)')
        print(f'    CDELT1:     {header.get("CDELT1", 0)*3600:.4f} arcsec/pix')
        print(f'    Beam area:  {ba_arcsec2:.4f} arcsec^2  =  {ba_pix:.2f} pix')
        results[ant] = {'beam_area_pix': ba_pix}

    return results


# ===================================================================
# Section 4: Compare flux extraction methods
# ===================================================================

def section4_compare_methods():
    """Compare raw vs corrected flux extraction + dirty image."""
    from astropy.io import fits

    print('\n' + '=' * 70)
    print('SECTION 4: Flux extraction method comparison')
    print('=' * 70)

    all_results = {}

    for ant in ANTENNA_CONFIGS:
        sid = f'{SOURCE_PREFIX}_{ant}_{TOTALTIME}'
        out_d = DEBUG_DIR / sid
        fits_path = out_d / f'{sid}.cube.image.fits'
        if not fits_path.exists():
            print(f'  WARNING: {fits_path} not found, skip {ant}')
            continue

        print(f'\n  --- {ant}  CLEANED (niter=1000) ---')
        hdul = fits.open(str(fits_path))
        data = hdul[0].data.squeeze()
        header = hdul[0].header

        vel, dv = build_velocity_axis(header)
        sp = extract_spectra(data, hdul)
        hdul.close()
        fit_B = fit_gaussian_spectrum(vel, sp['method_B'])
        lm = fit_B['line_mask']

        int_A = np.sum(sp['method_A'][lm]) * dv
        int_B = np.sum(sp['method_B'][lm]) * dv

        print(f'    Beam area:        {sp["beam_area_pix"]:.2f} pix')
        print(f'    FWHM:             {fit_B["fwhm"]:.1f} km/s')
        print(f'    Line channels:    {np.sum(lm)}/{len(lm)}')
        print(f'    RMS (line-free):  {fit_B["rms"]*1e3:.4f} mJy')
        print(f'    Method A (raw):   {int_A:.6f}  [Jy/beam-sum * km/s]')
        print(f'    Method B (Jy):    {int_B:.6f}  Jy km/s')

        # --- Dirty image ---
        dirty_fits = out_d / f'{sid}_dirty.cube.image.fits'
        cfg_bc = BASE_CONFIG.copy()
        cfg_bc['source_id'] = sid

        if not dirty_fits.exists():
            print(f'    Generating dirty image...')
            ms_path = out_d / f'{sid}.ms'
            dirty_fits = run_tclean_dirty(ms_path, cfg_bc, out_d)

        hdul_d = fits.open(str(dirty_fits))
        data_d = hdul_d[0].data.squeeze()
        header_d = hdul_d[0].header

        vel_d, dv_d = build_velocity_axis(header_d)
        sp_d = extract_spectra(data_d, hdul_d)
        hdul_d.close()
        fit_dB = fit_gaussian_spectrum(vel_d, sp_d['method_B'])
        lm_d = fit_dB['line_mask']

        int_dA = np.sum(sp_d['method_A'][lm_d]) * dv_d
        int_dB = np.sum(sp_d['method_B'][lm_d]) * dv_d

        print(f'    Dirty A (raw):    {int_dA:.6f}')
        print(f'    Dirty B (Jy):     {int_dB:.6f}  Jy km/s')

        all_results[ant] = {
            'clean': {
                'method_A': int_A, 'method_B': int_B,
                'fwhm': fit_B['fwhm'], 'rms': fit_B['rms'],
                'beam_area_pix': sp['beam_area_pix'],
                'vel': vel, 'spec_A': sp['method_A'], 'spec_B': sp['method_B'],
                'line_mask': lm,
            },
            'dirty': {
                'method_A': int_dA, 'method_B': int_dB,
                'fwhm': fit_dB['fwhm'], 'rms': fit_dB['rms'],
                'beam_area_pix': sp_d['beam_area_pix'],
                'vel': vel_d, 'spec_A': sp_d['method_A'], 'spec_B': sp_d['method_B'],
                'line_mask': lm_d,
            },
        }

    # --- Summary table ---
    print(f'\n  {"=" * 72}')
    hdr = (f'  {"Config":<14} {"Meth A (raw)":>14} {"Meth B (Jy)":>14} '
           f'{"Dirty B (Jy)":>14} {"Target":>10}')
    print(hdr)
    print(f'  {"-" * 72}')
    for ant in ANTENNA_CONFIGS:
        r = all_results.get(ant)
        if r is None:
            continue
        print(f'  {ant:<14} {r["clean"]["method_A"]:>14.4f} '
              f'{r["clean"]["method_B"]:>14.4f} '
              f'{r["dirty"]["method_B"]:>14.4f} '
              f'{TARGET_FLUX_JY_KMS:>10.4f}')

    return all_results


# ===================================================================
# Section 5: MS-level flux (bypass tclean)
# ===================================================================

def section5_ms_level_flux():
    """Zero-spacing flux from visibilities — should be identical for all configs."""
    from casatasks import split, visstat
    from galmockuv.casa_utils import average_uvdata, suppress_casa_logs

    print('\n' + '=' * 70)
    print('SECTION 5: MS-level flux (bypass tclean)')
    print('=' * 70)

    uv_bin_width = 10.0
    uv_max = 500.0
    uv_bins = np.arange(0, uv_max + uv_bin_width, uv_bin_width)
    ms_results = {}

    for ant in ANTENNA_CONFIGS:
        sid = f'{SOURCE_PREFIX}_{ant}_{TOTALTIME}'
        ms_path = DEBUG_DIR / sid / f'{sid}.ms'
        if not ms_path.exists():
            print(f'  WARNING: {ms_path} not found')
            continue

        print(f'\n  --- {ant} ---')

        # Average all channels
        avg_ms = str(DEBUG_DIR / sid / 'debug_avg.ms')
        avg_ms_p = Path(avg_ms)
        if avg_ms_p.exists():
            shutil.rmtree(str(avg_ms_p))

        split(vis=str(ms_path), outputvis=avg_ms,
              datacolumn='corrected', timebin='1e8', combine='scan', width=9999)

        # Binned UV amplitudes
        try:
            uv_dist, amp, _, _ = average_uvdata(
                vis=avg_ms, datacolumn='data',
                uvbins=uv_bins, units='klambda',
                axis='real', skip_empty=True, verbose=False,
            )
        except Exception as e:
            print(f'    average_uvdata failed: {e}')
            continue

        zero_flux = amp[0] / 1000.0 if len(amp) > 0 else 0  # mJy -> Jy
        print(f'    Zero-spacing flux: {zero_flux:.4f} Jy')
        print(f'    UV bins with data: {len(uv_dist)}')

        # visstat median
        try:
            with suppress_casa_logs():
                vs = visstat(vis=avg_ms, axis='real', datacolumn='data')
            keys = list(vs.keys())
            if keys:
                med = vs[keys[0]].get('median', 0)
                print(f'    visstat median:     {med*1000:.2f} mJy')
        except Exception as e:
            print(f'    visstat failed: {e}')

        ms_results[ant] = {
            'zero_spacing_flux': zero_flux,
            'uv_dist': uv_dist,
            'amp': amp,
        }

        # Cleanup avg MS
        if avg_ms_p.exists():
            shutil.rmtree(str(avg_ms_p))

    # Compare
    print('\n  --- Zero-spacing flux comparison ---')
    fluxes = [ms_results[a]['zero_spacing_flux']
              for a in ANTENNA_CONFIGS if a in ms_results]
    if fluxes:
        print(f'    Values:  {[f"{f:.4f}" for f in fluxes]}')
        spread = max(fluxes) - min(fluxes)
        mean_f = np.mean(fluxes)
        print(f'    Spread:  {spread:.6f} Jy ({spread/mean_f*100:.2f}%)')

    return ms_results


# ===================================================================
# Section 6: Diagnostic plots
# ===================================================================

def section6_generate_plots(ext_results, ms_results):
    """Generate 3 diagnostic figures."""
    print('\n' + '=' * 70)
    print('SECTION 6: Diagnostic plots')
    print('=' * 70)

    colors = {'alma.cycle7.5': '#e41a1c', 'alma.cycle7.3': '#377eb8',
              'alma.cycle7.1': '#4daf4a'}
    labels = {'alma.cycle7.5': 'C7.5 (~0.3")',
              'alma.cycle7.3': 'C7.3 (~0.6")',
              'alma.cycle7.1': 'C7.1 (~1.2")'}

    # --- Plot 1: Spectra comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ant in ANTENNA_CONFIGS:
        r = ext_results.get(ant)
        if r is None:
            continue
        c = colors[ant]
        l = labels[ant]
        vel = r['clean']['vel']
        axes[0].plot(vel, r['clean']['spec_A'] * 1e3, '--', color=c,
                     lw=1, alpha=0.7, label=l)
        axes[1].plot(vel, r['clean']['spec_B'] * 1e3, '-', color=c,
                     lw=1.5, label=l)

    for ax in axes:
        ax.axhline(0, color='gray', ls=':', lw=0.5)
        ax.set_xlabel('Velocity (km/s)')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Flux (mJy, raw sum)')
    axes[0].set_title('Method A: Raw Jy/beam pixel sums\n'
                      '(WRONG: scales with beam size)')
    axes[1].set_ylabel('Flux (mJy)')
    axes[1].set_title('Method B: Proper Jy/beam -> Jy\n'
                      '(CORRECT: beam-independent)')

    fig.suptitle('Spectrum Comparison Across Antenna Configurations',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(DEBUG_DIR / 'spectra_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print('  Saved: spectra_comparison.png')

    # --- Plot 2: Bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(ANTENNA_CONFIGS)
    x = np.arange(n)
    w = 0.18

    vals_A = [ext_results[a]['clean']['method_A']
              for a in ANTENNA_CONFIGS if a in ext_results]
    vals_B = [ext_results[a]['clean']['method_B']
              for a in ANTENNA_CONFIGS if a in ext_results]
    vals_dB = [ext_results[a]['dirty']['method_B']
               for a in ANTENNA_CONFIGS if a in ext_results]
    vals_z = [ms_results[a]['zero_spacing_flux']
              for a in ANTENNA_CONFIGS if a in ms_results]

    ax.bar(x - 1.5*w, vals_A, w, color='#ff7f00', edgecolor='k', lw=0.5,
           label='Method A (raw Jy/beam sum)')
    ax.bar(x - 0.5*w, vals_B, w, color='#377eb8', edgecolor='k', lw=0.5,
           label='Method B (correct Jy)')
    ax.bar(x + 0.5*w, vals_dB, w, color='#984ea3', edgecolor='k', lw=0.5,
           label='Dirty image (Method B)')
    ax.bar(x + 1.5*w, vals_z, w, color='#e41a1c', edgecolor='k', lw=0.5,
           label='MS zero-spacing flux')
    ax.axhline(TARGET_FLUX_JY_KMS, color='k', ls='--', lw=2,
               label=f'Input ({TARGET_FLUX_JY_KMS} Jy km/s)')

    ax.set_xlabel('Antenna Configuration')
    ax.set_ylabel('Integrated Line Flux (Jy km/s)')
    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in ANTENNA_CONFIGS
                        if a in ext_results], fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Integrated Flux: Method Comparison\n'
                 'Separating missing flux from unit conversion artifacts')

    plt.tight_layout()
    fig.savefig(DEBUG_DIR / 'flux_comparison_bar.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print('  Saved: flux_comparison_bar.png')

    # --- Plot 3: UV amplitude ---
    if ms_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ant in ANTENNA_CONFIGS:
            if ant not in ms_results:
                continue
            ax.plot(ms_results[ant]['uv_dist'],
                    ms_results[ant]['amp'] * 1e3, 'o-',
                    color=colors[ant], ms=3, lw=1, label=labels[ant])
        ax.set_xlabel('UV distance (k$\\lambda$)')
        ax.set_ylabel('Visibility Amplitude (mJy)')
        ax.set_title('UV Amplitude: Low-UV Region\n'
                     '(zero-baseline = total source flux)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(DEBUG_DIR / 'uv_amplitude_comparison.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print('  Saved: uv_amplitude_comparison.png')


# ===================================================================
# Main
# ===================================================================

def main():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    print('debug_flux_consistency.py')
    print(f'  Working dir:  {DEBUG_DIR}')
    print(f'  Configs:      {ANTENNA_CONFIGS}')
    print(f'  Totaltime:    {TOTALTIME}')
    print(f'  Python:       {sys.executable}')

    for mod in ['dysmalpy', 'casatasks', 'casatools']:
        try:
            __import__(mod)
            print(f'  {mod}:        OK')
        except ImportError:
            print(f'  {mod}:        NOT AVAILABLE')

    section1_generate_data()
    section2_verify_intrinsic()
    section3_check_headers()
    ext = section4_compare_methods()
    ms = section5_ms_level_flux()
    section6_generate_plots(ext, ms)

    print('\n' + '=' * 70)
    print('ALL SECTIONS COMPLETE')
    print(f'Output: {DEBUG_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()

"""CASA utility functions for UV data analysis.

Vendored from works/casa_utils.py for self-contained distribution.
"""

import numpy as np
import math
import re
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from casatasks import visstat, ft, uvmodelfit, split, concat
from casaplotms import plotms
from casatools import ms
from PIL import Image
import os
from astropy import units as u
from astropy.units import Quantity
from io import StringIO
from contextlib import contextmanager


@contextmanager
def suppress_casa_logs():
    """Context manager to suppress CASA log messages (SEVERE/WARN) on stderr.

    CASA's C++ log4cxx writes directly to fd 2, bypassing Python's sys.stderr,
    so contextlib.redirect_stderr is insufficient. Instead, we redirect fd 2 to
    a pipe and drain it after restoring, so buffered messages are discarded.
    """
    saved_stderr = os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(w_fd)
        # Drain the pipe to discard any buffered CASA log messages
        while True:
            data = os.read(r_fd, 4096)
            if not data:
                break
        os.close(r_fd)


def helper_read_visstat_output(visstat_result):
    '''
    Use the first available key (different UV ranges may
    return different DATA_DESC_ID keys)
    '''
    available_keys = list(visstat_result.keys())
    if not available_keys:
        raise KeyError("No keys found in visstat result")
    
    first_key = available_keys[0]
    if 'median' not in visstat_result[first_key]:
        raise KeyError(f"'median' not found in visstat result for key '{first_key}'")
    
    # Check for NaN or None values
    median_val = visstat_result[first_key]["median"]
    q1_val = visstat_result[first_key]["firstquartile"]
    q3_val = visstat_result[first_key]["thirdquartile"]
    npts_val = visstat_result[first_key]["npts"]

    results = {
        'median': median_val,
        'q1': q1_val,
        'q3': q3_val,
        'npts': npts_val
    }

    return results


def helper_robustness_checks(vis, axis, uvbins, units, datacolumn, verbose=False):
    '''
    Auto-detect tmp_key
    '''
    # Try different UV ranges to find one with data
    detected = False
    for i in range(len(uvbins) - 1):
        test_uvrange = f"{uvbins[i]}~{uvbins[i+1]} {units}"

        try:
            with suppress_casa_logs():
                test_result = visstat(vis=vis, axis=axis, uvrange=test_uvrange,
                                      datacolumn=datacolumn)

            # Get the first key from the result
            keys = list(test_result.keys())
            if not keys:
                continue

            tmp_key = keys[0]
            if verbose:
                print(f"  Detected tmp_key: '{tmp_key}'")
                print(f"  Available keys: {keys}")
            detected = True
            break

        except Exception:
            # This range didn't work, try the next one
            continue

    if not detected:
        raise RuntimeError(f"Failed to auto-detect tmp_key: could not find any UV range with data")


def average_uvdata(vis, datacolumn, uvbins, units="klambda", axis="real",
                   verbose=False, skip_empty=True):
    """
    Get the averaged uv data as a function of uv distance.
    Use visstat() in casa.

    Parameters
    ----------
    vis : string
        The name of the ms.
    datacolumn : string
        The data column to be used for visstat.
    uvbins : list
        The boundaries of the bins of the uv distance.
    units : string (default: "klambda")
        The unit of the uv distance.
    axis : string (default: "real")
        The axis of the data.
    tmp_key : string or None (default: None)
        The key of the output tmp dict. If None, will auto-detect.
    verbose : bool (default: False)
        Print auxiliary information if True.
    skip_empty : bool (default: True)
        If True, skip bins with no data. If False, raise an error.

    Returns
    -------
    uvpoints : array
        The uv distance, units following the input units.
    avg_amps : array
        The averaged amplitude of the visiblity, units: mJy.
    error_amps_lower : array
        The lower (negative) uncertainty of avg_amps, units: mJy.
        Based on (median - Q1) / sqrt(n).
    error_amps_upper : array
        The upper (positive) uncertainty of avg_amps, units: mJy.
        Based on (Q3 - median) / sqrt(n).

    Raises
    ------
    ValueError
        If no valid data is found in the MS or if visstat fails.
    RuntimeError
        If unable to auto-detect tmp_key.
    """
    if verbose:
        print(f"average_uvdata: Processing {vis}")
        print(f"  datacolumn: {datacolumn}")
        print(f"  axis: {axis}")
        print(f"  units: {units}")
        print(f"  uvbins: {uvbins[0]:.2f} - {uvbins[-1]:.2f} ({len(uvbins)-1} bins)")

    # Ensure CASA environment is properly set up for visstat
    helper_robustness_checks(
        vis, axis, uvbins, units, datacolumn, 
        verbose)  

    # Calculate uvpoints for all bins first
    all_uvpoints = uvbins[:-1] + np.diff(uvbins) / 2  # get array of uvmidpoints over which avg taken

    avg_amps = []  # define an empty list in which to put the averaged amplitudes
    q1_vals = []  # first quartile values
    q3_vals = []  # third quartile values
    numpoints = []
    uvpoints = []  # track which uvpoints actually have data

    if verbose:
        print(f"  Processing {len(all_uvpoints)} UV bins...")

    # Iterate over the uvrange & build up the binned data values
    skipped_bins = []
    for loop in range(len(all_uvpoints)):
        uvrange = "{0}~{1} {2}".format(uvbins[loop], uvbins[loop+1], units)

        try:
            with suppress_casa_logs():
                tmp = visstat(vis=vis, 
                              axis=axis, 
                              uvrange=uvrange, 
                              datacolumn=datacolumn,
                              timeaverage=False)

            res = helper_read_visstat_output(tmp)
            mean_val = res['median']
            q1_val = res['q1']
            q3_val = res['q3']
            npts_val = res['npts']  # Assuming npts is the same for real and imag

            if mean_val is None or np.isnan(mean_val):
                if verbose:
                    print(f"  WARNING: Bin {loop+1}/{len(all_uvpoints)} ({uvrange}) has NaN mean")
                avg_amps.append(0.0)
                q1_vals.append(0.0)
                q3_vals.append(0.0)
                numpoints.append(0)
                uvpoints.append(all_uvpoints[loop])
            elif npts_val == 0:
                if verbose:
                    print(f"  WARNING: Bin {loop+1}/{len(all_uvpoints)} ({uvrange}) has no data points")
                avg_amps.append(0.0)
                q1_vals.append(0.0)
                q3_vals.append(0.0)
                numpoints.append(0)
                uvpoints.append(all_uvpoints[loop])
            else:
                avg_amps.append(mean_val)
                q1_vals.append(q1_val)
                q3_vals.append(q3_val)
                numpoints.append(npts_val)
                uvpoints.append(all_uvpoints[loop])

                if verbose:
                    print(f"  Bin {loop+1}/{len(all_uvpoints)} ({uvrange}): "
                          f"npts={npts_val}, mean={mean_val:.6f}")

        except RuntimeError as e:
            # Check if this is a "no data" error
            if "zero rows" in str(e) or "NullSelection" in str(e):
                if skip_empty:
                    if verbose:
                        print(f"  SKIP: Bin {loop+1}/{len(all_uvpoints)} ({uvrange}) - no data")
                    skipped_bins.append(loop+1)
                    continue
                else:
                    if verbose:
                        print(f"  ERROR: Bin {loop+1}/{len(all_uvpoints)} ({uvrange}) - no data")
                        print(f"    Hint: Set skip_empty=True to skip bins with no data")
                    raise ValueError(f"UV bin {loop+1} ({uvrange}) has no data. "
                                   f"Your data may not cover this UV range. "
                                   f"Try reducing uvmax or setting skip_empty=True.")
            else:
                # Other RuntimeError
                if verbose:
                    print(f"  ERROR in bin {loop+1}/{len(all_uvpoints)} ({uvrange}): {e}")
                raise ValueError(f"Failed to process UV bin {loop+1} ({uvrange}): {e}")

        except Exception as e:
            if verbose:
                print(f"  ERROR in bin {loop+1}/{len(all_uvpoints)} ({uvrange}): {e}")
            raise ValueError(f"Failed to process UV bin {loop+1} ({uvrange}): {e}")

    # Check if we got any data at all
    if len(avg_amps) == 0:
        raise ValueError(f"No valid data found in {vis}. Check your uvrange and datacolumn.")

    # Convert to numpy arrays and scale to mJy
    avg_amps = np.array(avg_amps) * 1000.       # units: mJy
    q1_amps = np.array(q1_vals) * 1000.          # units: mJy
    q3_amps = np.array(q3_vals) * 1000.          # units: mJy
    numpoints_arr = np.array(numpoints, dtype=float)
    uvpoints = np.array(uvpoints)

    # Calculate asymmetric errors from IQR, divided by sqrt(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_amps_lower = (avg_amps - q1_amps) / np.sqrt(numpoints_arr)
        error_amps_upper = (q3_amps - avg_amps) / np.sqrt(numpoints_arr)
    error_amps_lower = np.nan_to_num(error_amps_lower)
    error_amps_upper = np.nan_to_num(error_amps_upper)

    # Warn about skipped or zero-point bins
    if skipped_bins and verbose:
        print(f"  INFO: Skipped {len(skipped_bins)} bins with no data: {skipped_bins}")

    zero_bins = np.sum(np.array(numpoints) == 0)
    if zero_bins > 0 and verbose:
        print(f"  WARNING: {zero_bins} bins had zero data points")

    if verbose:
        print(f"  Complete: {len(avg_amps)} data points")
        print(f"  UV range: {uvpoints[0]:.2f} - {uvpoints[-1]:.2f} {units}")
        print(f"  Amplitude range: {np.min(avg_amps):.3f} - {np.max(avg_amps):.3f} mJy")

    return uvpoints, avg_amps, error_amps_lower, error_amps_upper


def create_concat_ms(ms_list, concat_ms, plotfile=None):
    '''
    Concatenate multiple Measurement Sets (MS) into a single MS.

    Parameters:
    ms_list (list of str): List of paths to the input MS files.
    concat_ms (str): Path to the output concatenated MS file.
    '''
    os.system('rm -rf ' + concat_ms)

    if len(ms_list) == 0:
        print("No MS files found to concatenate.")
        return

    print ("Concatenating origin line MS into ", concat_ms)
    concat(vis=ms_list, concatvis=concat_ms)

    if plotfile:
        plot_ms(concat_ms, xaxis="freq", yaxis="amp", separate_spw=True,
                plotfile=plotfile, verbose=True, highres=True)


def get_spw_range(spw):
    '''
    Get the frequency range of a spectral window in GHz.
    '''
    s1 = spw['Chan1Freq'] / 1e9
    s2 = (spw['Chan1Freq'] + spw['TotalWidth']) / 1e9
    return s1, s2


def split_line_core(concat_ms, avg_data, line_nu):
    '''
    Split the line core data from the concatenated measurement set.

    Parameters:
    - concat_ms: str, path to the concatenated measurement set
    - avg_data: str, path to the output averaged measurement set
    - line_nu: tuple, (nu0, nu1) in GHz, the frequency range of the line core
    '''
    nu0, nu1 = line_nu

    ms_file = ms()
    ms_file.open(concat_ms)
    
    # Check the spectral window information
    spw_list = []
    spw_info = ms_file.getspectralwindowinfo()
    for spw_id, spw in spw_info.items():
        s1, s2 = get_spw_range(spw)
        if s1 > nu1 or s2 < nu0:
            #print(f"  SKIP SPW {spw_id}: {s1:.3f}~{s2:.3f} GHz is outside the line range {nu0:.3f}~{nu1:.3f} GHz")
            pass
        else:
            use0 = max(s1, nu0)
            use1 = min(s2, nu1)
            #print(f"  USE SPW {spw_id}: {s1:.3f}~{s2:.3f} GHz, use {use0:.3f}~{use1:.3f} GHz")
            spw_list.append(f"{spw_id}:{use0:.3f}~{use1:.3f}GHz")
    
    spw = ', '.join(spw_list)
    print(concat_ms)
    print(f"{len(spw_list)} SPWs used: {spw}")

    #-> Get the line core data; test if frequency works!
    os.system("rm -rf {0}".format(avg_data))
    split(vis=concat_ms, outputvis=avg_data,
          datacolumn="data", timebin="1e8", combine="scan", width=5,
          spw=spw) #

def plot_ms(vis, xaxis="freq", yaxis="amp", separate_spw=False,
            plotfile=None, spw="", avgtime="1e8", avgscan=True,
            coloraxis="field", xselfscale=True, highres=True,
            width=600, height=350, overwrite=True, verbose=False, **kwargs):
    """
    Wrapper function for plotms with common defaults for ALMA data visualization.

    Parameters
    ----------
    vis : string
        The name of the measurement set.
    xaxis : string (default: "freq")
        The x-axis of the plot. Common options: "freq", "channel", "uvdist".
    yaxis : string (default: "amp")
        The y-axis of the plot. Common options: "amp", "phase", "real", "imag".
    separate_spw : bool (default: False)
        If True, plot each spectral window in separate files and combine to PDF.
        If False, merge all spectral windows on one plot.
    plotfile : string or None (default: None)
        Output file path for the plot. If None and separate_spw is False,
        auto-generates filename based on vis and axes.
        If separate_spw is True, this becomes the PDF filename.
    spw : string (default: "")
        Spectral window selection. Ignored if separate_spw=True.
    avgtime : string (default: "1e8")
        Time averaging interval.
    avgscan : bool (default: True)
        Average over scans.
    coloraxis : string (default: "field")
        Axis to use for coloring data.
    xselfscale : bool (default: True)
        Scale x-axis independently for each panel.
    highres : bool (default: True)
        High resolution output.
    width : int (default: 600)
        Plot width in pixels.
    height : int (default: 350)
        Plot height in pixels.
    overwrite : bool (default: True)
        Overwrite existing plot file.
    verbose : bool (default: False)
        Print progress messages if True.
    **kwargs
        Additional keyword arguments passed to plotms.

    Returns
    -------
    string or None
        Returns the PDF filename if separate_spw=True, otherwise returns the
        plotfile path. Returns None if no file was saved.
    """
    from casatools import ms

    # Handle separate SPWs - generate individual files and combine to PDF
    if separate_spw:
        # Get number of SPWs
        mstool = ms()
        mstool.open(vis)
        spw_info = mstool.getspectralwindowinfo()
        nspw = len(spw_info)
        mstool.close()

        if verbose:
            print(f"Found {nspw} spectral windows")

        # Auto-generate PDF filename if not provided
        vis_name = Path(vis).stem
        if plotfile is None:
            pdf_file = f"{vis_name}_{yaxis}_{xaxis}_all_spw.pdf"
        else:
            pdf_file = plotfile
            if not pdf_file.endswith('.pdf'):
                pdf_file = str(Path(pdf_file).with_suffix('.pdf'))

        # Create temporary directory for individual plots
        temp_dir = Path(vis).parent / f"{vis_name}_temp_plots"
        temp_dir.mkdir(exist_ok=True)

        # Generate individual plot for each SPW
        plot_files = []
        for i in range(nspw):
            spw_plotfile = str(temp_dir / f"{vis_name}_spw{i}_{yaxis}_{xaxis}.jpg")
            plot_files.append(spw_plotfile)

            if verbose:
                print(f"Plotting SPW {i}/{nspw-1} -> {spw_plotfile}")

            plotms(vis=vis, spw=str(i), xaxis=xaxis, yaxis=yaxis,
                   avgtime=avgtime, avgscan=avgscan, coloraxis=coloraxis,
                   xselfscale=xselfscale, showgui=False,
                   plotfile=spw_plotfile, highres=highres,
                   width=width, height=height, overwrite=overwrite, **kwargs)

        # Combine images into multi-page PDF
        if verbose:
            print(f"Combining {len(plot_files)} plots into {pdf_file}")

        combine_images_to_pdf(plot_files, pdf_file)

        # Clean up temporary files
        if verbose:
            print(f"Cleaning up temporary files in {temp_dir}")

        for f in plot_files:
            if os.path.exists(f):
                os.remove(f)
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

        return pdf_file

    else:
        # Single plot with all SPWs merged
        # Auto-generate plotfile if not provided
        if plotfile is None:
            vis_name = Path(vis).stem
            plotfile = f"{vis_name}_{yaxis}_{xaxis}.jpg"

        # Always use showgui=False for script usage
        showgui = False

        # Build the plotms call
        plotms(vis=vis, spw=spw, xaxis=xaxis, yaxis=yaxis,
               avgtime=avgtime, avgscan=avgscan, coloraxis=coloraxis,
               xselfscale=xselfscale, showgui=showgui,
               plotfile=plotfile, highres=highres, width=width, height=height,
               overwrite=overwrite, **kwargs)

        return plotfile


def combine_images_to_pdf(image_paths, output_pdf):
    """
    Combine multiple images into a multi-page PDF file.

    Parameters
    ----------
    image_paths : list of string
        List of paths to image files to combine.
    output_pdf : string
        Path to the output PDF file.

    Returns
    -------
    None
        PDF file is created at output_pdf path.
    """
    # Open all images and convert to RGB if necessary
    images = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue
        img = Image.open(img_path)
        # Convert to RGB for PDF compatibility (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)

    if not images:
        raise ValueError("No valid image files found to combine")

    # Save first image as PDF, then append the rest
    images[0].save(
        output_pdf,
        save_all=True,
        append_images=images[1:],
        resolution=150.0,
        quality=95
    )


def check_model_column(vis, expected_flux_jy=None, verbose=True):
    """
    Diagnose the MODEL column in a measurement set after uvmodelfit.

    Checks if the MODEL column exists, contains data, and matches expected values.

    Parameters
    ----------
    vis : string
        The name of the measurement set.
    expected_flux_jy : float or None (default: None)
        Expected flux density in Jy from uvmodelfit. If provided, compares
        with the actual model values.
    verbose : bool (default: True)
        Print diagnostic information.

    Returns
    -------
    dict
        Dictionary with diagnostic results:
        - 'model_exists': bool, whether MODEL column exists
        - 'has_data': bool, whether MODEL column has non-zero data
        - 'sample_values': array of sample model values in Jy
        - 'mean_model': float, mean of model values in Jy
        - 'std_model': float, std of model values in Jy
        - 'matches_expected': bool or None, whether model matches expected flux
    """
    result = {
        'model_exists': False,
        'has_data': False,
        'sample_values': None,
        'mean_model': None,
        'std_model': None,
        'matches_expected': None
    }

    if verbose:
        print(f"\n=== Diagnosing MODEL column in {vis} ===")

    # Open MS and check for MODEL column
    mstool = ms()
    mstool.open(vis)

    try:
        # Get column names
        colnames = mstool.getcolnames()

        if verbose:
            print(f"Available columns: {colnames}")

        # Check if MODEL column exists
        if 'MODEL_DATA' not in colnames and 'model' not in colnames:
            if verbose:
                print("ERROR: MODEL column not found in MS!")
                print("  uvmodelfit may have failed to write the model.")
                print("  Check uvmodelfit output for errors.")
            mstool.close()
            return result

        result['model_exists'] = True
        model_col = 'MODEL_DATA' if 'MODEL_DATA' in colnames else 'model'

        if verbose:
            print(f"✓ MODEL column found: '{model_col}'")

        # Get some sample data from MODEL column
        # Use a simple selection to get a few rows
        mstool.initselect(dds=0)  # Select first DATA_DESC_ID

        # Try to get some data
        try:
            data = mstool.getdata(['model'])
            model_data = data['model']

            if model_data is None or len(model_data) == 0:
                if verbose:
                    print("ERROR: MODEL column exists but is empty!")
                mstool.close()
                return result

            # Get real part of visibilities (complex data)
            if len(model_data.shape) == 3:
                # Shape is typically [nchan, npol, nrow] or similar
                model_real = np.real(model_data).flatten()
            else:
                model_real = np.real(model_data.flatten())

            # Remove NaN values
            model_real = model_real[~np.isnan(model_real)]

            if len(model_real) == 0:
                if verbose:
                    print("ERROR: MODEL column exists but contains only NaN!")
                mstool.close()
                return result

            result['has_data'] = True

            # Get statistics
            result['mean_model'] = np.mean(model_real)
            result['std_model'] = np.std(model_real)
            result['sample_values'] = model_real[:min(10, len(model_real))]

            if verbose:
                print(f"✓ MODEL column contains data")
                print(f"  Sample values (first 10, Jy): {result['sample_values']}")
                print(f"  Mean: {result['mean_model']:.6f} Jy")
                print(f"  Std:  {result['std_model']:.6f} Jy")
                print(f"  Min:  {np.min(model_real):.6f} Jy")
                print(f"  Max:  {np.max(model_real):.6f} Jy")

            # Check if model matches expected flux
            if expected_flux_jy is not None:
                # Check if mean is close to expected (within 10%)
                if abs(result['mean_model'] - expected_flux_jy) / expected_flux_jy < 0.1:
                    result['matches_expected'] = True
                    if verbose:
                        print(f"✓ Model matches expected flux of {expected_flux_jy:.6f} Jy")
                else:
                    result['matches_expected'] = False
                    if verbose:
                        print(f"✗ Model does NOT match expected flux!")
                        print(f"  Expected: {expected_flux_jy:.6f} Jy")
                        print(f"  Got:      {result['mean_model']:.6f} Jy")
                        print(f"  Ratio:    {result['mean_model']/expected_flux_jy:.2f}x")
                        print("\n  Possible causes:")
                        print("  1. uvmodelfit wrote to wrong column or spw")
                        print("  2. MS has multiple DATA_DESC_IDs and model is in wrong one")
                        print("  3. Model was not successfully written by uvmodelfit")
                        print("  4. Expected flux is from different spw/selection")

            # Additional diagnostic: check if model is constant
            if result['std_model'] < 1e-6:
                if verbose:
                    print("\n⚠ WARNING: MODEL column appears to be constant!")
                    print("  This suggests uvmodelfit did not properly write the model.")
                    print("  The model should vary with UV distance for a Gaussian source.")

        except Exception as e:
            if verbose:
                print(f"ERROR reading MODEL column: {e}")
            mstool.close()
            return result

    except Exception as e:
        if verbose:
            print(f"ERROR accessing MS: {e}")
        mstool.close()
        return result

    finally:
        try:
            mstool.close()
        except:
            pass

    return result


def get_uv_range(vis, datacolumn="data", verbose=False):
    """
    Get the UV distance range (min and max) from a measurement set.

    Uses visstat to probe different UV ranges and find where data exists.

    Parameters
    ----------
    vis : string
        The name of the measurement set.
    datacolumn : string (default: "data")
        The data column to use for statistics.
    verbose : bool (default: False)
        Print information if True.

    Returns
    -------
    dict
        Dictionary with UV distance information:
        - 'uvdist_min': minimum UV distance in klambda
        - 'uvdist_max': maximum UV distance in klambda
        - 'nrows': number of rows
        - 'spw_info': spectral window information
    """
    from casatools import ms

    if verbose:
        print(f"\n=== Getting UV distance range from {vis} ===")

    result = {
        'uvdist_min': None,
        'uvdist_max': None,
        'nrows': 0,
        'spw_info': None
    }

    mstool = ms()
    try:
        mstool.open(vis)

        # Get spectral window info
        spw_info = mstool.getspectralwindowinfo()
        result['spw_info'] = spw_info

        if verbose:
            nspw = len(spw_info)
            print(f"Number of SPWs: {nspw}")

        # Get number of rows
        n_rows = mstool.nrow()
        result['nrows'] = n_rows

        if verbose:
            print(f"Number of rows: {n_rows}")

        mstool.close()

    except Exception as e:
        if verbose:
            print(f"ERROR getting MS info: {e}")
        try:
            mstool.close()
        except:
            pass

    # Use visstat with narrow ranges to find actual UV coverage
    if verbose:
        print("Searching for UV distance range using visstat...")

    found_min = None
    found_max = None

    with suppress_casa_logs():
        # For minimum: test increasingly small ranges
        for test_uv in [10, 5, 2, 1, 0.5, 0.2, 0.1]:
            test_range = f"0~{test_uv} klambda"
            try:
                test_result = visstat(vis=vis, axis="real", uvrange=test_range,
                                    datacolumn=datacolumn)
                if test_result:
                    key = list(test_result.keys())[0]
                    npts = test_result[key]['npts']
                    if npts > 0:
                        found_min = 0.1  # We know there's data below test_uv
                        break
            except:
                continue

        # For maximum: search upward from where we know data exists
        # Start with a reasonable range and increase
        test_uv = 50
        while test_uv < 100000:
            # Test a narrow range around test_uv
            test_range = f"{test_uv*0.8}~{test_uv*1.2} klambda"
            try:
                test_result = visstat(vis=vis, axis="real", uvrange=test_range,
                                    datacolumn=datacolumn)
                if test_result:
                    key = list(test_result.keys())[0]
                    npts = test_result[key]['npts']
                    if npts > 0:
                        found_max = test_uv
                        test_uv *= 1.5  # Search higher
                    else:
                        # No data in this range, we're past the maximum
                        # Now narrow down
                        break
            except:
                # No data, we're past the maximum
                break

        # If we found a general max but not the exact one, narrow it down
        if found_max and found_max > 100:
            # Binary search for the exact max
            lower = found_max / 1.5
            upper = found_max * 1.2

            for _ in range(20):
                test_uv = (lower + upper) / 2
                test_range = f"{test_uv*0.95}~{test_uv*1.05} klambda"

                try:
                    test_result = visstat(vis=vis, axis="real", uvrange=test_range,
                                        datacolumn=datacolumn)
                    if test_result:
                        key = list(test_result.keys())[0]
                        npts = test_result[key]['npts']
                        if npts > 0:
                            found_max = test_uv
                            lower = test_uv
                        else:
                            upper = test_uv
                except:
                    upper = test_uv

                if upper - lower < 1:
                    break

    result['uvdist_min'] = found_min if found_min else 1.0
    result['uvdist_max'] = found_max if found_max else 150.0

    if verbose:
        print(f"UV distance range detected:")
        print(f"  Min: {result['uvdist_min']:.2f} kλ")
        print(f"  Max: {result['uvdist_max']:.2f} kλ")

    return result


def suggest_uv_bins(uvmin, uvmax, n_bins=10, binning_type='log',
                    verbose=False):
    """
    Suggest optimal UV distance bins for visibility fitting.

    Parameters
    ----------
    uvmin : float
        Minimum UV distance in klambda.
    uvmax : float
        Maximum UV distance in klambda.
    n_bins : int (default: 10)
        Number of bins to create.
    binning_type : string (default: 'log')
        Type of binning: 'log' or 'linear'.
        Logarithmic is usually better for UV data.
    verbose : bool (default: False)
        Print information if True.

    Returns
    -------
    array
        UV bin edges (n_bins + 1 values).
    """
    if verbose:
        print(f"\n=== Suggesting UV bins ===")
        print(f"  UV range: {uvmin:.2f} - {uvmax:.2f} kλ")
        print(f"  Number of bins: {n_bins}")
        print(f"  Binning type: {binning_type}")

    if binning_type == 'log':
        # Logarithmic spacing - better for UV data
        bins = np.logspace(np.log10(uvmin), np.log10(uvmax), n_bins + 1)
    else:
        # Linear spacing
        bins = np.linspace(uvmin, uvmax, n_bins + 1)

    if verbose:
        print(f"  Bin edges:")
        for i, b in enumerate(bins):
            print(f"    {i}: {b:.2f} kλ")

    return bins


def parse_uvmodelfit_output(output_text, comptype="G"):
    """
    Parse uvmodelfit terminal output to extract parameter uncertainties and statistics.

    This is a workaround for the CASA bug where componentlist does not contain
    parameter uncertainties. uvmodelfit calculates them and displays in output,
    but doesn't write them to the componentlist file.

    Also extracts fit statistics (DOF, reduced chi2) from the output.

    Parameters
    ----------
    output_text : str
        The terminal output from uvmodelfit containing the results
    comptype : str (default: "G")
        Component type to determine number of parameters for BIC calculation

    Returns
    -------
    dict
        Parsed uncertainties and statistics with keys:
        - flux: {'value': float, 'error': float}
        - offset: {'x': {'value': float, 'error': float},
                   'y': {'value': float, 'error': float}}
        - size: {'bmaj': {'value': float, 'error': float},
                 'axrat': {'value': float, 'error': float},
                 'pa': {'value': float, 'error': float}}
        - statistics: {'dof': int, 'reduced_chi2': float, 'npts': int, 'n_params': int}
          Note: n_params is the actual number of fitted parameters, accounting for
          any fixed parameters specified via varpar.
    """
    result = {
        'flux': {'value': None, 'error': None},
        'offset': {
            'x': {'value': None, 'error': None},
            'y': {'value': None, 'error': None}
        },
        'size': {
            'bmaj': {'value': None, 'error': None},
            'axrat': {'value': None, 'error': None},
            'pa': {'value': None, 'error': None}
        },
        'statistics': {
            'dof': None,
            'reduced_chi2': None,
            'npts': None,
            'n_params': None  # Number of fitted parameters (accounts for varpar)
        }
    }

    # Pattern for uvmodelfit final output:
    # I = 0.00086484 +/- 5.55629e-05
    # x = -0.0730515 +/- 0.059827 arcsec
    # y = -0.0426891 +/- 0.058128 arcsec
    # a = 1.10391 +/- 0.285941 arcsec
    # r = 6.92496e-17 +/- 2.93741e+15
    # p = 48.1678 +/- 19.8283 deg

    patterns = {
        'flux': r"I\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        'x': r"x\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*arcsec",
        'y': r"y\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*arcsec",
        'bmaj': r"a\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*arcsec",
        'axrat': r"r\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
        'pa': r"p\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*deg"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if match:
            value = float(match.group(1))
            error = float(match.group(2))

            if key == 'flux':
                result['flux']['value'] = value
                result['flux']['error'] = error
            elif key == 'x':
                result['offset']['x']['value'] = value
                result['offset']['x']['error'] = error
            elif key == 'y':
                result['offset']['y']['value'] = value
                result['offset']['y']['error'] = error
            elif key == 'bmaj':
                result['size']['bmaj']['value'] = value
                result['size']['bmaj']['error'] = error
            elif key == 'axrat':
                result['size']['axrat']['value'] = value
                result['size']['axrat']['error'] = error
            elif key == 'pa':
                result['size']['pa']['value'] = value
                result['size']['pa']['error'] = error

    # Extract statistics from uvmodelfit output
    # Pattern: "There are 35320 - 6 = 35314 degrees of freedom."
    dof_pattern = r"There are (\d+) - (\d+) = (\d+) degrees of freedom"
    dof_match = re.search(dof_pattern, output_text)
    if dof_match:
        npts = int(dof_match.group(1))  # Total number of data points
        n_params = int(dof_match.group(2))  # Number of fitted parameters (accounts for varpar)
        dof = int(dof_match.group(3))  # Degrees of freedom
        result['statistics']['npts'] = npts
        result['statistics']['n_params'] = n_params
        result['statistics']['dof'] = dof

    # Extract reduced chi2 from the final iteration
    # Pattern: "iter=N: reduced chi2=X.XXX:" (last iteration)
    # Find all iterations and get the last one
    chi2_pattern = r"iter=\d+:\s+reduced chi2=([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?):"
    chi2_matches = re.findall(chi2_pattern, output_text)
    if chi2_matches:
        # Get the last one (final iteration)
        reduced_chi2 = float(chi2_matches[-1])
        result['statistics']['reduced_chi2'] = reduced_chi2

    return result


def calculate_chi2_from_ms(vis, datacolumn="data", modelcolumn="model_data",
                           axis="real", sample_size=100000, verbose=False):
    """
    Calculate chi-squared from MS data using sampling.

    χ² = Σ[(data - model)² × weight]

    Parameters
    ----------
    vis : string
        Measurement set name.
    datacolumn : string (default: "data")
        Data column to use (lowercase).
    modelcolumn : string (default: "model_data")
        Model column to use (lowercase, MS column is MODEL_DATA).
    axis : string (default: "real")
        Axis to use ('real' or 'imag' or 'amp').
    sample_size : int (default: 100000)
        Maximum number of samples to use.
    verbose : bool (default: False)
        Print progress information.

    Returns
    -------
    dict
        Dictionary with:
        - 'chi2': chi-squared value
        - 'npts': total number of data points
        - 'sum_squared_residuals': sum of (data - model)²
    """
    from casatools import ms

    if verbose:
        print(f"\n=== Calculating χ² from {vis} ===")

    mstool = ms()
    result = {
        'chi2': None,
        'npts': 0,
        'sum_squared_residuals': 0.0
    }

    try:
        mstool.open(vis)
        n_rows = mstool.nrow()

        if verbose:
            print(f"Total rows: {n_rows}")
            print(f"Sampling up to {sample_size} points...")

        # Use iteration to read data (CASA 6.x API)
        # Initialize iteration
        mstool.iterinit()
        mstool.iterorigin()

        residuals_list = []
        total_npts = 0
        n_samples = 0

        while n_samples < sample_size:
            # Advance to next iteration (CASA 6.x API)
            retval = mstool.iternext()

            # Check if we've reached the end
            if retval <= 0:
                break

            # Get data for current iteration
            # MS columns are uppercase (DATA, MODEL_DATA, WEIGHT)
            # But getdata returns lowercase keys (data, model_data, weight)
            ms_columns = [datacolumn.upper(), 'MODEL_DATA', 'WEIGHT']
            data = mstool.getdata(ms_columns)

            # Check if we got data
            if not data or len(data) == 0:
                break

            # Debug: print first iteration
            if verbose and n_samples == 0:
                print(f"  Keys returned by getdata: {list(data.keys())}")
                print(f"  Looking for: '{datacolumn}', '{modelcolumn}'")

            # getdata returns lowercase keys
            if datacolumn in data and modelcolumn in data:
                data_vals = np.array(data[datacolumn]).copy()  # Make explicit copy
                model_vals = np.array(data[modelcolumn]).copy()  # Make explicit copy
                weights = data.get('weight', None)

                # Debug: print first few iterations
                if verbose and n_samples < 3:
                    print(f"  Row {n_samples}: data shape: {data_vals.shape}, model shape: {model_vals.shape}")
                    print(f"    Data finite: {np.sum(np.isfinite(data_vals))}/{data_vals.size}")
                    print(f"    Model finite: {np.sum(np.isfinite(model_vals))}/{model_vals.size}")

                # Flatten if needed
                if hasattr(data_vals, 'flatten'):
                    data_vals = data_vals.flatten()
                if hasattr(model_vals, 'flatten'):
                    model_vals = model_vals.flatten()
                if weights is not None and hasattr(weights, 'flatten'):
                    weights = weights.flatten()

                # Debug after flatten
                if verbose and n_samples == 0:
                    print(f"  After flatten - data: {data_vals.shape}, model: {model_vals.shape}")

                # Select axis
                if axis == 'real':
                    data_vals = np.real(data_vals)
                    model_vals = np.real(model_vals)
                elif axis == 'imag':
                    data_vals = np.imag(data_vals)
                    model_vals = np.imag(model_vals)
                elif axis == 'amp':
                    data_vals = np.abs(data_vals)
                    model_vals = np.abs(model_vals)

                # Debug after axis selection
                if verbose and n_samples == 0:
                    print(f"  After axis '{axis}' - data: {data_vals.shape}, model: {model_vals.shape}")

                # Remove NaN and Inf
                mask = np.isfinite(data_vals) & np.isfinite(model_vals)
                if weights is not None:
                    mask = mask & np.isfinite(weights)

                data_vals = data_vals[mask]
                model_vals = model_vals[mask]

                if weights is not None:
                    weights = weights[mask]
                    # Debug before subtraction
                    if verbose and n_samples == 0:
                        print(f"  Before subtraction - data: {data_vals.shape}, model: {model_vals.shape}, weights: {weights.shape}")
                    try:
                        residuals = data_vals - model_vals
                        residuals_list.extend((residuals**2 * weights).tolist())
                        total_npts += len(residuals)
                    except Exception as e:
                        raise ValueError(f"Subtraction failed with data shape {data_vals.shape}, model shape {model_vals.shape}: {e}")
                else:
                    # Debug before subtraction
                    if verbose and n_samples == 0:
                        print(f"  Before subtraction - data: {data_vals.shape}, model: {model_vals.shape}")
                    try:
                        residuals = data_vals - model_vals
                        residuals_list.extend((residuals**2).tolist())
                        total_npts += len(residuals)
                    except Exception as e:
                        raise ValueError(f"Subtraction failed with data shape {data_vals.shape}, model shape {model_vals.shape}: {e}")

                n_samples += 1

        # End iteration
        mstool.iterend()
        mstool.close()

        # Calculate chi2
        sum_squared_residuals = np.sum(residuals_list)

        # Check if we got any valid data
        if total_npts == 0 or n_samples == 0:
            raise ValueError("No valid data points found in MS (all masked or empty)")

        # Scale up to full MS size
        if n_samples < n_rows:
            scale_factor = n_rows / n_samples
            chi2 = sum_squared_residuals * scale_factor
            estimated_npts = int(total_npts * scale_factor)
        else:
            chi2 = sum_squared_residuals
            estimated_npts = total_npts

        result['chi2'] = chi2
        result['npts'] = estimated_npts
        result['sum_squared_residuals'] = sum_squared_residuals

        if verbose:
            print(f"Sampled {n_samples} rows")
            print(f"Points in sample: {total_npts}")
            print(f"Estimated total points: {estimated_npts}")
            print(f"χ²: {chi2:.2f}")

    except Exception as e:
        if verbose:
            print(f"ERROR calculating χ²: {e}")
            import traceback
            traceback.print_exc()
        try:
            mstool.close()
        except:
            pass

    return result


def _extract_quantity(comp, value_key, error_key=None, target_unit=None, verbose=False):
    """
    Extract a quantity from a componentlist with proper units.

    Handles different patterns for value and error:
    - Pattern 1: value in comp[value_key], error in comp[error_key] (separate keys)
    - Pattern 2: value and error in comp[value_key]['value'] and ['error']

    Parameters
    ----------
    comp : dict
        Componentlist component dictionary
    value_key : str
        Key for the value (e.g., 'majoraxis', 'flux', 'm0')
    error_key : str or None
        Key for the error if separate (e.g., 'majoraxiserror')
    target_unit : astropy.units.Unit or None
        Target unit to convert to
    verbose : bool
        Print debug information

    Returns
    -------
    dict
        {
            'value': float, numeric value in target unit
            'error': float, error in target unit
            'quantity': astropy.Quantity, value with original unit
            'unit': str, string representation of target unit
        }
    """
    # Map CASA unit strings to astropy units
    unit_map = {
        'Jy': u.Jy,
        'mJy': u.mJy,
        'uJy': u.uJy,
        'arcsec': u.arcsec,
        'arcmin': u.arcmin,
        'deg': u.degree,
        'rad': u.rad,
        'mas': u.mas,
        '': u.dimensionless_unscaled
    }

    # Extract value
    if value_key in comp:
        value_struct = comp[value_key]
        if isinstance(value_struct, dict) and 'value' in value_struct:
            value = value_struct['value']
            unit_str = value_struct.get('unit', '')
        else:
            value = value_struct
            unit_str = ''
    else:
        value = 0
        unit_str = ''

    # Extract error - try multiple patterns
    error = 0
    if error_key is not None and error_key in comp:
        # Pattern 1: Separate error key (e.g., majoraxiserror)
        error_struct = comp[error_key]
        if isinstance(error_struct, dict) and 'value' in error_struct:
            error = error_struct['value']
            # If error has different unit than value, we'll need to handle it
            error_unit_str = error_struct.get('unit', '')
        else:
            error = error_struct
    elif value_key in comp and isinstance(comp[value_key], dict):
        # Pattern 2: Error in same struct
        value_struct = comp[value_key]
        if 'error' in value_struct:
            error_raw = value_struct['error']
            if isinstance(error_raw, dict):
                # Pattern 2a: Error has latitude/longitude (for direction)
                if 'latitude' in error_raw:
                    error = error_raw['latitude'].get('value', 0)
                elif 'longitude' in error_raw:
                    error = error_raw['longitude'].get('value', 0)
                else:
                    error = 0
            elif hasattr(error_raw, '__len__'):
                # Pattern 2b: Error is an array (for flux)
                error = error_raw[0] if len(error_raw) > 0 else 0
            else:
                error = error_raw

    # Get the original unit
    original_unit = unit_map.get(unit_str, u.dimensionless_unscaled)

    if verbose:
        print(f"    Extracting {value_key}: value={value}, error={error}, unit='{unit_str}' -> {original_unit}")
        if error_key and error_key in comp:
            print(f"      (error from separate key '{error_key}')")

    # Create Quantity with original unit
    quantity = value * original_unit
    error_quantity = error * original_unit

    # Convert to target unit if specified
    if target_unit is not None and original_unit != u.dimensionless_unscaled:
        try:
            quantity_converted = quantity.to(target_unit)
            error_converted = error_quantity.to(target_unit)
            return {
                'value': quantity_converted.value,
                'error': error_converted.value,
                'quantity': quantity_converted,
                'unit': str(target_unit)
            }
        except u.UnitConversionError:
            if verbose:
                print(f"    Warning: Could not convert {original_unit} to {target_unit}")
            return {
                'value': value,
                'error': error,
                'quantity': quantity,
                'unit': unit_str
            }

    return {
        'value': value,
        'error': error,
        'quantity': quantity,
        'unit': unit_str
    }


def plot_uvbins(vis, datacolumn='corrected', avg_axis='real',
                uvbin_params=None,
                save_uvdata=None,
                plotfile=None,
                target_name=None,
                fit_results=None,
                verbose=False):
    """
    Compute binned UV visibility data, optionally save to file and/or create a plot.

    This is a standalone function that can be used independently or is called
    internally by fit_uv_model(). When fit_results are provided, the model UV
    curve and fit statistics are included in the plot and saved data.

    Parameters
    ----------
    vis : string
        Measurement set path.
    datacolumn : string (default: 'corrected')
        Data column for average_uvdata().
    avg_axis : string (default: 'real')
        Axis for averaging.
    uvbin_params : dict or None (default: None)
        Parameters for suggest_uv_bins(). Default: {'n_bins': 10, 'binning_type': 'log'}.
    save_uvdata : string or None (default: None)
        If truthy, save binned data to this filename.
    plotfile : string or None or False (default: None)
        Output plot path. If None, auto-generate from target_name.
        If explicitly False, skip plotting (compute data only).
    target_name : string or None (default: None)
        Target name for plot title and auto-generated filenames.
    fit_results : dict or None (default: None)
        The return dict from fit_uv_model(). If provided:
        - Model UV curve is plotted (red line)
        - Key fitted values with errors + fit statistics are annotated.
    verbose : bool (default: False)
        Print progress information.

    Returns
    -------
    dict
        {
            'uvpoints': np.array,
            'data': np.array,       # mJy
            'err_lower': np.array,  # mJy
            'err_upper': np.array,  # mJy
            'model': np.array or None,  # mJy
        }
    """
    with suppress_casa_logs():
        if verbose:
            print(f"\n=== plot_uvbins: {vis} ===")

        # Detect UV range
        uv_info = get_uv_range(vis, verbose=verbose)
        if uv_info['uvdist_min'] is None or uv_info['uvdist_max'] is None:
            if verbose:
                print("Could not detect UV range, using defaults")
            uvmin = 3
            uvmax = 150
        else:
            uvmin = max(3, uv_info['uvdist_min'] * 0.95)
            uvmax = uv_info['uvdist_max'] * 1.05

        # Set binning parameters
        if uvbin_params is None:
            uvbin_params = {'n_bins': 10, 'binning_type': 'log'}

        # Create bins
        uvbins = suggest_uv_bins(uvmin, uvmax,
                                  n_bins=uvbin_params.get('n_bins', 10),
                                  binning_type=uvbin_params.get('binning_type', 'log'),
                                  verbose=verbose)

        # Average data
        uvpoints, avg_amps, err_lo, err_hi = average_uvdata(
            vis, datacolumn=datacolumn, uvbins=uvbins, axis=avg_axis, verbose=verbose)

        # Average model (if fit_results provided, try to get model column)
        mod_amps = None
        if fit_results is not None:
            try:
                uvpoints, mod_amps, _, _ = average_uvdata(
                    vis, datacolumn="model", uvbins=uvbins, axis=avg_axis, verbose=verbose)
            except Exception:
                if verbose:
                    print("  No MODEL column available, skipping model curve")
                mod_amps = None

    # Save binned data to file
    if save_uvdata:
        if verbose:
            print(f"Saving averaged UV data to {save_uvdata}...")

        if mod_amps is not None:
            uvdata = np.array([uvpoints, avg_amps, err_lo, err_hi, mod_amps]).transpose()
            np.savetxt(save_uvdata, uvdata, fmt="%.5f",
                       header="uvdist(klambda), data(mJy), err_lower(mJy), err_upper(mJy), model(mJy)")
        else:
            uvdata = np.array([uvpoints, avg_amps, err_lo, err_hi]).transpose()
            np.savetxt(save_uvdata, uvdata, fmt="%.5f",
                       header="uvdist(klambda), data(mJy), err_lower(mJy), err_upper(mJy)")

        if verbose:
            print(f"  Saved {len(uvpoints)} binned data points")

    # Create plot
    if plotfile is not False:
        if verbose:
            print("Creating UV plot...")

        fig = plt.figure(figsize=(7, 7))
        ax = plt.gca()

        ax.errorbar(uvpoints, avg_amps, yerr=[err_lo, err_hi], mfc='k', fmt='o',
                    label='data', capsize=3)

        if mod_amps is not None:
            ax.plot(uvpoints, mod_amps, 'r-', label='model')

        # Annotate fit results
        if fit_results is not None and fit_results.get('statistics', {}).get('dof') is not None:
            stats = fit_results['statistics']
            chi2 = stats['chi2']
            bic = stats['bic']
            dof = stats['dof']
            comptype = fit_results.get('comptype', 'P')

            flux_mjy = fit_results['flux']['value'] * 1000
            flux_err_mjy = fit_results['flux']['error'] * 1000

            if comptype in ["G", "D"] and fit_results.get('size') is not None:
                size = fit_results['size']
                bmaj_val = size['bmaj']['value']
                bmaj_err = size['bmaj']['error']
                axrat_val = size['axrat']['value']
                axrat_err = size['axrat']['error']

                text  = f"Flux = {flux_mjy:.2f} \u00b1 {flux_err_mjy:.2f} mJy\n"
                text += f"bmaj = {bmaj_val:.3f} \u00b1 {bmaj_err:.3f}\"\n"
                text += f"axrat = {axrat_val:.2f} \u00b1 {axrat_err:.2f}\n"
                text += f"\u03c7\u00b2 = {chi2:.1f}\n"
                text += f"BIC = {bic:.1f}\n"
                text += f"DOF = {dof}"
            else:
                text  = f"Flux = {flux_mjy:.2f} \u00b1 {flux_err_mjy:.2f} mJy\n"
                text += f"\u03c7\u00b2 = {chi2:.1f}\n"
                text += f"BIC = {bic:.1f}\n"
                text += f"DOF = {dof}"

            ax.text(1.02, 1.02, text, transform=ax.transAxes, va='top', ha='left', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.legend(loc='best', numpoints=1)
        ax.set_xlabel('UV Distance (k\u03bb)')
        ax.set_ylabel('Real Visibility (mJy)')

        if target_name:
            ax.set_title(f"{target_name}")

        # Determine plot file name
        if plotfile is None:
            if target_name:
                plotfile = f"{target_name}_uvfit.pdf"
            else:
                plotfile = "uvfit.pdf"

        plt.savefig(plotfile, bbox_inches="tight")
        plt.close()

        if verbose:
            print(f"  Saved plot to {plotfile}")

    return {
        'uvpoints': uvpoints,
        'data': avg_amps,
        'err_lower': err_lo,
        'err_upper': err_hi,
        'model': mod_amps,
    }


def fit_uv_model(vis,
                 comptype="P",
                 sourcepar=None,
                 varpar=None,
                 outfile="fit.cl",
                 niter=20,
                 save_uvdata=None,
                 make_plot=False,
                 plotfile=None,
                 target_name=None,
                 uvbin_params=None,
                 datacolumn='corrected',
                 avg_axis='real',
                 verbose=True,
                 **kwargs):
    """
    Run uvmodelfit on a measurement set and optionally save/plot results.

    This function captures uvmodelfit terminal output to extract parameter
    uncertainties, working around the CASA bug where componentlist does not
    contain error values.

    Parameters
    ----------
    vis : string
        Measurement set name.
    comptype : string (default: "P")
        Component type: 'P' (Point source), 'G' (Gaussian), or 'D' (Disk)
    sourcepar : list or None (default: None)
        Initial parameters. Format depends on comptype:
        - P: [flux, xoff, yoff] - flux (Jy), offsets (arcsec)
        - G: [flux, xoff, yoff, bmaj, axrat, pa] - add bmaj (arcsec), axrat (<1), pa (deg)
        - D: [flux, xoff, yoff, bmaj, axrat, pa] - same as Gaussian
    varpar : list or None (default: None)
        Which parameters to vary (0-based indices). If None, all vary.
        Parameters not in this list will be held fixed during fitting.
        Point source (P): [0=flux, 1=x, 2=y]
        Gaussian/Disk (G/D): [0=flux, 1=x, 2=y, 3=bmaj, 4=axrat, 5=pa]
        Note: This is converted to CASA's 'varypar' boolean list internally.
    outfile : string (default: "fit.cl")
        Output componentlist file.
    niter : int (default: 20)
        Maximum number of iterations.
    save_uvdata : string or None (default: None)
        If provided, save averaged UV data to this file.
    make_plot : bool (default: False)
        Whether to create a UV fit plot.
    plotfile : string or None (default: None)
        Output plot file name.
    target_name : string or None (default: None)
        Target name for plot title and file naming.
    uvbin_params : dict or None (default: None)
        Parameters for UV binning (n_bins, binning_type).
    verbose : bool (default: True)
        Print progress information.
    **kwargs
        Additional parameters for uvmodelfit.

    Returns
    -------
    dict
        Fit results containing:
        - 'success': bool, whether fit succeeded
        - 'comptype': str, component type fitted
        - 'flux': {'value': float, 'error': float, 'quantity': Quantity, 'unit': 'Jy'}
        - 'offset': {'x': {'value': float, 'error': float, 'quantity': Quantity, 'unit': 'arcsec'},
                     'y': {'value': float, 'error': float, 'quantity': Quantity, 'unit': 'arcsec'}}
        - 'size': dict or None (only for G/D types):
            {'bmaj': {'value': float, 'error': float, 'quantity': Quantity, 'unit': 'arcsec'},
             'axrat': {'value': float, 'error': float, 'unit': ''},
             'pa': {'value': float, 'error': float, 'quantity': Quantity, 'unit': 'deg'}}
        - 'statistics': {'dof': int, 'chi2': float, 'bic': float, 'n_params': int, 'npts': int}
          Note: n_params is the actual number of fitted parameters, accounting for
          any fixed parameters specified via varpar.
        - 'uv_data': dict or None, averaged UV data if save_uvdata
        - 'fit_stats': {'niter': int, 'converged': bool}
        - 'outfile': str, path to componentlist file

    Notes
    -----
    All parameter values and uncertainties are extracted from uvmodelfit terminal
    output. The componentlist file is still created because it is needed by ft()
    to write the MODEL_DATA column into the MS (used for model UV curve comparison).

    Offsets (x, y) are relative to the phase center in arcsec (East, North).
    For Gaussian and Disk types, axrat is the axis ratio (minor/major, < 1).
    PA is position angle in degrees (East of North).

    The varpar parameter specifies which parameters to vary during fitting
    (using 0-based indices: P=[0=flux,1=x,2=y], G/D=[0=flux,1=x,2=y,3=bmaj,4=axrat,5=pa]).
    Statistics (DOF, BIC) correctly account for fixed parameters - n_params reflects
    the actual number of fitted parameters, not the total available.

    For even more robust uncertainties, use fit_uv_model_bootstrap().

    References:
    - CASA Known Issues: https://casa.nrao.edu/release_ki.shtml
    - uvmodelfit documentation: https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.manipulation.uvmodelfit.html
    """
    result = {
        'success': False,
        'comptype': comptype,
        'flux': {'value': None, 'error': None, 'unit': 'Jy'},
        'offset': {'x': {'value': None, 'error': None, 'unit': 'arcsec'},
                   'y': {'value': None, 'error': None, 'unit': 'arcsec'}},
        'size': None,  # Only for G/D types
        'statistics': {'dof': None, 'chi2': None, 'bic': None, 'n_params': None, 'npts': None},
        'uv_data': None,
        'fit_stats': {'niter': None, 'converged': False},
        'outfile': outfile
    }

    # Set default sourcepar based on comptype if not provided
    if sourcepar is None:
        if comptype == "P":
            # Point source: 1 Jy, at phase center
            sourcepar = [1.0, 0., 0.]
        elif comptype in ["G", "D"]:
            # Gaussian/Disk: 200 mJy, center, 3" circular
            sourcepar = [0.2, 0., 0., 3., 0.9, 45.]
        else:
            raise ValueError(f"Unknown comptype: {comptype}. Must be 'P', 'G', or 'D'")

    if verbose:
        print(f"\n=== Fitting UV Model for {vis} ===")
        print(f"Component type: {comptype}")
        print(f"Initial parameters: {sourcepar}")

    # Remove old fit file if exists
    if os.path.exists(outfile):
        os.system(f"rm -rf {outfile}")

    # Run uvmodelfit with output capture
    if verbose:
        print("\nRunning uvmodelfit...")

    # Capture output using file descriptor redirection
    import sys
    log_file = "uvmodelfit_output.txt"

    # Save original stdout file descriptor
    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)

    try:
        # Open file to capture output
        captured_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

        # Redirect stdout to file
        os.dup2(captured_fd, stdout_fd)

        # Also redirect to a tee so we can see it on screen
        # (Actually, let's just let it go to the file, we'll read it back)

        # Run uvmodelfit (output goes to file)
        # Build kwargs, explicitly including varypar if varpar is provided
        # Note: CASA's uvmodelfit uses 'varypar' (boolean list), but our function
        # uses 'varpar' (list of indices to vary) for better usability
        uvmodelfit_kwargs = {
            'vis': vis,
            'niter': niter,
            'comptype': comptype,
            'outfile': outfile,
            'sourcepar': sourcepar,
        }

        # Convert varpar (indices) to varypar (boolean list) for CASA
        if varpar is not None:
            # Determine total number of parameters
            if comptype == "P":
                n_total_params = 3
            elif comptype in ["G", "D"]:
                n_total_params = 6
            else:
                n_total_params = 6

            # Create boolean list: True for vary, False for fixed
            varypar = [False] * n_total_params
            for idx in varpar:
                if 0 <= idx < n_total_params:
                    varypar[idx] = True
            uvmodelfit_kwargs['varypar'] = varypar

        # Add any additional kwargs
        uvmodelfit_kwargs.update(kwargs)

        uvmodelfit(**uvmodelfit_kwargs)

        # Restore original stdout
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(captured_fd)

        if verbose:
            print("  Fitting complete")

    except Exception as e:
        # Restore stdout on error
        os.dup2(saved_stdout_fd, stdout_fd)
        try:
            os.close(captured_fd)
        except:
            pass
        if verbose:
            print(f"ERROR: uvmodelfit failed: {e}")
        return result
    finally:
        # Always restore the saved fd
        try:
            os.close(saved_stdout_fd)
        except:
            pass

    # Read captured output
    output_text = ""
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            output_text = f.read()
        # Don't clean up temp log file for debugging
        # try:
        #     os.remove(log_file)
        # except:
        #     pass

    if verbose and output_text:
        print(f"  Captured {len(output_text)} characters of output")

    # Parse uncertainties and statistics from output
    uncertainties = parse_uvmodelfit_output(output_text, comptype=comptype)

    if verbose and uncertainties['flux']['error'] is not None:
        print("  ✓ Extracted parameter uncertainties from uvmodelfit output")
    elif verbose:
        print("  ⚠ Warning: Could not extract uncertainties from output")
        print("    (Use fit_uv_model_bootstrap() for robust errors)")
        if output_text:
            print("  Debug: Output file saved as", log_file)

    # Write MODEL column using ft
    if verbose:
        print("Writing MODEL column using ft...")

    try:
        ft(vis, complist=outfile)
    except Exception as e:
        if verbose:
            print(f"WARNING: ft failed: {e}")

    # Build result from parsed uvmodelfit output
    # All parameter values and uncertainties come from parsing the terminal output.
    # We do NOT read from the componentlist because:
    #   - CASA bug: componentlist errors are all 0.0
    #   - componentlist stores minoraxis as FWHM, not as axis ratio
    #   - componentlist stores absolute coordinates, not phase-center offsets
    # The outfile is still needed for ft() to write MODEL_DATA into the MS.
    if verbose:
        print("\nBuilding result from uvmodelfit output...")

    # Check that parsing succeeded
    if uncertainties['flux']['value'] is None:
        if verbose:
            print("ERROR: Could not parse flux from uvmodelfit output")
        return result

    # Flux
    flux_val = uncertainties['flux']['value']
    flux_err = uncertainties['flux']['error'] if uncertainties['flux']['error'] is not None else 0.0
    result['flux'] = {
        'value': flux_val,
        'error': flux_err,
        'quantity': flux_val * u.Jy,
        'unit': 'Jy'
    }

    # Offsets
    x_val = uncertainties['offset']['x']['value']
    y_val = uncertainties['offset']['y']['value']
    x_err = uncertainties['offset']['x']['error']
    y_err = uncertainties['offset']['y']['error']
    result['offset']['x'] = {
        'value': x_val if x_val is not None else 0.0,
        'error': x_err if x_err is not None else 0.0,
        'quantity': (x_val if x_val is not None else 0.0) * u.arcsec,
        'unit': 'arcsec'
    }
    result['offset']['y'] = {
        'value': y_val if y_val is not None else 0.0,
        'error': y_err if y_err is not None else 0.0,
        'quantity': (y_val if y_val is not None else 0.0) * u.arcsec,
        'unit': 'arcsec'
    }

    # Size (for Gaussian and Disk)
    if comptype in ["G", "D"]:
        bmaj_val = uncertainties['size']['bmaj']['value']
        bmaj_err = uncertainties['size']['bmaj']['error'] if uncertainties['size']['bmaj']['error'] is not None else 0.0
        axrat_val = uncertainties['size']['axrat']['value']
        axrat_err = uncertainties['size']['axrat']['error'] if uncertainties['size']['axrat']['error'] is not None else 0.0
        pa_val = uncertainties['size']['pa']['value']
        pa_err = uncertainties['size']['pa']['error'] if uncertainties['size']['pa']['error'] is not None else 0.0

        result['size'] = {
            'bmaj': {
                'value': bmaj_val if bmaj_val is not None else 0.0,
                'error': bmaj_err,
                'quantity': (bmaj_val if bmaj_val is not None else 0.0) * u.arcsec,
                'unit': 'arcsec'
            },
            'axrat': {
                'value': axrat_val if axrat_val is not None else 1.0,
                'error': axrat_err,
                'unit': ''
            },
            'pa': {
                'value': pa_val if pa_val is not None else 0.0,
                'error': pa_err,
                'quantity': (pa_val if pa_val is not None else 0.0) * u.degree,
                'unit': 'deg'
            }
        }

    result['success'] = True

    if verbose:
        flux = result['flux']
        offset = result['offset']
        print(f"Flux: {flux['value']:.6f} +/- {flux['error']:.6f} {flux['unit']}")
        print(f"Offset: x={offset['x']['value']:.6f} +/- {offset['x']['error']:.6f} {offset['x']['unit']}")
        print(f"         y={offset['y']['value']:.6f} +/- {offset['y']['error']:.6f} {offset['y']['unit']}")
        if comptype in ["G", "D"] and result.get('size'):
            size = result['size']
            print(f"Size: bmaj={size['bmaj']['value']:.6f} +/- {size['bmaj']['error']:.6f} {size['bmaj']['unit']}")
            print(f"      axrat={size['axrat']['value']:.6f} +/- {size['axrat']['error']:.6f}")
            print(f"      pa={size['pa']['value']:.2f} +/- {size['pa']['error']:.2f} {size['pa']['unit']}")

    # Calculate statistics from parsed uvmodelfit output
    if verbose:
        print("\nFit statistics from uvmodelfit:")

    if uncertainties['statistics']['dof'] is not None:
        dof = uncertainties['statistics']['dof']
        npts = uncertainties['statistics']['npts']
        reduced_chi2 = uncertainties['statistics']['reduced_chi2']

        # Calculate chi2 from reduced chi2
        chi2 = reduced_chi2 * dof

        # Determine number of fitted parameters
        # Priority: 1) Use parsed n_params from uvmodelfit output (accounts for varpar)
        #           2) Calculate from varpar length if provided
        #           3) Fall back to comptype-based default
        n_params = uncertainties['statistics'].get('n_params', None)
        if n_params is None:
            if varpar is not None:
                n_params = len(varpar)
                if verbose:
                    print(f"  Using varpar to determine n_params: {n_params}")
            else:
                if comptype in ["G", "D"]:
                    n_params = 6  # I, x, y, bmaj, axrat, pa
                elif comptype == "P":
                    n_params = 3  # I, x, y
                else:
                    n_params = 6  # Default assumption
                if verbose:
                    print(f"  Using comptype-based n_params: {n_params}")

        # Calculate BIC: BIC = k*ln(N) + chi2
        import math
        bic = n_params * math.log(npts) + chi2

        result['statistics'] = {
            'dof': dof,
            'reduced_chi2': reduced_chi2,
            'chi2': chi2,  # Include chi2 for reference
            'bic': bic,
            'n_params': n_params,  # Include number of fitted parameters
            'npts': npts
        }

        if verbose:
            print(f"  N (data points): {npts}")
            print(f"  k (fitted parameters): {n_params}")
            print(f"  DOF: {dof}")
            print(f"  Reduced χ²: {reduced_chi2:.4f}")
            print(f"  BIC: {bic:.2f}")
    else:
        if verbose:
            print("  ⚠ Could not extract statistics from uvmodelfit output")
            print("    (Iteration information may not be available)")

        # Still record n_params based on varpar if available
        n_params = None
        if varpar is not None:
            n_params = len(varpar)
        elif comptype in ["G", "D"]:
            n_params = 6
        elif comptype == "P":
            n_params = 3
        else:
            n_params = 6

        result['statistics'] = {
            'dof': None,
            'chi2': None,
            'bic': None,
            'n_params': n_params  # Record expected number of parameters
        }

    # Optionally save averaged UV data and/or create plot
    if make_plot:
        uv_data = plot_uvbins(
            vis, datacolumn=datacolumn, avg_axis=avg_axis,
            uvbin_params=uvbin_params,
            save_uvdata=save_uvdata,
            plotfile=plotfile,
            target_name=target_name,
            fit_results=result,
            verbose=verbose,
        )
        result['uv_data'] = uv_data

    return result


# ============================================================================
# Beam Utilities for FITS Cubes
# ============================================================================

def get_beam_from_fits(hdul):
    """Extract beam (BMAJ, BMIN in arcsec) from a CASA FITS file.

    CASA 6.7+ tclean stores per-channel beam info in a BEAMS binary table
    extension (HDU name ``'BEAMS'``) with units of arcsec.  Older CASA
    versions put beam info in the PRIMARY header ``BMAJ``/``BMIN`` keywords
    in degrees.  For per-channel beams the median across channels is returned.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Open FITS file handle.

    Returns
    -------
    bmaj_arcsec, bmin_arcsec : float
        Beam major and minor axis in arcsec.
    """
    from astropy.io import fits

    # Try PRIMARY header first (older CASA, units = degrees)
    header = hdul[0].header
    bmaj = header.get('BMAJ', None)
    bmin = header.get('BMIN', None)

    if bmaj is not None and bmin is not None and bmaj > 0 and bmin > 0:
        return float(bmaj) * 3600, float(bmin) * 3600  # deg -> arcsec

    # Try BEAMS binary table extension (CASA 6.7+, units = arcsec)
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
    """Beam area in pixels from a CASA FITS file.

    Gaussian beam area::

        omega = pi * BMAJ * BMIN / (4 * ln(2))
        beam_area_pix = omega / pixscale^2

    Reads beam from BEAMS table extension when BMAJ/BMIN are missing from
    the PRIMARY header (CASA 6.7+).

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Open FITS file handle.

    Returns
    -------
    beam_area_pix : float
        Beam area in units of pixels.
    beam_area_arcsec2 : float
        Beam area in square arcsec.
    bmaj_arcsec : float
        Beam major axis in arcsec.
    bmin_arcsec : float
        Beam minor axis in arcsec.
    """
    bmaj_arcsec, bmin_arcsec = get_beam_from_fits(hdul)
    header = hdul[0].header
    pixscale_arcsec = abs(header.get('CDELT1', 1)) * 3600

    beam_area_arcsec2 = (np.pi * bmaj_arcsec * bmin_arcsec
                         / (4.0 * np.log(2)))
    beam_area_pix = beam_area_arcsec2 / pixscale_arcsec**2
    return beam_area_pix, beam_area_arcsec2, bmaj_arcsec, bmin_arcsec


# ============================================================================
# Bootstrap Functions for Robust Error Estimation
# ============================================================================

def compute_bootstrap_stats(samples, confidence_level=0.68):
    """
    Compute bootstrap statistics from sample values.

    Parameters
    ----------
    samples : array-like
        Bootstrap sample values (e.g., flux values from each iteration)
    confidence_level : float (default: 0.68)
        Confidence level for uncertainty bounds.
        0.68 ≈ 1σ, 0.95 ≈ 2σ, 0.997 ≈ 3σ

    Returns
    -------
    dict
        {
            'median': float, median of bootstrap samples
            'lower': float, lower confidence bound
            'upper': float, upper confidence bound
            'std': float, standard deviation
            'samples': array, raw bootstrap samples
        }
    """
    samples = np.array(samples)

    if len(samples) == 0:
        return {
            'median': np.nan,
            'lower': np.nan,
            'upper': np.nan,
            'std': np.nan,
            'samples': samples
        }

    # Compute percentiles for confidence interval
    alpha = 1.0 - confidence_level
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    return {
        'median': np.median(samples),
        'lower': np.percentile(samples, lower_pct),
        'upper': np.percentile(samples, upper_pct),
        'std': np.std(samples),
        'samples': samples
    }


def resample_ms(vis_original, vis_resampled, seed=None, verbose=False):
    """
    Create a resampled measurement set by sampling rows with replacement.

    This function reads all visibility data from the original MS and creates
    a new MS with the same structure but with rows resampled with replacement.
    This is the core operation for bootstrap resampling.

    Parameters
    ----------
    vis_original : str
        Path to original measurement set
    vis_resampled : str
        Path for output resampled measurement set
    seed : int or None (default: None)
        Random seed for reproducibility
    verbose : bool (default: False)
        Print progress information

    Returns
    -------
    int
        Number of rows in the resampled MS

    Raises
    ------
    RuntimeError
        If resampling fails
    IOError
        If file operations fail
    """
    from casatools import table
    tb = table()

    if verbose:
        print(f"  Resampling {vis_original} -> {vis_resampled}")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Open original MS
    if verbose:
        print("    Reading original MS...")

    try:
        tb.open(vis_original, nomodify=True)
    except Exception as e:
        raise RuntimeError(f"Failed to open original MS: {e}")

    # Get the number of rows
    n_rows = tb.nrows()
    if verbose:
        print(f"    Original MS has {n_rows} rows")

    # Read all necessary columns
    # DATA: complex visibilities [nchan, npol, nrow]
    # WEIGHT: weights [npol, nrow]
    # FLAG: flags [nchan, npol, nrow]
    # UVW: baseline coordinates [3, nrow]
    # TIME, ANTENNA1, ANTENNA2: metadata [nrow]
    # DATA_DESC_ID, SCAN_NUMBER: metadata [nrow]

    try:
        data = tb.getcol("DATA")
        weight = tb.getcol("WEIGHT")
        flag = tb.getcol("FLAG")
        uvw = tb.getcol("UVW")
        time = tb.getcol("TIME")
        antenna1 = tb.getcol("ANTENNA1")
        antenna2 = tb.getcol("ANTENNA2")
        data_desc_id = tb.getcol("DATA_DESC_ID")
        scan_number = tb.getcol("SCAN_NUMBER")

        # Try to get optional columns
        try:
            sigma = tb.getcol("SIGMA")
        except RuntimeError:
            sigma = None

        try:
            flag_row = tb.getcol("FLAG_ROW")
        except RuntimeError:
            flag_row = None

    except Exception as e:
        tb.close()
        raise RuntimeError(f"Failed to read columns from MS: {e}")

    tb.close()

    # Generate random indices for resampling (with replacement)
    indices = np.random.choice(n_rows, size=n_rows, replace=True)

    if verbose:
        print(f"    Generated resampling indices")

    # Create resampled MS by copying original structure
    # and replacing data with resampled rows
    if verbose:
        print(f"    Creating resampled MS...")

    # Remove existing resampled MS if present
    if os.path.exists(vis_resampled):
        if verbose:
            print(f"    Removing existing {vis_resampled}")
        try:
            os.system(f"rm -rf {vis_resampled}")
        except Exception as e:
            raise IOError(f"Failed to remove existing MS: {e}")

    # Copy the original MS to create structure
    try:
        os.system(f"cp -r {vis_original} {vis_resampled}")
    except Exception as e:
        raise IOError(f"Failed to copy MS: {e}")

    # Open the new MS for writing
    try:
        tb.open(vis_resampled, nomodify=False)
    except Exception as e:
        raise RuntimeError(f"Failed to open resampled MS for writing: {e}")

    # Replace data with resampled rows
    try:
        tb.putcol("DATA", data[:, :, indices])
        tb.putcol("WEIGHT", weight[:, indices])
        tb.putcol("FLAG", flag[:, :, indices])
        tb.putcol("UVW", uvw[:, indices])
        tb.putcol("TIME", time[indices])
        tb.putcol("ANTENNA1", antenna1[indices])
        tb.putcol("ANTENNA2", antenna2[indices])
        tb.putcol("DATA_DESC_ID", data_desc_id[indices])
        tb.putcol("SCAN_NUMBER", scan_number[indices])

        if sigma is not None:
            tb.putcol("SIGMA", sigma[:, indices])

        if flag_row is not None:
            tb.putcol("FLAG_ROW", flag_row[indices])

    except Exception as e:
        tb.close()
        raise RuntimeError(f"Failed to write resampled data: {e}")

    tb.close()

    if verbose:
        print(f"    Created resampled MS with {n_rows} rows")

    return n_rows


def fit_uv_model_bootstrap(vis,
                            comptype="G",
                            sourcepar=None,
                            varpar=None,
                            n_samples=100,
                            confidence_level=0.68,
                            niter=20,
                            temp_dir=None,
                            keep_temp_files=False,
                            seed=None,
                            verbose=True,
                            **kwargs):
    """
    Bootstrap UV model fitting to estimate robust parameter uncertainties.

    ✅ RECOMMENDED for reliable parameter uncertainties.

    This function performs bootstrap resampling of the visibility data and
    fits a model to each resampled dataset. The distribution of fitted
    parameters provides robust uncertainty estimates that don't rely on
    assumptions about Gaussian errors or linear approximations.

    Why use bootstrap instead of fit_uv_model()?
    --------------------------------------------
    fit_uv_model() returns error values of 0.0 due to a known CASA bug where
    uvmodelfit does not write parameter uncertainties to the componentlist.
    Bootstrap uncertainties are more reliable anyway because they:
    - Capture non-Gaussian error distributions
    - Don't rely on linear approximations
    - Account for correlations between parameters
    - Work with the actual data structure

    Parameters
    ----------
    vis : str
        Measurement set name
    comptype : str (default: "G")
        Component type: 'G' (Gaussian), 'P' (Point), 'D' (Disk), etc.
    sourcepar : list or None (default: None)
        Initial parameters [flux, x, y, ...].
        For Gaussian: [flux, x, y, bmaj, bmin, pa] in Jy, arcsec, arcsec,
        arcsec, arcsec, deg
    varpar : list or None (default: None)
        Which parameters to vary (0-based indices). If None, all vary.
        Parameters not in this list will be held fixed during fitting.
        Point source (P): [0=flux, 1=x, 2=y]
        Gaussian/Disk (G/D): [0=flux, 1=x, 2=y, 3=bmaj, 4=axrat, 5=pa]
        Note: This is converted to CASA's 'varypar' boolean list internally.
    n_samples : int (default: 100)
        Number of bootstrap samples to generate
    confidence_level : float (default: 0.68)
        Confidence level for uncertainty bounds (0.68 = 1σ, 0.95 = 2σ)
    niter : int (default: 20)
        Maximum number of iterations for each fit
    temp_dir : str or None (default: None)
        Directory for temporary resampled MS files. If None, creates
        'bootstrap_temp_<pid>' in current directory.
    keep_temp_files : bool (default: False)
        If True, keep temporary MS files for debugging
    seed : int or None (default: None)
        Random seed for reproducibility. Each bootstrap sample uses
        seed+i as its seed.
    verbose : bool (default: True)
        Print progress information
    **kwargs
        Additional parameters passed to uvmodelfit

    Returns
    -------
    dict
        Bootstrap results containing:
        - 'success': bool, overall success
        - 'n_successful': int, number of successful fits
        - 'n_failed': int, number of failed fits
        - 'confidence_level': float, confidence level used
        - 'flux': {
            'median': float, median bootstrap value in Jy
            'lower': float, lower confidence bound in Jy
            'upper': float, upper confidence bound in Jy
            'std': float, standard deviation in Jy
            'samples': array, all bootstrap values
          }
        - 'position': {'ra': {...}, 'dec': {...}} in arcsec
        - 'size': {'bmaj': {...}, 'bmin': {...}, 'pa': {...}} for Gaussian
        - 'original_fit': dict, results from fit to original data
        - 'temp_dir': str, location of temporary files
        - 'success_rate': float, proportion of successful fits
    """
    import tempfile

    # Set default sourcepar if not provided
    if sourcepar is None:
        sourcepar = [0.2, 0., 0., 3., 3., 45.]

    # Initialize result structure
    result = {
        'success': False,
        'n_successful': 0,
        'n_failed': 0,
        'confidence_level': confidence_level,
        'flux': {
            'median': None, 'lower': None, 'upper': None, 'std': None, 'samples': []
        },
        'position': {
            'ra': {'median': None, 'lower': None, 'upper': None, 'std': None, 'samples': []},
            'dec': {'median': None, 'lower': None, 'upper': None, 'std': None, 'samples': []}
        },
        'size': {
            'bmaj': {'median': None, 'lower': None, 'upper': None, 'std': None, 'samples': []},
            'bmin': {'median': None, 'lower': None, 'upper': None, 'std': None, 'samples': []},
            'pa': {'median': None, 'lower': None, 'upper': None, 'std': None, 'samples': []}
        } if comptype == "G" else None,
        'original_fit': None,
        'temp_dir': temp_dir,
        'success_rate': 0.0
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"Bootstrap UV Model Fitting")
        print(f"{'='*70}")
        print(f"Input MS: {vis}")
        print(f"Component type: {comptype}")
        print(f"Number of bootstrap samples: {n_samples}")
        print(f"Confidence level: {confidence_level*100:.1f}%")
        print(f"Random seed: {seed}")
        print(f"{'='*70}\n")

    # Step 1: Fit the original data to get baseline parameters
    if verbose:
        print("Step 1: Fitting original data...")

    original_result = fit_uv_model(
        vis=vis,
        comptype=comptype,
        sourcepar=sourcepar,
        varpar=varpar,
        niter=niter,
        verbose=False,
        **kwargs
    )

    result['original_fit'] = original_result

    if not original_result['success']:
        if verbose:
            print("ERROR: Original fit failed! Cannot proceed with bootstrap.")
        return result

    if verbose:
        print("  Original fit successful")
        print(f"  Flux: {original_result['flux']['value']*1000:.3f} mJy")

    # Step 2: Set up temporary directory
    if temp_dir is None:
        temp_dir = f"bootstrap_temp_{os.getpid()}"

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        if verbose:
            print(f"\nStep 2: Created temporary directory: {temp_dir}")
    else:
        if verbose:
            print(f"\nStep 2: Using existing temporary directory: {temp_dir}")

    result['temp_dir'] = temp_dir

    # Step 3: Run bootstrap iterations
    if verbose:
        print(f"\nStep 3: Running {n_samples} bootstrap iterations...")
        print(f"{'Sample':>8} {'Status':>10} {'Flux (mJy)':>15}")
        print(f"{'-'*40}")

    # Storage for bootstrap samples
    flux_samples = []
    ra_samples = []
    dec_samples = []
    bmaj_samples = []
    bmin_samples = []
    pa_samples = []

    # Run bootstrap samples
    for i in range(n_samples):
        sample_seed = seed + i if seed is not None else None

        if verbose and (i % 10 == 0 or i == n_samples - 1):
            print(f"{i+1:8d} ", end="", flush=True)

        # Create resampled MS
        vis_boot = os.path.join(temp_dir, f"bootstrap_{i:04d}.ms")

        try:
            # Resample the MS
            resample_ms(vis, vis_boot, seed=sample_seed, verbose=False)

            # Fit the resampled data
            boot_result = fit_uv_model(
                vis=vis_boot,
                comptype=comptype,
                sourcepar=sourcepar,  # Use same initial params for all fits
                varpar=varpar,
                niter=niter,
                verbose=False,
                **kwargs
            )

            # Store results if successful
            if boot_result['success']:
                flux_samples.append(boot_result['flux']['value'])
                ra_samples.append(boot_result['position']['ra']['value'])
                dec_samples.append(boot_result['position']['dec']['value'])

                if comptype == "G":
                    bmaj_samples.append(boot_result['size']['bmaj']['value'])
                    bmin_samples.append(boot_result['size']['bmin']['value'])
                    pa_samples.append(boot_result['size']['pa']['value'])

                result['n_successful'] += 1

                if verbose and (i % 10 == 0 or i == n_samples - 1):
                    flux_mjy = boot_result['flux']['value'] * 1000
                    print(f"{'OK':>10} {flux_mjy:13.3f}")
            else:
                result['n_failed'] += 1
                if verbose and (i % 10 == 0 or i == n_samples - 1):
                    print(f"{'FAIL':>10} {'---':>15}")

        except Exception as e:
            result['n_failed'] += 1
            if verbose and (i % 10 == 0 or i == n_samples - 1):
                print(f"{'ERROR':>10} {'---':>15}")

            if verbose and i < 5:  # Only print first few errors
                print(f"    Error: {e}")

        # Clean up temp MS
        if not keep_temp_files:
            try:
                os.system(f"rm -rf {vis_boot}")
            except:
                pass

    # Step 4: Compute statistics from bootstrap samples
    if verbose:
        print(f"{'-'*40}")
        print(f"\nStep 4: Computing bootstrap statistics...")

    result['success_rate'] = result['n_successful'] / n_samples

    if verbose:
        print(f"  Successful fits: {result['n_successful']}/{n_samples} "
              f"({result['success_rate']*100:.1f}%)")

    if result['n_successful'] < 10:
        if verbose:
            print(f"\nWARNING: Too few successful bootstrap samples ({result['n_successful']})")
            print(f"         Results may be unreliable!")
        # Still return what we have, but mark as partial success

    # Compute statistics for each parameter
    if len(flux_samples) > 0:
        result['flux'] = compute_bootstrap_stats(flux_samples, confidence_level)
        result['position']['ra'] = compute_bootstrap_stats(ra_samples, confidence_level)
        result['position']['dec'] = compute_bootstrap_stats(dec_samples, confidence_level)

        if comptype == "G":
            result['size']['bmaj'] = compute_bootstrap_stats(bmaj_samples, confidence_level)
            result['size']['bmin'] = compute_bootstrap_stats(bmin_samples, confidence_level)
            result['size']['pa'] = compute_bootstrap_stats(pa_samples, confidence_level)

        result['success'] = True

    # Step 5: Print summary
    if verbose and result['success']:
        print(f"\n{'='*70}")
        print(f"Bootstrap Results Summary")
        print(f"{'='*70}")
        print(f"\nFlux:")
        # Use Quantity for flux conversion
        flux_med_qty = result['flux']['median'] * original_result['flux']['quantity'].unit
        flux_lower_qty = result['flux']['lower'] * original_result['flux']['quantity'].unit
        flux_upper_qty = result['flux']['upper'] * original_result['flux']['quantity'].unit
        flux_std_qty = result['flux']['std'] * original_result['flux']['quantity'].unit

        flux_med_mjy = flux_med_qty.to(u.mJy)
        flux_lower_mjy = flux_lower_qty.to(u.mJy)
        flux_upper_mjy = flux_upper_qty.to(u.mJy)
        flux_std_mjy = flux_std_qty.to(u.mJy)

        print(f"  Median: {flux_med_mjy.value:.3f} mJy")
        print(f"  {confidence_level*100:.0f}% CI: +{flux_upper_mjy.value-flux_med_mjy.value:.3f} / -{flux_med_mjy.value-flux_lower_mjy.value:.3f} mJy")
        print(f"  Std: {flux_std_mjy.value:.3f} mJy")

        print(f"\nPosition:")
        ra_unit = result['position']['ra'].get('unit', 'arcsec')
        print(f"  RA:  {result['position']['ra']['median']:.6f} {ra_unit}")
        print(f"       +{result['position']['ra']['upper']-result['position']['ra']['median']:.6f} / "
              f"-{result['position']['ra']['median']-result['position']['ra']['lower']:.6f} {ra_unit}")
        dec_unit = result['position']['dec'].get('unit', 'arcsec')
        print(f"  Dec: {result['position']['dec']['median']:.6f} {dec_unit}")
        print(f"       +{result['position']['dec']['upper']-result['position']['dec']['median']:.6f} / "
              f"-{result['position']['dec']['median']-result['position']['dec']['lower']:.6f} {dec_unit}")

        if comptype == "G":
            print(f"\nSize:")
            size_unit = result['size']['bmaj'].get('unit', 'arcsec')
            print(f"  Major axis: {result['size']['bmaj']['median']:.6f} {size_unit}")
            print(f"               +{result['size']['bmaj']['upper']-result['size']['bmaj']['median']:.6f} / "
                  f"-{result['size']['bmaj']['median']-result['size']['bmaj']['lower']:.6f} {size_unit}")
            print(f"  Minor axis: {result['size']['bmin']['median']:.6f} {size_unit}")
            print(f"               +{result['size']['bmin']['upper']-result['size']['bmin']['median']:.6f} / "
                  f"-{result['size']['bmin']['median']-result['size']['bmin']['lower']:.6f} {size_unit}")
            pa_unit = result['size']['pa'].get('unit', 'deg')
            print(f"  PA: {result['size']['pa']['median']:.2f} {pa_unit}")
            print(f"       +{result['size']['pa']['upper']-result['size']['pa']['median']:.2f} / "
                  f"-{result['size']['pa']['median']-result['size']['pa']['lower']:.2f} {pa_unit}")

        print(f"\nComparison with original fit:")
        orig_flux_mjy = original_result['flux']['quantity'].to(u.mJy)
        print(f"  Original flux: {orig_flux_mjy.value:.3f} mJy")
        print(f"  Bootstrap median: {flux_med_mjy.value:.3f} mJy")
        print(f"  Difference: {flux_med_mjy.value-orig_flux_mjy.value:+.3f} mJy "
              f"({(flux_med_mjy.value/orig_flux_mjy.value-1)*100:+.1f}%)")

        print(f"\nFit quality:")
        print(f"  Success rate: {result['success_rate']*100:.1f}%")
        if result['success_rate'] < 0.8:
            print(f"  WARNING: Low success rate may indicate problems!")

        print(f"\nTemporary files: {temp_dir}")
        if not keep_temp_files:
            print(f"  (Temporary MS files have been cleaned up)")
        else:
            print(f"  (Temporary MS files preserved for debugging)")

        print(f"{'='*70}\n")

    return result

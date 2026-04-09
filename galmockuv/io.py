"""I/O utilities: config loading, directory setup, metadata I/O."""

import yaml
import json
from pathlib import Path
from astropy.table import Table
import numpy as np


def load_config(path):
    """Load a configuration file (TOML or YAML).

    Uses ``tomllib`` (Python 3.11+ stdlib) for ``.toml`` files and
    ``yaml.safe_load`` for ``.yaml`` / ``.yml`` files.

    Parameters
    ----------
    path : str or Path
        Path to the config file.

    Returns
    -------
    config : dict
        Parsed configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == '.toml':
        import tomllib
        with open(path, 'rb') as f:
            config = tomllib.load(f)
        config = _flatten_config(config)
    elif suffix in ('.yaml', '.yml'):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix} (use .toml or .yaml)")

    return config


def _flatten_config(d, parent_key='', sep='_'):
    """Flatten a nested dict by joining keys with ``sep``.

    E.g. ``{'source': {'redshift': 2.5}}`` becomes ``{'source_redshift': 2.5}``.

    Only dict values are recursed into; all other types are kept as-is.
    Top-level keys that are already strings (not dicts) are kept unchanged,
    so flat configs pass through unmodified.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def setup_output_dir(base_dir, source_id):
    """Create the output directory tree for one source.

    Parameters
    ----------
    base_dir : str or Path
        Base output directory (e.g. ``outputs/single``).
    source_id : str
        Source identifier.

    Returns
    -------
    output_dir : Path
        Path to the created source directory.
    """
    output_dir = Path(base_dir) / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_metadata(metadata, output_dir, filename='metadata.yaml'):
    """Save metadata dictionary to a YAML file.

    Parameters
    ----------
    metadata : dict
        Metadata to save.
    output_dir : str or Path
        Directory to save the file in.
    filename : str
        Output filename.
    """
    output_dir = Path(output_dir)
    outpath = output_dir / filename
    with open(outpath, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    return outpath


def load_metadata(output_dir, filename='metadata.yaml'):
    """Load metadata from a YAML file.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing the metadata file.
    filename : str
        Metadata filename.

    Returns
    -------
    metadata : dict
    """
    outpath = Path(output_dir) / filename
    with open(outpath, 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata


def save_json(data, output_dir, filename):
    """Save a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
    output_dir : str or Path
    filename : str

    Returns
    -------
    Path
        Path to the saved file.
    """
    outpath = Path(output_dir) / filename
    with open(outpath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    return outpath


def load_json(output_dir, filename):
    """Load a dictionary from a JSON file.

    Parameters
    ----------
    output_dir : str or Path
    filename : str

    Returns
    -------
    dict
    """
    outpath = Path(output_dir) / filename
    with open(outpath, 'r') as f:
        return json.load(f)


def collect_batch_results(batch_dir):
    """Scan per-source JSON measurement files and assemble a master catalog.

    Parameters
    ----------
    batch_dir : str or Path
        Directory containing per-source subdirectories.

    Returns
    -------
    catalog : astropy.table.Table
        Master catalog with one row per source.
    """
    batch_dir = Path(batch_dir)
    rows = []
    for source_dir in sorted(batch_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        meas_file = source_dir / 'measurements.json'
        if not meas_file.exists():
            continue
        data = load_json(source_dir, 'measurements.json')
        # Flatten metadata + measurements into one row
        row = {}
        meta_file = source_dir / 'metadata.yaml'
        if meta_file.exists():
            meta = load_metadata(source_dir, 'metadata.yaml')
            for k, v in meta.items():
                if k not in row:
                    row[k] = v
        row.update(data)
        rows.append(row)

    if not rows:
        return Table()

    catalog = Table(rows)
    return catalog


def make_source_id(prefix, index, params=None):
    """Generate a source ID string from index and optional key parameters.

    Parameters
    ----------
    prefix : str
    index : int
    params : dict, optional
        Key parameters to include in the ID.

    Returns
    -------
    str
    """
    sid = f"{prefix}_{index:03d}"
    return sid

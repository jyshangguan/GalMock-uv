"""Environment detection for CASA and DysmalPy."""


def is_casa_env():
    """Check if CASA tools are available."""
    try:
        import casatools
        return True
    except ImportError:
        return False


def is_dysmalpy_env():
    """Check if DysmalPy is available."""
    try:
        import dysmalpy
        return True
    except ImportError:
        return False


def get_env_type():
    """Return the current environment type.

    Returns
    -------
    str
        One of 'casa', 'dysmalpy', or 'unknown'.
    """
    if is_casa_env():
        return 'casa'
    if is_dysmalpy_env():
        return 'dysmalpy'
    return 'unknown'

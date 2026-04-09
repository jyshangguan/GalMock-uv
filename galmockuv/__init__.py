"""galmockuv: Mock ALMA CO observation pipeline for high-z galaxies."""

__version__ = '0.1.0'


def __getattr__(name):
    """Lazy imports for key functions."""
    if name in ('load_config', 'save_metadata', 'load_metadata', 'setup_output_dir'):
        from . import io
        return getattr(io, name)
    if name in ('run_pipeline',):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

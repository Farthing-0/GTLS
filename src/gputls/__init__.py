"""GTLS: GPU-accelerated Transit Least Squares.

By exoplanet lovers, for exoplanet lovers.
"""
from .main import gtls
from .results import gtlsResult
from .version import __version__

__all__ = ["gtls", "gtlsResult", "__version__"]

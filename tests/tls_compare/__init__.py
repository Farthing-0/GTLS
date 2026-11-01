"""Benchmark framework comparing GTLS with TLS 1.x and TLS 2.x on lightkurve-prepared
mock Kepler / TESS pipeline light curves. See README.md in this directory."""
from .compare import plot_periodograms, render_markdown, run_comparison
from .engines import DEFAULT_ENGINES, TLS1_VERSION, TLS2_VERSION, registry
from .scenarios import SCENARIOS, SUITES

__all__ = ["run_comparison", "render_markdown", "plot_periodograms", "registry",
           "DEFAULT_ENGINES", "TLS1_VERSION", "TLS2_VERSION", "SCENARIOS", "SUITES"]

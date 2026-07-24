"""Bayesian multi-exposure image fusion for robust HDR imaging."""

import importlib
from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "BayesianMEF",
    "ConventionalMEF",
    "LaunchMEF",
    "enable_x64",
    "set_fluxes",
    "set_threshold",
]

try:
    __version__ = version("bayes-mef")
except PackageNotFoundError:  # not installed, e.g. running from a source checkout
    __version__ = "0.0.0"

_LAZY = {
    "BayesianMEF": "algos",
    "ConventionalMEF": "algos",
    "LaunchMEF": "mef_launcher",
    "set_fluxes": "mef_launcher",
    "set_threshold": "mef_launcher",
    "enable_x64": "_compat",
}


def __getattr__(name):
    """Import submodules on first use, so `import bayes_mef` does not load JAX.

    Keeps the import cheap, and leaves a window for callers to configure JAX
    (precision, platform) before the backend initialises.
    """
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

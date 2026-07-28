"""Bayesian multi-exposure image fusion for robust HDR imaging."""

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Never executed. The public names are bound lazily below, which leaves type
    # checkers and editors with nothing to resolve, so "go to definition" on
    # `from bayes_mef import BayesianMEF` lands nowhere. These imports exist for
    # them alone; at runtime the lazy path is still the only one.
    from ._compat import enable_x64
    from .algos import BayesianMEF, ConventionalMEF
    from .mef_launcher import LaunchMEF
    from .utils import set_fluxes, set_threshold

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
    "set_fluxes": "utils",
    "set_threshold": "utils",
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

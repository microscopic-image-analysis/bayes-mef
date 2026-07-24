"""Bayesian multi-exposure image fusion for robust HDR imaging."""

import importlib
import os
import sys
from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "BayesianMEF",
    "ConventionalMEF",
    "LaunchMEF",
    "enable_x64",
    "set_cpu_devices",
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


def set_cpu_devices(n: int) -> None:
    """Expose `n` CPU devices to JAX so scan positions can be sharded across cores.

    JAX otherwise presents the whole host as a single device, which leaves most
    cores idle. Must be called before JAX initialises its backend -- i.e. before
    the first `import jax` or any bayes_mef class. Setting
    ``XLA_FLAGS=--xla_force_host_platform_device_count=<n>`` in the environment
    does the same thing and is the more reliable option for scripts.
    """
    if "jax" in sys.modules:
        raise RuntimeError(
            "jax is already imported, so its backend may be initialised and the "
            "device count can no longer change. Call set_cpu_devices() earlier, "
            "or set XLA_FLAGS=--xla_force_host_platform_device_count="
            f"{n} in the environment."
        )
    flag = f"--xla_force_host_platform_device_count={n}"
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()


def __getattr__(name):
    """Import submodules on first use so JAX is not loaded by `import bayes_mef`.

    This is what makes `set_cpu_devices` usable: it has to run before the JAX
    backend initialises.
    """
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)

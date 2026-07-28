# Bayesian MEF
[![PyPI](https://img.shields.io/pypi/v/bayes_mef)](https://pypi.org/project/bayes_mef/)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)
[![DOI](https://img.shields.io/badge/DOI-10.1364%2FOE.524284-blue.svg)](https://doi.org/10.1364/OE.524284)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-purple.svg)](https://opensource.org/licenses/BSD-3-Clause)

Bayesian multi-exposure image fusion (MEF) is a general-purpose algorithm for robust
high dynamic range (HDR) imaging under low SNR or varying illumination — in particular
for phase retrieval in coherent diffractive imaging. The paper,
["Bayesian multi-exposure image fusion for robust high dynamic range ptychography"](https://doi.org/10.1364/OE.524284),
details the method and its benefits for ptychography ([reproducing the results](#reproducing-results)).

![demo_mef](https://github.com/microscopic-image-analysis/bayes-mef/assets/64919085/d00a8c5e-5e53-4b7e-856b-381cc99523ba)

This small library is implemented in [JAX](https://docs.jax.dev), so the same code runs on CPU/GPU/TPU. Inputs may be NumPy arrays, Python lists or JAX arrays; results come back as NumPy arrays.

```bash
pip install bayes_mef
pip install "bayes_mef[cuda13]"  # GPU with CUDA 13; `cuda12` flag for older GPUs
```

After a GPU install, check the GPU is picked up with `bayes_mef check gpu`.

## Usage

A minimal example, simulating some data
([Colab](https://colab.research.google.com/github/microscopic-image-analysis/bayes-mef/blob/main/demo.ipynb)):

```python
from bayes_mef import BayesianMEF, ConventionalMEF
from skimage.data import camera
import numpy as np

truth = camera()
background = 60
times = np.array([0.1, 1, 10])  # exposure times / flux factors
threshold = 1500                # detector limit

# overexposed Poisson data from the image-formation model
data = [np.random.poisson(t * truth + background) for t in times]
data_saturated = np.clip(data, None, threshold, dtype="float")

mef_em = BayesianMEF(data_saturated, threshold, times, background)
mef_em.run(n_iter=100)
fused_em = mef_em.fused_image.copy()

# ConventionalMEF is the paper's MLE baseline, with the same interface
mef_mle = ConventionalMEF(data_saturated, threshold, times, background)
mef_mle.mle()
fused_mle = mef_mle.fused_image.copy()
```

### Precision

Single precision by default (accurate for censoring thresholds up to ~4096). For 16-bit detectors, switch to double precision *before* creating any array; a warning flags the risk otherwise:

```python
import bayes_mef
bayes_mef.enable_x64()  # or set JAX_ENABLE_X64=1
```

### When the exposures are not known

Omit `times` (and `threshold`) and they are estimated from the data, initialised from the non-saturated pixels (`flux_init="matched"`). The estimate is only a starting point, so **pass `update_fluxes=True` with it** and let EM refine it:

```python
mef_em = BayesianMEF(data_saturated, background=background, update_fluxes=True)
mef_em.run(n_iter=200)
```

`update_fluxes` defaults to False — `times` are used exactly as given, estimated or not — and estimating them without it warns. On the simulation above, the unrefined estimate correlates 0.16 with the truth against 0.989 once refined. `flux_init="uncensored"` selects the v0.1.9 initialiser, whose ratios compress under heavy censoring.

Once the background dominates the signal — the weak, low-SNR regime the [paper](https://doi.org/10.1364/OE.524284) targets — summed counts carry almost no information about the exposures, and it is EM's iterative background handling, not the initialiser, that recovers the range; a warning is raised when the estimate comes out nearly flat. Estimated fluxes are relative, so the fused image is on a relative scale. **Supply the real exposure times whenever you know them.**

### Fusing a 4D ptychogram dataset

For ptychography, we record multiple exposures per scan position. `LaunchMEF` fuses every scan position with a single vectorised program (CPU or GPU):

```python
from bayes_mef import LaunchMEF

launch_mef = LaunchMEF(
    ptychogram_stack,    # (n_exposures, n_scans, dp_x, dp_y)
    background,          # a number, one dark frame, or one per exposure
    times=None,          # None -> estimated from the data
    threshold=None,      # None -> estimated automatically
    update_fluxes=False, # True -> EM refines the fluxes; pair with estimated times
    flux_init="matched",
)

# returns fused patterns (n_scans, dp_x, dp_y) and the flux factors
fused_ptyem_stack, em_flux_factors = launch_mef.run_em(n_iter=150)

# or the conventional MLE baseline over the whole stack (just the fused patterns)
fused_ptymle_stack = launch_mef.run_mle()
```

Backgrounds are taken however they were recorded, on `LaunchMEF` and on the single stack classes alike: a scalar, one offset per exposure `(n_exposures,)`, a single dark frame `(dp_x, dp_y)`, one frame per exposure `(n_exposures, dp_x, dp_y)`, or one per image (the full stack shape). Each is given its exposure axis explicitly, so a per-exposure vector is never spread along the image columns, and a mismatched shape raises instead of broadcasting into something that quietly means the wrong thing.

The old `n_cpus` argument is accepted but ignored (it warns), since there are no worker processes to size any more.

Scans are fused in chunks sized to the device's free memory, so a stack larger than device memory still works. Set `batch_size` yourself if you hit a `MemoryError` or fuse alongside other work (results do not depend on it):

```python
fused, fluxes = launch_mef.run_em(n_iter, batch_size=8)
```

See [synthetic_mef.py](scripts/synthetic_mef.py) for detailed usage on synthetic
ptychography data, and [benchmarks/FINDINGS.md](benchmarks/FINDINGS.md) for a study
of when each method helps and for performance benchmarks.

## Reproducing results

To reproduce the ptychographic reconstructions from the paper:

1. Clone the repo:
   ```bash
   git clone https://github.com/microscopic-image-analysis/bayes-mef.git
   cd bayes-mef
   ```
2. Install the pinned dependencies with [uv](https://docs.astral.sh/uv/), then prefix
   commands with `uv run` (e.g. `uv run python scripts/synthetic_mef.py`):
   ```bash
   uv sync --locked --group scripts
   ```
   A `requirements.txt` is also exported for pip users (`pip install -r requirements.txt`).
3. Download the data from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10964222):
   ```bash
   ./download_data.sh
   ```
4. Optional: install [`cupy`](https://docs.cupy.dev/en/stable/install.html) for faster
   GPU reconstructions.
5. Run files from [scripts/](scripts) to plot the results.

## Citation
If this algorithm or the publication was useful, please cite:
```tex
@article{Kodgirwar:24,
author = {Shantanu Kodgirwar and Lars Loetgering and Chang Liu and Aleena Joseph and Leona Licht and Daniel S. Penagos Molina and Wilhelm Eschen and Jan Rothhardt and Michael Habeck},
journal = {Opt. Express},
number = {16},
pages = {28090--28099},
publisher = {Optica Publishing Group},
title = {Bayesian multi-exposure image fusion for robust high dynamic range ptychography},
volume = {32},
month = {Jul},
year = {2024},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-32-16-28090},
doi = {10.1364/OE.524284},
}
```

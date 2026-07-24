# Bayesian MEF
[![PyPI](https://img.shields.io/pypi/v/bayes_mef)](https://pypi.org/project/bayes_mef/)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)
[![DOI](https://zenodo.org/badge/769216428.svg)](https://zenodo.org/doi/10.5281/zenodo.11103003)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-purple.svg)](https://opensource.org/licenses/BSD-3-Clause)

Bayesian multi-exposure image fusion (MEF) is a general-purpose algorithm to achieve robust high dynamic range (HDR) imaging, particularly in scenarios with low signal-to-noise ratio (SNR) or variations in illumination intensity. This approach could especially be useful for high quality phase retrieval in coherent diffractive imaging (CDI). The algorithm, detailed in this publication ["Bayesian multi-exposure image fusion for robust high dynamic range ptychography"](https://doi.org/10.1364/OE.524284), explains the method and demonstrates the benefits for ptychography. To reproduce the results in the paper, see this [section](#reproducing-results). However, to get started, please install the package and check the demo usage below.

![demo_mef](https://github.com/microscopic-image-analysis/bayes-mef/assets/64919085/d00a8c5e-5e53-4b7e-856b-381cc99523ba)

The algorithms are implemented in [JAX](https://docs.jax.dev), so the same code runs on CPU or GPU. Inputs may be NumPy arrays, Python lists or JAX arrays; results always come back as NumPy arrays.

To install this package from PyPI,
```bash
pip install bayes_mef
```

For GPU (CUDA 12, Linux), install the extra — note it pulls in several GB of NVIDIA wheels:
```bash
pip install "bayes_mef[cuda12]" # Optionally `cuda13` for newer GPUs
```

## Usage

A minimal example demonstrating the usage of `BayesianMEF` by simulating some data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microscopic-image-analysis/bayes-mef/blob/main/demo.ipynb)

```python
from bayes_mef import BayesianMEF
from skimage.data import camera
import numpy as np

# simulation params
truth = camera()
background = 60  # some background
times = np.array([0.1, 1, 10])  # exposure times or equivalently flux factors
threshold = 1500 # detector limit

# poisson data based on image formation model that is overexposed
data = [np.random.poisson(time * truth + background) for time in times]
data_saturated = np.clip(data, None, threshold, dtype="float")

# Bayesian MEF with optional field `update_fluxes`. Set it to `True` when
# flux factors (exposure times) are not accurately known.
mef_em = BayesianMEF(data_saturated, threshold, times, background, update_fluxes=False)
mef_em.run(n_iter=100)
fused_em = mef_em.fused_image.copy()
```

### Precision

Computation runs in single precision by default, which is accurate for censoring
thresholds up to roughly 4096. Above that -- 16-bit detectors, for instance --
the censored-Poisson expectation loses accuracy for rates near the threshold, and
a warning is raised. Switch to double precision *before* creating any array:

```python
import bayes_mef
bayes_mef.enable_x64()   # or set JAX_ENABLE_X64=1 in the environment
```

Additionally, one can also use the `ConventionalMEF` method as given in the paper.

```python
from bayes_mef import ConventionalMEF
mef_mle = ConventionalMEF(data_saturated, threshold, times, background)
mef_mle.mle()
fused_mle = mef_mle.fused_image.copy()
```
### Fusing a whole ptychogram

In ptychography, one needs to fuse diffraction patterns for every scan position.
Processing this for data over all scan positions can be slow for an iterative
algorithm, so `LaunchMEF` fuses every scan position with a single vectorised
program. The same code runs on CPU or GPU.

```python
from bayes_mef import LaunchMEF

launch_mef = LaunchMEF(
    ptychogram_stack,    # ptychogram shape (n_exposures, n_scans, dp_x, dp_y)
    background,          # background shape (n_exposures, n_scans, dp_x, dp_y)
    times=None,          # if set to `None`, calculates automatically
    threshold=None,      # if set to `None`, calculates automatically
    update_fluxes=False, # set to `True` if you want to update fluxes
)

# Fuses every scan position with one vectorised program.
# Returns the fused diffraction patterns with shape (n_scans, dp_x, dp_y)
# and the updated flux factors.
n_iter = 150
fused_ptyem_stack, em_flux_factors = launch_mef.run_em(n_iter)
```

There are no worker processes to size any more, so the old `n_cpus` argument is
accepted but ignored (it warns, so existing scripts keep running).

#### Memory

Scan positions are fused in chunks, and only the current chunk is held on the
device, so a stack far larger than device memory still works. The chunk size is
chosen from the device's free memory, but you can set it yourself when fusing
alongside other work, or if you hit a `MemoryError`:

```python
fused, fluxes = launch_mef.run_em(n_iter, batch_size=8)
```

Lower `batch_size` reduces peak memory; higher values are faster up to the point
where the device is saturated. Results do not depend on it.

### Performance

Fusing 256 scan positions of 6 exposures at 256x256, 150 EM iterations, on a
64-core host with an NVIDIA H200 (`benchmarks/bench_mef.py`, single precision):

| backend                               | time    | speedup |
| ------------------------------------- | ------- | ------- |
| NumPy + joblib, 32 processes (v0.1.9) | 41.1 s  | 1.0x    |
| JAX, CPU                              | 17.4 s  | 2.4x    |
| JAX, H200 GPU                         |  3.5 s  | 11.8x   |

The gain comes from vectorising the whole scan stack into one compiled program:
plain JAX on CPU already beats 32 joblib worker processes, which is why there is
no longer a separate parallel implementation to choose between.

For a detailed usage, please check [synthetic_mef.py](scripts/synthetic_mef.py) that uses synthetic ptychography data.

## Reproducing results

To reproduce the ptychographic reconstruction results from the paper, please install the package in development mode and download the necessary data. This is explained in the following steps:

1. Please clone this repository.
   ```bash
   git clone https://github.com/microscopic-image-analysis/bayes-mef.git
   cd bayes-mef
   ```
2. Install the *pinned* dependencies with [uv](https://docs.astral.sh/uv/), which
   provisions the right Python version and environment from `uv.lock`.
   ```bash
   uv sync --locked --group scripts
   ```
   Then prefix commands with `uv run`, e.g. `uv run python scripts/synthetic_mef.py`.
> [!NOTE]
> This project is managed with uv. A `requirements.txt` is still exported from
> `uv.lock` for pip users: `pip install -r requirements.txt`.

3. Download the data from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10964222) with the following command:
   ```bash
   ./download_data.sh
   ```

4. Optional: For faster ptychographic reconstructions using GPU, please install `cupy` as given under its [installation guide](https://docs.cupy.dev/en/stable/install.html).

5. Run files from the [scripts/](scripts) directory for plotting the results.

## Citation
If you found this algorithm or the publication useful, please cite us at:
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



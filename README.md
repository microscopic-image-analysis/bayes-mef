# Bayesian MEF
[![PyPI](https://img.shields.io/pypi/v/bayes_mef)](https://pypi.org/project/bayes_mef/)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10809893.svg)](https://doi.org/10.5281/zenodo.10809893)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-purple.svg)](https://opensource.org/licenses/BSD-3-Clause)

Bayesian multi-exposure image fusion (MEF) is a general-purpose algorithm to achieve robust high dynamic range (HDR) imaging, particularly in scenarios with low signal-to-noise ratio (SNR) or variations in illumination intensity. This approach is especially crucial for high quality phase retrieval in coherent diffractive imaging (CDI). The algorithm, detailed in the [arXiv preprint](https://arxiv.org/abs/2403.11344), primarily focuses on its implementation and demonstrates the benefits in ptychography. 

![demo_mef](https://github.com/microscopic-image-analysis/bayes-mef/assets/64919085/d00a8c5e-5e53-4b7e-856b-381cc99523ba)

To install the package and its dependencies, 
```bash
pip install bayes_mef
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

Additionally, one can also use the `ConventionalMEF` method as given in the paper.

```python
from bayes_mef import ConventionalMEF
mef_mle = ConventionalMEF(data_saturated, threshold, times)
mef_mle.mle()
fused_mle = mef_mle.fused_image.copy()
```
### Parallelized implementatation

In ptychography, one needs to fuse diffraction patterns for every scan position. Processing this for data over all scan positions can be slow for an iterative algorithm. Therefore, one can use the parallelized implementation for MEF `LaunchMEF`. Check the example below

```python
from bayes_mef import LaunchMEF

launch_mef = LaunchMEF(
    ptychogram_stack,    # ptychogram shape (n_exposures, n_scans, dp_x, dp_y)
    background,          # background shape (n_exposures, n_scans, dp_x, dp_y)
    flux_factors=None,   # if set to `None`, calculates automatically
    threshold=None,      # if set to `None`, calculates automatically
    update_fluxes=False, # set to `True` if you want to update fluxes
)

# runs Bayesian MEF in parallel by defining the number of CPUs `n_cpus`;
# returns the fused diffraction patterns with shape (n_scans, dp_x, dp_y) and updated flux factors
n_cpus = 20
n_iter = 150
fused_ptyem_stack, em_flux_factors = launch_mef.run_em(n_iter, n_cpus)
```

For a detailed usage, please check [synthetic_mef.py](scripts/synthetic_mef.py) that uses synthetic ptychography data.

## Publication results

Ptychography data used for the publication results can be found at [Zenodo](https://doi.org/10.5281/zenodo.10809893). It also includes the code which is in this repository. Therefore, to replicate the publication results, please follow the below steps:
1. Download and unzip the dataset and the code from [Zenodo](https://doi.org/10.5281/zenodo.10809893).
2. Create a virtual environment and install the dependencies as
   
   ```bash
   cd bayes-mef
   conda create --name bayes-mef-venv python=3.10.13 # or python version satisfying ">=3.9, <3.12" 
   conda activate bayes-mef-venv
   pip install -e .
   ```
3. An additional dependency of `ptylab` is required for processing ptychography data
   
   ```bash
   pip install git+https://github.com/PtyLab/PtyLab.py.git@main
   ```
4. Please run python scripts under [scripts/](scripts) directory to replicate results in the publication.
5. Optional: For faster ptychographic reconstructions using GPU, please install `cupy` as given under its [installation guide](https://docs.cupy.dev/en/stable/install.html).

## Citation
If you found this algorithm or the publication useful, please cite us at:
```tex
@misc{Kodgirwar:24,
      title={Bayesian multi-exposure image fusion for robust high dynamic range ptychography}, 
      author={Shantanu Kodgirwar and Lars Loetgering and Chang Liu and Aleena Joseph and Leona Licht
              and Daniel S. Penagos Molina and Wilhelm Eschen and Jan Rothhardt and Michael Habeck},
      year={2024},
      eprint={2403.11344},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```



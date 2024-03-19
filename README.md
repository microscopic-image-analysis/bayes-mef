# Bayesian MEF
[![PyPI](https://img.shields.io/pypi/v/bayes_mef)](https://pypi.org/project/bayes_mef/)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10809893.svg)](https://doi.org/10.5281/zenodo.10809893)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-purple.svg)](https://opensource.org/licenses/BSD-3-Clause)

Bayesian multi-exposure image fusion (MEF) is a general purpose MEF algorithm suitable for any imaging scheme requiring high dynamic range (HDR) treatment. The details on the algorithm and its implementation in the context of ptychography can be found in this [arXiv preprint](https://arxiv.org/abs/2403.11344) version. 

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
fused_im = mef_em.fused_image.copy()
```

Under [scripts/](scripts) directory, MEF with ptychographic data and subsequent reconstructions used in the publication can be tested using the data uploaded at [Zenodo](https://doi.org/10.5281/zenodo.10809893). These are based on the package `ptylab` that can be installed additionally.

```bash
pip install git+https://github.com/PtyLab/PtyLab.py.git@main
```

If you found this algorithm or the publication useful, please cite us at:
```bash
@misc{Kodgirwar:24,
      title={Bayesian multi-exposure image fusion for robust high dynamic range ptychography}, 
      author={Shantanu Kodgirwar and Lars Loetgering and Chang Liu and Aleena Joseph and Leona Licht and Daniel S. Penagos Molina and Wilhelm Eschen and Jan Rothhardt and Michael Habeck},
      year={2024},
      eprint={2403.11344},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

For faster reconstructions using GPU, please install `cupy` as given under its [installation guide](https://docs.cupy.dev/en/stable/install.html).



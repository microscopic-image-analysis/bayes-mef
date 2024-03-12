# Bayesian MEF

Bayesian multi-exposure image fusion (MEF) is a general purpose MEF algorithm suitable for any imaging scheme requiring high dynamic range (HDR) treatment. Implementation of the algorithm in the context of ptychography has been published as "Bayesian multi-exposure image fusion for robust high dynamic range preprocessing in ptychography".

![demo_mef](https://github.com/microscopic-image-analysis/bayes-mef/assets/64919085/44f06535-7eac-47a0-b6db-62ef40456993)


To install the package and its dependencies, 
```bash
pip install bayes_mef
```

## Usage

A minimal example demonstrating the usage of `BayesianMEF` by simulating some data.
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

Under [scripts/](scripts) directory, MEF with ptychographic data and subsequent reconstructions used in the publication can be tested. These are based on the package `ptylab` that can be installed additionally.

```bash
pip install git+https://github.com/PtyLab/PtyLab.py.git@main
```

For faster reconstructions using GPU, please install `cupy` as given under its [installation guide](https://docs.cupy.dev/en/stable/install.html).



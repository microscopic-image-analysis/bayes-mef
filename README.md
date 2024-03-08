# Bayesian MEF

Implementation of the algorithm referred to as "Bayesian MEF" under publication "Bayesian multi-exposure image fusion for robust high dynamic range preprocessing in ptychography" 


Clone the repository and go to the root folder

```bash
git clone https://github.com/microscopic-image-analysis/bayes-mef.git
cd bayes-mef
```

Installation of dependencies in a separate environment is easiest with `conda`

```bash
conda create --name bayes-mef-venv python=3.10.13 # or python version satisfying ">=3.9, <3.12" 
conda activate bayes-mef-venv
pip install -e .
```

For faster ptychographic reconstructions with GPU, install `cupy` as

```bash
conda install -c conda-forge cupy
```
# This file tests recorded experimental data for MEF. The Bayesian MEF corrects the flux factors for every
# scan position, averages them over all scans and reruns the procedure with these fixed corrected flux factors.

import os

import h5py
import numpy as np

from bayes_mef.mef_launcher import LaunchMEF, set_fluxes, set_threshold


def load_data(filepath, pty_key="ptychogram"):
    with h5py.File(filepath, "r") as hf:
        ptychogram = np.asarray(hf[pty_key])

    return ptychogram


def return_params(loadpath):
    params = {}
    with h5py.File(loadpath, "r") as hf:
        for key in hf.keys():
            params[key] = hf[key][()]

    return params


if __name__ == "__main__":
    # set the main data path
    PATH = "data/experimental/"

    # Runs Bayesian MEF by updating flux factors as well
    update_fluxes = True

    # add the paths
    pty = []
    bg = []
    for i in range(3):
        for j in range(3):
            # acquisitions
            acq_filename = f"acq{i+1}_mea{j+1}.hdf5"
            load_path = os.path.join(PATH, f"acq{i+1}/{acq_filename}")

            # background
            bg_filename = f"bg_acq{i+1}_mea{j+1}.hdf5"
            bg_path = os.path.join(PATH, f"bg_acq{i+1}/{bg_filename}")

            pty.append(load_data(load_path))
            bg.append(load_data(bg_path))

    pty = np.array(pty)
    bg = np.array(bg)

    # set censoring threshold and flux factors
    threshold = set_threshold(pty)
    fluxes = set_fluxes(pty, threshold)

    print(f"Fluxes: {fluxes}")
    print(f"Threshold: {threshold}")

    # params for Parallel EM method
    em_niter = 200  # n_iter for Full EM
    ncpus = 35  # parallel computation

    # instantiate the MEF launcher, letting it estimate
    # times and threshold internally
    launch_mef = LaunchMEF(
        pty,
        np.mean(bg, axis=1),
        times=fluxes,
        threshold=threshold,
        prior=None,
        update_fluxes=update_fluxes,
    )

    # extract experimental params from one of the preprocessed file
    params = return_params(load_path)

    # runs MLE and EM in parallel and saves the fused ptychogram for all scans
    savename = "fused"
    savepath = os.path.join(PATH, savename)
    os.makedirs(savepath, exist_ok=True)
    launch_mef.run_and_save(
        savepath,
        savename,
        params,
        pty_key="ptychogram",
        mle_switch=True,
        em_switch=True,
        n_iter_em=em_niter,
        ncpus_em=ncpus,
    )

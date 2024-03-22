# This generates synthetic ptychogram data and saves the em and mle fusion results for every varying rho.
# however, this is only fast with parallel implementation over many CPU cores. Instead of running this file,
# one can also download the synthetic fusion results that consists of the data as well.
import os

import numpy as np

from bayes_mef.mef_launcher import LaunchMEF
from scripts.synthetic_data_generation import save_ground_truth, simulate_ptychogram


def main(PATH, rho=0.5):
    """run MEF for every value of beta and setting remaining parameters such as censoring threshold, flux factors"""

    print(f"Running MEF for scattering parameter: {rho}")

    # set main directory
    os.makedirs(PATH, exist_ok=True)

    # set threshold
    threshold = 2**11

    # set fluxes (exposure times)
    exposures = np.repeat(np.array([7, 10, 13]), 6)  # every exposure repeated 6 times
    flux_factors = [2**bit for bit in exposures]

    # remaining params
    scan_radius = 100
    n_scans = 200

    # simulates ptychogram data based on the forward model
    simdata = simulate_ptychogram(
        flux_factors,
        threshold,
        verbose=False,
        radius=scan_radius,
        show_input=False,
        phase_object=True,
        a=rho,
        return_object=False,
        n_scans=n_scans,
    )

    # data and background same dimension
    ptychogram_noisefree = simdata["ptychogram_noisefree"]
    ptychogram_noisy = simdata["ptychogram_noisy"]
    background = simdata["background"]

    # Parallel EM by setting the no. of CPU cores based on your computer
    ncpus = 35  # parallel computation

    # save noisefree data
    noisefree_filename = f"truth/truth_synthetic_rho{rho}.hdf5"
    filepath_noisfree = os.path.join(PATH, noisefree_filename)
    os.makedirs(os.path.dirname(filepath_noisfree), exist_ok=True)

    if not os.path.exists(f"{filepath_noisfree}"):
        save_ground_truth(simdata, filepath_noisfree)

    # instantiate the MEF launcher (flux factors not updated)
    launch_mef = LaunchMEF(
        ptychogram_noisy,
        background,
        flux_factors,
        threshold,
        prior=None,
        pty_noisefree_stack=ptychogram_noisefree,
        update_fluxes=False,
    )

    # run and save the fused ptychogram for all the methods
    launch_mef.run_and_save(
        PATH,
        filename=f"synthetic_rho{rho}",
        params=simdata,
        pty_key="ptychogram",
        mle_switch=True,
        em_switch=True,
        n_iter_em=200,
        ncpus_em=ncpus,
        save_misc=True,
        separate_fused_dir=True,
    )


if __name__ == "__main__":

    # fuses data for every rho
    run_all_rhos = False
    PATH = "data/synthetic/"

    if run_all_rhos:
        rhos = np.round(np.arange(0.3, 1.1, 0.1), decimals=1)

        [main(PATH, rho) for rho in rhos]
    else:
        rho = 0.4
        main(PATH, rho)

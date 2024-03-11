# synthetic reconstruction for data with background based on a specific beta
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PtyLab import (
    Engines,
    ExperimentalData,
    Monitor,
    Params,
    Reconstruction,
    easyInitialize,
)


def save_reconstruction(
    reconstruction: Reconstruction,
    monitor: Monitor,
    PATH: str,
    filename: str,
    method: str = None,
    save_filename: str = None,
) -> None:
    """
    Uses internal reconstruction state to store the result as well as png result,
    kind of convenient because when there are more than 1 mode (npsm > 1) in probe,
    it is a multidimesional complex array.
    """

    # some checks before saving
    assert isinstance(reconstruction, Reconstruction)
    assert isinstance(monitor, Monitor)

    # define the `reconstruction` directory
    if method is not None:
        savedir = os.path.join(PATH, f"reconstructed/{method}")
    else:
        savedir = os.path.join(PATH, "reconstructed")

    # create directory to save
    os.makedirs(savedir, exist_ok=True)

    # if true, create a subdirectory inside 'reconstructed' and save the results there.
    if save_filename is not None:
        savefile_hdf5 = save_filename
    else:
        savefile_hdf5 = filename.split(".hdf5")[0] + "_recon.hdf5"

    # save the result as hdf5
    writepath_hdf5 = os.path.join(savedir, savefile_hdf5)
    reconstruction.saveResults(writepath_hdf5)

    print(f"saving the reconstruction result as {savefile_hdf5}")


def data_loader(filepath, pty_key="ptychogram") -> ExperimentalData:
    """
    Loads data from the given file and returns an instance of
    `ExperimentalData` class
    """

    # loading filepath
    print(f"Loading file {filepath} for reconstruction")

    # instance of `ExperimentalData` class
    experimentalData = ExperimentalData(operationMode="CPM")

    # experimental data class
    with h5py.File(filepath, "r") as hf:
        experimentalData.ptychogram = np.array(hf[pty_key])
        experimentalData.wavelength = hf["wavelength"][()]
        experimentalData.encoder = np.array(hf["encoder"])
        experimentalData.dxd = hf["dxd"][()]
        experimentalData.zo = hf["zo"][()]
        experimentalData.entrancePupilDiameter = hf["entrancePupilDiameter"][()]

    # experimentalData.ptychogram /= experimentalData.ptychogram
    experimentalData.spectralDensity = None
    experimentalData.theta = None
    experimentalData._setData()

    return experimentalData


def synthetic_recon_params(filepath, pty_key):

    # load the data using the `ExperimentalData` class
    experimentalData = data_loader(filepath, pty_key)

    # initialize the `Params` class
    params = Params()

    # main parameters for reconstruction
    params.intensityConstraint = "standard"  # thin single-layer object
    params.positionOrder = "random"  # 'sequential' or 'random'
    params.propagatorType = "Fraunhofer"  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP

    params.probePowerCorrectionSwitch = True  # probe normalization to measured PSD
    params.comStabilizationSwitch = False  # center of mass stabilization for probe
    params.orthogonalizationSwitch = False  # probe orthogonalization
    params.absorbingProbeBoundary = True  # if probe has periodic boundary conditions

    params.absObjectSwitch = True  # force the object to be abs-only
    params.absObjectBeta = 1e-2  # relaxation parameter for abs-only

    # standard fluctuation exponential poission
    # couple adjacent wavelengths (relaxation parameter)
    params.couplingAleph = 1

    # initialize the reconstruction class
    reconstruction = Reconstruction(experimentalData, params)

    return experimentalData, params, reconstruction


def experimental_recon_params(filepath):
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(
        filepath, operationMode="CPM"
    )

    reconstruction.No = np.array([4000, 4000])
    reconstruction.copyAttributesFromExperiment(experimentalData)

    reconstruction.computeParameters()
    experimentalData.entrancePupilDiameter = 600e-6
    # main parameters for reconstruction
    params.intensityConstraint = "standard"  # thin single-layer object
    params.positionOrder = "random"  # 'sequential' or 'random'
    params.propagatorType = "Fraunhofer"  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP

    params.probePowerCorrectionSwitch = True
    params.comStabilizationSwitch = True
    params.orthogonalizationSwitch = True
    params.absorbingProbeBoundary = True

    return experimentalData, params, reconstruction, monitor


def set_recon_parameters(
    filepath,
    seedpath=None,
    pty_key="ptychogram",
    gpu_switch=True,
    data_source="synthetic",
):
    """providing all the necessary parameters for reconstruction with all necessary params"""

    if data_source == "synthetic":
        experimentalData, params, reconstruction = synthetic_recon_params(
            filepath, pty_key
        )
        # initialize the `Monitor` class
        monitor = Monitor()
    elif data_source == "experimental":
        experimentalData, params, reconstruction, monitor = experimental_recon_params(
            filepath
        )
    else:
        print("Needs to be either synthetic or experimental!")
        raise KeyError

    # remaining parms common for both the data types
    params.gpuSwitch = gpu_switch  # whether to run on a CPU or GPU

    # create an object to hold everything we're eventually interested in
    reconstruction.npsm = 1  # Number of probe modes to reconstruct
    reconstruction.nosm = 1  # Number of object modes to reconstruct
    reconstruction.nlambda = 1  # Number of mixed wavelengths
    reconstruction.nslice = 1  # Number of object slice

    # initialize the reconstruction
    init_probe = "circ" if seedpath is None else "recon"

    reconstruction.initialProbe = init_probe  # initialize probe
    reconstruction.initialProbe_filename = seedpath
    reconstruction.initialObject = "ones"  # initialize params
    reconstruction.initializeObjectProbe()

    # monitor the reconstruction by setting these properties
    monitor.figureUpdateFrequency = 20  # frequency of reconstruction monitor
    monitor.objectPlot = "complex"  # complex abs angle
    monitor.verboseLevel = "low"  # high: plot two figures, low: plot only one figure
    monitor.objectZoom = 2  # control object plot FoV
    monitor.probeZoom = 0.5  # control probe plot FoV

    recon_params = {
        "experimentalData": experimentalData,
        "reconstruction": reconstruction,
        "params": params,
        "monitor": monitor,
    }

    return recon_params


def launch_recon(
    PATH,
    filename,
    save_recon=False,
    seedpath=None,
    seed_switch=False,
    gpu_switch=True,
    data_source="synthetic",
):
    """Launches reconstruction for multiple files"""
    plt.close("all")

    mPIE_niter = 150
    savename = filename.split(".hdf5")[0] + "_recon.hdf5"

    # load params
    filepath = os.path.join(PATH, filename)

    seedpath = seedpath if seed_switch else None
    recon_params = set_recon_parameters(
        filepath, gpu_switch=gpu_switch, data_source=data_source
    )
    experimentalData = recon_params["experimentalData"]
    reconstruction = recon_params["reconstruction"]
    params = recon_params["params"]
    monitor = recon_params["monitor"]

    # start mPIE for reconstruction
    mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
    mPIE.numIterations = mPIE_niter
    print(
        f"Using {mPIE.numIterations} iterations of {Engines.mPIE.__name__} for reconstruction"
    )
    mPIE.betaProbe = 0.25
    mPIE.betaObject = 0.25
    mPIE.reconstruct()

    # save the reconstruction result (object, probe, error) as hdf5 and png
    if save_recon:
        save_reconstruction(
            reconstruction, monitor, PATH, filename, save_filename=savename
        )


def recon_synthetic_all_rhos(PATH):

    # runs and saves reconstructions for MEF methods and ground
    # truth corresponding to all rhos (24 reconstructions).
    # CATION: runs every reconstruction serially, and takes long

    # rhos varying from 0.3 to 1.0
    rhos = np.round(np.arange(0.3, 1.1, 0.1), decimals=1)

    for rho in rhos:
        for method in ["noisefree", "mle", "em"]:

            if method == "noisefree":
                savepath = os.path.join(PATH, "truth/reconstructed")
            elif method == "mle":
                savepath = os.path.join(PATH, "mle_fused/reconstructed")
            elif method == "em":
                savepath = os.path.join(PATH, "em_fused/reconstructed")
            else:
                print("Provide relevant method")

            # fused filename
            recon_path = os.path.join(
                savepath,
                f"{method}_synthetic_rho{rho:.1f}_recon.hdf5",
            )
            if os.path.exists(recon_path):
                print(f"{recon_path} exists, moving to the next\n")
                continue

            # launch recon
            filename = f"{method}_synthetic_rho{rho}.hdf5"
            launch_recon(PATH, filename, save_recon=True)


if __name__ == "__main__":

    synthetic_datapath = "data/synthetic/"

    # runs recontruction for a single value of rho
    recon_synthetic = False
    rho = 1.0
    if recon_synthetic:
        # select one of the three filepaths for reconstruction
        filename = [
            f"mle_fused/mle_synthetic_rho{rho}.hdf5",
            f"em_fused/em_synthetic_rho{rho}.hdf5",
            f"truth/truth_synthetic_rho{rho}.hdf5",
        ][1]

        # reconstruction using mPIE and saves the result if `save_recon=True`
        launch_recon(synthetic_datapath, filename, save_recon=False)

    # CAUTION: runs and saves all reconstructions for synthetic data with
    # varying rhos, slow to run!
    switch_all_rhos = False
    if switch_all_rhos:
        recon_synthetic_all_rhos(synthetic_datapath)

    # experimental data reconstruction (sets some parameters differently)
    recon_experimental_data = False
    experimental_datapath = "data/experimental/fused"
    fused_name = "em_fused.hdf5"
    launch_recon(
        experimental_datapath, fused_name, save_recon=False, data_source="experimental"
    )

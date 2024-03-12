# This file simulates raw (noisy) data for MEF tests
import h5py
import matplotlib.pylab as plt
import numpy as np
from PtyLab.Operators.Operators import aspw
from PtyLab.utils.scanGrids import GenerateNonUniformFermat
from PtyLab.utils.utils import cart2pol, circ, fft2c, gaussian2D
from PtyLab.utils.visualisation import hsvplot
from scipy.signal import convolve2d


class PhysicalProps:
    def __init__(self) -> None:
        # detector coordinates
        self.wavelength = 632.8e-9  # wavelength
        self.zo = 5e-2  # object-detector distance
        self.nlambda = 1  # single-wavelength
        self.npsm = 1
        self.nosm = 1
        self.nslice = 1
        self.binningFactor = 1
        self.Nd = 2**8  # no. of samples in detector plane
        # detector pixel size (equivalent to 8-fold binning)
        self.dxd = 2**11 / self.Nd * 4.5e-6
        self.Ld = self.Nd * self.dxd  # length of the detector

        # probe/pupil coordinates
        self.dxp = self.wavelength * self.zo / self.Ld  # probe sampling step size
        # number of samples in probe field of view
        self.Np = self.Nd
        # field of view in pinhole plane
        self.Lp = self.dxp * self.Np
        self.xp = (
            np.arange(-self.Np // 2, self.Np // 2) * self.dxp
        )  # 1D coordinates (probe)
        # 2D coordinates (probe)
        self.Xp, self.Yp = np.meshgrid(self.xp, self.xp)
        self.zp = 1e-2  # pinhole-object distance

        # object coordinates
        self.No = 2**10  # no. of samples in object plane
        self.dxo = self.dxp  # object pixel size
        self.Lo = self.dxo * self.No  # object field of view
        self.xo = (
            np.arange(-self.No // 2, self.No // 2) * self.dxo
        )  # 1D coordinates (object)
        # 2D coordinates (object)
        self.Xo, self.Yo = np.meshgrid(self.xo, self.xo)


def generate_illumination():
    """
    Simulates focused beam passing through an aperture that is used to scan
    the object.
    """
    props = PhysicalProps()

    # goal: 1:1 image iris through (low-NA) lens with focal length f onto an object
    f = 10e-3  # focal length of lens, creating a focused probe

    pinhole = circ(props.Xp, props.Yp, props.Lp / 3)
    pinhole = convolve2d(pinhole, gaussian2D(5, 1).astype(np.float32), mode="same")

    # propagate to lens
    probe = aspw(pinhole, 2 * f, props.wavelength, props.Lp, is_FT=False)[0]

    # specify aperture (important setting!)
    blur = gaussian2D(5, 3).astype(np.float32)
    aperture = np.exp(
        -(props.Xp**2 + props.Yp**2) / (2 * (3 * props.Lp / 4 / 2.355) ** 2)
    )
    aperture = convolve2d(aperture, blur, mode="same")

    # probe modified as product of quadratic phase and aperture
    phase = np.exp(
        -1.0j * 2 * np.pi / props.wavelength * (props.Xp**2 + props.Yp**2) / (2 * f)
    )
    probe = probe * phase * aperture

    # repropagate the modified probe
    probe = aspw(probe, 2 * f, props.wavelength, props.Lp, is_FT=False)[0]

    return probe


def generate_object(d=1e-3, b=33, phase_object=False, a=None):
    """generates an object from analytical equations

    Args:
        d(float): The smaller this parameter the larger the spatial
                    frequencies
        b(int): Topological charge
    """
    # access object properties
    props = PhysicalProps()
    Xo, Yo = props.Xo, props.Yo
    Lo, dxo = props.Lo, props.dxo

    # special function to create our simulated object
    theta, rho = cart2pol(Xo, Yo)
    phaseFun = 1
    t0 = (1 + np.sign(np.sin(b * theta + 2 * np.pi * (rho / d) ** 2))) / 2
    t1 = t0 * circ(Xo, Yo, Lo) * (1 - circ(Xo, Yo, 160 * dxo)) * phaseFun
    t2 = circ(Xo, Yo, 150 * dxo)
    t3 = (
        np.abs(
            1 + np.exp(1.0j * 2 * np.pi / (props.wavelength * 1e-2) * (Xo**2 + Yo**2))
        )
        ** 2
    )
    t3 = 1 / 4 * t3 > 1 / 2

    # construct object for this function
    obj = t1 + t2 * t3
    obj = convolve2d(obj, gaussian2D(2, 1), mode="same")  # smooth edges

    if phase_object and a is not None:
        obj = np.exp(1j * a * obj)

    return obj


def scan_grid(n_scans=200, radius=100, return_grid=False, verbose=True):
    """
    generate Non uniform fermat grid and optimize path

    Args:
        numPoints (int, optional): number of points. Defaults to 200.
        radius (int, optional): radius of final scan grid. Defaults to 100.

    Returns:
        scan_props(dict): dictionary consisting important scanning properties
    """

    props = PhysicalProps()
    probe = generate_illumination()

    # generate positions based on nonuniform Fermat grid
    p = 1  # p = 1 is standard Fermat;  p > 1 yields more points towards the center of grid
    R, C = GenerateNonUniformFermat(n_scans, radius=radius, power=p)

    if return_grid:
        return R, C

    # optimizing scan grid
    encoder = np.vstack((R * props.dxo, C * props.dxo)).T

    # prevent negative indices by centering spiral coordinates on object
    positions = np.round(encoder / props.dxo)
    offset = np.array([50, 20])
    positions = (positions + props.No // 2 - props.Np // 2 + offset).astype(int)

    # get number of positions
    numFrames = len(R)

    # calculate estimated overlap
    # expected beam size, required to calculate overlap (expect Gaussian-like beam, derive from second moment)
    Xp, Yp = props.Xp, props.Yp
    beamSize = (
        np.sqrt(np.sum((Xp**2 + Yp**2) * np.abs(probe) ** 2) / np.sum(abs(probe) ** 2))
        * 2.355
    )
    distances = np.sqrt(np.diff(R) ** 2 + np.diff(C) ** 2) * props.dxo
    averageDistance = np.mean(distances) * 1e6

    if verbose:
        print("average step size: %.1f (um)" % averageDistance)
        print(f"probe diameter: {beamSize*1e6:.2f}")
        print("number of scan points: %d" % numFrames)

    scan_props = {
        "positions": positions,
        "numFrames": numFrames,
        "encoder": encoder,
        "beamSize": beamSize,
    }

    return scan_props


def object_with_probe(d=1e-3, b=33, n_scans=200, radius=100):
    """shows scan grid on object"""

    # get important params
    props = PhysicalProps()
    scan_props = scan_grid(n_scans, radius)
    positions = scan_props["positions"]
    beamSize = scan_props["beamSize"]

    # object that will be overlayed on grid
    obj = generate_object(d, b)

    # show scan grid on object
    plt.figure(figsize=(5, 5))
    ax1 = plt.axes()
    hsvplot(np.squeeze(obj), ax=ax1)

    pos_pix = positions + props.Np // 2
    dia_pix = beamSize / props.dxo
    ax1.plot(
        pos_pix[:, -1],
        pos_pix[:, -2],
        "ro",
        alpha=0.9,
    )
    ax1.set_xlim(pos_pix[:, 1].min() - 100, pos_pix[:, 1].max() + 100)
    ax1.set_ylim(pos_pix[:, 0].max() + 100, pos_pix[:, 0].min() - 100)
    # indicate the probe with the typical diameter
    for p in pos_pix:
        c = plt.Circle(p, radius=dia_pix / 2, color="black", fill=False, alpha=0.5)
        ax1.add_artist(c)
    ax1.set_title("object with probe positions")


def generate_ptychogram(scan_props: dict, obj):
    """simulates ptychogram by scanning the object with overlapping probe"""

    props = PhysicalProps()
    # obj = generate_object(d, b)
    probe = generate_illumination()

    # generating ptychogram
    numFrames = scan_props["numFrames"]
    positions = scan_props["positions"]

    ptychogram = np.zeros((numFrames, props.Nd, props.Nd))
    for i in range(numFrames):
        # get object patch
        row, col = positions[i]
        sy = slice(row, row + props.Np)
        sx = slice(col, col + props.Np)

        # note that object patch has size of probe array
        objectPatch = obj[..., sy, sx].copy()
        # multiply each probe mode with object patch
        esw = objectPatch * probe
        # generate diffraction data, propagate the esw to the detector plane
        ESW = fft2c(esw)

        # save data in ptychogram
        ptychogram[i] = abs(ESW) ** 2

    return ptychogram


def visualize_input(
    d=1e-3, b=33, n_scans=200, radius=100, show_grid=False, show_object=False
):
    props = PhysicalProps()

    # show object
    if show_object:
        obj = generate_object(d, b)
        plt.figure(figsize=(5, 5), num=2)
        ax = plt.axes()
        hsvplot(np.squeeze(obj), ax=ax)
        ax.set_title("object")

    # show probe
    probe = generate_illumination()
    plt.figure(figsize=(10, 5), num=1)
    ax1 = plt.subplot(121)
    hsvplot(probe, ax=ax1, pixelSize=props.dxp)
    ax1.set_title("complex probe")
    plt.subplot(122)
    plt.imshow(abs(probe) ** 2)
    plt.title("probe intensity")

    # show scan grid overlapped with object
    object_with_probe(d, b, n_scans, radius)

    # show scan grid
    if show_grid:
        R, C = scan_grid(n_scans, radius, return_grid=True, verbose=False)
        plt.figure(figsize=(5, 5), num=99)
        plt.plot(R, C, "o")
        plt.xlabel("um")
        plt.title("scan grid")

    plt.show()


def simulate_background(
    bg_shape: tuple,
    mean: float = 100.0,
    var: float = 0.8,
    seed: int = 123,
):
    """Generating background that is a certain fraction of the threshold."""

    rng = np.random.default_rng(seed)

    # a gaussian distribution around a certain fraction of threshold
    background = rng.normal(mean, var, size=bg_shape)

    return np.abs(background)


def simulate_ptychogram(
    times,
    threshold,
    scale=1,
    d=1e-3,
    b=33,
    n_scans=200,
    radius=100,
    verbose=True,
    show_input=False,
    seed=123,
    phase_object=False,
    a=None,
    return_object=False,
):
    """
    models data as poisson process (shot noise).
    `image = poisson(exp_time * (true_image + dark_current) + readout_noise)`

    This is later overexposed for the given threshold or detector bitdepth

    Args:
        times (list[float]): exposure times
        threshold (int): camera bit depth, all pixels above this value are saturated.
        maxNumCountsPerDiff (int, optional): Scale for noise free true signal (ptychogram). Defaults to 2**6
        baselevel (int, optional): Parameter that scales the background noise
        d(float): The smaller this parameter the larger the spatial frequencies
        b(int): Topological charge
        n_scans (int, optional): number of scan points. Defaults to 200.
        radius (int, optional): radius of final scan grid. Defaults to 100.
        show_input(bool): Shows input images that were created like scans, object, probe

    """
    # if seed is given, random generator fixed
    rng = np.random.default_rng(seed)

    props = PhysicalProps()
    scan_props = scan_grid(n_scans, radius, verbose=verbose)
    obj = generate_object(d, b, phase_object, a)

    # show visualization
    if show_input:
        visualize_input(d, b, n_scans, radius)

    # generate noisefree ptychogram
    ptychogram = generate_ptychogram(scan_props, obj)
    ptychogram /= ptychogram.max()
    pty_noisefree = ptychogram * scale

    # simulate background
    bg_shape = (len(times), *pty_noisefree.shape)
    background = simulate_background(bg_shape)

    # Poisson noise
    pty_noisy = np.zeros((len(times), *pty_noisefree.shape))
    for i, time in enumerate(times):
        # simulate data
        pty_noisy[i] = rng.poisson(pty_noisefree * time + background[i])

        # overexpose the data
        pty_noisy[i] = np.clip(pty_noisy[i], None, threshold)

    # create a dictionary to store result and other important fields
    result = {
        "ptychogram_noisy": pty_noisy,
        "exposures": times,
        "threshold": threshold,
        "background": background,
        "ptychogram_noisefree": pty_noisefree,
        "encoder": scan_props["encoder"],
        "binningFactor": props.binningFactor,
        "dxd": props.dxd,
        "Nd": props.Nd,
        "No": props.No,
        "zo": props.zo,
        "wavelength": props.wavelength,
        "entrancePupilDiameter": scan_props["beamSize"],
    }

    if return_object:
        return result, obj

    return result


def save_ground_truth(simdata: dict, filepath: str):
    """saving noisefree ptychogram for comparison"""

    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("ptychogram", data=simdata["ptychogram_noisefree"], dtype="f")
        hf.create_dataset("encoder", data=simdata["encoder"], dtype="f")
        hf.create_dataset("binningFactor", data=simdata["binningFactor"], dtype="i")
        hf.create_dataset("dxd", data=(simdata["dxd"],), dtype="f")
        hf.create_dataset("Nd", data=(simdata["Nd"],), dtype="i")
        hf.create_dataset("No", data=(simdata["No"],), dtype="i")
        hf.create_dataset("zo", data=(simdata["zo"],), dtype="f")
        hf.create_dataset("wavelength", data=(simdata["wavelength"],), dtype="f")
        hf.create_dataset(
            "entrancePupilDiameter", data=(simdata["entrancePupilDiameter"],), dtype="f"
        )
        hf.create_dataset("orientation", data=0)

    print(f"Saved noisfree ptychogram at {filepath}")

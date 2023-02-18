import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
from joblib import Parallel, delayed


def phase2frequency(
    phase_arrays, TEs, load_bar=False, keep_dims=True, return_fit_results=False, njobs=4
):
    """Fit frequency map into array of phase images.

    Does a pixel-wise linear fit through the echo images and requies an input
    array of the following shape:

        N x M x K    or    N x M x K x J

    with N echo images, that is len(TEs) == N, and M,K,J spatial dimensions.

    Parameters
    ----------
    phase_arrays : np.ndarray
        Three or four dimensional array with the first dimension being the
        number of echoes and two or three spatial dimensions.

    TEs : np.ndarray
        One dimensional array containing the echo times corresponding to the
        phase images.

    load_bar : bool, optional
        Wether or not a progress bar should be displayed. Default is False.

    keep_dims : bool, optional
        Only relevant for 2D phase images. Wether or not the spatial dimensions
        should be kept or filled up to three dimensions. Default is True.

    return_fit_results : bool, optional
        When True, returns all fit parameters m, m_err, t, t_err as direct fit-
        results from fitting to the equation f(x) = m * x + t. Default is False.

    njobs : int, optional
        The number of CPUs used to calculate the fmap. The pixel-wise fit
        process (which is based on scipy.optimize.curve_fit) is distributed to
        N=njobs CPUs using the joblib library.
        Use njobs=1 for a sequential execution and njobs=-1 to use all CPUs
        available.

    Returns
    ----------
    out : list (2 or 3 elements)
        A list containing two np.ndarrays. In case return_fit_results=True the
        list contains three array.

            out[0]  <-- frequency map [Hz]
            out[1]  <-- error map to frequency map [Hz]
          ( out[2]  <-- fit results containing m, m_err, t, t_err  )
    """
    assert len(phase_arrays) == len(TEs), "Echo dimension must be at axis=0."

    if phase_arrays[0].ndim == 2:
        phase_arrays = np.expand_dims(phase_arrays, axis=3)

    def linear_fit_func(x, m, t):
        return m * x + t

    _, x_dim, y_dim, z_dim = phase_arrays.shape

    # generate a list of all voxel indices
    index_list = list(np.ndindex((x_dim, y_dim, z_dim)))

    if load_bar:  # pragma: no cover
        index_list = tqdm(index_list, desc="Pixel-wise fitting", leave=True)

    def fit_to_voxel(idx):
        """Does a linear fit of a single voxel over time, that is, all TEs."""
        popt, pcov = curve_fit(linear_fit_func, TEs, phase_arrays[(..., *idx)])

        m = popt[0]
        m_err = np.sqrt(np.diag(pcov))[0]

        t = popt[1]
        t_err = np.sqrt(np.diag(pcov))[1]

        return m, m_err, t, t_err

    serial_results = Parallel(n_jobs=njobs)(
        delayed(fit_to_voxel)(i) for i in index_list
    )

    fit_results = np.array(serial_results).reshape(x_dim, y_dim, z_dim, 4)

    if keep_dims:
        # in case of 2D: remove slice dimension again
        fit_results = np.squeeze(fit_results)

    frequency_map_Hz = fit_results[..., 0] * 1e3 / (2 * np.pi)
    frequency_map_Hz_err = fit_results[..., 1] * 1e3 / (2 * np.pi)

    out = [frequency_map_Hz, frequency_map_Hz_err]

    if return_fit_results:
        out.append(fit_results)

    return out

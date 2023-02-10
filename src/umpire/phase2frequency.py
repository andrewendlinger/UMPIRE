import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import curve_fit


def phase2frequency(
    phase_arrays, TEs, load_bar=False, keep_dims=True, return_fit_results=False
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

    fit_results = np.zeros(
        (x_dim, y_dim, z_dim, 4)
    )  # last dim contains m, m_err, t, t_err

    x_range, y_range, z_range = range(x_dim), range(y_dim), range(z_dim)

    if load_bar:  # pragma: no cover
        x_range = tqdm(x_range, position=0, desc="Pixel-wise fitting", leave=True)
        # nested for loops are not displayed correctly
        # y_range = tqdm(y_range, position=1, desc="y-dim", leave=True)
        # z_range = tqdm(z_range, position=2, desc="z-dim", leave=True)

    for x in x_range:
        for y in y_range:
            for z in z_range:
                popt, pcov = curve_fit(linear_fit_func, TEs, phase_arrays[:, x, y, z])

                m = popt[0]
                m_err = np.sqrt(np.diag(pcov))[0]

                t = popt[1]
                t_err = np.sqrt(np.diag(pcov))[1]

                fit_results[x, y, z] = m, m_err, t, t_err

    if keep_dims:
        # in case of 2D: remove slice dimension again
        fit_results = np.squeeze(fit_results)

    frequency_map_Hz = fit_results[..., 0] * 1e3 / (2 * np.pi)
    frequency_map_Hz_err = fit_results[..., 1] * 1e3 / (2 * np.pi)

    out = [frequency_map_Hz, frequency_map_Hz_err]

    if return_fit_results:
        out.append(fit_results)

    return out

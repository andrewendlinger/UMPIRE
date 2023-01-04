import numpy as np
from scipy.optimize import curve_fit

def fit_frequency(phase_arrays, TEs, load_bar=False):

    for arr in phase_arrays:
        dimension = len(arr.shape)
        if dimension not in (2, 3):
            raise AttributeError(
                f"Only 2D and 3D arrays allowed. {dimension} dimension(s) found instead."
            )
        if arr.shape != phase_arrays[0].shape:
            raise AttributeError(
                f"Array shapes do not match. {arr.shape} != {arr[0].shape}"
            )

    if dimension == 2:
        examination_array_tmp = np.stack(phase_arrays, axis=2)
        examination_array = np.expand_dims(examination_array_tmp, axis=3)

    else:  # dimension == 3
        examination_array = np.stack(phase_arrays, axis=2)

    def linear_fit_func(x, m, t):
        return m * x + t

    x_dim, y_dim, TE_dim, slice_dim = examination_array.shape

    m_phase = np.zeros(
        (x_dim, y_dim, slice_dim, 4)
    )  # last dim contains m, m_err, t, t_err

    if load_bar:
        iter_var = tqdm(range(slice_dim))
    else:
        iter_var = range(slice_dim)

    for i in iter_var:
        for x in range(x_dim):
            for y in range(y_dim):
                popt, pcov = curve_fit(
                    linear_fit_func, TEs, examination_array[x, y, :, i]
                )

                m = popt[0]
                m_err = np.sqrt(np.diag(pcov))[0]

                t = popt[1]
                t_err = np.sqrt(np.diag(pcov))[1]

                m_phase[x, y, i] = m, m_err, t, t_err

    # in case of 2D: remove slice dimension again
    m_phase = np.squeeze(m_phase)

    slope_frequency_map = m_phase[..., 0] * 1e3 / (2 * np.pi)
    slope_frequency_map_err = m_phase[..., 1] * 1e3 / (2 * np.pi)

    out = [slope_frequency_map, slope_frequency_map_err, m_phase]

    return out

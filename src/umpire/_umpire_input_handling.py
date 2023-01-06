"""
Here, the users input arguments of the UMPIRE function are checked for validity.

Additionaly the default DPD_filter_func is generated here and the data type of
the provided 'echo_scan' determined.
"""
"""
Note: scipy.ndimage dependency
      In case you don't want the scipy.ndimage dependency and/or don't care
      about smoothing the DPD image, you can apply these three steps to get rid
      of the dependendy:
        1. Comment out the line 'from scipy.ndimage import median_filter'
        2. Comment out the function 'default_DPD_filter_func'
        3. Replace 'out_func = default_DPD_filter_func()' with 'pass'
"""
import numpy as np
from scipy.ndimage import median_filter


class UmpireError(Exception):
    pass


def default_DPD_filter_func(kernel_size=3):
    """The default DPD filter function using median_filter from scipy.ndimage"""

    def filter_func(input):
        return median_filter(input, kernel_size)

    return filter_func


def __handle_UMPIRE_input(
    echo_scans, TEs, DPD_filter_func, magnitude_weighted_omega_star, debug_return_step
):
    """Ensures echo-image arrays and corresponding TEs are both of valid format.

    This function helps to avoid some false input errors. It checks the
    echo_scans arrays, TEs length and some other stuff.

    Parameters
    ----------
    echo_scans : array_like (N, ...), complex or real
        See UMPIRE-function documentation.

    TEs : array_like (N)
        See UMPIRE-function documentation.

    magnitude_weighted_omega_star : bool
        See UMPIRE-function documentation.

    Returns
    -------
    out : str {"real" or "complex"}
        Returns string corresponding to data type of given 'echo_scans' arrays.

    Raises
    ------
    UmpireError
        If echo-image arrays from 'echo_scans' are of invalid shape or data type
        or if length of 'TEs' and 'echo_scans' do not match.
    """
    # check if lenght of echo_scans and TEs match
    if len(echo_scans) != len(TEs):
        raise UmpireError(
            "Length of arguments 'echo_scans' and 'TEs' must be equal. "
            + f"{len(echo_scans)} != {len(TEs)}"
        )

    # check if lenght of echo_scans is at least 3
    if len(echo_scans) < 3:
        raise UmpireError(
            "You must provide at least three echo images. Only found "
            + f"{len(echo_scans)}."
        )

    for scan in echo_scans:
        # check for every scan to be an instance of numpy.ndarray class
        if not isinstance(scan, np.ndarray):
            raise UmpireError(
                "Arrays from arg. 'echo_scans' must be of type numpy.ndarray."
            )
        # check for all scan arrays to be of equal shape
        if scan.shape != echo_scans[0].shape:
            raise UmpireError(
                "Array shapes inside 'echo_scans' argument do not match. "
                + f"{scan.shape} != {echo_scans[0].shape}"
            )
        # ensure data types of all scan arrays match
        if scan.dtype != echo_scans[0].dtype:
            raise UmpireError(
                "Array data types inside 'echo_scans' argument do not match. "
                + f"{scan.dtype} != {echo_scans[0].dtype}"
            )
    # we only accept 2D and 3D numpy arrays as echo images
    if echo_scans[0].ndim not in (2, 3):
        raise UmpireError(
            "Only 2D and 3D arrays allowed. "
            + f"{scan.ndim} axis (=dimensions) found instead."
        )

    # check if our data is real or complex set 'out_type' accordingly
    if "complex" in str(scan.dtype):
        out_type = "complex"
    else:
        out_type = "real"

    # check wether DPD_filter_func argument is correctly assigned.
    if DPD_filter_func == "default":
        out_func = default_DPD_filter_func()
    elif DPD_filter_func in (False, None):
        out_func = lambda x: x
    elif callable(DPD_filter_func):
        out_func = DPD_filter_func
    else:
        raise UmpireError(
            'DPD_filter_func argument must be None, "Default" or a function.'
        )

    # for magnitude weighted omega star images, the scan arrays must be complex
    if magnitude_weighted_omega_star and out_type == "real":
        raise UmpireError(
            "To set the argument magnitude_weighted_omega_star=True, "
            + "complex-valued arrays must be provided."
        )

    # check if debug return step is a valid integer
    if debug_return_step != None:
        if not isinstance(debug_return_step, int) or not 0 < debug_return_step < 13:
            raise UmpireError(
                "The debug_return_step has to be an integer between 1 and 12."
                + f"Found {debug_return_step} instead."
                + " (Note: False == 0, use None instead.)"
            )

    return out_type, out_func

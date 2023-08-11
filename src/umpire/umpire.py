"""
Implementation of the UMPIRE algorithm from the following paper:

    Simon Robinson, Horst Schödl, Siegfried Trattnig, 2014 July,
    "A method for unwrapping highly wrapped multi-echo phase images at
    very high field: UMPIRE", Magnetic Resonance in Medicine, 72(1):80-92
    DOI: 10.1002/mrm.24897

Implemented by: Andre Wendlinger
"""
import numpy as np
from ._umpire_input_handling import __handle_UMPIRE_input


def UMPIRE(
    echo_scans,
    TEs,
    DPD_filter_func="default",
    magnitude_weighted_omega_star=False,
    debug_return_step=None,
    axis_TE=0,
):
    """Performs phase unwrapping of 3+ echo images using the UMPIRE algorithm.

    This function is an implementation of the UMPIRE algorithm from paper [1].
    It performs a pixel-wise, temporal phase unwrapping by using three (or more)
    echo images. The corresponding echo times TE1, TE2, TE3 are unevenly spaced,
    such that:

    DELTA_TE = TE2 - TE1
    delta_TE = TE3 - TE2 - DELTA_TE

    With the constraint that no phase wraps occcure during the time delta_TE.
    The algorithm follows the implementation suggested in paper [1] and is split
    into 13 steps.

    Parameters
    ----------
    echo_scans : array_like (N, ...), complex or real
        Input array of N echo images, with N ≥ 3. N is assumed to be at axis 0,
        if that is not the case, specify the axis using the 'axis_TE' keyword.
        The images can be two or three dimensional and their dtype can be real,
        i.e. phase images, or complex.
        Note: In case the optional argument magnitude_weighted_omega_star=True
              the arrays are must be complex-valued.

    TEs : array_like (N)
        Array of the N echo times in milliseconds, corresponding to the given
        N 'echo_scans'. The echo times should to be chosen according to [1].
        The small delta_T delay can be placed within the first three echos.

    DPD_filter_func : {None, "default", tuple, function}, optional
        This argument allows for an optional image filtering. Its purpose is to
        smooth the DPD image (see [1]), usually by convolution with a kernel.
        The options are:

        - "default":
            The default, which returns a scipy.ndimages.median_filter with
            kernel size of 3.

        - None (or False):
            No filter is applied.

        - tuple:
            Specifyies a kernel size for the default scipy.ndimages.median_filter
            function.

        - custom function:
            If a custom function is provided, it is blindly applied to the DPD
            image. The image filter function should accept arrays of similair
            size to a single echo image from 'echo_scans'.

    magnitude_weighted_omega_star : bool, optional
        Requires echo_scans to be complex-valued arrays. If true, computes a
        magnitude weighted image of omega_star (see [1]). Default is false.

    debug_return_step: {None, int}, optional
        Integer i in the range of 1 ≤ i ≤ 12. This option is for debugging. It
        results in a premature return of the processed 'echo_scans' after i
        steps of the UMPIRE algorithm (see [1]). Default is None.
        Important return steps might be:
            3 : DPD image (unfiltered and filtered in case of filter)
            9 : Unwrapped phase image before reciever offset calculations

    axis_TE: int, optional
        Default is zero. Used to specify the axis/dimension of N echo images.

    Returns
    -------
    out : ndarray (N, ...)
        An array containing N unwrapped phase images, extracted from the given
        N echo images using the UMPIRE algorithm (see [1]). The phase images are
        of the same shape as the input arrays from 'echo_scans' and real-valued.

    Raises
    ------
    UmpireError
        If echo-image arrays from 'echo_scans' are of invalid shape or data type
        or if length of 'TEs' and 'echo_scans' do not match.

    Notes
    -----
    1. Echo spacing and small delta_T location:
        As briefly mentioned in [1], the additional small delta_T delay does not
        have to occure between the second and third image. Especially in low SNR
        regimes it might be benificial to insert it between the first and second
        echo image.

    2. magnitude weighted omega star calculation:
        TODO

    References
    ----------
    .. [1] Simon Robinson, Horst Schödl, Siegfried Trattnig, 2014 July,
           "A method for unwrapping highly wrapped multi-echo phase images at
           very high field: UMPIRE", Magnetic Resonance in Medicine, 72(1):80-92
           DOI: 10.1002/mrm.24897
    """
    # This will raise an error in case of any invalid input,
    # otherwise returns 'echo_scan' data type as string {"real" or "complex"}
    # and the filter_func for the DPD image.
    data_type, DPD_filter_func = __handle_UMPIRE_input(
        echo_scans,
        TEs,
        DPD_filter_func,
        magnitude_weighted_omega_star,
        debug_return_step,
        axis_TE,
    )

    # actually swap dimensions
    echo_scans = np.moveaxis(echo_scans, axis_TE, 0)

    # --------------------------------------------------------------------------
    # STEP 0: Extract echo times, small delta T and large delta T.
    # --------------------------------------------------------------------------
    TE1, TE2, TE3 = TEs[:3]

    DELTA_TE = TE2 - TE1
    delta_TE = TE3 - TE2 - DELTA_TE
    assert abs(delta_TE) > 1e-3, f"Very small delta_TE detected: {delta_TE}"

    # --------------------------------------------------------------------------
    # STEP 1: Phase images $\theta_i$ are reconstructed for each echo time
    #         $T_{Ei}$.
    # --------------------------------------------------------------------------
    thetas = np.zeros_like(echo_scans, dtype="float")
    magnitudes = np.zeros_like(echo_scans, dtype="float")

    # standard case --> complex-valued echo images are provided
    if data_type == "complex":
        for i, scan in enumerate(echo_scans):
            thetas[i], magnitudes[i] = np.angle(scan), np.abs(scan)

    # data_type == 'real' --> real-valued phase images are provided
    else:
        for i, scan in enumerate(echo_scans):
            thetas[i] = scan

    if debug_return_step == 1:
        return thetas

    # --------------------------------------------------------------------------
    # STEP 2: Phase Difference Images are calculated.
    # --------------------------------------------------------------------------
    PDs = np.zeros((len(thetas) - 1, *thetas[0].shape))  # (N-1, ...)

    def phase_difference(ph1, ph2):
        return np.angle(np.exp(1j * (ph2 - ph1)))

    for i in range(len(thetas) - 1):
        PDs[i] = phase_difference(thetas[i], thetas[i + 1])

    if debug_return_step == 2:
        return PDs

    # --------------------------------------------------------------------------
    # STEP 3: Difference Between Phase Difference Images is calculated.
    # --------------------------------------------------------------------------
    DPD_raw = np.angle(np.exp(1j * (thetas[2] - 2 * thetas[1] + thetas[0])))

    # The filter func can be: 1. median_filter      (default)
    #                         2. lambda x: x        (does nothing)
    #                         3. a custom function  (not supervised)
    DPD_filtered = DPD_filter_func(DPD_raw)

    if debug_return_step == 3:
        return DPD_raw, DPD_filtered

    DPD = DPD_filtered

    # --------------------------------------------------------------------------
    #  STEP 4: Convert DPD to $\omega$-image.
    # --------------------------------------------------------------------------
    omega = DPD / delta_TE

    if debug_return_step == 4:
        return omega

    # --------------------------------------------------------------------------
    # STEP 5: Identify wraps in PD image using $\omega$.
    # --------------------------------------------------------------------------
    n_PDs = np.zeros_like(PDs)

    for i, PD in enumerate(PDs):
        n_PDs[i] = np.round(
            (PD - (TEs[i + 1] - TEs[i]) * omega) / (2 * np.pi), decimals=0
        )

    if debug_return_step == 5:
        return n_PDs

    # --------------------------------------------------------------------------
    # STEP 6: Unwrapping PD images (by pixelwise substraction of $2\pi\cdot n$).
    # --------------------------------------------------------------------------
    PDs_prime = np.zeros_like(PDs)

    for i in range(len(PDs_prime)):
        PDs_prime[i] = PDs[i] - 2 * np.pi * n_PDs[i]

    if debug_return_step == 6:
        return PDs_prime

    # --------------------------------------------------------------------------
    # STEP 7: Obtain higher SNR estimate of $\omega^\ast$ using unwrapped PD
    #         images.
    # --------------------------------------------------------------------------
    # pick the very first PDs_prime as basis for omega_star
    omega_star = PDs_prime[0] / DELTA_TE

    if magnitude_weighted_omega_star:
        # take the average magnitude of the two echo images each PD consists of,
        # and use them as weights for a magnitude-weighted averaged PDs_prime.
        weights = np.array(
            [np.mean(magnitudes[i : i + 2], axis=0) for i in range(len(magnitudes) - 1)]
        )  # mag[i], mag[i+1] = mag[i:i+2]

        omega_star_weighted = np.average(
            [
                pdp / dt for pdp, dt in zip(PDs_prime, np.diff(TEs))
            ],  # every PD has its own delta T
            axis=0,
            weights=weights,
        )

    if debug_return_step == 7:
        if magnitude_weighted_omega_star:  # pragma: no cover
            return omega_star, omega_star_weighted
        else:
            return omega_star

    if magnitude_weighted_omega_star:
        omega_star = omega_star_weighted

    # --------------------------------------------------------------------------
    # STEP 8: Use refined $\omega^\ast$-map to identify wraps in original phase
    #         images.
    # --------------------------------------------------------------------------
    n_thetas = np.zeros_like(thetas)
    for i in range(len(n_thetas)):
        n_thetas[i] = np.round(
            ((thetas[i] - TEs[i] * omega_star) / (2 * np.pi)), decimals=0
        )

    if debug_return_step == 8:
        return n_thetas

    # --------------------------------------------------------------------------
    #  STEP 9: Remove wraps that occured during TE.
    # --------------------------------------------------------------------------
    thetas_prime = np.zeros_like(thetas)

    for i in range(len(thetas_prime)):
        thetas_prime[i] = thetas[i] - 2 * np.pi * n_thetas[i]

    if debug_return_step == 9:
        return thetas_prime

    # --------------------------------------------------------------------------
    # STEP 10: Calculate Reciever Phase Offset R.
    # --------------------------------------------------------------------------
    R = (TEs[0] * thetas_prime[1] - TEs[1] * thetas_prime[0]) / (TEs[0] - TEs[1])

    if debug_return_step == 10:
        return R

    # --------------------------------------------------------------------------
    #  STEP 11: Substract Reciever Offset from each phase image.
    # --------------------------------------------------------------------------
    thetas_zero = np.zeros_like(thetas)

    for i, theta in enumerate(thetas):
        thetas_zero[i] = np.angle(np.exp(1j * (theta - R)))

    if debug_return_step == 11:
        return thetas_zero

    # --------------------------------------------------------------------------
    # STEP 12: (See step 8) Re-estimate number of wraps that occured between 0
    #           and $T_Ei$.
    # --------------------------------------------------------------------------
    n_primes = np.zeros_like(thetas)

    for i in range(len(n_primes)):
        n_primes[i] = np.round(
            ((thetas_zero[i] - TEs[i] * omega_star) / (2 * np.pi)), decimals=0
        )

    if debug_return_step == 12:
        return n_primes

    # --------------------------------------------------------------------------
    # STEP 13: Finally yield images free of wraps due to both reciever offset
    #          and $\Delta B_0$ variations.
    # --------------------------------------------------------------------------
    thetas_primeprime = np.zeros_like(thetas)

    for i in range(len(thetas_primeprime)):
        thetas_primeprime[i] = thetas_zero[i] - 2 * np.pi * n_primes[i]

    out = np.array([arr for arr in thetas_primeprime])

    return out

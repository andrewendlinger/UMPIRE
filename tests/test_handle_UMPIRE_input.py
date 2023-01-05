import pytest
import numpy as np
from umpire import __handle_UMPIRE_input, UmpireError


def test_handle_UMPIRE_input():

    # test: lenght of echo_scans and TEs do not match
    with pytest.raises(UmpireError, match=r"Length of argument"):
        __handle_UMPIRE_input(
            echo_scans=np.zeros((3, 128, 128)),
            TEs=[1, 2, 3, 4],
            magnitude_weighted_omega_star=False,
        )

    # test: lenght of echo_scans is less than 3
    with pytest.raises(UmpireError, match=r"You must provide at least three"):
        __handle_UMPIRE_input(
            echo_scans=np.zeros((2, 128, 128)),
            TEs=[1, 2],
            magnitude_weighted_omega_star=False,
        )
    # test: not every scan to be an instance of numpy.ndarray class
    with pytest.raises(UmpireError, match=r"Arrays from arg. 'echo_scans'"):
        __handle_UMPIRE_input(
            echo_scans=[np.zeros((128, 128)), np.zeros((128, 128)), [55, 384]],
            TEs=[1, 2, 3],
            magnitude_weighted_omega_star=False,
        )

    # test: not all scan arrays are of equal shape
    with pytest.raises(UmpireError, match=r"Array shapes"):
        __handle_UMPIRE_input(
            echo_scans=[
                np.zeros((128, 128)),
                np.zeros((128, 128)),
                np.zeros((128, 256)),
            ],
            TEs=[1, 2, 3],
            magnitude_weighted_omega_star=False,
        )

    # test: not all scan arrays are of equal data type
    with pytest.raises(UmpireError, match=r"Array data types"):
        __handle_UMPIRE_input(
            echo_scans=[
                np.zeros((128, 128)),
                np.zeros((128, 128)),
                np.zeros((128, 128), dtype="complex"),
            ],
            TEs=[1, 2, 3],
            magnitude_weighted_omega_star=False,
        )
    # test: not a 2D and 3D numpy array as echo image
    with pytest.raises(UmpireError, match=r"Only 2D and 3D"):
        __handle_UMPIRE_input(
            echo_scans=[
                np.zeros((128, 128, 64, 4)),
                np.zeros((128, 128, 64, 4)),
                np.zeros((128, 128, 64, 4)),
            ],
            TEs=[1, 2, 3],
            magnitude_weighted_omega_star=False,
        )

    # test: magnitude weighted omega star images true but, no complex scans
    with pytest.raises(UmpireError, match=r"To set the argument"):
        __handle_UMPIRE_input(
            echo_scans=[
                np.zeros((128, 128)),
                np.zeros((128, 128)),
                np.zeros((128, 128)),
            ],
            TEs=[1, 2, 3],
            magnitude_weighted_omega_star=True,
        )

    out = __handle_UMPIRE_input(
        echo_scans=[
            np.zeros((128, 128)),
            np.zeros((128, 128)),
            np.zeros((128, 128)),
        ],
        TEs=[1, 2, 3],
        magnitude_weighted_omega_star=False,
    )
    assert out == "real"

    out = __handle_UMPIRE_input(
        echo_scans=[
            np.zeros((128, 128), dtype="complex"),
            np.zeros((128, 128), dtype="complex"),
            np.zeros((128, 128), dtype="complex"),
        ],
        TEs=[1, 2, 3],
        magnitude_weighted_omega_star=True,
    )
    assert out == "complex"

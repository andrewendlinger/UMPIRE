import pytest
import numpy as np
from umpire._umpire_input_handling import __handle_UMPIRE_input, UmpireError


def test_handle_UMPIRE_input():
    # test: lenght of echo_scans and TEs do not match
    with pytest.raises(UmpireError, match=r"Length of argument"):
        __handle_UMPIRE_input(
            echo_scans=np.zeros((3, 128, 128)),
            TEs=[1, 2, 3, 4],
            DPD_filter_func=None,
            magnitude_weighted_omega_star=False,
            debug_return_step=None,
        )

    # test: lenght of echo_scans is less than 3
    with pytest.raises(UmpireError, match=r"You must provide at least three"):
        __handle_UMPIRE_input(
            echo_scans=np.zeros((2, 128, 128)),
            TEs=[1, 2],
            DPD_filter_func=None,
            magnitude_weighted_omega_star=False,
            debug_return_step=None,
        )

    # test: not all scan arrays are of equal shape
    with pytest.raises(
        UmpireError, match=r"Failed to turn echo_scans into numpy array"
    ):
        print("Is this the faulty one?")
        __handle_UMPIRE_input(
            echo_scans=[
                np.zeros((128, 128)),
                np.zeros((128, 128)),
                np.zeros((128, 256)),
            ],
            TEs=[1, 2, 3],
            DPD_filter_func=None,
            magnitude_weighted_omega_star=False,
            debug_return_step=None,
        )

    # test: not a 2D and 3D numpy array as echo image
    # with pytest.raises(UmpireError, match=r"Only 2D and 3D"):
    #     __handle_UMPIRE_input(
    #         echo_scans=[
    #             np.zeros((128, 128, 64, 4)),
    #             np.zeros((128, 128, 64, 4)),
    #             np.zeros((128, 128, 64, 4)),
    #         ],
    #         TEs=[1, 2, 3],
    #         DPD_filter_func=None,
    #         magnitude_weighted_omega_star=False,
    #         debug_return_step=None,
    #     )

    # test: magnitude weighted omega star images true but, no complex scans
    with pytest.raises(UmpireError, match=r"To set the argument"):
        __handle_UMPIRE_input(
            echo_scans=[
                np.zeros((128, 128)),
                np.zeros((128, 128)),
                np.zeros((128, 128)),
            ],
            TEs=[1, 2, 3],
            DPD_filter_func=None,
            magnitude_weighted_omega_star=True,
            debug_return_step=None,
        )

    # test: DPD_filter_func argument is wrongly assigned.
    with pytest.raises(UmpireError, match=r"DPD_filter_func"):
        __handle_UMPIRE_input(
            echo_scans=[
                np.zeros((128, 128)),
                np.zeros((128, 128)),
                np.zeros((128, 128)),
            ],
            TEs=[1, 2, 3],
            DPD_filter_func=np.array(1),
            magnitude_weighted_omega_star=False,
            debug_return_step=None,
        )

    # test: debug_return_step is wrong
    for wrong_step_input in [0, 13, 1.1, "3", False]:
        with pytest.raises(UmpireError, match=r"The debug_return_step"):
            __handle_UMPIRE_input(
                echo_scans=[
                    np.zeros((128, 128)),
                    np.zeros((128, 128)),
                    np.zeros((128, 128)),
                ],
                TEs=[1, 2, 3],
                DPD_filter_func=None,
                magnitude_weighted_omega_star=False,
                debug_return_step=wrong_step_input,
            )

    # test: wether we get the correct datatype string as ouput (real)
    out, _ = __handle_UMPIRE_input(
        echo_scans=[
            np.zeros((128, 128)),
            np.zeros((128, 128)),
            np.zeros((128, 128)),
        ],
        TEs=[1, 2, 3],
        DPD_filter_func=None,
        magnitude_weighted_omega_star=False,
        debug_return_step=None,
    )
    assert out == "real"

    # test: wether we get the correct datatype string as ouput (complex)
    out, _ = __handle_UMPIRE_input(
        echo_scans=[
            np.zeros((128, 128), dtype="complex"),
            np.zeros((128, 128), dtype="complex"),
            np.zeros((128, 128), dtype="complex"),
        ],
        TEs=[1, 2, 3],
        DPD_filter_func=None,
        magnitude_weighted_omega_star=True,
        debug_return_step=None,
    )
    assert out == "complex"

    # test: axis_TE wrong
    with pytest.raises(UmpireError, match=r"axis_TE="):
        __handle_UMPIRE_input(
            echo_scans=[
                np.zeros((128, 128, 64, 4)),
                np.zeros((128, 128, 64, 4)),
                np.zeros((128, 128, 64, 4)),
            ],
            TEs=[1, 2, 3],
            DPD_filter_func=None,
            magnitude_weighted_omega_star=False,
            debug_return_step=None,
            axis_TE=5,
        )

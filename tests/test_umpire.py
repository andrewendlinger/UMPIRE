import pytest
import numpy as np
from umpire import UMPIRE
from umpire._umpire_input_handling import default_DPD_filter_func


def generate_simulated_data_2D(img_dims, TEs, reciever_offset=True):
    """Generate len(TEs) simulated 2D phase images of shape img_dim.

    The phase images are generated according to the section "Simulated Data 1:
    Linear Gradients in Delta-B0 and Reciever Phase Offset." of the UMPIRE
    paper by Robinson et al.

    Parameters
    ----------
    img_dims: array_like of lenght 2
        Array containing dimensions of a phase image according to
        x_dim, y_dim = img_dims.

    TEs : array_like (N)
        Array of the 'echo_scans' N echo times in milliseconds. For each
        echo time a phase image is generated.

    reciever_offset : bool, optional
        Wether a static gradient along y, simulating a reciever offset R,
        is added or not.

    """
    assert len(img_dims) == 2, "Only two dimensional images."
    dim_y, dim_x = img_dims  # image axis are swapped for matplotlib

    x = np.linspace(-1, 1, img_dims[1])
    y = np.linspace(0, 1, img_dims[0])
    xx, yy = np.meshgrid(x, y)

    def phase_RecieverOffset_OFF(x, y, t_echo, Gx=np.pi, Gy=6 * np.pi):
        return Gx * x * t_echo

    def phase_RecieverOffset_ON(x, y, t_echo, Gx=np.pi, Gy=6 * np.pi):
        return Gx * x * t_echo + Gy * y

    if reciever_offset:
        phase_gen_func = phase_RecieverOffset_ON
    else:
        phase_gen_func = phase_RecieverOffset_OFF

    phase_imgs = np.zeros((len(TEs), *img_dims))

    for i in range(len(TEs)):
        phase_imgs[i] = phase_gen_func(xx, yy, TEs[i])

    return phase_imgs


def generate_wrapped_simulated_data_2D_complex(img_dims, TEs, reciever_offset=True):
    """Wrap phase from 'generate_simulated_data_2D()' and turn complex.

    Phase data is already wrapped for simplicity. Apart from that it is still
    equal to 'generate_simulated_data_2D()' output and the Magnitude data is
    uniformly set to one.
    Note: See  generate_simulated_data_2D for more information.
    """
    phase_imgs = generate_simulated_data_2D(img_dims, TEs, reciever_offset)
    phase_imgs_wrapped = wrap_phase(phase_imgs)
    magnitude_imgs = np.ones_like(phase_imgs)

    out = magnitude_imgs * np.exp(1j * phase_imgs_wrapped)

    return out


def generate_simulated_data_semi3D(img_dims, TEs, reciever_offset=True):
    """Stack 2D images to a semi-3D dataset of shape [len(TEs), x, y, z].

    We create a 3D dataset by repeating the generated 2D phase images
    along a new axis. There is no new information encoded along the
    z-direction, hence only semi-3D.

    Parameters
    ----------
    img_dims : array_like of lenght 3
        Array containing dimensions of a phase image according to
        x_dim, y_dim, z_dim = img_dims.


    TEs : array_like (N)
        Array of the 'echo_scans' N echo times in milliseconds. For each
        echo time a phase image is generated.

    reciever_offset : bool, optional
        Wether a static gradient along y, simulating a reciever offset R,
        is added or not.

    """
    assert len(img_dims) == 3, "Only three dimensional images."
    dim_x, dim_y, dim_z = img_dims

    # generate 2D images
    phase_imgs_2D = generate_simulated_data_2D((dim_x, dim_y), TEs, reciever_offset)

    # generate 3D images, using the broadcasting functionality from np.resize
    phase_imgs_3D = np.array(
        [np.resize(arr, (dim_z, dim_x, dim_y)) for arr in phase_imgs_2D]
    )

    return phase_imgs_3D


def wrap_phase(phase):
    """Wraps phase into [-pi, pi] range."""
    return phase - np.floor(phase / (2 * np.pi)) * 2 * np.pi


@pytest.mark.parametrize(
    "step",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
)
def test_debug_return_step(step):
    """Simply test if every step can be returned. Does no output control."""
    TEs = [5, 10, 16]
    img_dims = (128, 128)

    # generate phase images
    phase_imgs = generate_simulated_data_2D(img_dims, TEs, True)
    phase_imgs_wrapped = wrap_phase(phase_imgs)

    # apply umpire algorithm
    premature_output = UMPIRE(
        phase_imgs_wrapped,
        TEs,
        DPD_filter_func=False,
        magnitude_weighted_omega_star=False,
        debug_return_step=step,
    )


@pytest.mark.parametrize(
    "img_dims, TEs, reciver_offset",
    [
        pytest.param((128, 128), [5, 10, 16], False),
        pytest.param((128, 128), [5, 10, 16], True),
        pytest.param((128, 256), [5, 10, 16], True),
        pytest.param((128, 128), [5, 10, 16, 21, 26], True),
        pytest.param((128, 128), [5, 11, 16, 21, 26], True),
        pytest.param((128, 128), [6, 10, 15, 20, 25], True),
    ],
)
def test_umpire_2D(img_dims, TEs, reciver_offset):
    """"""
    # generate phase images
    phase_imgs = generate_simulated_data_2D(img_dims, TEs, reciver_offset)
    phase_imgs_wrapped = wrap_phase(phase_imgs)

    # generate umpire ground-truth images. Keep in mind that the results
    # from UMPIRE always correspond to phase images without reciever_offset!
    phase_imgs_groundtruth = generate_simulated_data_2D(img_dims, TEs, False)

    # apply umpire algorithm
    phase_imgs_umpire = UMPIRE(
        phase_imgs_wrapped,
        TEs,
        DPD_filter_func=False,
        magnitude_weighted_omega_star=False,
    )

    # ensure input shape is same as outputshape
    assert phase_imgs_umpire.shape == phase_imgs_wrapped.shape

    # Calculate absolute Error
    # Note: We cut off a 1 pixel border for the error calculation, because the
    #       pixel border sometimes contains random floating point errors.
    def total_abs_err(a, b):
        return np.sum(np.abs(a - b)[..., 1:-1, 1:-1])

    error = total_abs_err(phase_imgs_groundtruth, phase_imgs_umpire)

    assert error < 1e-9


@pytest.mark.parametrize(
    "img_dims, TEs, reciver_offset",
    [
        pytest.param((128, 128), [5, 10, 16], False),
        pytest.param((128, 128), [5, 10, 16], True),
        pytest.param((128, 256), [5, 10, 16], True),
        pytest.param((128, 128), [5, 10, 16, 21, 26], True),
        pytest.param((128, 128), [5, 11, 16, 21, 26], True),
        pytest.param((128, 128), [6, 10, 15, 20, 25], True),
    ],
)
def test_umpire_2D_complex(img_dims, TEs, reciver_offset):
    """"""
    # generate phase images
    phase_imgs_wrapped = generate_wrapped_simulated_data_2D_complex(
        img_dims, TEs, reciver_offset
    )

    # generate umpire ground-truth images. Keep in mind that the results
    # from UMPIRE always correspond to phase images without reciever_offset!
    phase_imgs_groundtruth = generate_simulated_data_2D(img_dims, TEs, False)

    # apply umpire algorithm
    phase_imgs_umpire = UMPIRE(
        phase_imgs_wrapped,
        TEs,
        DPD_filter_func=False,
        magnitude_weighted_omega_star=False,
    )

    # ensure input shape is same as outputshape
    assert phase_imgs_umpire.shape == phase_imgs_wrapped.shape

    # Calculate absolute Error
    # Note: We cut off a 1 pixel border for the error calculation, because the
    #       pixel border sometimes contains random floating point errors.
    def total_abs_err(a, b):
        return np.sum(np.abs(a - b)[..., 1:-1, 1:-1])

    error = total_abs_err(phase_imgs_groundtruth, phase_imgs_umpire)

    assert error < 1e-9


@pytest.mark.parametrize(
    "img_dims, TEs, reciver_offset, DPD_filter_func_arg",
    [
        pytest.param((64, 32), [5, 10, 16], True, "default"),
        pytest.param((64, 32), [5, 10, 16], True, (3, 3)),
        pytest.param((64, 32), [5, 10, 16], True, default_DPD_filter_func(3)),
    ],
)
def test_umpire_DPD_filter_func(img_dims, TEs, reciver_offset, DPD_filter_func_arg):
    """"""
    # generate phase images
    phase_imgs = generate_simulated_data_2D(img_dims, TEs, reciver_offset)
    phase_imgs_wrapped = wrap_phase(phase_imgs)

    # generate umpire ground-truth images. Keep in mind that the results
    # from UMPIRE always correspond to phase images without reciever_offset!
    phase_imgs_groundtruth = generate_simulated_data_2D(img_dims, TEs, False)

    # apply umpire algorithm
    phase_imgs_umpire = UMPIRE(
        phase_imgs_wrapped,
        TEs,
        DPD_filter_func=DPD_filter_func_arg,
        magnitude_weighted_omega_star=False,
    )

    # Calculate absolute Error
    # Note: We cut off a 1 pixel border for the error calculation, because the
    #       pixel border sometimes contains random floating point errors.
    def total_abs_err(a, b):
        return np.sum(np.abs(a - b)[..., 1:-1, 1:-1])

    error = total_abs_err(phase_imgs_groundtruth, phase_imgs_umpire)

    assert error < 1e-9


@pytest.mark.parametrize(
    "img_dims, TEs, reciver_offset",
    [
        pytest.param((128, 64, 32), [5, 10, 16], False),
        pytest.param((128, 64, 64), [5, 10, 16, 21], True),
        pytest.param((128, 64, 64), [5, 11, 16, 21], True),
    ],
)
def test_umpire_3D(img_dims, TEs, reciver_offset):
    """"""
    # generate phase images
    phase_imgs = generate_simulated_data_semi3D(img_dims, TEs, reciver_offset)
    phase_imgs_wrapped = wrap_phase(phase_imgs)

    # generate umpire ground-truth images. Keep in mind that the results
    # from UMPIRE always correspond to phase images without reciever_offset!
    phase_imgs_groundtruth = generate_simulated_data_semi3D(img_dims, TEs, False)

    # apply umpire algorithm
    phase_imgs_umpire = UMPIRE(
        phase_imgs_wrapped,
        TEs,
        DPD_filter_func=False,
        magnitude_weighted_omega_star=False,
    )

    # ensure input shape is same as outputshape
    assert phase_imgs_umpire.shape == phase_imgs_wrapped.shape

    # Calculate absolute Error
    # Note: We cut off a 1 pixel border for the error calculation, because the
    #       pixel border sometimes contains random floating point errors.
    def total_abs_err(a, b):
        return np.sum(np.abs(a - b)[..., 1:-1, 1:-1, 1:-1])

    error = total_abs_err(phase_imgs_groundtruth, phase_imgs_umpire)

    assert error < 1e-9 * img_dims[-1]  # error adds up for every z slice


@pytest.mark.parametrize(
    "img_dims, TEs, reciver_offset",
    [
        pytest.param((64, 64), [5, 10, 16], True),
        # pytest.param((64, 64, 32), [5, 10, 16], True),
        pytest.param((128, 64), [5, 10, 16, 21, 26], True),
    ],
)
def test_umpire_magnitude_weighted_omega_star(img_dims, TEs, reciver_offset):
    """"""
    # generate phase images
    phase_imgs_wrapped = generate_wrapped_simulated_data_2D_complex(
        img_dims, TEs, reciver_offset
    )

    # generate umpire ground-truth images. Keep in mind that the results
    # from UMPIRE always correspond to phase images without reciever_offset!
    phase_imgs_groundtruth = generate_simulated_data_2D(img_dims, TEs, False)

    # apply umpire algorithm
    phase_imgs_umpire = UMPIRE(
        phase_imgs_wrapped,
        TEs,
        DPD_filter_func=False,
        magnitude_weighted_omega_star=True,
    )

    # ensure input shape is same as outputshape
    assert phase_imgs_umpire.shape == phase_imgs_wrapped.shape

    # Calculate absolute Error
    # Note: We cut off a 1 pixel border for the error calculation, because the
    #       pixel border sometimes contains random floating point errors.
    def total_abs_err(a, b):
        return np.sum(np.abs(a - b)[..., 1:-1, 1:-1])

    error = total_abs_err(phase_imgs_groundtruth, phase_imgs_umpire)

    assert error < 1e-9

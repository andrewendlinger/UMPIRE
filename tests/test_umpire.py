import pytest
import numpy as np
from umpire import UMPIRE


def generate_simulated_data_2D(img_dims, TEs, reciever_offset=True):
    """Generate len(TEs) simulated 2D phase images of shape img_dim.

    The phase images are generated according to the section "Simulated Data 1:
    Linear Gradients in Delta-B0 and Reciever Phase Offset." of the UMPIRE
    paper by Robinson et al.
    """
    assert len(img_dims) == 2, "Only two dimensional images."
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


def wrap_phase(phase):
    """Wraps phase into [-pi, pi] range."""
    return phase - np.floor(phase / (2 * np.pi)) * 2 * np.pi


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
def test_umpire_DPD_filter_func(img_dims, TEs, reciver_offset):
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
        DPD_filter_func="default",
        magnitude_weighted_omega_star=False,
    )

    # Calculate absolute Error
    # Note: We cut off a 1 pixel border for the error calculation, because the
    #       pixel border sometimes contains random floating point errors.
    def total_abs_err(a, b):
        return np.sum(np.abs(a - b)[..., 1:-1, 1:-1])

    error = total_abs_err(phase_imgs_groundtruth, phase_imgs_umpire)

    assert error < 1e-9


def test_umpire_3D():
    pass


def test_umpire_magnitude_weighted_omega_star():
    pass

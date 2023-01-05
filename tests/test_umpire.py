import pytest
import numpy as np
from umpire import UMPIRE


def generate_simulated_data_2D(img_dims, TEs, reciever_offset=True):
    """Generate len(TEs) simulated 2D phase images of shape img_dim.

    The phase images are generated according to the section "Simulated Data 1:
    Linear Gradients in Delta-B0 and Reciever Phase Offset." of the UMPIRE
    paper by Robinson et al.
    """
    x = np.linspace(-1, 1, img_dims[0])
    y = np.linspace(0, 1, img_dims[1])
    xx, yy = np.meshgrid(x, y)

    def phi_noRecieverOffset(x, y, t_echo, Gx=np.pi, Gy=6 * np.pi):
        return Gx * x * t_echo

    def phi_all(x, y, t_echo, Gx=np.pi, Gy=6 * np.pi):
        return Gx * x * t_echo + Gy * y

    if reciever_offset:
        phase_gen_func = phi_all
        print("ON")
    else:
        phase_gen_func = phi_noRecieverOffset
        print("OFF")

    phis = np.zeros((len(TEs), *img_dims))

    for i in range(len(TEs)):
        phis[i] = phase_gen_func(xx, yy, TEs[i])

    return phis


@pytest.mark.parametrize(
    "img_dims, TEs, reciver_offset",
    [
        pytest.param((128, 128), [5, 10, 16], False, id=" RecieverOffset OFF"),
        pytest.param((128, 128), [5, 10, 16], True, id=" RecieverOffset ON"),
    ],
)
def test_umpire_3_images(img_dims, TEs, reciver_offset):

    phis = generate_simulated_data_2D(img_dims, TEs, reciver_offset)

    output = UMPIRE(
        phis, TEs, DPD_filter_func=False, magnitude_weighted_omega_star=False
    )

    def abs_err(a, b):
        return np.abs(a - b)

    error = np.sum([abs_err(phis[i], output[i]) for i in range(len(TEs))])

    print(error)
    assert error < 1e-10


def test_umpire_3plus_images():
    pass


def test_umpire_3_images_reversed():
    pass


def test_umpire_3plus_images_reversed():
    pass

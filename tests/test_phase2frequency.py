import pytest
import numpy as np
from umpire.phase2frequency import phase2frequency
from .test_umpire import generate_simulated_data_2D, generate_simulated_data_semi3D


@pytest.mark.parametrize(
    "img_dims, TEs, reciver_offset",
    [
        pytest.param((8, 8), [5, 10, 16], True),
    ],
)
def test_phase2frequency_2D(img_dims, TEs, reciver_offset):
    """"""
    # generate phase images, we cut off the pixel border with values of exact
    # -pi and pi which would result in unrealistic values
    phase_imgs = generate_simulated_data_2D(img_dims, TEs, reciver_offset)
    phase_imgs = phase_imgs[..., 1:-1, 1:-1]

    # Generate ground-truth images. The results from fitting to the phase data
    # will yield the temporal dependence from the phase image generation
    # function only, that is, the temporal derivative:
    #      phase images: f(x,y, t_echo) = Gx * x * t_echo + Gy * y
    #   phase fit yield: f'(x,y) = Gx  * x
    groundtruth = np.squeeze(generate_simulated_data_2D(img_dims, [1], False))
    groundtruth *= 1e3 / (2 * np.pi)  # convert from 'rad per ms' to 'Hz'
    groundtruth = groundtruth[..., 1:-1, 1:-1]

    # fit to generated phase images
    fmap, fmap_err, _ = phase2frequency(phase_imgs, TEs)

    error_fit = np.sum(np.abs(fmap_err))
    assert error_fit < 1e-10

    # Calculate absolute Error
    # Note: We cut off a 1 pixel border for the error calculation, because the
    #       pixel border sometimes contains random floating point errors.
    def total_abs_err(a, b):
        return np.sum(np.abs(a - b))

    error = total_abs_err(groundtruth, fmap)
    assert error < 1e-10


@pytest.mark.parametrize(
    "img_dims, TEs, reciver_offset",
    [
        pytest.param((4, 4, 4), [5, 10, 16], True),
    ],
)
def test_phase2frequency_3D(img_dims, TEs, reciver_offset):
    """"""
    # generate phase images, we cut off the pixel border with values of exact
    # -pi and pi which would result in unrealistic values
    phase_imgs = generate_simulated_data_semi3D(img_dims, TEs, reciver_offset)
    phase_imgs = phase_imgs[..., 1:-1, 1:-1, 1:-1]

    # Generate ground-truth images. The results from fitting to the phase data
    # will yield the temporal dependence from the phase image generation
    # function only, that is, the temporal derivative:
    #      phase images: f(x,y, t_echo) = Gx * x * t_echo + Gy * y
    #   phase fit yield: f'(x,y) = Gx  * x
    groundtruth = np.squeeze(generate_simulated_data_semi3D(img_dims, [1], False))
    groundtruth *= 1e3 / (2 * np.pi)  # convert from 'rad per ms' to 'Hz'
    groundtruth = groundtruth[..., 1:-1, 1:-1, 1:-1]

    # fit to generated phase images
    fmap, fmap_err, _ = phase2frequency(phase_imgs, TEs)

    error_fit = np.sum(np.abs(fmap_err))
    assert error_fit < 1e-9

    # Calculate absolute Error
    # Note: We cut off a 1 pixel border for the error calculation, because the
    #       pixel border sometimes contains random floating point errors.
    def total_abs_err(a, b):
        return np.sum(np.abs(a - b))

    error = total_abs_err(groundtruth, fmap)
    assert error < 1e-9

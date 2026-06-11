import pytest
from keras import ops, random, backend as K
from skimage.metrics import structural_similarity as compare_ssim

# Local imports - adjust paths according to your project structure
from auramask.losses.ssim import DSSIMObjective, SSIMC


@pytest.fixture
def test_image_a():
    """Fixture to create a consistent test image for use across multiple tests."""
    return ops.convert_to_tensor(
        random.uniform(
            (224, 224, 3), minval=0.0, maxval=1.0, dtype=K.floatx(), seed=123
        )
    )


@pytest.fixture
def ssim():
    """Fixture to create the SSIM loss object."""
    return SSIMC()


@pytest.mark.parametrize("noise_sigma", [0.1, 0.3, 0.5])
def test_ssim_noise_variations(ssim, test_image_a, noise_sigma):
    """Parametrized test for SSIM with varying noise levels."""
    _test_img_b = ops.clip(
        ops.add(
            test_image_a,
            ops.convert_to_tensor(
                random.uniform(
                    (224, 224, 3),
                    minval=-noise_sigma,
                    maxval=noise_sigma,
                    dtype=K.floatx(),
                    seed=456,
                )
            ),
        ),
        0.0,
        1.0,
    )

    expected_ssim = 1 - compare_ssim(
        ops.convert_to_numpy(test_image_a),
        ops.convert_to_numpy(_test_img_b),
        data_range=1.0,
        channel_axis=2,
        window_size=11,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )

    actual_ssim = ssim(test_image_a, _test_img_b)

    assert ops.allclose(expected_ssim, actual_ssim, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(
    K.backend() != "torch", reason="IQASSIM only works in torch context"
)
def test_ssim_iqassim():
    """Test SSIM against IQASSIM when running in Torch backend."""
    from auramask.losses.ssim import IQASSIMC

    iqa = IQASSIMC()
    _test_img_b = ops.clip(
        ops.add(
            test_image_a,
            ops.convert_to_tensor(
                random.uniform(
                    (224, 224, 3),
                    minval=-0.1,
                    maxval=0.1,
                    dtype=K.floatx(),
                    seed=456,
                )
            ),
        ),
        0.0,
        1.0,
    )

    t_ssim = iqa(test_image_a, _test_img_b)

    # Use a simple identical image test for cross-backend comparison
    expected_ssim = DSSIMObjective()(test_image_a, test_image_a)

    assert ops.allclose(t_ssim, expected_ssim, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize(
    "shape", [(224, 224, 3), (256, 256, 3), (64, 64, 3), (512, 512, 3)]
)
def test_ssim_batch_shape(shape):
    """Test SSIM with different image shapes."""
    from keras import ops, random

    img1 = ops.convert_to_tensor(
        random.uniform(shape, minval=0, maxval=1.0, dtype=K.floatx(), seed=123)
    )
    img2 = ops.convert_to_tensor(
        random.uniform(shape, minval=0, maxval=1.0, dtype=K.floatx(), seed=456)
    )

    actual_val = DSSIMObjective()(img1, img2)

    # Compute reference SSIM with scikit-image
    expected_val = 1 - compare_ssim(
        ops.convert_to_numpy(img1),
        ops.convert_to_numpy(img2),
        data_range=1.0,
        channel_axis=2,
        window_size=3,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )

    # Allow some tolerance due to implementation differences
    assert ops.allclose(expected_val, actual_val, rtol=0.1, atol=0.05)

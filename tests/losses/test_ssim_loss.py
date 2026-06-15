import pytest
from keras import ops, random, backend as K
from skimage.metrics import structural_similarity as compare_ssim

# Local imports - adjust paths according to your project structure
from auramask.losses.ssim import SSIMC


# Set floating point precision tolerances (absolute 0 and relative 2%)
@pytest.fixture
def atol_rtol():
    return {"rtol": 0.05, "atol": 5e-7}


@pytest.fixture
def image_data_format(request):
    K.set_image_data_format(request.param)
    return request.param


@pytest.mark.parametrize(
    "image_data_format", ["channels_last", "channels_first"], indirect=True
)
@pytest.mark.parametrize("img_shape", [(224, 224), (256, 256), (64, 64), (512, 512)])
@pytest.mark.parametrize("noise", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("gaussian_sigma", [0.1, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("k_params", [(0.01, 0.03), (0.02, 0.06), (0.03, 0.09)])
def test_ssim_noise_variations(
    image_data_format, img_shape, atol_rtol, noise, gaussian_sigma, k_params
):
    """Parametrized test for SSIM with varying noise levels."""
    # Generate channels first or channels last images
    if image_data_format == "channels_first":
        img_shape = (3,) + img_shape
    else:
        img_shape = img_shape + (3,)

    k1, k2 = k_params
    img1 = random.uniform(img_shape, minval=0, maxval=1.0, dtype=K.floatx(), seed=123)

    img2 = ops.clip(
        ops.add(
            img1,
            random.uniform(
                img_shape,
                minval=-noise,
                maxval=noise,
                dtype=K.floatx(),
                seed=456,
            ),
        ),
        0.0,
        1.0,
    )

    expected_ssim = 1 - compare_ssim(
        ops.convert_to_numpy(img1),
        ops.convert_to_numpy(img2),
        data_range=1.0,
        channel_axis=2 if image_data_format == "channels_last" else 0,
        sigma=gaussian_sigma,
        gaussian_weights=True,
        use_sample_covariance=False,
        K1=k1,
        K2=k2,
    )

    actual_ssim = SSIMC(k1=k1, k2=k2, filter_sigma=gaussian_sigma, dtype=K.floatx())(
        ops.expand_dims(img1, 0), ops.expand_dims(img2, 0)
    )

    assert ops.allclose(expected_ssim, actual_ssim, **atol_rtol)


# @pytest.mark.skipif(
#     K.backend() != "torch", reason="IQASSIM only works in torch context"
# )
# def test_ssim_iqassim():
#     """Test SSIM against IQASSIM when running in Torch backend."""
#     from auramask.losses.ssim import IQASSIMC

#     iqa = IQASSIMC()
#     _test_img_b = ops.clip(
#         ops.add(
#             test_image_a,
#             ops.convert_to_tensor(
#                 random.uniform(
#                     (224, 224, 3),
#                     minval=-0.1,
#                     maxval=0.1,
#                     dtype=K.floatx(),
#                     seed=456,
#                 )
#             ),
#         ),
#         0.0,
#         1.0,
#     )

#     t_ssim = iqa(test_image_a, _test_img_b)

#     # Use a simple identical image test for cross-backend comparison
#     expected_ssim = DSSIMObjective()(test_image_a, test_image_a)

#     assert ops.allclose(t_ssim, expected_ssim, rtol=1e-3, atol=1e-4)

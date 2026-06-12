import pytest
from keras import ops, random, backend as K
from skimage.metrics import structural_similarity as compare_ssim
import itertools

# Local imports - adjust paths according to your project structure
from auramask.losses.ssim import SSIMC

# SSIM test case parameters
_img_shape_perms = [(224, 224, 3), (256, 256, 3), (64, 64, 3), (512, 512, 3)]
_noise_sigma_perms = [0.1, 0.3, 0.5]
_guass_sigma_perms = [0.1, 0.5, 1.0, 1.5]


@pytest.mark.parametrize(
    "shape,noise_sigma,gauss_sigma", list(itertools.product(_img_shape_perms, _noise_sigma_perms, _guass_sigma_perms))
)
def test_ssim_noise_variations(shape, noise_sigma, gauss_sigma):
    """Parametrized test for SSIM with varying noise levels."""
    img1 = random.uniform(shape, minval=0, maxval=1.0, dtype=K.floatx(), seed=123)
    
    img2 = ops.clip(
        ops.add(
            img1,
            random.uniform(
                shape,
                minval=-noise_sigma,
                maxval=noise_sigma,
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
        channel_axis=2,
        sigma=gauss_sigma,
        gaussian_weights=True,
        use_sample_covariance=False,
    )

    actual_ssim = SSIMC(filter_sigma=gauss_sigma, dtype=K.floatx())(img1, img2)

    assert ops.allclose(expected_ssim, actual_ssim, rtol=1e-3, atol=1e-4)


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


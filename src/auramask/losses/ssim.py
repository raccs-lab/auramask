from keras import Loss, ops, backend as K, KerasTensor
from typing import Callable


### REIMPLEMENTATION OF tensorflow library SSIM in backend agnostic Keras
def _fspecial_gauss(size: int, sigma: float):
    """Function mimicking 'fspecial' gaussian MATLAB function. Taken from https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/image_ops_impl.py#L4385-L4476"""
    size = ops.convert_to_tensor(size, "int32")
    sigma = ops.convert_to_tensor(sigma)

    coords = ops.cast(ops.arange(size), sigma.dtype)
    coords -= ops.cast(size - 1, sigma.dtype) / 2.0

    g = ops.square(coords)
    g *= -0.5 / ops.square(sigma)

    g = ops.reshape(g, newshape=[1, -1]) + ops.reshape(g, newshape=[-1, 1])
    g = ops.reshape(g, newshape=[1, -1])
    g = ops.nn.softmax(g)
    return ops.reshape(g, newshape=[size, size, 1, 1])


def _ssim_helper(
    x: KerasTensor,
    y: KerasTensor,
    reducer: Callable,
    max_val: float,
    compensation: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
):
    """Helper function for computing SSIM.

    SSIM estimates covariances with weighted sums.  The default parameters
    use a biased estimate of the covariance:
    Suppose `reducer` is a weighted sum, then the mean estimators are
        \mu_x = \sum_i w_i x_i,
        \mu_y = \sum_i w_i y_i,
    where w_i's are the weighted-sum weights, and covariance estimator is
        cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    with assumption \sum_i w_i = 1. This covariance estimator is biased, since
        E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
    For SSIM measure with unbiased covariance estimators, pass as `compensation`
    argument (1 - \sum_i w_i ^ 2).

    Args:
        x: First set of images.
        y: Second set of images.
        reducer: Function that computes 'local' averages from the set of images. For
        non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
        for convolutional version, this is usually tf.nn.avg_pool2d or
        tf.nn.conv2d with weighted-sum kernel.
        max_val: The dynamic range (i.e., the difference between the maximum
        possible allowed value and the minimum allowed value).
        compensation: Compensation factor. See above.
        k1: Default value 0.01
        k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).

    Returns:
        A pair containing the luminance measure, and the contrast-structure measure.
    """

    c1 = ops.square(k1 * max_val)
    c2 = ops.square(k2 * max_val)

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = ops.square(mean0) + ops.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = reducer(x * y) * 2.0
    den1 = reducer(ops.square(x) + ops.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    # SSIM score is the product of the luminance and contrast-structure measures.
    return luminance, cs


def _ssim_per_channel(
    img1: KerasTensor,
    img2: KerasTensor,
    max_val: float = 1.0,
    kernel_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_index_map=False,
) -> tuple[KerasTensor, KerasTensor]:
    """Computes SSIM index between img1 and img2 per color channel. (Reimplementation from tensorflow library)

    This function matches the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.

    Details:
      - 11x11 Gaussian filter of width 1.5 is used.
      - k1 = 0.01, k2 = 0.03 as in the original paper.

    Args:
      img1: First image batch.
      img2: Second image batch.
      max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      kernel_size: Default value 11 (size of gaussian filter).
      filter_sigma: Default value 1.5 (width of gaussian filter).
      k1: Default value 0.01
      k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
      return_index_map: If True returns local SSIM map instead of the global mean.

    Returns:
      A pair of tensors containing and channel-wise SSIM and contrast-structure
      values. The shape is [..., channels].
    """
    kernel_size = ops.convert_to_tensor(kernel_size, dtype="int32")
    filter_sigma = ops.convert_to_tensor(filter_sigma, dtype=img1.dtype)

    shape1 = ops.shape(img1)

    # INFO: removed assertive checks from tensorflow
    # TODO: See if ops.image.gaussian_blur can be used for this to improve performance
    kernel = _fspecial_gauss(kernel_size, filter_sigma)
    if K.image_data_format() == "channels_last":
        kernel = ops.tile(kernel, repeats=[1, 1, shape1[-1], 1])
    else:
        kernel = ops.tile(kernel, repeats=[1, 1, shape1[-3], 1])
    
    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    compensation = 1.0

    # BUG: Even though variable data format has implied support, currently this only supports channels last
    def reducer(x):
        shape = ops.shape(x)
        x = ops.reshape(x, newshape=(-1,) + shape[-3:])
        y = ops.nn.depthwise_conv(
            x,
            kernel,
            strides=1,
            padding="valid",
        )
        return ops.reshape(y, newshape=(-1,) + shape[:-3] + ops.shape(y)[1:])

    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1, k2)

    # Average over the second and the third from the last: height, width.
    if return_index_map:
        ssim_val = luminance * cs
    else:
        axes = [-3, -2] if K.backend() == "channels_last" else [-2, -1]
        ssim_val = ops.mean(luminance * cs, axes)
        cs = ops.mean(cs, axes)
    return ssim_val, cs


def ssim(
    img1: KerasTensor,
    img2: KerasTensor,
    max_val: float,
    kernel_size: int = 11,
    filter_sigma: int = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_index_map: bool = False,
) -> KerasTensor:
    """Computes SSIM index between img1 and img2.

    This function is based on the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.

    Note: The true SSIM is only defined on grayscale.  This function does not
    perform any colorspace transform.  (If the input is already YUV, then it will
    compute YUV SSIM average.)

    Details:
        - 11x11 Gaussian filter of width 1.5 is used.
        - k1 = 0.01, k2 = 0.03 as in the original paper.

    The image sizes must be at least 11x11 because of the filter size.

    Example:

    ```python
        # Read images (of size 255 x 255) from file.
        im1 = tf.image.decode_image(tf.io.read_file('path/to/im1.png'))
        im2 = tf.image.decode_image(tf.io.read_file('path/to/im2.png'))
        tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
        tf.shape(im2)  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
        # Add an outer batch for each image.
        im1 = tf.expand_dims(im1, axis=0)
        im2 = tf.expand_dims(im2, axis=0)
        # Compute SSIM over tf.uint8 Tensors.
        ssim1 = tf.image.ssim(im1, im2, max_val=255, kernel_size=11,
                                filter_sigma=1.5, k1=0.01, k2=0.03)

        # Compute SSIM over tf.float32 Tensors.
        im1 = tf.image.convert_image_dtype(im1, tf.float32)
        im2 = tf.image.convert_image_dtype(im2, tf.float32)
        ssim2 = tf.image.ssim(im1, im2, max_val=1.0, kernel_size=11,
                                filter_sigma=1.5, k1=0.01, k2=0.03)
        # ssim1 and ssim2 both have type tf.float32 and are almost equal.
    ```

    Args:
        img1: First image batch. 4-D Tensor of shape `[batch, height, width,
        channels]` with only Positive Pixel Values.
        img2: Second image batch. 4-D Tensor of shape `[batch, height, width,
        channels]` with only Positive Pixel Values.
        max_val: The dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
        kernel_size: Default value 11 (size of gaussian filter).
        filter_sigma: Default value 1.5 (width of gaussian filter).
        k1: Default value 0.01
        k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
        it would be better if we took the values in the range of 0 < K2 < 0.4).
        return_index_map: If True returns local SSIM map instead of the global mean.

    Returns:
        A tensor containing an SSIM value for each image in batch or a tensor
        containing an SSIM value for each pixel for each image in batch if
        return_index_map is True. Returned SSIM values are in range (-1, 1], when
        pixel values are non-negative. Returns a tensor with shape:
        broadcast(img1.shape[:-3], img2.shape[:-3]) or broadcast(img1.shape[:-1],
        img2.shape[:-1]).
    """
    # Convert to tensor if needed.
    img1 = ops.convert_to_tensor(img1)
    img2 = ops.convert_to_tensor(img2)
    # TODO: Static verification

    # BUG: Assume images are float32 (0, 1) format.
    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    ssim_per_channel, _ = _ssim_per_channel(
        img1, img2, max_val, kernel_size, filter_sigma, k1, k2, return_index_map
    )

    # Compute average over color channels.
    return ops.mean(ssim_per_channel, [-1])

# @depreciated("This wrapper for the pyiqa metric will be removed in a future version.")
# TODO: Implement depreciation warnings for python versions before and after 3.14
class DSSIMObjective(Loss):
    """Difference of Structural Similarity (DSSIM loss function).
    Clipped between 0 and 0.5

    Note : You should add a regularization term like a l2 loss in addition to this one.
    Note : In theano, the `kernel_size` must be a factor of the output size. So 3 could
           not be the `kernel_size` for an output of 32.

    # Arguments
        k1: Parameter of the SSIM (default 0.01)
        k2: Parameter of the SSIM (default 0.03)
        kernel_size: Size of the sliding window (default 3)
        max_value: Max value of the output (default 1.0)
    """

    def __init__(
        self,
        k1=0.01,
        k2=0.03,
        kernel_size=3,
        max_value=1.0,
        name="DSSIMObjective",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.kernel_size = (kernel_size, kernel_size)
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.c1 = (self.k1 * self.max_value) ** 2
        self.c2 = (self.k2 * self.max_value) ** 2

    def call(self, y_true, y_pred):
        patches_pred = ops.image.extract_patches(
            y_pred, self.kernel_size, self.kernel_size, padding="valid"
        )

        patches_true = ops.image.extract_patches(
            y_true, self.kernel_size, self.kernel_size, padding="valid"
        )

        # Get mean
        u_true = ops.mean(patches_true, axis=-1)
        u_pred = ops.mean(patches_pred, axis=-1)
        # Get variance
        var_true = ops.var(patches_true, axis=-1)
        var_pred = ops.var(patches_pred, axis=-1)
        # Get std dev
        covar_true_pred = ops.subtract(
            ops.mean(ops.multiply(patches_true, patches_pred), axis=-1),
            ops.multiply(u_true, u_pred),
        )

        ssim = ops.multiply(
            ops.add(ops.multiply(ops.multiply(2.0, u_true), u_pred), self.c1),
            ops.add(ops.multiply(2.0, covar_true_pred), self.c2),
        )
        denom = ops.multiply(
            ops.add(ops.add(ops.square(u_true), ops.square(u_pred)), self.c1),
            ops.add(ops.add(var_pred, var_true), self.c2),
        )
        ssim = ops.divide(ssim, denom)
        ssim = ops.mean(ops.divide(ops.subtract(1.0, ssim), 2.0))
        return ssim

# @depreciated("This wrapper for the pyiqa metric will be removed in a future version.")
# TODO: Implement depreciation warnings for python versions before and after 3.14
class GRAYSSIMObjective(DSSIMObjective):
    def __init__(
        self,
        k1=0.01,
        k2=0.03,
        kernel_size=3,
        max_value=1,
        name="GSSIMObjective",
        **kwargs,
    ):
        super().__init__(k1, k2, kernel_size, max_value, name, **kwargs)

    def call(self, y_true, y_pred):
        y_t_gs = ops.image.rgb_to_grayscale(y_true)
        y_p_gs = ops.image.rgb_to_grayscale(y_pred)

        return super().call(y_t_gs, y_p_gs)


class SSIMC(Loss):
    """Loss wrapper around the `ssim` computation function.

    # Arguments
        k1: Parameter of the SSIM (default 0.01)
        k2: Parameter of the SSIM (default 0.03)
        kernel_size: Size of the sliding window (default 11)
        filter_sigma: Sigma of the gaussian filter (default 1.5)
        max_value: Max value of the output (default 1.0)
    """
    def __init__(
        self,
        name="SSIMC",
        k1=0.01,
        k2=0.03,
        kernel_size=11,
        filter_sigma=1.5,
        max_value=1.0,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.kernel_size = kernel_size
        self.filter_sigma = filter_sigma

    def call(self, y_true, y_pred):
        return 1 - ssim(y_true, y_pred, max_val=self.max_value, k1=self.k1, k2=self.k2, kernel_size=self.kernel_size, filter_sigma=self.filter_sigma)


# @depreciated("This wrapper for the pyiqa metric will be removed in a future version.")
# TODO: Implement depreciation warnings for python versions before and after 3.14
class IQASSIMC(Loss):
    def __init__(
        self,
        name="IQASSIMC",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if K.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        import pyiqa

        self.model = pyiqa.create_metric("ssimc", as_loss=True)

    def get_config(self):
        base_config = super().get_config()
        config = {"lower_better": self.model.lower_better}
        return {**base_config, **config}

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        # Library only supports channels first so change incoming data
        if K.image_data_format() == "channels_last":
            y_true = ops.moveaxis(y_true, -1, 1)
            y_pred = ops.moveaxis(y_pred, -1, 1)
        return 1 - self.model(ref=y_true, target=y_pred)

# @depreciated("This wrapper for the pyiqa metric will be removed in a future version.")
# TODO: Implement depreciation warnings for python versions before and after 3.14
class IQACWSSIM(Loss):
    def __init__(
        self,
        name="IQACWSSIM",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if K.backend() != "torch":
            raise Exception("IQA cannot be used in non-torch backend context")

        import pyiqa

        self.model = pyiqa.create_metric("cw_ssim", as_loss=True)

    def get_config(self):
        base_config = super().get_config()
        config = {"lower_better": self.model.lower_better}
        return {**base_config, **config}

    def call(
        self,
        y_true,  # reference_img
        y_pred,  # compared_img
    ):
        # Library only supports channels first so change incoming data
        if K.image_data_format() == "channels_last":
            y_true = ops.moveaxis(y_true, -1, 1)
            y_pred = ops.moveaxis(y_pred, -1, 1)
        return 1 - self.model(ref=y_true, target=y_pred)

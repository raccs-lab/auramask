from keras import Loss, ops, backend as K, KerasTensor

def ssim(
    img1: KerasTensor,
    img2: KerasTensor,
    max_val: float,
    window: int = 11,
    sigma: int = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    compensation:float = 1.0,
    return_index_map: bool = False,
    dtype:str = "float32"
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
        ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
                                filter_sigma=1.5, k1=0.01, k2=0.03)

        # Compute SSIM over tf.float32 Tensors.
        im1 = tf.image.convert_image_dtype(im1, tf.float32)
        im2 = tf.image.convert_image_dtype(im2, tf.float32)
        ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
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
        filter_size: Default value 11 (size of gaussian filter).
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
    img1 = ops.cast(img1, dtype=dtype)
    img2 = ops.cast(img2, dtype=dtype)

    _kernel = (window, window)
    _sigma = (sigma, sigma)

    # BUG: Assume images are float32 (0, 1) format.
    # Need to convert the images to float32.  Scale max_val accordingly so that
    # SSIM is computed correctly.
    c1 = ops.square(k1 * max_val)
    c2 = ops.square(k2 * max_val)

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    img1_gauss = ops.image.gaussian_blur(img1, kernel_size=_kernel, sigma=_sigma)
    img2_gauss = ops.image.gaussian_blur(img2, kernel_size=_kernel, sigma=_sigma)

    num0 = img1_gauss * img2_gauss * 2.0
    den0 = ops.square(img1_gauss) + ops.square(img2_gauss)
    luminance = (num0 + c1) / (den0 + c1)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = ops.image.gaussian_blur(img1 * img2, kernel_size=_kernel, sigma=_sigma) * 2.0
    den1 = ops.image.gaussian_blur(ops.square(img1) + ops.square(img2), kernel_size=_kernel, sigma=_sigma)

    c2 = ops.multiply(compensation, c2)
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    ssim_val = ops.multiply(luminance, cs)

    pad = (window - 1) // 2

    ssim_val = ops.image.crop_images(ssim_val, top_cropping=pad, left_cropping=pad, right_cropping=pad, bottom_cropping=pad)

    # Average over the second and the third from the last: height, width.
    if not return_index_map:
        axes = [-3, -2] if K.image_data_format() == "channels_last" else [-2, -1]
        ssim_val = ops.mean(ssim_val, axes)
        cs = ops.mean(cs, axes)

    ch_axis = [-1] if K.image_data_format() == "channels_last" else [-3]

    return ops.mean(ssim_val, axis=ch_axis)


class SSIMC(Loss):
    def __init__(
        self,
        name="SSIMC",
        k1=0.01,
        k2=0.03,
        filter_size=11,
        filter_sigma=1.5,
        max_value=1.0,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.k1 = k1
        self.k2 = k2
        self.max_value = max_value
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma

    def call(self, y_true, y_pred):
        return 1 - ssim(
            y_true,
            y_pred,
            max_val=self.max_value,
            k1=self.k1,
            k2=self.k2,
            window=self.filter_size,
            sigma=self.filter_sigma,
            dtype=self.dtype
        )


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

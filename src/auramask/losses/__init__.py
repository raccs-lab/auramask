# ruff: noqa: F401

from auramask.losses.aesthetic import AestheticLoss, IQAAestheticLoss
from auramask.losses.content import ContentLoss
from auramask.losses.embeddistance import (
    FaceEmbeddingAbsoluteLoss,
    FaceEmbeddingAbsoluteThresholdLoss,
    FaceEmbeddingLoss,
    FaceEmbeddingThresholdLoss,
)
from auramask.losses.histogram import HistogramMatchingLoss
from auramask.losses.ms_swd import MSSWD
from auramask.losses.perceptual import IQAPerceptual, PerceptualLoss
from auramask.losses.psnr import IQAPSNR
from auramask.losses.ssim import (
    IQACWSSIM,
    IQASSIMC,
    SSIMC,
)
from auramask.losses.style import StyleLoss, StyleRefs
from auramask.losses.topiq import SoftTopIQFR, TopIQFR, TopIQNR
from auramask.losses.variation import VariationLoss
from auramask.losses.zero_dce import (
    ColorConstancyLoss,
    ExposureControlLoss,
    IlluminationSmoothnessLoss,
    SpatialConsistencyLoss,
)

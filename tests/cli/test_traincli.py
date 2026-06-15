import pytest
from argparse import ArgumentParser

# Assuming the module structure allows direct import of configuration functions:
from auramask.cli.train import configure_parser, parse_args, initialize_loss, aurautils
from auramask import losses as auraloss
from keras import Loss, backend


# =============================================================================
# Setup Fixture
# =============================================================================
@pytest.fixture
def parser():
    """Provides a mock ArgumentParser instance used for testing."""
    # We return None here, but in practice, this fixture should create and return
    # a fully configured ArgumentParser object.
    return configure_parser(ArgumentParser("auramask"))


# =============================================================================
# Test Cases for Loss Arguments (--losses and -L)
# =============================================================================
@pytest.mark.parametrize("loss", [["mse"], ["mae", "mse"]])
@pytest.mark.parametrize("flag", ["-L", "--losses"])
def test_cli_parses_single_loss_no_lambda(mocker, parser, loss, flag):
    """
    Tests successful parsing when only a single loss type is specified (-L/--losses).
    The code path must correctly assume default weights (e.g., 1.0).
    """
    # Setup command line arguments simulation:
    mock_args = [
        "",
        "-m",
        "unet",  # --model-backbone
        "--dataset",
        "lfw",
        "--training",
        "0.8",
        "--testing",
        "0.2",
        flag,  # loss specified
    ] + loss

    mocker.patch("sys.argv", mock_args)
    mocker.patch("json.load", lambda x: {})

    hparams = parser.parse_args()

    # Call parse_args with mock_args:
    parsed_args = parse_args(parser, hparams)

    # Expected successful parsing verification:
    assert parsed_args.losses == loss
    assert parsed_args.lam == [1.0]  # Default weight assumed


# =============================================================================
# Test Cases for lambda Arguments (--lam and -l)
# =============================================================================
@pytest.mark.parametrize("loss", [["mse"], ["mae", "mse"]])
@pytest.mark.parametrize("lam", [[0.1], [2.0, 0.5]])
@pytest.mark.parametrize("flag", ["-l", "--lam"])
def test_cli_parses_multiple_losses_with_explicit_lambda(
    mocker, parser, loss, lam, flag
):
    """
    Tests successful parsing when multiple losses are specified, each with a unique lambda weight.
    """
    # Setup command line arguments simulation:
    mock_args = (
        [
            "",
            "-m",
            "unet",  # --model-backbone
            "--dataset",
            "lfw",
            "--training",
            "0.8",
            "--testing",
            "0.2",
            "-L",
        ]
        + loss
        + [flag]
        + [str(w) for w in lam]
    )  # Corresponding weights provided

    mocker.patch("sys.argv", mock_args)
    mocker.patch("json.load", lambda x: {})

    # Call parse_args with mock_args:

    # Ensure error raised if multiple lambda values given but shorter than list of losses
    if (len(loss) != len(lam)) and len(lam) > 1:
        try:
            hparams = parser.parse_args()
            parsed_args = parse_args(parser, hparams)
        except SystemExit as e:
            assert e.code == 2
            return

        # Assert that parse_args failed
        assert False
    # Ensure weights parsed correctly
    else:
        hparams = parser.parse_args()
        parsed_args = parse_args(parser, hparams)
        # Expected successful parsing verification:
        assert parsed_args.losses == loss
        assert parsed_args.lam == lam


# =============================================================================
# Test Cases for Loss initialization
# =============================================================================
@pytest.mark.parametrize(
    "losses",
    [
        "alex",
        "vgg",
        "squeeze",
        "mse",
        "mae",
        "ssim",
        "nima",
        "exposure",
        "color",
        "illumination",
        "spatial",
        "style",
        "content",
        "variation",
        "histogram",
        "none",
    ],
)
@pytest.mark.parametrize("lam", [0.1])
def test_cli_loss_initializing_with_params(mocker, parser, losses, lam):
    """
    Tests successful loss initialization (excluding embeddings for now)
    """

    # Set up hparams to pass into initialization function
    hparams = {
        "F": None,
        "threshold": False,
        "rho": 0,
        "metric": False,
        "losses": [losses],
        "lam": [lam],
        "color_space": aurautils.constants.ColorSpaceEnum.RGB,
        "lpips_spatial": False,
        "style_ref": auraloss.StyleRefs.STARRYNIGHT,
    }

    configd_losses, weights, cs_transform, metrics = initialize_loss(hparams)

    if losses != "none":
        assert len(configd_losses) == 1  # Only initializing one loss
        assert len(weights) == 1  # Only using a single weight
        assert issubclass(configd_losses[0].__class__, Loss)
        assert weights[0] == lam
    else:
        assert len(configd_losses) == 0  # no losses
        assert len(weights) == 0  # no losses


# =============================================================================
# Test Cases for Loss initialization (Torch Only)
# =============================================================================
@pytest.mark.parametrize(
    "losses",
    ["lpips", "ssimc", "cwssim", "iqanima", "psnr", "topiq", "topiqnr", "ms_swd"],
)
@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
def test_cli_loss_initializing_with_params_torch(mocker, parser, losses, lam):
    """
    Tests successful loss initialization (excluding embeddings for now)
    """

    # Set up hparams to pass into initialization function
    hparams = {
        "F": None,
        "threshold": False,
        "rho": 0,
        "metric": False,
        "losses": [losses],
        "lam": [lam],
        "color_space": aurautils.constants.ColorSpaceEnum.RGB,
        "lpips_spatial": False,
        "style_ref": auraloss.StyleRefs.STARRYNIGHT,
    }

    if backend.backend() != "torch":
        with pytest.raises(
            Exception, match="IQA cannot be used in non-torch backend context"
        ):
            initialize_loss(hparams)
    else:
        configd_losses, weights, cs_transform, metrics = initialize_loss(hparams)

        assert len(configd_losses) == 1  # Only initializing one loss
        assert len(weights) == 1  # Only using a single weight
        assert issubclass(configd_losses[0].__class__, Loss)
        assert weights[0] == lam

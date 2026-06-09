from argparse import ArgumentParser, BooleanOptionalAction
from keras.mixed_precision import set_dtype_policy
from auramask.utils import cli as cli_utils
from auramask.cli import test as test_cli, train as train_cli
from random import choice
from string import ascii_uppercase


def configure_parser():
    parser = ArgumentParser(
        prog="auramask",
        description="The AuraMask Toolkit for the development of Adversarial Transformation Networks to sabotage algorithmic inference.",
    )

    parser.add_argument(
        "-S",
        "--seed",
        type=str,
        default="".join(choice(ascii_uppercase) for _ in range(12)),
    )
    parser.add_argument(
        "--wandb", default=False, type=bool, action=BooleanOptionalAction
    )
    parser.add_argument("-v", "--verbose", default=1, type=int)
    parser.add_argument(
        "--note", default=False, type=bool, action=BooleanOptionalAction
    )
    parser.add_argument("--log-dir", default=None, type=cli_utils.dir_path)
    parser.add_argument(
        "--mixed-precision",
        default=True,
        type=bool,
        action=BooleanOptionalAction,
    )

    # Subcommand collection
    subparsers = parser.add_subparsers(
        help="choose to train new model or test existing one",
        dest="command",
        required=True,
    )

    # Training subcommand
    train_command = subparsers.add_parser(
        "train", description="Training interface for AuraMask toolkit."
    )

    train_cli.configure_parser(train_command)

    # Test subcommand
    test_command = subparsers.add_parser(
        "test", description="Evaluation interface for AuraMask toolkit."
    )

    test_cli.configure_parser(test_command)

    return parser


def main():
    parser = configure_parser()
    hparams = parser.parse_args()
    if hparams.command == "train":
        from json import load

        hparams.model_config = load(hparams.model_config)
    hparams = hparams.__dict__
    action = hparams.pop("command")

    if hparams["mixed_precision"]:
        print(f"Using mixed precision (f16) for {action}")
        set_dtype_policy("mixed_float16")

    # Obtain input if note specified
    hparams["log_dir"] = cli_utils.setup_logging(
        hparams.pop("wandb"), hparams["log_dir"], hparams["seed"]
    )

    cli_utils.set_seed(hparams["seed"])

    if action == "train":
        train_cli.apply_params(hparams)
    elif action == "test":
        test_cli.apply_params(hparams)
    else:
        exit(1)

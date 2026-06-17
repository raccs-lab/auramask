from argparse import ArgumentTypeError
from pathlib import Path
from keras.utils import set_random_seed
from hashlib import sha256
from os import path as ospath, environ
from datetime import datetime


def dir_path(path: str) -> str:
    """Creates a given path if it doesn't exist.

    Args:
        path (str): desired path as a string

    Raises:
        ArgumentTypeError: Raised if the given path has more than 2 missing parents or if the file already exists.

    Returns:
        str: Full path if created or "" if no path is provided.
    """
    if path:
        path = Path(path)
        try:
            if not path.parent.parent.exists():
                raise FileNotFoundError()
            path.mkdir(parents=True, exist_ok=True)
            return str(path.absolute())
        except FileNotFoundError:
            raise ArgumentTypeError(
                f"The directory {path} cannot have more than 2 missing parents."
            )
        except FileExistsError:
            raise ArgumentTypeError(f"The directory {path} exists as a file")
    return path


def set_seed(seed: str):
    """Converts the provided string to an integer for use as the random seed in the keras library.

    Args:
        seed (str): String to enforce as the underlying seed
    """
    seed_hash = int(sha256(seed.encode("utf-8")).hexdigest(), 16) % 10**8
    set_random_seed(seed_hash)


def setup_logging(wandb_online: bool, logdir: str, seed: str):
    """Sets up the log directory in the file system.

    Args:
        wandb_online (bool): online or offline mode for Weights and Biases
        logdir (str): directory to store the log files
        seed (str): seed string used for organizing runs
    """
    if not wandb_online:
        environ["WANDB_MODE"] = "offline"

    if not logdir:
        logdir = Path(
            ospath.join(
                ospath.curdir,
                "logs",
                datetime.now().strftime("%m-%d"),
                seed,
            )
        )
    else:
        logdir = Path(ospath.join(logdir))
    logdir.mkdir(parents=True, exist_ok=True)

    return logdir

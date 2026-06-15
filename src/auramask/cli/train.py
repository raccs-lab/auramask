import os
import keras
from keras.src.utils import file_utils
import wandb
import auramask
from auramask import utils as aurautils
from argparse import (
    ArgumentParser,
    FileType,
    BooleanOptionalAction,
)
import json
from ast import literal_eval

# Global hparams object
keras.config.disable_traceback_filtering()
# Normalize network to use channels last ordering
keras.backend.set_image_data_format("channels_last")


def configure_parser(parser: ArgumentParser):
    parser.add_argument(
        "-m",
        "--model-backbone",
        type=aurautils.constants.BaseModels,
        action=aurautils.constants.EnumAction,
        required=True,
    )
    parser.add_argument("--model-config", type=FileType("r"))
    parser.add_argument(
        "-F",
        type=aurautils.constants.FaceEmbedEnum,
        nargs="+",
        required=False,
        action=aurautils.constants.EnumAction,
    )
    parser.add_argument(
        "--threshold", default=True, type=bool, action=BooleanOptionalAction
    )
    parser.add_argument("-p", "--rho", type=float, default=1.0)
    parser.add_argument("-a", "--alpha", type=float, default=2e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("-e", "--epsilon", type=float, default=0.03)
    parser.add_argument("-B", "--batch-size", dest="batch", type=int, default=32)
    parser.add_argument("-E", "--epochs", type=int, default=5)
    parser.add_argument("-s", "--steps-per-epoch", type=int, default=-1)
    parser.add_argument("-d", "--dims", type=int, default=256)
    parser.add_argument(
        "--lpips-spatial",
        type=bool,
        required=False,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "-L",
        "--losses",
        type=str,
        default=["none"],
        choices=[
            "lpips",
            "alex",
            "vgg",
            "squeeze",
            "mse",
            "mae",
            "ssim",
            "ssimc",
            "cwssim",
            "nima",
            "iqanima",
            "psnr",
            "exposure",
            "color",
            "illumination",
            "spatial",
            "style",
            "content",
            "variation",
            "histogram",
            "topiq",
            "topiqnr",
            "ms_swd",
            "none",
        ],
        nargs="+",
    )
    parser.add_argument("-l", "--lam", type=float, default=[1.0], nargs="+")
    parser.add_argument(
        "--adaptive-loss",
        type=str,
        required=False,
        choices=["loss-weighted", "normalized", "base"],
    )
    parser.add_argument(
        "--adaptive-loss-frequency", type=str, required=False, default="epoch"
    )
    parser.add_argument(
        "--gradient-alteration", type=str, required=False, choices=["pc-grad"]
    )
    parser.add_argument(
        "--style-ref",
        type=auramask.losses.StyleRefs,
        action=aurautils.constants.EnumAction,
        default=auramask.losses.StyleRefs.STARRYNIGHT,
        required=False,
    )
    parser.add_argument("--training", type=float, required=True)
    parser.add_argument("--testing", type=float, required=True)
    parser.add_argument(
        "--eager", default=False, type=bool, action=BooleanOptionalAction
    )
    parser.add_argument(
        "-C",
        "--color-space",
        type=aurautils.constants.ColorSpaceEnum,
        action=aurautils.constants.EnumAction,
        default=aurautils.constants.ColorSpaceEnum.RGB,
        required=False,
    )
    parser.add_argument(
        "--checkpoint", default=False, type=bool, action=BooleanOptionalAction
    )
    parser.add_argument(
        "-D",
        "--dataset",
        default="lfw",
        type=aurautils.constants.DatasetEnum,
        action=aurautils.constants.EnumAction,
        required=True,
    )
    parser.add_argument(
        "--instagram-filter",
        type=aurautils.constants.InstaFilterEnum,
        action=aurautils.constants.EnumAction,
        required=False,
    )
    parser.add_argument(
        "--metric", type=bool, default=False, action=BooleanOptionalAction
    )

    return parser


def parse_args(parser: ArgumentParser):
    args = parser.parse_args()

    # Check if loss weights are appropriately sized
    if len(args.losses) != len(args.lam) and len(args.lam) > 1:
        parser.error(
            f"The length of lam values must equal that of losses argument. lam={len(args.lam)} != losses={len(args.losses)}"
        )

    from json import load

    args.model_config = load(args.model_config)

    return args


def load_data(hparams: dict):
    ds: aurautils.constants.DatasetEnum = hparams["dataset"]
    train_size, test_size = hparams["training"], hparams["testing"]

    # In the case that a number of samples is passed in instead of a percentage of the test split
    if train_size > 1.0:
        train_size = int(train_size)
    if test_size > 1.0:
        test_size = int(test_size)

    insta: aurautils.constants.InstaFilterEnum = hparams["instagram_filter"]
    t_ds, v_ds = ds.load_dataset(
        hparams["input"],
        train_size,
        test_size,
        hparams["batch"],
        insta.filter_transform if insta else None,
    )

    hparams["dataset"] = ds.name.lower()

    return t_ds, v_ds


def initialize_loss(hparams: dict):
    losses = []
    weights = []
    loss_config = {}
    cs_transforms = []
    metrics = []

    is_not_rgb = hparams["color_space"].name.casefold() != "rgb"
    F = hparams.pop("F")
    threshold = hparams.pop("threshold")
    rho = hparams.pop("rho")
    metr = hparams.pop("metric")
    if F:
        for f in F:
            if threshold:  # Loss with thresholding
                losses.append(
                    auramask.losses.FaceEmbeddingAbsoluteThresholdLoss(
                        f=f,
                        threshold=f.get_threshold(),
                        # negative_slope=0.5,
                    )
                )
                weights.append(rho)
            else:  # Loss as described by ReFace
                losses.append(auramask.losses.FaceEmbeddingAbsoluteLoss(f=f))
                weights.append(rho / len(F))

            if metr:
                metrics.append(auramask.metrics.CosineDistance(f=f))

            loss_config[losses[-1].name] = losses[-1].get_config() | {
                "weight": weights[-1]
            }
            cs_transforms.append(
                is_not_rgb
            )  # Determine if it needs to be transformed to rgb space

    if "none" not in hparams["losses"]:
        lam = hparams.pop("lam")
        loss_in = hparams.pop("losses")
        if len(lam) <= 1:
            w = lam[0] if len(lam) > 0 else 1.0
            iters = zip(loss_in, [w] * len(loss_in))
        else:
            iters = zip(loss_in, lam)

        for loss_i, w_i in iters:
            if loss_i == "mse":
                tmp_loss = keras.losses.MeanSquaredError()
                cs_transforms.append(False)
            elif loss_i == "mae":
                tmp_loss = keras.losses.MeanAbsoluteError()
                cs_transforms.append(False)
            elif loss_i == "ssim":
                tmp_loss = auramask.losses.SSIMC()
                cs_transforms.append(False)
            elif loss_i == "ssimc":
                tmp_loss = auramask.losses.IQASSIMC()
                cs_transforms.append(False)
            elif loss_i == "cwssim":
                tmp_loss = auramask.losses.IQACWSSIM()
                cs_transforms.append(False)
            elif loss_i == "nima":
                tmp_loss = auramask.losses.AestheticLoss(
                    name="NIMA-A", backbone="inceptionresnetv2"
                )
                cs_transforms.append(is_not_rgb)
            elif loss_i == "iqanima":
                tmp_loss = auramask.losses.IQAAestheticLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "exposure":
                tmp_loss = auramask.losses.ExposureControlLoss(mean_val=0.6)
                cs_transforms.append(is_not_rgb)
            elif loss_i == "color":
                tmp_loss = auramask.losses.ColorConstancyLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "illumination":
                tmp_loss = auramask.losses.IlluminationSmoothnessLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "spatial":
                tmp_loss = auramask.losses.SpatialConsistencyLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "style":
                style = hparams.pop("style_ref")
                tmp_loss = auramask.losses.StyleLoss(reference=style)
                cs_transforms.append(is_not_rgb)
            elif loss_i == "variation":
                tmp_loss = auramask.losses.VariationLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "content":
                tmp_loss = auramask.losses.ContentLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "topiq":
                tmp_loss = auramask.losses.TopIQFR()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "topiqnr":
                tmp_loss = auramask.losses.TopIQNR()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "histogram":
                tmp_loss = auramask.losses.HistogramMatchingLoss()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "psnr":
                tmp_loss = auramask.losses.IQAPSNR()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "lpips":
                tmp_loss = auramask.losses.IQAPerceptual()
                cs_transforms.append(is_not_rgb)
            elif loss_i == "ms_swd":
                style = hparams.pop("style_ref")
                tmp_loss = auramask.losses.MSSWD(reference=style)
                cs_transforms.append(is_not_rgb)
            else:
                spatial = hparams.pop("lpips_spatial")
                tmp_loss = auramask.losses.PerceptualLoss(
                    backbone=loss_i, spatial=spatial, tolerance=0.05
                )
                cs_transforms.append(is_not_rgb)

            losses.append(tmp_loss)
            weights.append(w_i)
            loss_config[tmp_loss.name] = tmp_loss.get_config() | {"weight": w_i}

    if not is_not_rgb:
        cs_transforms = None

    hparams["losses"] = loss_config

    return losses, weights, cs_transforms, metrics


def handle_environment_config() -> dict:
    """Retrieves configuration string from AURAMASK_CONFIG environment variable to apply it to the model configuration.

    Returns:
        dict: dictionary of values to update the existing model config with.
    """
    # Allows modifying the config at calling with the AURAMASK_CONFIG environment variable
    cfg_mod: dict = literal_eval(os.getenv("AURAMASK_CONFIG", "{}"))
    for key, val in cfg_mod.items():
        if isinstance(val, str) and val.lower() in ["true", "false"]:
            cfg_mod[key] = True if val.lower() == "true" else False
    return cfg_mod


def initialize_model(hparams: dict):
    losses, losses_w, losses_t, metrics = initialize_loss(hparams)

    adaptive_callback = []

    hparams["model_config"].update(handle_environment_config())

    # Ensure model name is in a parseable format
    hparams["model"] = hparams.pop("model_backbone").name.lower()

    # TODO: Refacor AuraMask model building script
    model = auramask.AuraMask(hparams)

    # keras.utils.plot_model(model, expand_nested=True, show_shapes=True)

    # TODO: Support more optimizers
    if hparams["optimizer"] == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=hparams["alpha"], clipnorm=1.0)
    elif hparams["optimizer"] == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=hparams["alpha"])
    else:
        raise Exception("Unrecognized optimizer")

    hparams["optimizer"] = optimizer.get_config()

    if hparams["adaptive_loss"] is not None:
        try:
            freq = int(hparams["adaptive_loss_frequency"])
        except ValueError:
            freq = hparams["adaptive_loss_frequency"]

        adaptive_callback = [
            auramask.callbacks.AdaptiveLossCallback(
                [lss.name for lss in losses],
                weights=losses_w,
                frequency=freq,
                algorithm=hparams["adaptive_loss"],
                clip_weights=True,
                backup_dir=os.path.join(hparams["log_dir"], "backup"),
            )
        ]

    if hparams["gradient_alteration"] is not None:
        grad_fn = auramask.pcgrad.compute_pc_grads
    else:
        grad_fn = None

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=losses_w,
        metrics=metrics,
        run_eagerly=hparams.pop("eager"),
        jit_compile=False,
        auto_scale_loss=True,
        gradient_alter=grad_fn,
    )

    return model, adaptive_callback


def get_sample_data(ds):
    if keras.backend.backend() == "tensorflow":
        for x in ds.take(1):
            inp = x[0]
    else:
        for batch in ds:
            inp = batch[0]
            break

    return inp


def init_callbacks(hparams: dict, sample, logdir, note: str = ""):
    checkpoint = hparams.pop("checkpoint")
    tmp_hparams = hparams
    tmp_hparams["color_space"] = (
        tmp_hparams["color_space"].name if tmp_hparams["color_space"] else "rgb"
    )
    tmp_hparams["input"] = str(tmp_hparams["input"])
    tmp_hparams["task_id"] = str(os.getenv("SLURM_JOB_ID", None))

    train_callbacks = []
    wandb.init(
        project="auramask",
        id=os.getenv("WANDB_RUN_ID", None),
        dir=logdir,
        config=tmp_hparams,
        name=os.getenv("SLURM_JOB_NAME", None),
        notes=note,
        resume="allow",
    )

    if checkpoint:
        train_callbacks.append(
            auramask.callbacks.AuramaskCheckpoint(
                filepath=os.path.join(logdir, "checkpoints"),
                freq_mode="epoch",
                save_weights_only=True,
                save_freq=int(os.getenv("AURAMASK_CHECKPOINT_FREQ", 100)),
            )
        )

    train_callbacks.append(auramask.callbacks.AuramaskWandbMetrics(log_freq="epoch"))
    train_callbacks.append(
        auramask.callbacks.AuramaskCallback(
            validation_data=sample,
            data_table_columns=["idx", "orig", "aug"],
            pred_table_columns=["epoch", "idx", "pred", "mask"],
            log_freq=int(os.getenv("AURAMASK_LOG_FREQ", 5)),
        )
    )
    # train_callbacks.append(
    #     keras.callbacks.ReduceLROnPlateau(
    #         monitor="val_loss", patience=10, verbose=1, cooldown=50, min_lr=2e-9,
    #     )
    # )
    train_callbacks.append(auramask.callbacks.AuramaskStopOnNaN())
    return train_callbacks


def apply_params(hparams: dict):
    dims = hparams.pop("dims")
    hparams["input"] = (dims, dims)
    verbose = hparams.pop("verbose")

    model, callbacks = initialize_model(hparams)

    note = hparams.pop("note")

    if note:
        note = input("Give a note for this run")
    else:
        note = ""

    hparams["auramask_params"] = hparams["model_config"]

    hparams["model_config"] = model.get_config()

    bkup_callback = keras.callbacks.BackupAndRestore(
        double_checkpoint=True,
        delete_checkpoint=False,
        backup_dir=os.path.join(hparams["log_dir"], "backup"),
    )

    if file_utils.exists(bkup_callback._training_metadata_path):
        with file_utils.File(bkup_callback._training_metadata_path, "r") as f:
            training_metadata = json.loads(f.read())
        epoch = training_metadata["epoch"]
    else:
        epoch = 0

    # Load the training and validation data
    t_ds, v_ds = load_data(hparams)
    v = get_sample_data(v_ds)

    # On resume, make sure the iterable dataset is shuffled according to the HF scheme
    t_ds.dataset.set_epoch(epoch)
    v_ds.dataset.set_epoch(0)
    model(v)

    callbacks.extend(init_callbacks(hparams, v, hparams.pop("log_dir"), note))
    callbacks.append(bkup_callback)

    training_history = model.fit(
        t_ds,
        callbacks=callbacks,
        epochs=hparams["epochs"],
        verbose=verbose,
        validation_data=v_ds,
        steps_per_epoch=hparams["steps_per_epoch"],
    )
    return training_history

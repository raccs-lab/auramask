import keras

from auramask import callbacks as callbacks
from auramask import layers as layers
from auramask import losses as losses
from auramask import metrics as metrics
from auramask import models as models
from auramask import utils as utils


def AuraMask(config: dict, weights: str | None = None):
    eps = config["epsilon"]
    base_model: utils.constants.BaseModels = utils.constants.BaseModels[
        config["model"].upper()
    ]
    model_config: dict = config["model_config"]

    activation: str = model_config["output_activation"]
    if activation.lower() == "tanh":
        activation_fn = keras.activations.tanh
    else:
        activation_fn = activation

    model_config["output_activation"] = None

    if base_model in [
        utils.constants.BaseModels.ZERODCE,
        utils.constants.BaseModels.RESZERODCE,
    ]:
        postproc = models.zero_dce.get_enhanced_image
        preproc = None
    else:

        def preproc(inputs):
            inputs = keras.layers.Rescaling(scale=2, offset=-1)(inputs)
            return inputs

        if model_config["n_labels"] == 24:
            activation_fn = "sigmoid"  # postprocessing step only works for 0-1 range
            postproc = models.zero_dce.get_enhanced_image
        elif eps < 1:

            def postproc(x: keras.KerasTensor, inputs: keras.KerasTensor):
                x = keras.ops.multiply(eps, x)
                out = keras.ops.add(x, inputs)
                out = keras.ops.clip(out, 0.0, 1.0)
                return [out, x]
        else:
            if activation == "tanh":

                def postproc(x: keras.KerasTensor, inputs: keras.KerasTensor):
                    x = keras.layers.Rescaling(scale=1 / 2.0, offset=0.5)(x)
                    return [
                        x,
                        keras.ops.stop_gradient(
                            keras.ops.abs(keras.ops.subtract(inputs, x))
                        ),
                    ]
            else:

                def postproc(x: keras.KerasTensor, inputs: keras.KerasTensor):
                    return [
                        x,
                        keras.ops.stop_gradient(
                            keras.ops.abs(keras.ops.subtract(inputs, x))
                        ),
                    ]

    dim = int(config["input"][0] * 0.875)

    model = base_model.build_backbone(
        model_config=model_config,
        input_shape=(dim, dim, 3)
        if keras.backend.image_data_format() == "channels_last"
        else (3, dim, dim),
        preprocess=preproc,
        activation_fn=activation_fn,
        post_processing=postproc,
        name="AuraMask",
    )

    if keras.backend.image_data_format() == "channels_last":
        model.build_from_config({"input_shape": (None, dim, dim, 3)})
    else:
        model.build_from_config({"input_shape": (None, 3, dim, dim)})

    if weights:
        model.load_weights(weights)

    model_config["output_activation"] = activation

    return model

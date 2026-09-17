import wandb
from keras import ops, preprocessing
from keras.callbacks import Callback
from wandb.sdk.lib import telemetry


class AuramaskCallback(Callback):
    def __init__(
        self,
        validation_data,
        num_samples=100,
        log_freq=1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # From wandb.integration.keras.callbacks.tables_builder.py
        if wandb.run is None:
            raise wandb.Error(
                "You must call `wandb.init()` first before using this callback."
            )

        with telemetry.context(run=wandb.run) as tel:
            tel.feature.keras_wandb_eval_callback = True

        self.x = validation_data[:num_samples]
        self.log_freq = log_freq

    def on_train_begin(self, logs: dict | None = None) -> None:
        if wandb.run.step > 1:
            pass
        else:
            wandb.log(
                {
                    "image": [
                        wandb.Image(
                            preprocessing.image.array_to_img(x_i * 255, scale=False)
                        )
                        for x_i in self.x
                    ],
                },
                step=0,
            )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if epoch % self.log_freq == 0:
            self.save_results()

    def save_results(self):
        y, mask = self.model(self.x, training=False)
        if ops.shape(mask)[-1] > 3 and ops.shape(mask)[-1] % 3 == 0:
            data = {}
            r = ops.split(mask, 8, axis=-1)
            y = ops.convert_to_numpy(y)
            r = ops.convert_to_numpy(r)

            for i, r_n in enumerate(r):
                data["r%d" % i] = [
                    wandb.Image(
                        preprocessing.image.array_to_img(m_i * 255, scale=False)
                    )
                    for m_i in r_n
                ]

            data["image"] = [
                wandb.Image(preprocessing.image.array_to_img(y_i * 255, scale=False))
                for y_i in y
            ]
            wandb.log(data, step=wandb.run.step)

        else:
            y = ops.convert_to_numpy(y)
            mask = ops.convert_to_numpy(mask)
            wandb.log(
                {
                    "image": [
                        wandb.Image(
                            preprocessing.image.array_to_img(y_i * 255, scale=False)
                        )
                        for y_i in y
                    ],
                    "mask": [
                        wandb.Image(
                            preprocessing.image.array_to_img(m_i * 255, scale=False)
                        )
                        for m_i in mask
                    ],
                },
                step=wandb.run.step,
            )

    def on_train_end(self, logs: dict | None = None) -> None:
        self.save_results()

from typing import Literal

from keras import backend as K
from keras import ops
from softadapt.callbacks import AdaptiveLossCallback


class AdaptiveLossCallback(AdaptiveLossCallback):
    def __init__(
        self,
        components: list[str],
        weights: list[float],
        frequency: Literal["epoch", "batch"] | int = "epoch",
        beta: float = 0.1,
        accuracy_order: int | None = None,
        algorithm: Literal["loss-weighted", "normalized", "base"] = "base",
        calculate_on_validation=False,
        clip_weights: bool = False,
        backup_dir: str | None = None,
    ):
        super().__init__(
            components,
            weights,
            frequency,
            beta,
            accuracy_order,
            algorithm,
            calculate_on_validation,
            backup_dir,
        )
        # self.debug = True
        self.clip_weights = clip_weights

    @property
    def weights(self):
        return self.model.loss_weights

    @weights.setter
    def weights(self, value: list):
        if self.clip_weights:
            self.model.loss_weights = [
                ops.maximum(w, K.epsilon()) for w in value
            ]  # Clip weights to avoid going below 0
        else:
            self.model.loss_weights = value

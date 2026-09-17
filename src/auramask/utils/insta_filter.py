from enum import Enum

import numpy as np
import pilgram2 as pilgram
from albumentations import CLAHE
from keras.utils import array_to_img, img_to_array


class InstaFilterEnum(Enum):
    _1977 = 0
    ADEN = 1
    ASHBY = 2
    AMARO = 3
    BRANNAN = 4
    BROOKLYN = 5
    CHARMES = 6
    CLARENDON = 7
    CREMA = 8
    DOGPATCH = 9
    EARLYBIRD = 10
    GINGHAM = 11
    GINZA = 12
    HEFE = 13
    HELENA = 14
    HUDSON = 15
    INKWELL = 16
    JUNO = 17
    KELVIN = 18
    LARK = 19
    LOFI = 20
    LUDWIG = 21
    MAVEN = 22
    MAYFAIR = 23
    MOON = 24
    NASHVILLE = 25
    PERPETUA = 26
    POPROCKET = 27
    REYES = 28
    RISE = 29
    SIERRA = 30
    SKYLINE = 31
    SLUMBER = 32
    STINSON = 33
    SUTRO = 34
    TOASTER = 35
    VALENCIA = 36
    WALDEN = 37
    WILLOW = 38
    XPRO2 = 39

    def filter_transform(self, images: np.ndarray) -> dict[str, np.ndarray]:
        fn = getattr(pilgram, self.name.lower())
        clahe = CLAHE(clip_limit=1.0, tile_grid_size=(8, 8))
        batch = {"image": None, "target": None}
        batch["image"] = np.stack([clahe(image=ex)["image"] for ex in images])
        batch["target"] = np.stack(
            [img_to_array(fn(array_to_img(f)), dtype="uint8") for f in batch["image"]]
        )
        return batch

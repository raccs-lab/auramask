from enum import Enum
import os
from typing import Callable, TypedDict
from datasets import load_dataset, Dataset
from keras import backend as K
from auramask.utils import preprocessing
from os import cpu_count
import numpy as np
from albumentations import CLAHE


class DatasetEnum(Enum):
    LFW = ("logasja/lfw", "default", ["image"])
    FDF256 = ("logasja/FDF", "default", ["image"])
    FDF = ("logasja/FDF", "fdf", ["image"])
    VGGFACE2 = ("logasja/VGGFace2", "256", ["image"])

    class LoaderConfig(TypedDict):
        w: int
        h: int
        crop: bool

    def fetch_dataset(self) -> Dataset:
        """Fetches the dataset from huggingface given the enumerated values above.

        Returns:
            Dataset: dataset with training split.
        """
        dataset, name, _ = self.value
        ds = load_dataset(
            dataset,
            name,
            split="train",
            num_proc=cpu_count(),
        )

        return ds

    class GeomConfig(TypedDict):
        augs_per_image: int
        rate: float

    class AugConfig(TypedDict):
        augs_per_image: int
        rate: float
        magnitude: float

    @staticmethod
    def get_augmenters(geom_config: GeomConfig, aug_config: AugConfig) -> dict:
        """Static method for the configuring of Albumentation-based image augmentation pipelines.

        Args:
            geom_config (GeomConfig): configuration dictionary for geometric augmentations
            aug_config (AugConfig): configuration dictionary for non-geometric augmentations

        Returns:
            dict: albumentations transform pipelines for geometric and non-geometric
        """
        return {
            "geom": preprocessing.gen_geometric_aug_layers(**geom_config),
            "aug": preprocessing.gen_non_geometric_aug_layers(**aug_config),
        }

    @staticmethod
    def data_collater(feature_batch: dict, args: LoaderConfig) -> dict:
        loader = preprocessing.gen_image_loading_layers(**args)

        batch = {}

        for k, values in feature_batch.items():
            first = values[0]
            if isinstance(first, np.ndarray) and np.ndim(first) == 3:
                batch[k] = np.array([loader(image=i)["image"] for i in values])
            else:
                batch[k] = np.array(values)
        del loader
        return batch

    def data_augmenter(self, examples: list[dict] | dict, geom, aug):
        cols = self.value[-1]
        if isinstance(examples, list):
            x = np.stack([ex[cols[0]] for ex in examples], dtype="float32")
            # Determine if desired output is a referenced output or the original image
            if len(cols) > 1:
                y = np.stack([ex[cols[1]] for ex in examples], dtype="float32")
            elif "target" in examples[0].keys():
                y = np.stack([ex["target"] for ex in examples], dtype="float32")
            else:
                y = np.copy(x)  # Separate out target
        else:
            x = np.stack(examples[cols[0]], dtype="float32")
            # Determine if desired output is a referenced output or the original image
            if len(cols) > 1:
                y = np.stack(examples[cols[1]], dtype="float32")
            elif "target" in examples.keys():
                y = np.stack(examples["target"], dtype="float32")
            else:
                y = np.copy(x)  # Separate out target

        a = [
            geom(image=i, mask=j) for i, j in zip(x, y)
        ]  # Apply geometric modifications
        x, y = np.stack([i["image"] for i in a]), np.stack([j["mask"] for j in a])

        x = np.stack([aug(image=i)["image"] for i in x])  # Pixel-level modifications
        return {
            "image": x,
            "target": y,
        }

    def load_dataset(
        self,
        dims: tuple[int, int],
        train_size: float | int | None,
        test_size: float | int | None,
        batch: int = 32,
        prefilter: Callable | None = None,
    ):
        # Fetch and split requested dataset
        ds = self.fetch_dataset()
        ds = ds.train_test_split(test_size=test_size, train_size=train_size)

        if prefilter:

            def transform_train(examples: dict):
                examples.update(prefilter(examples["image"]))
                examples = DatasetEnum.data_collater(
                    examples, {"w": dims[0], "h": dims[1]}
                )
                augmenters = self.get_augmenters(
                    {"augs_per_image": 1, "rate": 0.5},
                    {"augs_per_image": 1, "rate": 0.2, "magnitude": 0.5},
                )

                examples = self.data_augmenter(
                    examples, augmenters["geom"], augmenters["aug"]
                )
                return examples

            def transform_test(examples):
                examples.update(prefilter(examples["image"]))
                examples = DatasetEnum.data_collater(
                    examples, {"w": dims[0], "h": dims[1]}
                )
                return examples
        else:

            def transform_train(examples):
                if isinstance(examples, np.ndarray):
                    examples = {"image": np.astype(examples, np.uint8)}

                clahe = CLAHE(clip_limit=1.0, tile_grid_size=(8, 8))

                examples["target"] = np.stack(
                    [clahe(image=ex)["image"] for ex in examples["image"]]
                )
                examples = DatasetEnum.data_collater(
                    examples, {"w": dims[0], "h": dims[1]}
                )
                augmenters = self.get_augmenters(
                    {"augs_per_image": 1, "rate": 0.5},
                    {"augs_per_image": 1, "rate": 0.2, "magnitude": 0.5},
                )

                examples = self.data_augmenter(
                    examples, augmenters["geom"], augmenters["aug"]
                )
                return examples

            def transform_test(examples):
                if isinstance(examples, np.ndarray):
                    examples = {"image": np.astype(examples, np.uint8)}
                examples = DatasetEnum.data_collater(
                    examples, {"w": dims[0], "h": dims[1]}
                )
                if "target" not in examples.keys():
                    examples["target"] = np.copy(examples["image"])
                return examples

        ds["train"] = (
            ds["train"]
            .to_iterable_dataset(num_shards=int(os.getenv("DL_TRAIN_WORKERS", 8)))
            .select_columns(self.value[2])
            .with_format("numpy")
            .map(
                transform_train,
                batched=True,
                batch_size=batch,
                input_columns=self.value[2],
            )
            .shuffle(buffer_size=10_000)
        )
        ds["test"] = (
            ds["test"]
            .to_iterable_dataset(num_shards=int(os.getenv("DL_TEST_WORKERS", 8)))
            .select_columns(self.value[2])
            .with_format("numpy")
            .map(
                transform_test,
                batched=True,
                batch_size=batch,
                input_columns=self.value[2],
            )
        )

        if K.backend() == "torch":
            from torch.utils.data import DataLoader, _utils

            # TODO: `collate_fn` implementation is potentially fragile as it uses private part of pytorch library
            return DataLoader(
                ds["train"],
                batch_size=batch,
                collate_fn=lambda x: tuple(_utils.collate.default_collate(x).values()),
            ), DataLoader(
                ds["test"],
                batch_size=batch,
                collate_fn=lambda x: tuple(_utils.collate.default_collate(x).values()),
            )

    def _load_data_torch(
        self,
        train_ds: Dataset,
        test_ds: Dataset,
        batch: int,
        train_t: Callable,
        test_t: Callable,
    ):
        from torch.utils.data import DataLoader

        def collate_train(examples: list[dict]):
            examples = {k: [dic[k] for dic in examples] for k in examples[0]}
            examples = train_t(examples)
            return (examples["image"], examples["target"])

        train_ds = DataLoader(
            train_ds,
            batch,
            drop_last=True,
            collate_fn=collate_train,
            num_workers=int(os.getenv("DL_TRAIN_WORKERS", 8)),
        )

        def collate_test(examples: list[dict]):
            examples = {k: [dic[k] for dic in examples] for k in examples[0]}
            examples = test_t(examples)
            return (examples["image"], examples["target"])

        test_ds = DataLoader(
            test_ds,
            batch,
            drop_last=True,
            collate_fn=collate_test,
            num_workers=int(os.getenv("DL_TEST_WORKERS", 8)),
        )

        return train_ds, test_ds

    def _load_data_tf(self, train_ds: Dataset, test_ds: Dataset, batch: int):
        train_ds = train_ds.to_tf_dataset(
            columns=self.value[2],
            batch_size=batch,
            collate_fn=self.data_collater,
            # collate_fn_args={"args": {"w": dim[0], "h": dim[1]}},
            prefetch=False,
            shuffle=True,
            drop_remainder=True,
        )

        test_ds = (
            test_ds.to_tf_dataset(
                columns=self.value[2],
                batch_size=batch,
                collate_fn=self.data_collater,
                # collate_fn_args={"args": {"w": dim[0], "h": dim[1]}},
                prefetch=True,
                drop_remainder=True,
            )
            .cache()
            .prefetch(-1)
        )
        return train_ds, test_ds

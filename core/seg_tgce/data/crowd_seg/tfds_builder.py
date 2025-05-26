import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt

from seg_tgce.data.crowd_seg.__retrieve import (
    _BUCKET_NAME,
    MASKS_OBJECT_NAME,
    PATCHES_OBJECT_NAME,
)
from seg_tgce.data.crowd_seg.types import Stage

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

CLASSES_DEFINITION = {
    0: "Ignore",
    1: "Other",
    2: "Tumor",
    3: "Stroma",
    4: "Benign Inflammation",
    5: "Necrosis",
}

REAL_SCORERS = [
    "NP1",
    "NP2",
    "NP3",
    "NP4",
    "NP5",
    "NP6",
    "NP7",
    "NP8",
    "NP9",
    "NP10",
    "NP11",
    "NP12",
    "NP13",
    "NP14",
    "NP15",
    "NP16",
    "NP17",
    "NP18",
    "NP19",
    "NP20",
    "NP21",
]

AGGREGATED_SCORERS = ["MV", "STAPLE"]

ALL_SCORER_TAGS = REAL_SCORERS + AGGREGATED_SCORERS + ["expert"]

DEFAULT_IMG_SIZE = (512, 512)
METADATA_PATH = Path(__file__).resolve().parent / "metadata"


class CrowdSegDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for crowd segmentation dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(
        self,
        *,
        image_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    ):
        """Initialize the dataset builder.

        Args:
            image_size: Tuple[int, int] = DEFAULT_IMG_SIZE: Image size for the dataset.
        """
        self.image_size = image_size
        super().__init__()

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Crowd segmentation dataset for histology images.",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(*self.image_size, 3)),
                    "masks": tfds.features.Sequence(
                        tfds.features.Tensor(
                            shape=(*self.image_size, 1), dtype=tf.uint8
                        )
                    ),
                    "labelers": tfds.features.Sequence(tfds.features.Text()),
                }
            ),
            supervised_keys=("image", "masks"),
            homepage="https://github.com/your-repo/crowd-seg",
            citation="""@article{your-citation}""",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        patches_url = f"https://{_BUCKET_NAME}.s3.amazonaws.com/{PATCHES_OBJECT_NAME}"
        masks_url = f"https://{_BUCKET_NAME}.s3.amazonaws.com/{MASKS_OBJECT_NAME}"

        patches_path = dl_manager.download_and_extract(patches_url)
        masks_path = dl_manager.download_and_extract(masks_url)

        patches_dir = os.path.join(patches_path, "patches")
        masks_dir = os.path.join(masks_path, "masks")

        return {
            "train": self._generate_examples(
                os.path.join(patches_dir, "Train"),
                os.path.join(masks_dir, "Train"),
            ),
            "validation": self._generate_examples(
                os.path.join(patches_dir, "Val"),
                os.path.join(masks_dir, "Val"),
            ),
            "test": self._generate_examples(
                os.path.join(patches_dir, "Test"),
                os.path.join(masks_dir, "Test"),
            ),
        }

    def _generate_examples(self, image_dir: str, mask_dir: str):
        image_filenames = self._get_image_filenames(image_dir)

        for filename in image_filenames:
            image, masks, labelers = self._load_sample(filename, image_dir, mask_dir)
            yield filename, {
                "image": image,
                "masks": masks,
                "labelers": labelers,
            }

    def _get_image_filenames(self, image_dir: str) -> List[str]:
        return sorted(
            [
                filename
                for filename in os.listdir(image_dir)
                if filename.endswith(".png")
            ]
        )

    def _load_sample(
        self,
        filename: str,
        image_dir: str,
        mask_dir: str,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
        img_path = os.path.join(image_dir, filename)
        image = load_img(img_path, target_size=self.image_size)
        image = img_to_array(image, dtype=np.uint8)

        masks = []
        labelers = []

        for scorer_dir in ALL_SCORER_TAGS:
            scorer_mask_dir = os.path.join(mask_dir, scorer_dir)
            mask_path = os.path.join(scorer_mask_dir, filename)

            if os.path.exists(mask_path):
                mask_raw = load_img(
                    mask_path,
                    color_mode="grayscale",
                    target_size=self.image_size,
                )
                mask = img_to_array(mask_raw, dtype=np.uint8)

                if not np.all(np.isin(np.unique(mask), list(CLASSES_DEFINITION))):
                    LOGGER.warning(
                        "Mask %s contains invalid values. "
                        "Expected values: %s. "
                        "Values found: %s",
                        mask_path,
                        list(CLASSES_DEFINITION),
                        np.unique(mask),
                    )

                mask = mask.astype(np.uint8)
                if mask.ndim == 2:
                    mask = np.expand_dims(mask, axis=-1)
                masks.append(mask)
                labelers.append(scorer_dir)

        return image, masks, labelers


def get_crowd_seg_dataset_tfds(
    image_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    stages = ("train", "validation", "test")
    """Get crowd segmentation dataset.

    Args:
        image_size: Tuple[int, int] = DEFAULT_IMG_SIZE: Image size for the dataset.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: TensorFlow datasets for train, validation, and test.
    """
    builder = CrowdSegDataset(
        image_size=image_size,
    )
    builder.download_and_prepare()

    return builder.as_dataset(split=stages)


if __name__ == "__main__":
    target_size = (64, 64)
    train, validation, test = get_crowd_seg_dataset_tfds(
        image_size=target_size,
    )

    for batch in train.take(1):
        print("Image shape:", batch["image"].shape)
        print("Masks shape:", batch["masks"].shape)
        print("Labelers:", batch["labelers"])

        n_masks = len(batch["masks"])
        n_cols = 3
        n_rows = (n_masks + n_cols) // n_cols

        plt.figure(figsize=(15, 5 * n_rows))

        plt.subplot(n_rows, n_cols, 1)
        plt.imshow(batch["image"])
        plt.title("Original Image")

        for i, (mask, labeler) in enumerate(zip(batch["masks"], batch["labelers"])):
            plt.subplot(n_rows, n_cols, i + 2)
            plt.imshow(mask, cmap="viridis")
            plt.title(f"Mask from {labeler}")
            plt.colorbar()

        plt.tight_layout()
        plt.show()

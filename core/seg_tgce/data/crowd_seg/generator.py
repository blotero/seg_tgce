from enum import Enum
import logging
import os
from typing import List, Optional, Tuple, TypedDict

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import Sequence
from matplotlib import pyplot as plt
from tensorflow import Tensor, transpose, reshape
from tensorflow import argmax as tf_argmax

from .__retrieve import fetch_data, get_masks_dir, get_patches_dir
from .stage import Stage

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
DEFAULT_IMG_SIZE = (512, 512)


class ScorerNotFoundError(Exception):
    pass


class CustomPath(TypedDict):
    """Custom path for image and mask directories."""

    image_dir: str
    mask_dir: str


class DataSchema(Enum):
    """Data schema for the dataset.
    MA_RAW: Raw data for multiple annotators.
    MA_SPARSE: Processed data for multiple annotators. Sparse for fulfilling the
    required dimensions for consistency with the model.
    """

    MA_RAW = 0
    MA_SPARSE = 1


def get_image_filenames(image_dir: str) -> List[str]:
    return sorted(
        [filename for filename in os.listdir(image_dir) if filename.endswith(".png")]
    )


class ImageDataGenerator(Sequence):  # pylint: disable=too-many-instance-attributes
    """
    Data generator for crowd segmentation data.
    Delivered data is in the form of images, masks and scorers labels.
    Shapes are as follows:
    - images: (batch_size, image_size[0], image_size[1], 3)
    - masks: (batch_size, image_size[0], image_size[1], n_classes, n_scorers)"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        image_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        batch_size: int = 32,
        shuffle: bool = False,
        stage: Stage = Stage.TRAIN,
        paths: Optional[CustomPath] = None,
        schema: DataSchema = DataSchema.MA_RAW,
    ) -> None:
        if paths is not None:
            image_dir = paths["image_dir"]
            mask_dir = paths["mask_dir"]
        else:
            fetch_data()
            image_dir = get_patches_dir(stage)
            mask_dir = get_masks_dir(stage)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_filenames = get_image_filenames(image_dir)
        self.scorers_tags = sorted(os.listdir(mask_dir))
        self.on_epoch_end()
        self.schema = schema

    @property
    def classes_definition(self) -> dict[int, str]:
        """Returns classes definition."""
        return CLASSES_DEFINITION

    @property
    def n_classes(self) -> int:
        """Returns number of classes."""
        return len(self.classes_definition)

    @property
    def n_scorers(self) -> int:
        """Returns number of scorers."""
        return len(self.scorers_tags)

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def _get_items_raw(self, index: int) -> Tuple[Tensor, Tensor]:
        batch_filenames = self.image_filenames[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        return self.__data_generation(batch_filenames)

    def _get_items_sparse(self, index: int) -> Tuple[Tensor, Tensor]:
        batch_filenames = self.image_filenames[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        images, masks = self.__data_generation(batch_filenames)
        return images, reshape(masks, (self.batch_size, *self.image_size, -1))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        match self.schema:
            case DataSchema.MA_RAW:
                return self._get_items_raw(index)
            case DataSchema.MA_SPARSE:
                return self._get_items_sparse(index)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.image_filenames)

    def visualize_sample(
        self,
        batch_index: int = 0,
        sample_indexes: Optional[List[int]] = None,
        scorers: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Visualizes a sample from the dataset."""
        if scorers is None:
            scorers = self.scorers_tags
        images, masks = self._get_items_raw(batch_index)
        if sample_indexes is None:
            sample_indexes = [0, 1, 2, 3]

        fig, axes = plt.subplots(len(sample_indexes), len(scorers) + 1)
        for ax in axes.flatten():
            ax.axis("off")

        for i, sample_index in enumerate(sample_indexes):
            axes[i, 0].imshow(images[sample_index].astype(int))
            axes[i, 0].set_title(
                self.image_filenames[batch_index * self.batch_size + sample_index]
            )
            for j, scorer in enumerate(scorers):
                if scorer not in self.scorers_tags:
                    raise ScorerNotFoundError(
                        f"Scorer {scorer} not found in the dataset scorers "
                        f"({self.scorers_tags})."
                    )
                scorer_index = self.scorers_tags.index(scorer)
                axes[i, j + 1].imshow(
                    tf_argmax(masks[sample_index, :, :, :, scorer_index], axis=2),
                    cmap="viridis",
                )
                axes[i, j + 1].set_title(f"Label for {scorer}")

        plt.show()
        return fig

    def __data_generation(self, batch_filenames: List[str]) -> Tuple[Tensor, Tensor]:
        images = np.empty((self.batch_size, *self.image_size, 3))
        masks = np.empty(
            (
                self.batch_size,
                self.n_scorers,
                self.n_classes,
                *self.image_size,
            )
        )

        for batch, filename in enumerate(batch_filenames):
            img_path = os.path.join(self.image_dir, filename)
            for scorer, scorer_dir in enumerate(self.scorers_tags):
                scorer_mask_dir = os.path.join(self.mask_dir, scorer_dir)
                mask_path = os.path.join(scorer_mask_dir, filename)
                if os.path.exists(mask_path):
                    mask_raw = load_img(
                        mask_path,
                        color_mode="grayscale",
                        target_size=self.image_size,
                    )
                    mask = img_to_array(mask_raw)
                    if not np.all(
                        np.isin(np.unique(mask), list(self.classes_definition))
                    ):
                        LOGGER.warning(
                            "Mask %s contains invalid values. "
                            "Expected values: %s. "
                            "Values found: %s",
                            mask_path,
                            list(self.classes_definition),
                            np.unique(mask),
                        )
                    for class_num in self.classes_definition:
                        masks[batch][scorer][class_num] = np.where(
                            mask == class_num, 1, 0
                        ).reshape(*self.image_size)
                else:
                    LOGGER.debug(
                        "Mask not found for scorer %s and image %s",
                        scorer_dir,
                        filename,
                    )
                    masks[batch, scorer, 0] = np.ones(self.image_size)
                    masks[batch, scorer, 1:] = np.zeros(
                        (self.n_classes - 1, *self.image_size)
                    )

            image = load_img(img_path, target_size=self.image_size)
            image = img_to_array(image)

            images[batch] = image

        return images, transpose(masks, perm=[0, 3, 4, 2, 1])

import logging
import os
from typing import List, Optional, Tuple, TypedDict

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import Sequence
from matplotlib import pyplot as plt

from .retrieve import fetch_data, get_masks_dir, get_patches_dir
from .stage import Stage

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


class CustomPath(TypedDict):
    image_dir: str
    mask_dir: str


class ImageDataGenerator(Sequence):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        n_classes: int,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 32,
        shuffle: bool = True,
        stage: Stage = Stage.TRAIN,
        paths: Optional[CustomPath] = None,
    ):
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
        self.image_filenames = sorted(
            [
                filename
                for filename in os.listdir(image_dir)
                if filename.endswith(".png")
            ]
        )
        self.n_scorers = len(os.listdir(mask_dir))
        self.scorers_tags = sorted(os.listdir(mask_dir))
        LOGGER.info("Scorer tags: %s", self.scorers_tags)
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_filenames = self.image_filenames[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        images, masks = self.__data_generation(batch_filenames)
        return images, masks

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.image_filenames)

    def visualize_sample(
        self,
        scorers: List[str],
        batch_index: int = 1,
        sample_index: int = 1,
    ) -> plt.Figure:
        images, masks = self[batch_index]

        fig, axes = plt.subplots(len(scorers), self.n_classes + 1)
        for scorer_num, scorer in enumerate(scorers):
            for class_num in range(self.n_classes):
                axes[scorer_num][0].imshow(images[sample_index].astype(int))
                axes[scorer_num][class_num + 1].imshow(
                    masks[sample_index, scorer_num, class_num]
                )
                axes[scorer_num][0].axis("off")
                axes[scorer_num][class_num + 1].axis("off")
                axes[scorer_num][0].set_title(f"Image (ann {scorer})")
                axes[scorer_num][class_num + 1].set_title(f"Class {class_num}")

        plt.show()
        return fig

    def __data_generation(self, batch_filenames):
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
                    for class_num in range(self.n_classes):
                        masks[batch][scorer][class_num] = np.where(
                            mask == class_num, 1, 0
                        ).reshape(*self.image_size)
                    plt.show()
                else:
                    LOGGER.info(
                        (
                            "Mask not found for scorer %s and image %s "
                            "Filling up with zeros."
                        ),
                        scorer_dir,
                        filename,
                    )
                    masks[batch, scorer] = np.zeros((self.n_classes, *self.image_size))

            image = load_img(img_path, target_size=self.image_size)
            image = img_to_array(image)

            images[batch] = image

        return images, masks

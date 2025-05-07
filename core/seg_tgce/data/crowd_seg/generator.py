import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import Sequence
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from tensorflow import Tensor, reshape, transpose
from tensorflow import argmax as tf_argmax

# from seg_tgce.data.crowd_seg.types import InvertedMetadataRecord
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
METADATA_PATH = Path(__file__).resolve().parent / "metadata"


class ScorerNotFoundError(Exception):
    pass


class CustomPath(TypedDict):
    """Custom path for image and mask directories."""

    image_dir: str
    mask_dir: str


class DataSchema(str, Enum):
    """Data schema for the dataset.
    MA_RAW: Raw data for multiple annotators.
    MA_SPARSE: Processed data for multiple annotators. Sparse for fulfilling the
    required dimensions for consistency with the model.
    """

    MA_RAW = "ma_raw"
    MA_SPARSE = "ma_sparse"


def find_n_scorers(data: dict[str, dict[str, Any]], n: int) -> List[str]:
    # return a list of length n with the scorers that scored the most images
    scorers = sorted(data.keys(), key=lambda x: data[x]["total"], reverse=True)
    return scorers[:n]


def get_image_filenames(
    image_dir: str, stage: Stage, *, trim_n_scorers: int | None
) -> List[str]:
    if trim_n_scorers is None:
        return sorted(
            [
                filename
                for filename in os.listdir(image_dir)
                if filename.endswith(".png")
            ]
        )
    filenames: set[str] = set()
    inverted_data_path = f"{METADATA_PATH}/{stage.name.lower()}_inverted.json"
    with open(inverted_data_path, "r", newline="", encoding="utf-8") as json_file:
        inverted_data: dict[str, Any] = json.load(json_file)
        # trim to n scorers which scored the most images:
        trimmed_scorers = find_n_scorers(inverted_data, trim_n_scorers)

        LOGGER.info(
            "Limiting dataset to only images scored by the top %d scorers: %s",
            trim_n_scorers,
            trimmed_scorers,
        )
        for scorer in trimmed_scorers:
            filenames.update(inverted_data[scorer]["scored"])
    return list(filenames)


class ImageDataGenerator(Sequence):  # pylint: disable=too-many-instance-attributes
    """
    Data generator for crowd segmentation data.
    Delivered data is in the form of images, masks and scorers labels.
    Shapes are as follows:
    - images: (batch_size, image_size[0], image_size[1], 3)
    - masks: (batch_size, image_size[0], image_size[1], n_classes, n_scorers)

    Args:
    - image_size: Tuple[int, int] = DEFAULT_IMG_SIZE: Image size for the dataset.
    - batch_size: int = 32: Batch size for the generator.
    - shuffle: bool = False: Shuffle the dataset.
    - stage: Stage = Stage.TRAIN: Stage of the dataset.
    - paths: Optional[CustomPath] = None: Custom paths for image and mask directories.
    - schema: DataSchema = DataSchema.MA_RAW: Data schema for the dataset.
    - trim_n_scorers: int | None = None: Trim and leave only top n scorers

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        image_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        batch_size: int = 32,
        shuffle: bool = False,
        stage: Stage = Stage.TRAIN,
        paths: Optional[CustomPath] = None,
        schema: DataSchema = DataSchema.MA_RAW,
        trim_n_scorers: int | None = None,
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
        self.image_filenames = get_image_filenames(
            image_dir, stage, trim_n_scorers=trim_n_scorers
        )
        self.scorers_tags = sorted(os.listdir(mask_dir))
        self.on_epoch_end()
        self.schema = schema
        self.scorers_db = {
            filename: {scorer: False for scorer in self.scorers_tags}
            for filename in self.image_filenames
        }
        self.stage = stage

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

        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(12, 3 * len(sample_indexes)))

        # Create grid with more space for colorbar
        gs = fig.add_gridspec(
            len(sample_indexes),
            len(scorers) + 2,
            width_ratios=[1] * (len(scorers) + 1) + [0.3],
            wspace=0.3,  # Add more space between subplots
        )

        # Create subplots
        axes = np.array(
            [
                [fig.add_subplot(gs[i, j]) for j in range(len(scorers) + 2)]
                for i in range(len(sample_indexes))
            ]
        )

        for ax in axes.flatten():
            ax.axis("off")

        axes[0, 0].set_title("Slide", fontsize=12, pad=10)
        _ = [
            axes[0, i + 1].set_title(f"Label for {scorer}", fontsize=12, pad=10)
            for i, scorer in enumerate(scorers)
        ]

        # Create a custom colormap with named colors for each class
        class_colors = {
            0: "#440154",  # Dark purple for Ignore
            1: "#414487",  # Deep blue for Other
            2: "#2a788e",  # Teal for Tumor
            3: "#22a884",  # Turquoise for Stroma
            4: "#44bf70",  # Green for Benign Inflammation
            5: "#fde725",  # Yellow for Necrosis
        }

        # Convert hex colors to RGB
        colors = [plt.cm.colors.to_rgb(class_colors[i]) for i in range(self.n_classes)]
        cmap = ListedColormap(colors)

        for i, sample_index in enumerate(sample_indexes):
            axes[i, 0].imshow(images[sample_index].astype(int))
            for j, scorer in enumerate(scorers):
                if scorer not in self.scorers_tags:
                    raise ScorerNotFoundError(
                        f"Scorer {scorer} not found in the dataset scorers "
                        f"({self.scorers_tags})."
                    )
                scorer_index = self.scorers_tags.index(scorer)
                im = axes[i, j + 1].imshow(
                    tf_argmax(masks[sample_index, :, :, :, scorer_index], axis=2),
                    cmap=cmap,
                    vmin=0,
                    vmax=self.n_classes - 1,
                )

        # Add colorbar in the last column with class labels
        cbar_ax = axes[0, -1]
        cbar_ax.axis("on")  # Turn axis on for colorbar
        cbar = fig.colorbar(
            im, cax=cbar_ax, ticks=range(self.n_classes), orientation="vertical"
        )
        cbar.ax.tick_params(labelsize=10)  # Increase tick label size
        cbar.set_ticklabels(
            [CLASSES_DEFINITION[i] for i in range(self.n_classes)], fontsize=10
        )

        # Add a title to the colorbar
        cbar_ax.set_title("Classes", fontsize=12, pad=20)

        plt.tight_layout()
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
                    masks[batch, scorer, 0] = np.ones(self.image_size)
                    masks[batch, scorer, 1:] = np.zeros(
                        (self.n_classes - 1, *self.image_size)
                    )

            image = load_img(img_path, target_size=self.image_size)
            image = img_to_array(image)

            images[batch] = image

        return images, transpose(masks, perm=[0, 3, 4, 2, 1])

    def populate_metadata(self) -> None:
        for filename in self.image_filenames:
            for scorer in self.scorers_tags:
                scorer_mask_dir = os.path.join(self.mask_dir, scorer)
                mask_path = os.path.join(scorer_mask_dir, filename)
                if os.path.exists(mask_path):
                    self.scorers_db[filename][scorer] = True

    def store_metadata(self) -> None:
        LOGGER.info("Storing scorers database...")
        data_path = f"{METADATA_PATH}/{self.stage.name.lower()}_data.json"
        inverted_path = f"{METADATA_PATH}/{self.stage.name.lower()}_inverted.json"
        projected_data = {
            filename: [key for key, value in file_data.items() if value]
            for filename, file_data in self.scorers_db.items()
        }
        inverted_data: dict[str, Any] = {
            scorer: {"total": 0, "scored": []} for scorer in self.scorers_tags
        }
        for img_path, scorers in projected_data.items():
            for scorer in scorers:
                inverted_data[scorer]["total"] += 1
                inverted_data[scorer]["scored"].append(img_path)

        for data, json_path in zip(
            [projected_data, dict(inverted_data)], [data_path, inverted_path]
        ):
            with open(json_path, "w", newline="", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4)

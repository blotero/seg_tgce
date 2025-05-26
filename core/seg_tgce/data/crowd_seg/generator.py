import json
import logging
import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, TypedDict

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import Sequence
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from tensorflow import Tensor
from tensorflow import argmax as tf_argmax

from seg_tgce.data.crowd_seg.types import Stage

from .__retrieve import fetch_data, get_masks_dir, get_patches_dir

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

AGGREGATED_SCORERS = [
    "MV",
    "STAPLE",
]

ALL_SCORER_TAGS = (
    REAL_SCORERS
    + AGGREGATED_SCORERS
    + [
        "expert",
    ]
)

DEFAULT_IMG_SIZE = (512, 512)
METADATA_PATH = Path(__file__).resolve().parent / "metadata"


class CrowdSegItemData(TypedDict):
    x: Tensor
    y_labelers: Tensor
    labelers_mask: Tensor
    ground_truth: Tensor


class ScorerNotFoundError(Exception):
    pass


class CustomPath(TypedDict):
    """Custom path for image and mask directories."""

    image_dir: str
    mask_dir: str


def find_n_scorers(data: dict[str, dict[str, Any]], n: int) -> List[str]:
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
    inverted_data_path = f"{METADATA_PATH}/{stage}_inverted.json"
    with open(inverted_data_path, "r", newline="", encoding="utf-8") as json_file:
        inverted_data: dict[str, Any] = json.load(json_file)
        trimmed_scorers = find_n_scorers(inverted_data, trim_n_scorers)

        LOGGER.info(
            "Limiting dataset to only images scored by the top %d scorers: %s",
            trim_n_scorers,
            trimmed_scorers,
        )
        for scorer in trimmed_scorers:
            filenames.update(inverted_data[scorer]["scored"])
    return list(filenames)


class CrowdSegDataGenerator(Sequence):  # pylint: disable=too-many-instance-attributes
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
    - trim_n_scorers: int | None = None: Trim and leave only top n scorers

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        image_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
        batch_size: int = 32,
        shuffle: bool = False,
        stage: Stage = "train",
        paths: Optional[CustomPath] = None,
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
        self.scorers_tags = REAL_SCORERS
        self.on_epoch_end()
        self.stage = stage

        self._patch_labelers: dict[str, List[int]] = {}
        self._compute_patch_labelers()

    def _compute_patch_labelers(self) -> None:
        """Pre-compute which labelers annotated each patch."""
        for filename in self.image_filenames:
            active_labelers = []
            for scorer_idx, scorer_dir in enumerate(self.scorers_tags):
                scorer_mask_dir = os.path.join(self.mask_dir, scorer_dir)
                mask_path = os.path.join(scorer_mask_dir, filename)
                if os.path.exists(mask_path):
                    active_labelers.append(scorer_idx)
            self._patch_labelers[filename] = active_labelers

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

    def __getitem__(self, index: int) -> CrowdSegItemData:
        batch_filenames = self.image_filenames[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        x, y_labelers, labelers_mask, ground_truth = self.__data_generation(
            batch_filenames
        )
        return {
            "x": x,
            "y_labelers": y_labelers,
            "labelers_mask": labelers_mask,
            "ground_truth": ground_truth,
        }

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.image_filenames)

    def visualize_sample(
        self,
        batch_index: int = 0,
        sample_indexes: Optional[List[int]] = None,
    ) -> plt.Figure:
        """
        Visualizes a sample from the dataset.

        Args:
            batch_index: Index of the batch to visualize
            sample_indexes: List of sample indexes to visualize. If None, shows first 4 samples.
        """
        item_data = self.__getitem__(batch_index)
        images, masks, labeler_mask, ground_truth = (
            item_data["x"],
            item_data["y_labelers"],
            item_data["labelers_mask"],
            item_data["ground_truth"],
        )
        if sample_indexes is None:
            sample_indexes = [0, 1, 2, 3]

        unique_labelers: List[int] = []
        for sample_idx in sample_indexes:
            present_labelers = np.where(labeler_mask[sample_idx] == 1)[0]
            unique_labelers.extend(present_labelers)
        unique_labelers = sorted(set(unique_labelers))

        fig = plt.figure(figsize=(12, 3 * len(sample_indexes)))

        gs = fig.add_gridspec(
            len(sample_indexes),
            len(unique_labelers) + 3,  # +3 for image, ground truth, and colorbar
            width_ratios=[1] * (len(unique_labelers) + 2) + [0.3],
            wspace=0.3,
        )

        axes = np.array(
            [
                [fig.add_subplot(gs[i, j]) for j in range(len(unique_labelers) + 3)]
                for i in range(len(sample_indexes))
            ]
        )

        for ax in axes.flatten():
            ax.axis("off")

        axes[0, 0].set_title("Slide", fontsize=12, pad=10)
        axes[0, 1].set_title("Ground Truth", fontsize=12, pad=10)
        _ = [
            axes[0, i + 2].set_title(
                f"Label for {self.scorers_tags[labeler_idx]}", fontsize=12, pad=10
            )
            for i, labeler_idx in enumerate(unique_labelers)
        ]

        class_colors = {
            0: "#440154",  # Dark purple for Ignore
            1: "#414487",  # Deep blue for Other
            2: "#2a788e",  # Teal for Tumor
            3: "#22a884",  # Turquoise for Stroma
            4: "#44bf70",  # Green for Benign Inflammation
            5: "#fde725",  # Yellow for Necrosis
        }

        colors = [to_rgb(class_colors[i]) for i in range(self.n_classes)]
        cmap = ListedColormap(colors)

        im = None

        for i, sample_index in enumerate(sample_indexes):
            axes[i, 0].imshow(images[sample_index])
            # Show ground truth
            axes[i, 1].imshow(
                tf_argmax(ground_truth[sample_index], axis=-1),
                cmap=cmap,
                vmin=0,
                vmax=self.n_classes - 1,
            )
            for j, labeler_idx in enumerate(unique_labelers):
                if labeler_mask[sample_index, labeler_idx] == 1:
                    sample_mask = masks[sample_index, ..., labeler_idx]
                    im = axes[i, j + 2].imshow(
                        tf_argmax(sample_mask, axis=-1),
                        cmap=cmap,
                        vmin=0,
                        vmax=self.n_classes - 1,
                    )
                else:
                    axes[i, j + 2].imshow(np.zeros(self.image_size), cmap="gray")

        if im is not None:
            cbar_ax = axes[0, -1]
            cbar_ax.axis("on")
            cbar = fig.colorbar(
                im, cax=cbar_ax, ticks=range(self.n_classes), orientation="vertical"
            )
            cbar.ax.tick_params(labelsize=10)
            cbar.set_ticklabels(
                [CLASSES_DEFINITION[i] for i in range(self.n_classes)], fontsize=10
            )

            cbar_ax.set_title("Classes", fontsize=12, pad=20)

        plt.tight_layout()
        return fig

    def _load_sample(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single sample from disk."""
        img_path = os.path.join(self.image_dir, filename)
        image = load_img(img_path, target_size=self.image_size)
        image = img_to_array(image, dtype=np.float32)
        image = image / 255.0

        active_masks: dict[int, np.ndarray] = {}
        for scorer_idx in self._patch_labelers[filename]:
            scorer_dir = self.scorers_tags[scorer_idx]
            scorer_mask_dir = os.path.join(self.mask_dir, scorer_dir)
            mask_path = os.path.join(scorer_mask_dir, filename)

            mask_raw = load_img(
                mask_path,
                color_mode="grayscale",
                target_size=self.image_size,
            )
            mask = img_to_array(mask_raw, dtype=np.float32)
            if not np.all(np.isin(np.unique(mask), list(self.classes_definition))):
                LOGGER.warning(
                    "Mask %s contains invalid values. "
                    "Expected values: %s. "
                    "Values found: %s",
                    mask_path,
                    list(self.classes_definition),
                    np.unique(mask),
                )

            labeler_mask_for_scorer = np.zeros(
                (*self.image_size, self.n_classes), dtype=np.float32
            )
            if not (self.stage == "train" and scorer_dir == "expert"):
                for class_num in self.classes_definition:
                    labeler_mask_for_scorer[..., class_num] = np.where(
                        mask == class_num, 1.0, 0.0
                    ).reshape(*self.image_size)
            active_masks[scorer_idx] = labeler_mask_for_scorer

        # Load expert mask
        expert_mask = np.zeros((*self.image_size, self.n_classes), dtype=np.float32)
        expert_mask_path = os.path.join(self.mask_dir, "expert", filename)
        if os.path.exists(expert_mask_path):
            expert_mask_raw = load_img(
                expert_mask_path,
                color_mode="grayscale",
                target_size=self.image_size,
            )
            expert_mask_array = img_to_array(expert_mask_raw, dtype=np.float32)
            for class_num in self.classes_definition:
                expert_mask[..., class_num] = np.where(
                    expert_mask_array == class_num, 1.0, 0.0
                ).reshape(*self.image_size)

        full_masks = np.zeros(
            (*self.image_size, self.n_classes, len(self.scorers_tags)), dtype=np.float32
        )
        for labeler_idx, mask in active_masks.items():
            full_masks[..., labeler_idx] = mask

        return image, full_masks, expert_mask

    def __data_generation(
        self, batch_filenames: List[str]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        current_batch_size = len(batch_filenames)

        images = np.empty((current_batch_size, *self.image_size, 3), dtype=np.float32)
        masks = np.zeros(
            (
                current_batch_size,
                *self.image_size,
                self.n_classes,
                len(self.scorers_tags),
            ),
            dtype=np.float32,
        )
        labeler_mask = np.zeros(
            (current_batch_size, len(self.scorers_tags)), dtype=np.float32
        )
        ground_truth = np.zeros(
            (current_batch_size, *self.image_size, self.n_classes), dtype=np.float32
        )

        for i, filename in enumerate(batch_filenames):
            image, sample_masks, expert_mask = self._load_sample(filename)
            images[i] = image
            masks[i] = sample_masks
            labeler_mask[i, self._patch_labelers[filename]] = 1.0
            ground_truth[i] = expert_mask

        return images, masks, labeler_mask, ground_truth

    def populate_metadata(self) -> None:
        for filename in self.image_filenames:
            for scorer in self.scorers_tags:
                scorer_mask_dir = os.path.join(self.mask_dir, scorer)
                mask_path = os.path.join(scorer_mask_dir, filename)
                if os.path.exists(mask_path):
                    self.scorers_db[filename][scorer] = True

    def store_metadata(self) -> None:
        LOGGER.info("Storing scorers database...")
        data_path = f"{METADATA_PATH}/{self.stage}_data.json"
        inverted_path = f"{METADATA_PATH}/{self.stage}_inverted.json"
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

    def take(self, count: int) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Take a specified number of samples from the dataset.

        Args:
            count: Number of samples to take from the dataset.

        Returns:
            List of tuples containing (image, mask, labeler_mask, ground_truth) pairs.
        """
        samples = []
        for i in range(min(count, len(self.image_filenames))):
            batch_filenames = self.image_filenames[i : i + 1]
            images, masks, labeler_mask, ground_truth = self.__data_generation(
                batch_filenames
            )
            samples.append((images[0], masks[0], labeler_mask[0], ground_truth[0]))
        return samples


def get_crowd_seg_data(
    image_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    trim_n_scorers: int | None = None,
) -> Tuple[CrowdSegDataGenerator, ...]:
    """
    Retrieve all data generators for the crowd segmentation task.
    returns a tuple of ImageDataGenerator instances for the train, val, and test stages.
    """
    stages: tuple[Stage, ...] = ("train", "val", "test")
    return tuple(
        CrowdSegDataGenerator(
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            stage=stage,
            trim_n_scorers=trim_n_scorers,
        )
        for stage in stages
    )

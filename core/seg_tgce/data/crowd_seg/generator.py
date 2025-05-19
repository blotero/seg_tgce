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
from matplotlib.colors import ListedColormap, to_rgb
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
    - schema: DataSchema = DataSchema.MA_RAW: Data schema for the dataset.
    - trim_n_scorers: int | None = None: Trim and leave only top n scorers
    - use_cache: bool = False: Whether to use caching for loaded data
    - cache_size: int = 100: Maximum number of samples to keep in cache

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
        use_cache: bool = False,
        cache_size: int = 100,
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
        # Use predefined SCORER_TAGS instead of reading from directory
        self.scorers_tags = ALL_SCORER_TAGS
        self.on_epoch_end()
        self.schema = schema
        self.stage = stage

        # Cache-related attributes
        self.use_cache = use_cache
        self.cache_size = cache_size
        self._cache: dict[str, Tuple[np.ndarray, dict[int, np.ndarray]]] = {}
        self._cache_order: List[str] = []  # To implement LRU cache

        # Pre-compute labeler information for each patch
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

    def _get_items_raw(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        batch_filenames = self.image_filenames[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        return self.__data_generation(batch_filenames)

    def _get_items_sparse(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        batch_filenames = self.image_filenames[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        images, masks, labelers = self.__data_generation(batch_filenames)
        # Use the actual batch dimension from the generated masks for reshaping
        actual_batch_dim = masks.shape[0]
        return (
            images,
            reshape(masks, (actual_batch_dim, *self.image_size, -1)),
            labelers,
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
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
    ) -> plt.Figure:
        """
        Visualizes a sample from the dataset.

        Args:
            batch_index: Index of the batch to visualize
            sample_indexes: List of sample indexes to visualize. If None, shows first 4 samples.
        """
        images, masks, labeler_mask = self._get_items_raw(batch_index)
        if sample_indexes is None:
            sample_indexes = [0, 1, 2, 3]

        # Get unique labelers from the samples we're visualizing
        unique_labelers: List[int] = []
        for sample_idx in sample_indexes:
            # Get indices of labelers that are present (labeler_mask == 1)
            present_labelers = np.where(labeler_mask[sample_idx] == 1)[0]
            unique_labelers.extend(present_labelers)
        unique_labelers = sorted(set(unique_labelers))

        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=(12, 3 * len(sample_indexes)))

        # Create grid with more space for colorbar
        gs = fig.add_gridspec(
            len(sample_indexes),
            len(unique_labelers) + 2,
            width_ratios=[1] * (len(unique_labelers) + 1) + [0.3],
            wspace=0.3,  # Add more space between subplots
        )

        # Create subplots
        axes = np.array(
            [
                [fig.add_subplot(gs[i, j]) for j in range(len(unique_labelers) + 2)]
                for i in range(len(sample_indexes))
            ]
        )

        for ax in axes.flatten():
            ax.axis("off")

        axes[0, 0].set_title("Slide", fontsize=12, pad=10)
        _ = [
            axes[0, i + 1].set_title(
                f"Label for {self.scorers_tags[labeler_idx]}", fontsize=12, pad=10
            )
            for i, labeler_idx in enumerate(unique_labelers)
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
        colors = [to_rgb(class_colors[i]) for i in range(self.n_classes)]
        cmap = ListedColormap(colors)

        # Initialize im for colorbar
        im = None

        for i, sample_index in enumerate(sample_indexes):
            axes[i, 0].imshow(images[sample_index])
            for j, labeler_idx in enumerate(unique_labelers):
                if labeler_mask[sample_index, labeler_idx] == 1:
                    # Get the mask for this sample and labeler
                    # Shape: (h, w, classes, scorers) -> (h, w, classes) for this labeler
                    sample_mask = masks[sample_index, ..., labeler_idx]
                    im = axes[i, j + 1].imshow(
                        tf_argmax(sample_mask, axis=-1),
                        cmap=cmap,
                        vmin=0,
                        vmax=self.n_classes - 1,
                    )
                else:
                    # If labeler didn't label this sample, show blank image
                    axes[i, j + 1].imshow(np.zeros(self.image_size), cmap="gray")

        # Add colorbar in the last column with class labels
        if im is not None:  # Only add colorbar if we have at least one mask
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

    def _load_sample(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single sample from disk or cache."""
        if self.use_cache and filename in self._cache:
            # Update cache order (move to end as most recently used)
            self._cache_order.remove(filename)
            self._cache_order.append(filename)
            cached_image, cached_masks = self._cache[filename]

            # Reconstruct full mask tensor with zeros for non-labeled masks
            full_masks = np.zeros(
                (*self.image_size, self.n_classes, len(self.scorers_tags)),
                dtype=np.float32,
            )
            for labeler_idx, mask in cached_masks.items():
                full_masks[..., labeler_idx] = mask
            return cached_image, full_masks

        # Load image
        img_path = os.path.join(self.image_dir, filename)
        image = load_img(img_path, target_size=self.image_size)
        image = img_to_array(image, dtype=np.float32)
        image = image / 255.0  # Normalize to [0, 1]

        # Load only masks from active labelers
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

            # Create one-hot encoded mask for this labeler
            labeler_mask_for_scorer = np.zeros(
                (*self.image_size, self.n_classes), dtype=np.float32
            )
            # If stage is TRAIN and current scorer is "expert", this mask remains all zeros.
            # Otherwise, populate it from the loaded mask file.
            if not (self.stage == Stage.TRAIN and scorer_dir == "expert"):
                for class_num in self.classes_definition:
                    labeler_mask_for_scorer[..., class_num] = np.where(
                        mask == class_num, 1.0, 0.0
                    ).reshape(*self.image_size)
            active_masks[scorer_idx] = labeler_mask_for_scorer

        if self.use_cache:
            # If cache is full, remove least recently used item
            if len(self._cache) >= self.cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]

            # Add to cache
            self._cache[filename] = (image, active_masks)
            self._cache_order.append(filename)

        # Create full mask tensor with zeros for non-labeled masks
        full_masks = np.zeros(
            (*self.image_size, self.n_classes, len(self.scorers_tags)), dtype=np.float32
        )
        for labeler_idx, mask in active_masks.items():
            full_masks[..., labeler_idx] = mask

        return image, full_masks

    def __data_generation(
        self, batch_filenames: List[str]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        current_batch_size = len(batch_filenames)  # Determine actual batch size

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

        for i, filename in enumerate(
            batch_filenames
        ):  # Iterate up to current_batch_size
            image, sample_masks = self._load_sample(filename)
            images[i] = image
            masks[i] = sample_masks
            # Set labeler mask based on pre-computed active labelers
            labeler_mask[i, self._patch_labelers[filename]] = 1.0

        return images, masks, labeler_mask

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

    def take(self, count: int) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """Take a specified number of samples from the dataset.

        Args:
            count: Number of samples to take from the dataset.

        Returns:
            List of tuples containing (image, mask, labeler_mask) pairs.
        """
        samples = []
        for i in range(min(count, len(self.image_filenames))):
            batch_filenames = self.image_filenames[i : i + 1]  # Get single sample
            images, masks, labeler_mask = self.__data_generation(batch_filenames)
            samples.append(
                (images[0], masks[0], labeler_mask[0])
            )  # Remove batch dimension
        return samples

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_order.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_masks = sum(len(masks) for _, masks in self._cache.values())
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "total_cached_masks": total_masks,
            "average_masks_per_sample": (
                total_masks / len(self._cache) if self._cache else 0
            ),
        }

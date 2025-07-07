import logging
from pathlib import Path
from typing import Iterator, List, Optional, TypedDict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

from seg_tgce.data.crowd_seg.__retrieve import (
    _BUCKET_NAME,
    MASKS_OBJECT_NAME,
    PATCHES_OBJECT_NAME,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


CLASSES_DEFINITION = {
    0: "Ignore",
    1: "Other",
    2: "Tumor",
    3: "Stroma",
    4: "B. Inflammation",  # Benign Inflammation
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

ALL_SCORER_TAGS = REAL_SCORERS + AGGREGATED_SCORERS + ["ground_truth"]

DEFAULT_IMG_SIZE = (512, 512)
METADATA_PATH = Path(__file__).resolve().parent / "metadata"

N_CLASSES = len(CLASSES_DEFINITION)
N_REAL_SCORERS = len(REAL_SCORERS)


class SampleData(TypedDict):
    image: tf.Tensor
    masks: tf.Tensor
    labelers: tf.Tensor
    filename: tf.Tensor


class ProcessedSampleData(TypedDict):
    image: tf.Tensor
    masks: tf.Tensor
    labelers_mask: tf.Tensor
    ground_truth: tf.Tensor


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    return tf.cast(image, tf.float32) / 255.0


def create_one_hot_mask(mask: tf.Tensor) -> tf.Tensor:
    mask = tf.cast(mask, tf.uint8)
    return tf.one_hot(mask, N_CLASSES, dtype=tf.float32)


def create_labeler_mask(labelers: tf.Tensor) -> tf.Tensor:
    labeler_mask = tf.zeros(N_REAL_SCORERS, dtype=tf.float32)

    for i, scorer in enumerate(REAL_SCORERS):
        labeler_mask = tf.tensor_scatter_nd_update(
            labeler_mask,
            [[i]],
            [tf.cast(tf.reduce_any(tf.equal(labelers, scorer)), tf.uint8)],
        )

    return labeler_mask


@tf.function
def map_sample(sample: SampleData, image_size: tuple[int, int]) -> ProcessedSampleData:
    image = normalize_image(sample["image"])

    real_scorer_indices = tf.where(
        tf.reduce_any(
            tf.equal(tf.expand_dims(sample["labelers"], 1), tf.constant(REAL_SCORERS)),
            axis=1,
        )
    )

    masks = tf.squeeze(
        tf.squeeze(tf.gather(sample["masks"], real_scorer_indices), axis=1), axis=-1
    )
    labelers = tf.gather(sample["labelers"], real_scorer_indices)

    masks = tf.map_fn(
        create_one_hot_mask,
        masks,
        fn_output_signature=tf.TensorSpec(
            shape=(*image_size, N_CLASSES), dtype=tf.float32
        ),
    )

    labeler_mask = create_labeler_mask(labelers)

    expanded_masks = tf.zeros((N_REAL_SCORERS, *image_size, N_CLASSES), dtype=tf.float32)

    active_indices = tf.where(tf.equal(labeler_mask, 1))[:, 0]

    active_indices = tf.reshape(active_indices, [-1, 1])

    masks = tf.reshape(masks, [-1, *image_size, N_CLASSES])

    expanded_masks = tf.tensor_scatter_nd_update(expanded_masks, active_indices, masks)

    expanded_masks = tf.transpose(expanded_masks, perm=[1, 2, 3, 0])

    gt_mask = sample["masks"][-1]
    gt_mask = tf.reshape(gt_mask, image_size)

    return {
        "image": image,
        "masks": expanded_masks,
        "labelers_mask": labeler_mask,
        "ground_truth": create_one_hot_mask(gt_mask),
    }


class CrowdSegDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for crowd segmentation dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(
        self,
        *,
        image_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    ):
        """Initialize the dataset builder.

        Args:
            image_size: tuple[int, int] = DEFAULT_IMG_SIZE: Image size for the dataset.
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
                        tfds.features.Tensor(shape=(*self.image_size, 1), dtype=tf.uint8)
                    ),
                    "labelers": tfds.features.Sequence(tfds.features.Text()),
                    "filename": tfds.features.Text(),
                }
            ),
            supervised_keys=("image", "masks"),
            homepage="https://github.com/your-repo/crowd-seg",
            citation="""@article{your-citation}""",
        )

    def _split_generators(
        self, dl_manager: tfds.download.DownloadManager
    ) -> dict[str, Iterator[tuple[str, SampleData]]]:
        patches_url = f"https://{_BUCKET_NAME}.s3.amazonaws.com/{PATCHES_OBJECT_NAME}"
        masks_url = f"https://{_BUCKET_NAME}.s3.amazonaws.com/{MASKS_OBJECT_NAME}"

        patches_path = dl_manager.download_and_extract(patches_url)
        masks_path = dl_manager.download_and_extract(masks_url)

        patches_dir = Path(patches_path) / "patches"
        masks_dir = Path(masks_path) / "masks"

        return {
            "train": self._generate_examples(
                patches_dir / "Train",
                masks_dir / "Train",
            ),
            "validation": self._generate_examples(
                patches_dir / "Valid",
                masks_dir / "Valid",
            ),
            "test": self._generate_examples(
                patches_dir / "Test",
                masks_dir / "Test",
            ),
        }

    def _generate_examples(
        self, image_dir: Path, mask_dir: Path
    ) -> Iterator[tuple[str, SampleData]]:
        image_filenames = self._get_image_filenames(image_dir)

        for filename in sorted(image_filenames):
            image, masks, labelers = self._load_sample(filename, image_dir, mask_dir)
            yield (
                filename,
                {
                    "image": image,
                    "masks": masks,
                    "labelers": labelers,
                    "filename": filename,
                },
            )

    def _get_image_filenames(self, image_dir: Path) -> List[str]:
        return sorted(
            [
                filename.name
                for filename in image_dir.iterdir()
                if filename.suffix == ".png"
            ]
        )

    def _load_sample(
        self,
        filename: str,
        image_dir: Path,
        mask_dir: Path,
    ) -> tuple[np.ndarray, List[np.ndarray], List[str]]:
        img_path = image_dir / filename
        image = load_img(img_path, target_size=self.image_size)
        image = img_to_array(image, dtype=np.uint8)

        masks = []
        labelers = []

        for scorer_dir in ALL_SCORER_TAGS:
            scorer_mask_dir = mask_dir / scorer_dir

            if not (scorer_mask_dir / "class_0" / filename).exists():
                continue

            scorer_mask = np.zeros(self.image_size, dtype=np.uint8)

            for class_idx in range(N_CLASSES):
                class_dir = scorer_mask_dir / f"class_{class_idx}"
                mask_path = class_dir / filename

                mask_raw = load_img(
                    mask_path,
                    color_mode="grayscale",
                    target_size=self.image_size,
                )
                class_mask = img_to_array(mask_raw, dtype=np.uint8)
                class_mask = class_mask[:, :, 0]
                scorer_mask[class_mask > 0] = class_idx

            scorer_mask = scorer_mask[..., np.newaxis]
            masks.append(scorer_mask)
            labelers.append(scorer_dir)

        return image, masks, labelers


def visualize_sample(
    dataset: tf.data.Dataset,
    image_size: tuple[int, int],
    batch_index: int = 0,
    sample_indexes: Optional[List[int]] = None,
    sparse_labelers: bool = True,
) -> plt.Figure:
    if sample_indexes is None:
        sample_indexes = [0, 1, 2, 3]

    batch = next(iter(dataset.skip(batch_index).take(1)))

    images = batch["image"]
    masks = batch["masks"]
    labeler_mask = batch["labelers_mask"]
    ground_truth = batch["ground_truth"]

    masks_class = tf.argmax(masks, axis=-2)
    ground_truth_class = tf.argmax(ground_truth, axis=-1)

    unique_labelers = []
    for sample_idx in sample_indexes:
        present_labelers = tf.where(labeler_mask[sample_idx] == 1)[:, 0]
        unique_labelers.extend(present_labelers.numpy())
    unique_labelers = sorted(set(unique_labelers))

    fig = plt.figure(figsize=(12, 3 * len(sample_indexes)), constrained_layout=True)

    n_cols = len(unique_labelers) + 3 if sparse_labelers else 4

    gs = fig.add_gridspec(
        len(sample_indexes),
        n_cols,
        width_ratios=[1] * (n_cols - 1) + [0.3],
        wspace=0.1,
    )

    axes = np.array(
        [
            [fig.add_subplot(gs[i, j]) for j in range(n_cols)]
            for i in range(len(sample_indexes))
        ]
    )

    for ax in axes.flatten():
        ax.axis("off")

    axes[0, 0].set_title("Slide", fontsize=16, pad=10)
    axes[0, 1].set_title("Expert", fontsize=16, pad=10)

    if sparse_labelers:
        _ = [
            axes[0, i + 2].set_title(
                f"Label for {REAL_SCORERS[labeler_idx]}", fontsize=14, pad=10
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

    colors = [to_rgb(class_colors[i]) for i in range(N_CLASSES)]
    cmap = ListedColormap(colors)

    im = None

    for i, sample_index in enumerate(sample_indexes):
        axes[i, 0].imshow(images[sample_index])
        axes[i, 1].imshow(
            ground_truth_class[sample_index],
            cmap=cmap,
            vmin=0,
            vmax=N_CLASSES - 1,
        )
        if sparse_labelers:
            for j, labeler_idx in enumerate(unique_labelers):
                if labeler_mask[sample_index, labeler_idx] == 1:
                    sample_mask = masks_class[sample_index, ..., labeler_idx]
                    im = axes[i, j + 2].imshow(
                        sample_mask,
                        cmap=cmap,
                        vmin=0,
                        vmax=N_CLASSES - 1,
                    )
                else:
                    axes[i, j + 2].imshow(np.zeros(image_size), cmap="gray")
        else:
            non_expert_labeler_idx = tf.where(labeler_mask[sample_index] == 1).numpy()[0][
                0
            ]
            im = axes[i, 2].imshow(
                masks_class[sample_index, ..., non_expert_labeler_idx],
                cmap=cmap,
                vmin=0,
                vmax=N_CLASSES - 1,
            )
            axes[i, 2].set_title(
                f"Label for {REAL_SCORERS[non_expert_labeler_idx]}",
                fontsize=14,
                pad=10,
            )

    if im is not None:
        cbar_ax = axes[0, -1]
        cbar_ax.axis("on")
        cbar = fig.colorbar(
            im, cax=cbar_ax, ticks=range(N_CLASSES), orientation="vertical"
        )
        cbar.ax.tick_params(labelsize=10)
        cbar.set_ticklabels(
            [CLASSES_DEFINITION[i] for i in range(N_CLASSES)], fontsize=12
        )

        cbar_ax.set_title("Classes", fontsize=16, pad=20)

    return fig


def get_crowd_seg_dataset_tfds(
    image_size: tuple[int, int] = DEFAULT_IMG_SIZE,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Get crowd segmentation dataset.

    Args:
        image_size: tuple[int, int] = DEFAULT_IMG_SIZE: Image size for the dataset.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: TensorFlow datasets for train, validation, and test.
    """
    builder = CrowdSegDataset(
        image_size=image_size,
    )
    builder.download_and_prepare()

    return builder.as_dataset(split=("train", "validation", "test"))


def get_training_augmentation(
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    hue_shift_limit: float = 0.05,
    sat_shift_limit: float = 0.1,
) -> tf.keras.Sequential:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.GaussianNoise(0.1),
            tf.keras.layers.RandomBrightness(brightness_limit),
            tf.keras.layers.RandomContrast(contrast_limit),
            tf.keras.layers.RandomSaturation(sat_shift_limit),
            tf.keras.layers.RandomHue(hue_shift_limit),
            tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.clip_by_value(x * 255, 0, 255), tf.uint8)
            ),
        ]
    )

    return augmentation


def get_processed_data(
    image_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = 32,
    use_augmentation: bool = False,
    augmentation_factor: int = 2,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train, validation, test = get_crowd_seg_dataset_tfds(
        image_size=image_size,
    )

    print(f"Original dataset train length: {len(train)}")

    processed_train = train.cache()

    if use_augmentation:
        augmentation = get_training_augmentation()

        for _ in range(augmentation_factor):
            augmented = processed_train.map(
                lambda x: {**x, "image": augmentation(x["image"])},
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            processed_train = processed_train.concatenate(augmented)
        print(f"Augmented dataset train length before batching: {len(processed_train)}")

    processed_train = processed_train.map(
        lambda x: map_sample(x, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    processed_validation = validation.map(
        lambda x: map_sample(x, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    processed_test = test.map(
        lambda x: map_sample(x, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return (
        processed_train.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        processed_validation.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        processed_test.batch(batch_size).prefetch(tf.data.AUTOTUNE),
    )


def get_processed_data_baseline(
    image_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = 32,
    use_augmentation: bool = False,
    augmentation_factor: int = 2,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    def map_dataset_to_baseline(x):
        expert_mask = x["masks"][-1]
        expert_mask = tf.reshape(expert_mask, image_size)
        return (x["image"], create_one_hot_mask(expert_mask))

    train, validation, test = get_crowd_seg_dataset_tfds(image_size=image_size)

    processed_train = train.map(map_dataset_to_baseline).cache()

    if use_augmentation:
        augmentation = get_training_augmentation()

        for _ in range(augmentation_factor):
            augmented = processed_train.map(
                lambda x, y: (augmentation(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            processed_train = processed_train.concatenate(augmented)

    processed_train = processed_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    processed_validation = (
        validation.map(map_dataset_to_baseline)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    processed_test = (
        test.map(map_dataset_to_baseline)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return processed_train, processed_validation, processed_test


def print_processed_sample_info(data: dict) -> None:
    print(f"Filename: {data['filename']}")
    print(f"Image shape: {data['image'].shape}")
    print(f"Masks shape: {data['masks'].shape}")
    print(f"Labelers mask shape: {data['labelers_mask'].shape}")
    print(f"Ground truth shape: {data['ground_truth'].shape}")
    print("-" * 50)


def print_sample_info(data: dict) -> None:
    print(f"Filename: {data['filename']}")
    print(
        f"Labelers: {[labeler.numpy().decode('utf-8') for labeler in data['labelers']]}"
    )
    print(f"Image shape: {data['image'].shape}")
    print(f"Masks shape: {data['masks'].shape}")
    print("-" * 50)


def main_processed() -> None:
    target_size = (128, 128)
    batch_size = 16

    train, validation, test = get_processed_data(
        image_size=target_size, batch_size=batch_size, use_augmentation=False
    )
    print(f"Dataset train length: {len(train)}")

    fig = visualize_sample(
        validation,
        target_size,
        batch_index=1,
        sample_indexes=[0, 1, 3, 4],
        sparse_labelers=False,
    )
    fig.savefig(
        "/home/brandon/unal/maestria/master_thesis/Cap1/Figures/multiannotator-segmentation.png"
    )


def main_raw():
    train, validation, test = get_crowd_seg_dataset_tfds(
        image_size=(64, 64),
    )
    for i, data in enumerate(train):
        print(f"\nSample {i}:")
        print_sample_info(data)


if __name__ == "__main__":
    main_processed()

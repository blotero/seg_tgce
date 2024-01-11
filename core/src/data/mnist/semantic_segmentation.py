# pylint: disable=wrong-import-order
import typing
from typing import Any, NamedTuple, Tuple

import matplotlib
import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

from .array_overlay import overlay_arrays
from .mnist import download_mnist, preprocess_mnist

plt.rcParams["figure.facecolor"] = "white"


class SegmentationDatasetConfig(NamedTuple):
    num_train_samples: int
    num_test_samples: int
    image_shape: Tuple[int, int] = (60, 60)
    min_num_digits_per_image: int = 2
    max_num_digits_per_image: int = 4
    num_classes: int = 10
    max_iou: float = 0.2
    labels_are_exclusive: bool = False
    target_is_whole_bounding_box: bool = False
    proportion_of_mnist: float = 1.0


class SegmentationDatasetFromDigitsConfig(NamedTuple):
    digits: np.ndarray[Any, np.dtype[np.float64]]
    digit_labels: np.ndarray[Any, np.dtype[np.float64]]
    num_samples: int
    image_shape: tuple[int, int]
    min_num_digits_per_image: int
    max_num_digits_per_image: int
    num_classes: int
    max_iou: float
    labels_are_exclusive: bool = False
    target_is_whole_bounding_box: bool = False


class SegmentationTargetConfig(NamedTuple):
    images: np.ndarray[Any, np.dtype[np.float64]]
    labels: np.ndarray[Any, np.dtype[np.float64]]
    bounding_boxes: np.ndarray[Any, np.dtype[np.float64]]
    image_shape: tuple[int, int]
    num_classes: int
    labels_are_exclusive: bool = False
    target_is_whole_bounding_box: bool = False


class AxConfig(NamedTuple):
    title: str = ""
    ax: matplotlib.axes.Axes | None = None


def create_semantic_segmentation_dataset(
    config: SegmentationDatasetConfig,
) -> Tuple[
    np.ndarray[typing.Any, np.dtype[np.float64]],
    np.ndarray[typing.Any, np.dtype[np.float64]],
    np.ndarray[typing.Any, np.dtype[np.float64]],
    np.ndarray[typing.Any, np.dtype[np.float64]],
]:
    (train_images, train_labels), (test_images, test_labels) = download_mnist()

    train_images, train_labels = preprocess_mnist(
        images=train_images,
        labels=train_labels,
        proportion=config.proportion_of_mnist,
        num_classes=config.num_classes,
        normalise=True,
    )

    test_images, test_labels = preprocess_mnist(
        images=test_images,
        labels=test_labels,
        proportion=config.proportion_of_mnist,
        num_classes=config.num_classes,
        normalise=True,
    )

    dataset_from_digits_config_train = SegmentationDatasetFromDigitsConfig(
        digits=train_images,
        digit_labels=train_labels,
        num_samples=config.num_train_samples,
        image_shape=config.image_shape,
        min_num_digits_per_image=config.min_num_digits_per_image,
        max_num_digits_per_image=config.max_num_digits_per_image,
        num_classes=config.num_classes,
        max_iou=config.max_iou,
        labels_are_exclusive=config.labels_are_exclusive,
        target_is_whole_bounding_box=config.target_is_whole_bounding_box,
    )

    x_train, y_train = create_semantic_segmentation_data_from_digits(
        config=dataset_from_digits_config_train
    )
    dataset_from_digits_config_test = SegmentationDatasetFromDigitsConfig(
        digits=test_images,
        digit_labels=test_labels,
        num_samples=config.num_test_samples,
        image_shape=config.image_shape,
        min_num_digits_per_image=config.min_num_digits_per_image,
        max_num_digits_per_image=config.max_num_digits_per_image,
        num_classes=config.num_classes,
        max_iou=config.max_iou,
        labels_are_exclusive=config.labels_are_exclusive,
        target_is_whole_bounding_box=config.target_is_whole_bounding_box,
    )

    x_test, y_test = create_semantic_segmentation_data_from_digits(
        config=dataset_from_digits_config_test
    )

    return x_train, y_train, x_test, y_test


def create_semantic_segmentation_data_from_digits(
    config: SegmentationDatasetFromDigitsConfig,
) -> Tuple[
    np.ndarray[typing.Any, np.dtype[np.float64]],
    np.ndarray[typing.Any, np.dtype[np.float64]],
]:
    input_data = []
    target_data = []

    for _ in range(config.num_samples):
        num_digits = np.random.randint(
            config.min_num_digits_per_image, config.max_num_digits_per_image + 1
        )

        (
            input_array,
            arrays_overlaid,
            labels_overlaid,
            bounding_boxes_overlaid,
        ) = overlay_arrays(
            array_shape=config.image_shape + (1,),
            input_arrays=config.digits,
            input_labels=config.digit_labels,
            max_array_value=1,
            num_input_arrays_to_overlay=num_digits,
            max_iou=config.max_iou,
        )

        segmentation_target_config = SegmentationTargetConfig(
            images=arrays_overlaid,
            labels=labels_overlaid,
            bounding_boxes=bounding_boxes_overlaid,
            image_shape=config.image_shape,
            num_classes=config.num_classes,
            labels_are_exclusive=config.labels_are_exclusive,
            target_is_whole_bounding_box=config.target_is_whole_bounding_box,
        )

        target_array = create_segmentation_target(
            config=segmentation_target_config)

        input_data.append(input_array)
        target_data.append(target_array)

    return np.stack(input_data), np.stack(target_data)


def create_segmentation_target(
    config: SegmentationTargetConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if len(config.bounding_boxes) != len(config.labels) != len(config.images):
        raise ValueError(
            f"The length of bounding_boxes must be the same as the length of labels. "
            f"Received shapes: {config.bounding_boxes.shape}!={config.labels.shape}"
        )

    target = np.zeros(config.image_shape + (config.num_classes,))

    if config.labels_are_exclusive:
        exclusivity_mask = np.zeros(config.image_shape)

    for box_num, label in enumerate(config.labels):
        xmin, ymin, xmax, ymax = config.bounding_boxes[box_num]

        if config.target_is_whole_bounding_box:
            target[ymin:ymax, xmin:xmax, [label]] = 1
        else:
            target[ymin:ymax, xmin:xmax, [label]] = (
                config.images[box_num] + target[ymin:ymax, xmin:xmax, [label]]
            )

        if config.labels_are_exclusive:
            target[..., label] = np.where(
                exclusivity_mask, 0, target[..., label])
            exclusivity_mask = np.logical_or(
                exclusivity_mask, target[..., label])

    return target


def display_grayscale_array(
    array: np.ndarray[Any, np.dtype[np.float64]],
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> None:
    axes: matplotlib.axes.Axes = ax or plt.gca()

    if len(array.shape) == 3:
        array = array[..., 0]

    axes.imshow(array, cmap="binary")
    axes.axes.set_yticks([])  # type:ignore
    axes.axes.set_xticks([])  # type:ignore

    if title:
        axes.set_title(title)

    plt.show()


def display_segmented_image(
    y: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float = 0.5,
    input_image: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    alpha_input_image: float = 0.2,
    ax_config: AxConfig = AxConfig("", None),
) -> None:
    ax = ax_config.ax or plt.gca()

    base_array = np.ones((y.shape[0], y.shape[1], 3)) * 1
    legend_handles = []

    for k in range(y.shape[-1]):
        # pylint: disable=no-member
        colour = plt.cm.jet(k / y.shape[-1])[:-1]  # type: ignore
        base_array[y[..., k] > threshold] = colour
        legend_handles.append(mpatches.Patch(color=colour, label=str(k)))

    ax.imshow(base_array)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(ax_config.title)

    if input_image is not None:
        ax.imshow(
            input_image[..., 0],
            cmap="binary",
            alpha=alpha_input_image,
        )

    plt.show()


def plot_class_masks(
    y_true: np.ndarray[Any, np.dtype[np.float64]],
    y_predicted: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    title: str = "",
) -> None:
    num_rows = 2 if y_predicted is not None else 1

    num_classes = y_true.shape[-1]
    fig, axes = plt.subplots(
        num_rows, num_classes, figsize=(num_classes * 4, num_rows * 4)
    )
    axes = axes.reshape(-1, num_classes)
    fig.suptitle(title)
    plt.tight_layout()

    for label in range(num_classes):
        axes[0, label].imshow(y_true[..., label], cmap="binary")
        axes[0, label].axes.set_yticks([])
        axes[0, label].axes.set_xticks([])

        if label == 0:
            axes[0, label].set_ylabel("Target")

        if y_predicted is not None:
            if label == 0:
                axes[1, label].set_ylabel("Predicted")

            axes[1, label].imshow(y_predicted[..., label], cmap="binary")
            axes[1, label].set_xlabel(f"Label: {label}")
            axes[1, label].axes.set_yticks([])
            axes[1, label].axes.set_xticks([])
        else:
            axes[0, label].set_xlabel(f"Label: {label}")

    plt.show()


if __name__ == "__main__":
    dataset_config = SegmentationDatasetConfig(
        num_train_samples=100, num_test_samples=10, image_shape=(60, 60), num_classes=5
    )
    train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(
        config=dataset_config
    )
    i = np.random.randint(len(train_x))
    display_grayscale_array(array=train_x[i])
    plot_class_masks(train_y[i])
    display_segmented_image(y=train_y[i])

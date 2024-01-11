from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

plt.rcParams["figure.facecolor"] = "white"


def download_mnist() -> (
    tuple[
        tuple[
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]],
        ],
        tuple[
            np.ndarray[Any, np.dtype[np.float64]],
            np.ndarray[Any, np.dtype[np.float64]],
        ],
    ]
):
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()  # pylint: disable=no-member

    return (train_images, train_labels), (test_images, test_labels)


def preprocess_mnist(
    images: np.ndarray[Any, np.dtype[np.float64]],
    labels: np.ndarray[Any, np.dtype[np.float64]],
    proportion: float,
    num_classes: int,
    normalise: bool = True,
) -> Tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    indices = np.random.randint(
        len(images), size=int(len(images) * proportion))
    images = images[indices]
    labels = labels[indices]

    valid_examples = np.zeros_like(labels)
    for i in range(0, num_classes):
        valid_examples = np.logical_or(labels == i, valid_examples)

    images = images[valid_examples]
    labels = labels[valid_examples]

    if normalise:
        images = images / 255.0

    images = np.expand_dims(images, -1)

    return images, labels


def display_digits(
    images: np.ndarray[Any, np.dtype[np.float64]],
    labels: np.ndarray[Any, np.dtype[np.float64]],
    num_to_display: int = 25,
    random: bool = True,
) -> None:
    num_columns = 5
    num_rows = int(np.ceil(num_to_display / num_columns))

    plt.figure(figsize=(num_columns * 2, num_rows * 2))

    indices = (
        np.random.randint(len(images), size=num_to_display)
        if random
        else range(num_to_display)
    )

    for i, index in enumerate(indices):  # type: ignore
        ax = plt.subplot(num_rows, num_columns, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        if len(images.shape) == 4:
            ax.imshow(images[index, ..., 0], cmap="binary")
        else:
            ax.imshow(images[index], cmap="binary")

        ax.set_xlabel(labels[index])

    plt.show()

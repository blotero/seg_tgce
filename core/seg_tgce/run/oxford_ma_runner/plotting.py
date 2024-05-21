from typing import List

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import Tensor


def epoch_progress_plotter(  # pylint: disable=too-many-arguments
    x: Tensor,
    y: Tensor,
    predictions: Tensor,
    num_img: int,
    noise_values: List[float],
    num_annotators: int,
) -> None:
    _, axes = plt.subplots(4, num_annotators, figsize=(15, 9))

    axes[0, 0].imshow(x[num_img, :, :, :])
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    if num_annotators > 2:
        hide_axis = [axes[0, sub_ind] for sub_ind in range(1, num_annotators - 1)]
        for ax in hide_axis:
            ax.axis("off")

    prediction_ax = axes[0, -1]
    prediction_ax.imshow(predictions[num_img, :, :, 0])
    prediction_ax.set_title("Prediction")
    prediction_ax.axis("off")

    mask_axes = [axes[1, sub_ind] for sub_ind in range(num_annotators)]
    rmap_axes = [axes[2, sub_ind] for sub_ind in range(num_annotators)]
    rmap_product_axes = [axes[3, sub_ind] for sub_ind in range(num_annotators)]

    for i, ax in enumerate(rmap_axes):
        ax.imshow(predictions[num_img, :, :, i + 2])

        ax.set_title(
            f"Reliability map, $\lambda={np.mean(predictions[num_img, :, :, i + 2]):.3f}$"  # pylint: disable=anomalous-backslash-in-string
        )
        ax.axis("off")

    for i, ax in enumerate(rmap_product_axes):
        product = predictions[num_img, :, :, i + 2] * y[num_img, :, :, 0, 0]
        ax.imshow(product)

        ax.set_title(
            f"Rmap times mask $\lambda={np.mean(product):.3f}$"  # pylint: disable=anomalous-backslash-in-string
        )
        ax.axis("off")

    for i, ax in enumerate(mask_axes):
        ax.imshow(y[num_img, :, :, 0, i])
        ax.set_title(f"Mask for ann: {i}. SNR={noise_values[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_losses_and_metrics(
    losses: list[float],
    dices: list[float],
    val_losses: list[float],
    val_dices: list[float],
    title: str,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss {title}")
    plt.subplot(1, 2, 2)
    plt.plot(dices, label="Training Dice")
    plt.plot(val_dices, label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.show()

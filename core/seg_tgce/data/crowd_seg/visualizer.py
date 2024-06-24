import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread  # pylint: disable=no-name-in-module


class BaseDirectoryNotFoundError(Exception):
    pass


def visualize_data(  # pylint: disable=too-many-locals, too-many-arguments
    x_ini_values: Tuple[int, ...],
    y_ini_values: Tuple[int, ...],
    labelers: Tuple[str, str],
    base_path: str,
    save_path: str,
    patch_tag: str,
) -> None:
    """
    Simple routine for visualizing some patches and masks
    """

    fig, axes = plt.subplots(len(x_ini_values), 3)

    for i, (x_ini, y_ini) in enumerate(zip(x_ini_values, y_ini_values)):
        if not os.path.exists(base_path):
            raise BaseDirectoryNotFoundError(
                f"Could not find base directory: {base_path}"
            )

        img_path = (
            f"{base_path}/patches/Train/{patch_tag}_x_ini_{x_ini}_y_ini_{y_ini}.png"
        )

        img = imread(img_path)
        axes[i, 0].imshow(img)
        img_title = ("Histology patch \n" if i == 0 else "") + img_path.split("/")[-1]
        axes[i, 0].set_title(img_title)
        axes[i, 0].axis("off")

        for p, (labeler) in enumerate(labelers):
            mask_path = (
                f"{base_path}/masks/Train/{labeler}/"
                f"{patch_tag}_x_ini_{x_ini}_y_ini_{y_ini}.png"
            )
            mask = imread(mask_path, -1)
            axes[i, p + 1].imshow(mask, cmap="Pastel1")
            axes[i, p + 1].axis("off")
            mask_title = (
                f"Mask for: {labeler}\n" if i == 0 else ""
            ) + f"{np.unique(mask)}"
            axes[i, p + 1].set_title(mask_title)
    fig.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    x_ini_values = (1790, 2148)
    y_ini_values = (716, 358)
    labelers = ("STAPLE", "expert")
    BASE_PATH = "../../../datasets/Histology Data"
    SAVE_PATH = "../docs/source/resources/crowd-seg-example-instances.png"
    visualize_data(
        x_ini_values=x_ini_values,
        y_ini_values=y_ini_values,
        labelers=labelers,
        base_path=BASE_PATH,
        save_path=SAVE_PATH,
        patch_tag="eval_A73Y_LL",
    )

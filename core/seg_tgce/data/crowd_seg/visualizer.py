import os
from typing import Tuple

import matplotlib.pyplot as plt
from cv2 import imread  # pylint: disable=no-name-in-module


class BaseDirectoryNotFoundError(Exception):
    pass


def visualize_data(  # pylint: disable=too-many-locals
    x_ini_values: Tuple[int, ...],
    y_ini_values: Tuple[int, ...],
    labelers: Tuple[str, str],
    base_path: str,
    save_path: str,
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
            f"{base_path}/patches/Train/core_A0AL_AN_x_ini_{x_ini}_y_ini_{y_ini}.png"
        )
        non_expert_mask_path = (
            f"{base_path}/masks/Train/{labelers[0]}/"
            f"core_A0AL_AN_x_ini_{x_ini}_y_ini_{y_ini}.png"
        )
        expert_mask_path = (
            f"{base_path}/masks/Train/{labelers[1]}/"
            f"core_A0AL_AN_x_ini_{x_ini}_y_ini_{y_ini}.png"
        )

        img = imread(img_path)
        non_expert_mask = imread(non_expert_mask_path, -1)
        expert_mask = imread(expert_mask_path, -1)

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(non_expert_mask, cmap="Pastel1")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(expert_mask, cmap="Pastel1")
        axes[i, 2].axis("off")

        if i == 0:
            axes[i, 0].set_title("Histology patch")
            axes[i, 1].set_title("Non expert label mask")
            axes[i, 2].set_title("Expert label mask")
    fig.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    x_ini_values = 1074, 1432, 2148
    y_ini_values = 1074, 1432, 2148
    labelers = "NP1", "expert"
    BASE_PATH = "../../../datasets/Histology Data"
    SAVE_PATH = "../docs/source/resources/crowd-seg-example-instances.png"
    visualize_data(
        x_ini_values=x_ini_values,
        y_ini_values=y_ini_values,
        labelers=labelers,
        base_path=BASE_PATH,
        save_path=SAVE_PATH,
    )

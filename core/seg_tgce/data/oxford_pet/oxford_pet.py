import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model

from seg_tgce.data.oxford_pet.disturbance.model import (
    ResizeToInput,
    download_base_model,
    find_last_encoder_conv_layer,
    produce_disturbed_models,
)
from seg_tgce.data.utils import (
    LabelerAssignmentManager,
    map_dataset_multiple_annotators,
)

from .oxford_iiit_pet import OxfordIiitPet

MODEL_ORIGINAL_SHAPE = (512, 512)
DEFAULT_SEED = 42


def fetch_models(noise_levels_snr: list[float], seed: int = DEFAULT_SEED) -> list[Model]:
    model_path = download_base_model()
    model_ann = load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"ResizeToInput": ResizeToInput},
    )

    last_conv_encoder_layer = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        noise_levels_snr, model_path, last_conv_encoder_layer, seed=seed
    )
    print(f"Measured snr values from produced models: {measured_snr_values}")
    return disturbance_models


def get_data_multiple_annotators(
    annotation_models: list[Model],
    target_shape: tuple[int, int] = (256, 256),
    batch_size: int = 32,
    labeling_rate: float = 0.7,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Get datasets with multiple annotators and ground truth, with optional labeler assignments.

    Args:
        annotation_models: List of models to generate noisy annotations
        target_shape: Target shape for images and masks
        batch_size: Batch size for the datasets
        labeling_rate: Probability that a labeler will label a sample (default: 0.7)
    """
    dataset = OxfordIiitPet()
    train_dataset, val_dataset, test_dataset = dataset()

    train_size = sum(1 for _ in train_dataset)

    train_labeler_manager = LabelerAssignmentManager(
        num_samples=train_size,
        num_labelers=len(annotation_models),
        labeling_rate=labeling_rate,
        seed=42,
    )

    train_data, val_data, test_data = (
        map_dataset_multiple_annotators(
            dataset=data,
            target_shape=target_shape,
            model_shape=MODEL_ORIGINAL_SHAPE,
            batch_size=batch_size,
            disturbance_models=annotation_models,
            labeler_manager=labeler_manager,
        )
        for data, labeler_manager in (
            (
                train_dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE),
                train_labeler_manager,
            ),
            (val_dataset.prefetch(tf.data.AUTOTUNE), None),
            (test_dataset, None),
        )
    )
    return train_data, val_data, test_data


def visualize_data(
    dataset: tf.data.Dataset,
    num_samples: int = 4,
    batch_index: int = 0,
    noise_levels: list[float] = [],
    save_path: str | None = None,
) -> None:
    """
    Visualize samples from the Oxford Pet dataset with their segmentation masks.

    Args:
        dataset: TensorFlow dataset containing images, masks, labeler assignments, and ground truth
        num_samples: Number of samples to visualize (default: 4)
        batch_index: Index of the batch to visualize (default: 0)
        noise_levels: List of noise levels for each annotator
        save_path: Path to save the figure (default: None, which shows the figure interactively)
    """
    for i, (images, masks, labeled_by, ground_truth) in enumerate(dataset):
        if i == batch_index:
            break
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {masks.shape}")
    print(f"Labeler mask shape: {labeled_by.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")

    num_annotators = masks.shape[-1]
    num_classes = masks.shape[-2]

    # Add one column for ground truth
    num_columns = num_annotators + 2

    fig, axes = plt.subplots(
        num_samples, num_columns, figsize=(3 * num_columns, 3 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    class_colors = {
        -1: "#000000",  # Unlabeled
        0: "#b31051",  # Pet
        1: "#440154",  # Background
        2: "#2a788e",  # Border
    }

    colors = [mcolors.to_rgb(color) for color in class_colors.values()]
    cmap = mcolors.ListedColormap(colors)

    for i in range(num_samples):
        if i >= len(images):
            break

        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        # Plot noisy annotations
        for j in range(num_annotators):
            if labeled_by[i, j] == 1:
                mask_class = tf.argmax(masks[i, :, :, :, j], axis=-1)
                mask_class = tf.where(tf.reduce_all(mask_class == 0), -1, mask_class)
                axes[i, j + 1].imshow(
                    mask_class, cmap=cmap, vmin=-1, vmax=num_classes - 1
                )
                if i == 0:
                    axes[i, j + 1].set_title(f"Ann {j + 1} (SNR: {noise_levels[j]}dB)")
            else:
                axes[i, j + 1].imshow(np.zeros(masks.shape[1:3]), cmap="gray")
                if i == 0:
                    axes[i, j + 1].set_title(f"Ann {j + 1} (SNR: {noise_levels[j]}dB)")
            axes[i, j + 1].axis("off")

        # Plot ground truth in the last column
        gt_class = tf.argmax(ground_truth[i], axis=-1)
        gt_class = tf.where(tf.reduce_all(gt_class == 0), -1, gt_class)
        axes[i, -1].imshow(gt_class, cmap=cmap, vmin=-1, vmax=num_classes - 1)
        if i == 0:
            axes[i, -1].set_title("Ground Truth")
        axes[i, -1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    noise_levels_snr = [-20.0, 0.0, 20.0]

    disturbance_models = fetch_models(noise_levels_snr, seed=42)
    train_data, val_data, test_data = get_data_multiple_annotators(
        annotation_models=disturbance_models,
        target_shape=(256, 256),
        batch_size=16,
        labeling_rate=0.7,
    )

    print("Visualizing training data samples...")
    visualize_data(
        train_data,
        num_samples=6,
        batch_index=0,
        noise_levels=noise_levels_snr,
        # save_path="oxford_pet_training_data.png",
    )

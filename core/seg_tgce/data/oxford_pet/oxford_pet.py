import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, load_model

from seg_tgce.data.oxford_pet.disturbance.model import (
    ResizeToInput,
    download_base_model,
    find_last_encoder_conv_layer,
    produce_disturbed_models,
)
from seg_tgce.data.utils import map_dataset_multiple_annotators

from .oxford_iiit_pet import OxfordIiitPet

MODEL_ORIGINAL_SHAPE = (512, 512)


def fetch_models(noise_levels_snr: list[float]) -> list[Model]:
    model_path = download_base_model()
    model_ann = load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={"ResizeToInput": ResizeToInput},
    )

    last_conv_encoder_layer = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        noise_levels_snr, model_path, last_conv_encoder_layer
    )
    print(f"Measured snr values from produced models: {measured_snr_values}")
    return disturbance_models


def get_data_multiple_annotators(
    annotation_models: list[Model],
    target_shape: tuple[int, int] = (256, 256),
    batch_size: int = 32,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    dataset = OxfordIiitPet()
    train_dataset, val_dataset, test_dataset = dataset()
    train_data, val_data, test_data = (
        map_dataset_multiple_annotators(
            dataset=data,
            target_shape=target_shape,
            model_shape=MODEL_ORIGINAL_SHAPE,
            batch_size=batch_size,
            disturbance_models=annotation_models,
        )
        for data in (train_dataset, val_dataset, test_dataset)
    )
    return train_data, val_data, test_data


def visualize_data(
    dataset: tf.data.Dataset,
    num_samples: int = 4,
    batch_index: int = 0,
    show_annotators: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Visualize samples from the Oxford Pet dataset with their segmentation masks.

    Args:
        dataset: TensorFlow dataset containing images and masks
        num_samples: Number of samples to visualize (default: 4)
        batch_index: Index of the batch to visualize (default: 0)
        show_annotators: Whether to show multiple annotator masks if available (default: True)
        save_path: Path to save the figure (default: None, which shows the figure interactively)
    """
    # Get a batch of data
    for i, (images, masks) in enumerate(dataset):
        if i == batch_index:
            break

    # Get the actual number of annotators from the masks tensor
    num_annotators = masks.shape[-1] - 1  # Subtract 1 for ground truth

    # Calculate number of columns needed
    num_columns = 2  # Image + ground truth
    if show_annotators and num_annotators > 0:
        num_columns += num_annotators  # Add columns for each annotator

    # Create figure with subplots
    fig, axes = plt.subplots(
        num_samples, num_columns, figsize=(3 * num_columns, 3 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Define class colors for visualization
    class_colors = {
        0: "#440154",  # Background
        1: "#414487",  # Pet
        2: "#2a788e",  # Border
    }

    # Convert hex colors to RGB
    colors = [plt.cm.colors.to_rgb(class_colors[i]) for i in range(3)]
    cmap = plt.cm.colors.ListedColormap(colors)

    for i in range(num_samples):
        if i >= len(images):
            break

        # Show original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        # Show ground truth mask
        axes[i, 1].imshow(masks[i, :, :, 0], cmap=cmap, vmin=0, vmax=2)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        # Show all annotator masks if available and requested
        if show_annotators and num_annotators > 0:
            for j in range(num_annotators - 1):
                axes[i, j + 2].imshow(masks[i, :, :, j + 1], cmap=cmap, vmin=0, vmax=2)
                axes[i, j + 2].set_title(f"Annotator {j + 1} Mask")
                axes[i, j + 2].axis("off")

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["Background", "Pet", "Border"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    noise_levels_snr = [-20.0, -10.0, 0.0, 10.0]

    disturbance_models = fetch_models(noise_levels_snr)
    train_data, val_data, test_data = get_data_multiple_annotators(
        annotation_models=disturbance_models,
        target_shape=(256, 256),
        batch_size=4,
    )

    print("Visualizing training data samples...")
    visualize_data(
        train_data,
        num_samples=4,
        batch_index=0,
        show_annotators=True,
    )

    print("\nVisualizing validation data samples...")
    visualize_data(
        val_data,
        num_samples=4,
        batch_index=0,
        show_annotators=True,
    )

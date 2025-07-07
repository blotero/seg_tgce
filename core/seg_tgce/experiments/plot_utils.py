import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import History
from keras.models import Model


def plot_training_history(
    history: History,
    title: str,
    metrics: list[str] = [
        "loss",
        "segmentation_output_dice_coefficient",
        "segmentation_output_jaccard_coefficient",
    ],
    save_path: str | None = None,
) -> None:
    """Plot training and validation metrics from model history.

    Args:
        history: Model training history object
        title: Title for the plot
        save_path: Optional path to save the plot. If None, plot is shown instead.
    """

    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        plt.plot(history.history[metric], label=f"Training {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def print_test_metrics(model: Model, test_data: tf.data.Dataset, model_name: str) -> None:
    print(f"\nEvaluating {model_name} model:")
    results = model.evaluate(test_data)

    # Get metric names from model
    metric_names = [m.name for m in model.metrics]

    # Print results
    print("\nTest Results:")
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")

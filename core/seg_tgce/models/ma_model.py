import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from keras.models import Model
from tensorflow import GradientTape, is_symbolic_tensor

from seg_tgce.loss.tgce import TcgeScalar


class VisualizationCallback(Callback):
    """Callback to visualize model predictions during evaluation."""

    def __init__(self, validation_data, reliability_type="scalar"):
        super().__init__()
        self.validation_data = validation_data
        self.reliability_type = reliability_type

    def on_epoch_end(self, epoch, logs=None):
        # Get a batch of validation data
        x_val, y_val = next(iter(self.validation_data))

        # Get predictions
        y_pred, lambda_r = self.model(x_val, training=False)

        # Visualize first sample
        self._visualize_results(x_val[0], y_val[0], y_pred[0], lambda_r[0])

    def _visualize_results(self, x, y, y_pred, lambda_r):
        """Visualize the results."""
        # Get number of annotators from lambda_r shape
        n_annotators = lambda_r.shape[0]

        # Create figure with subplots
        fig, axes = plt.subplots(2, n_annotators + 1, figsize=(15, 6))

        # Plot input image
        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        # Plot predicted segmentation
        pred_seg = np.argmax(y_pred, axis=-1)
        axes[1, 0].imshow(pred_seg)
        axes[1, 0].set_title("Predicted Segmentation")
        axes[1, 0].axis("off")

        print(f"Computed reliabilities: {lambda_r}")

        # Plot masks for each annotator
        for i in range(n_annotators):
            # Get mask for current annotator
            mask = np.argmax(y[..., i], axis=-1)
            axes[0, i + 1].imshow(mask)
            axes[0, i + 1].set_title(f"Annotator {i+1} Mask")
            axes[0, i + 1].axis("off")

            # Plot reliability value
            rel_value = float(lambda_r[i])
            axes[1, i + 1].text(
                0.5,
                0.5,
                f"λ = {rel_value:.3f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes[1, i + 1].transAxes,
                fontsize=12,
            )
            axes[1, i + 1].axis("off")

        plt.tight_layout()
        plt.show()


class ModelMultipleAnnotators(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reliability_type = kwargs.get("reliability_type", "pixel")

    def train_step(self, data):
        x, y = data

        with GradientTape() as tape:
            y_pred, lambda_r = self(x, training=True)
            loss = self.loss_fn.call(y_true=y, y_pred=y_pred, lambda_r=lambda_r)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred, lambda_r = self(x, training=False)
        loss = self.loss_fn.call(y_true=y, y_pred=y_pred, lambda_r=lambda_r)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

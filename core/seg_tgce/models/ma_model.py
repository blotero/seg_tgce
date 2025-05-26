import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from keras.models import Model
from tensorflow import GradientTape, is_symbolic_tensor

from seg_tgce.loss.tgce import TcgeScalar


class ScalarVisualizationCallback(Callback):
    """Callback to visualize model predictions during evaluation."""

    def __init__(self, validation_data, reliability_type="scalar"):
        super().__init__()
        self.validation_data = validation_data
        self.reliability_type = reliability_type

    def on_epoch_end(self, epoch, logs=None):
        data_iter = next(iter(self.validation_data))
        if isinstance(data_iter, dict):
            x_val, y_val, labeler_mask, y_ground_truth = (
                data_iter["x"],
                data_iter["y_labelers"],
                data_iter["labelers_mask"],
                data_iter["ground_truth"],
            )
        else:
            x_val, y_val, labeler_mask, y_ground_truth = data_iter
        y_pred, lambda_r = self.model(x_val, training=False)

        self._visualize_results(
            x_val[0], y_val[0], y_pred[0], lambda_r[0], labeler_mask[0]
        )

    def _visualize_results(self, x, y, y_pred, lambda_r, labeler_mask):
        """Visualize the results."""
        # Get number of active annotators for this sample
        active_annotators = np.where(labeler_mask == 1)[0]
        n_annotators = len(active_annotators)

        fig, axes = plt.subplots(2, n_annotators + 1, figsize=(15, 6))

        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        pred_seg = np.argmax(y_pred, axis=-1)
        axes[1, 0].imshow(pred_seg)
        axes[1, 0].set_title("Predicted Segmentation")
        axes[1, 0].axis("off")

        print(f"Computed reliabilities: {lambda_r}")

        for i, annotator_idx in enumerate(active_annotators):
            # Get mask for this annotator (shape: h, w, classes)
            mask = y[..., annotator_idx]
            mask_seg = np.argmax(mask, axis=-1)
            axes[0, i + 1].imshow(mask_seg)
            axes[0, i + 1].set_title(f"Annotator {annotator_idx+1} Mask")
            axes[0, i + 1].axis("off")

            rel_value = float(lambda_r[annotator_idx])
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


class FeatureVisualizationCallback(Callback):
    """Callback to visualize model predictions during evaluation for feature-based reliability."""

    def __init__(self, validation_data, reliability_type="features"):
        super().__init__()
        self.validation_data = validation_data
        self.reliability_type = reliability_type

    def on_epoch_end(self, epoch, logs=None):
        data_iter = next(iter(self.validation_data))
        if isinstance(data_iter, dict):
            x_val, y_val, labeler_mask, y_ground_truth = (
                data_iter["x"],
                data_iter["y_labelers"],
                data_iter["labelers_mask"],
                data_iter["ground_truth"],
            )
        else:
            x_val, y_val, labeler_mask, y_ground_truth = data_iter
        y_pred, lambda_r = self.model(x_val, training=False)

        self._visualize_results(
            x_val[0], y_val[0], y_pred[0], lambda_r[0], labeler_mask[0]
        )

    def _visualize_results(self, x, y, y_pred, lambda_r, labeler_mask):
        """Visualize the results."""
        # Get number of active annotators for this sample
        active_annotators = np.where(labeler_mask == 1)[0]
        n_annotators = len(active_annotators)

        fig, axes = plt.subplots(2, n_annotators + 1, figsize=(15, 9))

        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        pred_seg = np.argmax(y_pred, axis=-1)
        axes[1, 0].imshow(pred_seg)
        axes[1, 0].set_title("Predicted Segmentation")
        axes[1, 0].axis("off")

        for i, annotator_idx in enumerate(active_annotators):
            # Get mask for this annotator (shape: h, w, classes)
            mask = y[..., annotator_idx]
            mask_seg = np.argmax(mask, axis=-1)
            axes[0, i + 1].imshow(mask_seg)
            axes[0, i + 1].set_title(f"Annotator {annotator_idx+1} Mask")
            axes[0, i + 1].axis("off")

            # Use grayscale colormap for reliability map
            im = axes[1, i + 1].imshow(
                lambda_r[..., annotator_idx], cmap="gray", vmin=0, vmax=1
            )
            axes[1, i + 1].set_title(f"Reliability Map {annotator_idx+1}")
            axes[1, i + 1].axis("off")
            plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


class PixelVisualizationCallback(Callback):
    """Callback to visualize model predictions during evaluation for pixel-wise reliability."""

    def __init__(self, validation_data, reliability_type="pixel"):
        super().__init__()
        self.validation_data = validation_data
        self.reliability_type = reliability_type

    def on_epoch_end(self, epoch, logs=None):
        data_iter = next(iter(self.validation_data))
        if isinstance(data_iter, dict):
            x_val, y_val, labeler_mask, y_ground_truth = (
                data_iter["x"],
                data_iter["y_labelers"],
                data_iter["labelers_mask"],
                data_iter["ground_truth"],
            )
        else:
            x_val, y_val, labeler_mask, y_ground_truth = data_iter
        y_pred, lambda_r = self.model(x_val, training=False)

        self._visualize_results(
            x_val[0], y_val[0], y_pred[0], lambda_r[0], labeler_mask[0]
        )

    def _visualize_results(self, x, y, y_pred, lambda_r, labeler_mask):
        """Visualize the results."""
        # Get number of active annotators for this sample
        active_annotators = np.where(labeler_mask == 1)[0]
        n_annotators = len(active_annotators)

        fig, axes = plt.subplots(3, n_annotators + 1, figsize=(15, 9))

        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        pred_seg = np.argmax(y_pred, axis=-1)
        axes[1, 0].imshow(pred_seg)
        axes[1, 0].set_title("Predicted Segmentation")
        axes[1, 0].axis("off")

        # Calculate mean reliability only for active annotators
        # Convert tensor to numpy array first
        lambda_r_np = lambda_r.numpy()
        mean_reliability = np.mean(lambda_r_np[..., active_annotators], axis=-1)
        im = axes[2, 0].imshow(mean_reliability, cmap="gray", vmin=0, vmax=1)
        axes[2, 0].set_title("Mean Reliability Map")
        axes[2, 0].axis("off")
        plt.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)

        for i, annotator_idx in enumerate(active_annotators):
            # Get mask for this annotator (shape: h, w, classes)
            mask = y[..., annotator_idx]
            mask_seg = np.argmax(mask, axis=-1)
            axes[0, i + 1].imshow(mask_seg)
            axes[0, i + 1].set_title(f"Annotator {annotator_idx+1} Mask")
            axes[0, i + 1].axis("off")

            weighted_seg = pred_seg * lambda_r_np[..., annotator_idx]
            axes[1, i + 1].imshow(weighted_seg)
            axes[1, i + 1].set_title(
                f"Weighted Segmentation (λ={np.mean(lambda_r_np[..., annotator_idx]):.3f})"
            )
            axes[1, i + 1].axis("off")

            # Use grayscale colormap for reliability map
            im = axes[2, i + 1].imshow(
                lambda_r_np[..., annotator_idx], cmap="gray", vmin=0, vmax=1
            )
            axes[2, i + 1].set_title(f"Reliability Map {annotator_idx+1}")
            axes[2, i + 1].axis("off")
            plt.colorbar(im, ax=axes[2, i + 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


class ModelMultipleAnnotators(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reliability_type = kwargs.get("reliability_type", "pixel")

    def train_step(self, data):
        if isinstance(data, dict):
            x = data["x"]
            y = data["y_labelers"]
            labeler_mask = data["labelers_mask"]
            y_ground_truth = data["ground_truth"]
        else:
            x, y, labeler_mask, y_ground_truth = data

        with GradientTape() as tape:
            y_pred, lambda_r = self(x, training=True)
            loss = self.loss_fn.call(
                y_true=y, y_pred=y_pred, lambda_r=lambda_r, labeler_mask=labeler_mask
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y_ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, dict):
            x = data["x"]
            y = data["y_labelers"]
            labeler_mask = data["labelers_mask"]
            y_ground_truth = data["ground_truth"]
        else:
            x, y, labeler_mask, y_ground_truth = data

        y_pred, lambda_r = self(x, training=False)
        loss = self.loss_fn.call(
            y_true=y, y_pred=y_pred, lambda_r=lambda_r, labeler_mask=labeler_mask
        )

        return_metrics = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                return_metrics[metric.name] = metric.result()
            else:
                metric.update_state(y_ground_truth, y_pred)
                result = metric.result()
                if isinstance(result, dict):
                    return_metrics.update(result)
                else:
                    return_metrics[metric.name] = result
        return return_metrics

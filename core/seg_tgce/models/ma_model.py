import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.models import Model
from tensorflow import GradientTape


class BaselineVisualizationCallback(Callback):
    def __init__(
        self, validation_data: tf.data.Dataset, save_dir: str | None = None
    ) -> None:
        super().__init__()
        self.validation_data = validation_data
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        data_iter = next(iter(self.validation_data))
        if isinstance(data_iter, dict):
            x_val: tf.Tensor = data_iter["image"]
            y_val: tf.Tensor = data_iter["ground_truth"]
        else:
            x_val, y_val = data_iter
        y_pred: tf.Tensor = self.model(tf.expand_dims(x_val[0], 0), training=False)

        self._visualize_results(x_val[0], y_val[0], y_pred[0], epoch)

    def _visualize_results(
        self, x: tf.Tensor, y: tf.Tensor, y_pred: tf.Tensor, epoch: int
    ) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

        ax1.imshow(x)
        ax1.set_title("Input Image")
        ax1.axis("off")

        pred_seg: np.ndarray = np.argmax(y_pred, axis=-1)
        ax2.imshow(pred_seg)
        ax2.set_title("Predicted Segmentation")
        ax2.axis("off")

        ax3.imshow(np.argmax(y, axis=-1))
        ax3.set_title("Ground Truth")
        ax3.axis("off")

        plt.tight_layout()
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/baseline_vis_epoch_{epoch}.png")
            plt.close()
        else:
            plt.show()


class ScalarVisualizationCallback(Callback):
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        reliability_type: str = "scalar",
        save_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.validation_data = validation_data
        self.reliability_type = reliability_type
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        data_iter = next(iter(self.validation_data))
        if isinstance(data_iter, dict):
            x_val: tf.Tensor = data_iter["image"]
            y_val: tf.Tensor = data_iter["masks"]
            labeler_mask: tf.Tensor = data_iter["labelers_mask"]
            y_ground_truth: tf.Tensor = data_iter["ground_truth"]
        else:
            x_val, y_val, labeler_mask, y_ground_truth = data_iter
        y_pred: tf.Tensor
        lambda_r: tf.Tensor
        y_pred, lambda_r = self.model(tf.expand_dims(x_val[0], 0), training=False)

        self._visualize_results(
            x_val[0], y_val[0], y_pred[0], lambda_r[0], labeler_mask[0], epoch
        )

    def _visualize_results(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        y_pred: tf.Tensor,
        lambda_r: tf.Tensor,
        labeler_mask: tf.Tensor,
        epoch: int,
    ) -> None:
        active_annotators: np.ndarray = np.where(labeler_mask == 1)[0]
        n_annotators: int = len(active_annotators)

        fig, axes = plt.subplots(2, n_annotators + 1, figsize=(15, 6))

        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        pred_seg: np.ndarray = np.argmax(y_pred, axis=-1)
        axes[1, 0].imshow(pred_seg)
        axes[1, 0].set_title("Predicted Segmentation")
        axes[1, 0].axis("off")

        print(f"Computed reliabilities: {lambda_r}")

        for i, annotator_idx in enumerate(active_annotators):
            mask: tf.Tensor = y[..., annotator_idx]
            mask_seg: np.ndarray = np.argmax(mask, axis=-1)
            axes[0, i + 1].imshow(mask_seg)
            axes[0, i + 1].set_title(f"Annotator {annotator_idx + 1} Mask")
            axes[0, i + 1].axis("off")

            rel_value: float = float(lambda_r[annotator_idx])
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
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/scalar_vis_epoch_{epoch}.png")
            plt.close()
        else:
            plt.show()


class FeatureVisualizationCallback(Callback):
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        reliability_type: str = "features",
        save_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.validation_data = validation_data
        self.reliability_type = reliability_type
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        data_iter = next(iter(self.validation_data))
        if isinstance(data_iter, dict):
            x_val: tf.Tensor = data_iter["image"]
            y_val: tf.Tensor = data_iter["masks"]
            labeler_mask: tf.Tensor = data_iter["labelers_mask"]
            y_ground_truth: tf.Tensor = data_iter["ground_truth"]
        else:
            x_val, y_val, labeler_mask, y_ground_truth = data_iter
        y_pred: tf.Tensor
        lambda_r: tf.Tensor
        y_pred, lambda_r = self.model(tf.expand_dims(x_val[0], 0), training=False)

        self._visualize_results(
            x_val[0], y_val[0], y_pred[0], lambda_r[0], labeler_mask[0], epoch
        )

    def _visualize_results(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        y_pred: tf.Tensor,
        lambda_r: tf.Tensor,
        labeler_mask: tf.Tensor,
        epoch: int,
    ) -> None:
        active_annotators: np.ndarray = np.where(labeler_mask == 1)[0]
        n_annotators: int = len(active_annotators)

        fig, axes = plt.subplots(2, n_annotators + 1, figsize=(15, 9))

        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        pred_seg: np.ndarray = np.argmax(y_pred, axis=-1)
        axes[1, 0].imshow(pred_seg)
        axes[1, 0].set_title("Predicted Segmentation")
        axes[1, 0].axis("off")

        for i, annotator_idx in enumerate(active_annotators):
            mask: tf.Tensor = y[..., annotator_idx]
            mask_seg: np.ndarray = np.argmax(mask, axis=-1)
            axes[0, i + 1].imshow(mask_seg)
            axes[0, i + 1].set_title(f"Annotator {annotator_idx + 1} Mask")
            axes[0, i + 1].axis("off")

            im = axes[1, i + 1].imshow(
                lambda_r[..., annotator_idx], cmap="gray", vmin=0, vmax=1
            )
            axes[1, i + 1].set_title(f"Reliability Map {annotator_idx + 1}")
            axes[1, i + 1].axis("off")
            plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/feature_vis_epoch_{epoch}.png")
            plt.close()
        else:
            plt.show()


class PixelVisualizationCallback(Callback):
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        reliability_type: str = "pixel",
        save_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.validation_data = validation_data
        self.reliability_type = reliability_type
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        data_iter = next(iter(self.validation_data))
        if isinstance(data_iter, dict):
            x_val: tf.Tensor = data_iter["image"]
            y_val: tf.Tensor = data_iter["masks"]
            labeler_mask: tf.Tensor = data_iter["labelers_mask"]
            y_ground_truth: tf.Tensor = data_iter["ground_truth"]
        else:
            x_val, y_val, labeler_mask, y_ground_truth = data_iter
        y_pred: tf.Tensor
        lambda_r: tf.Tensor
        y_pred, lambda_r = self.model(tf.expand_dims(x_val[0], 0), training=False)

        self._visualize_results(
            x_val[0], y_val[0], y_pred[0], lambda_r[0], labeler_mask[0], epoch
        )

    def _visualize_results(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        y_pred: tf.Tensor,
        lambda_r: tf.Tensor,
        labeler_mask: tf.Tensor,
        epoch: int,
    ) -> None:
        active_annotators: np.ndarray = np.where(labeler_mask == 1)[0]
        n_annotators: int = len(active_annotators)

        fig, axes = plt.subplots(3, n_annotators + 1, figsize=(15, 9))

        axes[0, 0].imshow(x)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        pred_seg: np.ndarray = np.argmax(y_pred, axis=-1)
        axes[1, 0].imshow(pred_seg)
        axes[1, 0].set_title("Predicted Segmentation")
        axes[1, 0].axis("off")

        lambda_r_np: np.ndarray = lambda_r.numpy()
        mean_reliability: np.ndarray = np.mean(
            lambda_r_np[..., active_annotators], axis=-1
        )
        im = axes[2, 0].imshow(mean_reliability, cmap="gray", vmin=0, vmax=1)
        axes[2, 0].set_title("Mean Reliability Map")
        axes[2, 0].axis("off")
        plt.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)

        for i, annotator_idx in enumerate(active_annotators):
            mask: tf.Tensor = y[..., annotator_idx]
            mask_seg: np.ndarray = np.argmax(mask, axis=-1)
            axes[0, i + 1].imshow(mask_seg)
            axes[0, i + 1].set_title(f"Annotator {annotator_idx + 1} Mask")
            axes[0, i + 1].axis("off")

            weighted_seg: np.ndarray = pred_seg * lambda_r_np[..., annotator_idx]
            axes[1, i + 1].imshow(weighted_seg)
            axes[1, i + 1].set_title(
                f"Weighted Segmentation (λ={np.mean(lambda_r_np[..., annotator_idx]):.3f})"
            )
            axes[1, i + 1].axis("off")

            im = axes[2, i + 1].imshow(
                lambda_r_np[..., annotator_idx], cmap="gray", vmin=0, vmax=1
            )
            axes[2, i + 1].set_title(f"Reliability Map {annotator_idx + 1}")
            axes[2, i + 1].axis("off")
            plt.colorbar(im, ax=axes[2, i + 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if self.save_dir:
            plt.savefig(f"{self.save_dir}/pixel_vis_epoch_{epoch}.png")
            plt.close()
        else:
            plt.show()


class ModelMultipleAnnotators(Model):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reliability_type: str = kwargs.get("reliability_type", "pixel")

    def train_step(
        self, data: dict[str, tf.Tensor] | tuple[tf.Tensor, ...]
    ) -> dict[str, float]:
        if isinstance(data, dict):
            x: tf.Tensor = data["image"]
            y: tf.Tensor = data["masks"]
            labeler_mask: tf.Tensor = data["labelers_mask"]
            y_ground_truth: tf.Tensor = data["ground_truth"]
        else:
            x, y, labeler_mask, y_ground_truth = data

        with GradientTape() as tape:
            y_pred: tf.Tensor
            lambda_r: tf.Tensor
            y_pred, lambda_r = self(x, training=True)
            loss: tf.Tensor = self.loss_fn.call(
                y_true=y, y_pred=y_pred, lambda_r=lambda_r, labeler_mask=labeler_mask
            )

        trainable_vars: list[tf.Variable] = self.trainable_variables
        gradients: list[tf.Tensor] = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y_ground_truth, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(
        self, data: dict[str, tf.Tensor] | tuple[tf.Tensor, ...]
    ) -> dict[str, float | dict[str, float]]:
        if isinstance(data, dict):
            x: tf.Tensor = data["image"]
            y: tf.Tensor = data["masks"]
            labeler_mask: tf.Tensor = data["labelers_mask"]
            y_ground_truth: tf.Tensor = data["ground_truth"]
        else:
            x, y, labeler_mask, y_ground_truth = data

        y_pred: tf.Tensor
        lambda_r: tf.Tensor
        y_pred, lambda_r = self(x, training=False)
        loss: tf.Tensor = self.loss_fn.call(
            y_true=y, y_pred=y_pred, lambda_r=lambda_r, labeler_mask=labeler_mask
        )

        return_metrics: dict[str, float | dict[str, float]] = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                return_metrics[metric.name] = metric.result()
            else:
                metric.update_state(y_ground_truth, y_pred)
                result: float | dict[str, float] = metric.result()
                if isinstance(result, dict):
                    return_metrics.update(result)
                else:
                    return_metrics[metric.name] = result
        return return_metrics

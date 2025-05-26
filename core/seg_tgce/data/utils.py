from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow import Tensor


def disturb_mask(
    model: Model,
    image: Tensor,
    model_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Tensor:
    return tf.image.resize(model(tf.image.resize(image, model_shape)), target_shape)


def add_noisy_annotators(
    img: Tensor,
    models: list[Tensor],
    model_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> Tensor:
    return tf.transpose(
        [
            disturb_mask(model, img, model_shape=model_shape, target_shape=target_shape)
            for model in models
        ],
        [2, 3, 1, 4, 0],
    )


class LabelerAssignmentManager:
    def __init__(
        self,
        num_samples: int,
        num_labelers: int,
        labeling_rate: float = 0.7,
        seed: int = 42,
    ):
        """
        Initialize the labeler assignment manager.

        Args:
            num_samples: Number of samples in the dataset
            num_labelers: Number of available labelers
            labeling_rate: Probability that each labeler will label a sample (default: 0.7)
            seed: Random seed for reproducibility (default: 42)
        """
        self.num_samples = num_samples
        self.num_labelers = num_labelers
        self.labeling_rate = labeling_rate

        np.random.seed(seed)

        self.assignments = np.zeros((num_samples, num_labelers), dtype=bool)

        for labeler_idx in range(num_labelers):
            labeler_mask = np.random.random(num_samples) < labeling_rate
            self.assignments[:, labeler_idx] = labeler_mask

        self.assignments_tf = tf.convert_to_tensor(self.assignments, dtype=tf.float32)

    def get_labeler_mask(self, sample_idx: int | tf.Tensor) -> tf.Tensor:
        """
        Get a binary mask indicating which labelers are assigned to a sample.

        Args:
            sample_idx: Index of the sample (can be int or tf.Tensor)

        Returns:
            Tensor of shape (num_labelers,) with 1s for assigned labelers and 0s for unassigned
        """
        if isinstance(sample_idx, tf.Tensor):
            return tf.gather(self.assignments_tf, sample_idx)
        else:
            return tf.convert_to_tensor(self.assignments[sample_idx], dtype=tf.float32)

    def get_assignment_matrix(self) -> tf.Tensor:
        """
        Get the full assignment matrix.

        Returns:
            Tensor of shape (num_samples, num_labelers) with 1s for assignments and 0s for non-assignments
        """
        return self.assignments_tf

    def get_labeler_stats(self) -> Dict[str, float]:
        """
        Get statistics about the labeler assignments.

        Returns:
            Dictionary containing:
            - 'avg_samples_per_labeler': Average number of samples each labeler is assigned to
            - 'avg_labelers_per_sample': Average number of labelers assigned to each sample
        """
        samples_per_labeler = np.sum(self.assignments, axis=0)
        labelers_per_sample = np.sum(self.assignments, axis=1)

        return {
            "avg_samples_per_labeler": float(np.mean(samples_per_labeler)),
            "avg_labelers_per_sample": float(np.mean(labelers_per_sample)),
        }


def map_dataset_multiple_annotators(
    dataset: Tensor,
    target_shape: tuple[int, int],
    model_shape: tuple[int, int],
    batch_size: int,
    disturbance_models: list[Model],
    labeler_manager: LabelerAssignmentManager | None = None,
) -> Tensor:
    """
    Map dataset to include multiple annotator masks and ground truth as separate elements.

    Args:
        dataset: Input dataset
        target_shape: Target shape for images and masks
        model_shape: Shape expected by the disturbance models
        batch_size: Batch size for the dataset
        disturbance_models: List of models to generate noisy annotations
        labeler_manager: Optional manager for labeler assignments
    """
    dataset_ = dataset.map(
        lambda img, mask, label, id_img: (img, mask),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.map(
        lambda img, mask: (
            tf.image.resize(img, target_shape),
            tf.image.resize(mask, target_shape),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if labeler_manager is not None:
        dataset_ = dataset_.enumerate()

        dataset_ = dataset_.map(
            lambda idx, data: (
                data[0],
                add_noisy_annotators(
                    tf.expand_dims(data[0], 0),
                    disturbance_models,
                    model_shape=model_shape,
                    target_shape=target_shape,
                ),
                data[1],  # Keep the ground truth mask
                labeler_manager.get_labeler_mask(idx),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Apply labeler mask to noisy annotations
        dataset_ = dataset_.map(
            lambda img, masks, gt_mask, labeler_mask: (
                img,
                tf.multiply(masks, tf.reshape(labeler_mask, [1, 1, 1, -1])),
                labeler_mask,
                gt_mask,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset_ = dataset_.map(
            lambda img, mask: (
                img,
                add_noisy_annotators(
                    tf.expand_dims(img, 0),
                    disturbance_models,
                    model_shape=model_shape,
                    target_shape=target_shape,
                ),
                mask,  # Keep the ground truth mask
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Add all ones labeler mask for noisy annotations
        dataset_ = dataset_.map(
            lambda img, masks, gt_mask: (
                img,
                masks,
                tf.ones(tf.shape(masks)[-1]),  # All labelers assigned
                gt_mask,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset_ = dataset_.map(
        lambda img, mask, labeler_mask, gt_mask: (
            img,
            tf.squeeze(mask, axis=2),
            labeler_mask,
            gt_mask,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset_ = dataset_.batch(batch_size)
    return dataset_

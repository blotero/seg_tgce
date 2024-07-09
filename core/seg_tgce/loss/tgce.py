from dataclasses import dataclass
from typing import Any

import keras.backend as K
import tensorflow as tf
from keras.losses import Loss
from tensorflow import cast, Tensor
from tensorflow import float32 as tf_float32

TARGET_DATA_TYPE = tf_float32


def safe_divide(
    numerator: Tensor, denominator: Tensor, epsilon: float = 1e-8
) -> Tensor:
    """Safely divide two tensors, avoiding division by zero."""
    return tf.math.divide(
        numerator, tf.clip_by_value(denominator, epsilon, tf.reduce_max(denominator))
    )


def stable_pow(x: Tensor, p: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Compute x^p safely by ensuring x is within a valid range."""
    return tf.pow(tf.clip_by_value(x, epsilon, 1.0 - epsilon), p)


@dataclass
class TcgeConfig:
    """
    TCGE configuration parameters.
    """

    num_annotators: int = 5
    num_classes: int = 2
    gamma: float = 0.1


def binary_entropy(target: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    Adds binary entropy to the loss.
    """
    pred_probs = tf.sigmoid(pred)
    hadamard_product = target * pred_probs
    epsilon = 1e-12
    entropy = -hadamard_product * K.log(hadamard_product + epsilon) - (
        1 - hadamard_product
    ) * K.log(1 - hadamard_product + epsilon)
    return K.mean(entropy)


class TcgeSs(Loss):
    """
    Truncated generalized cross entropy
    for semantic segmentation loss.
    """

    def __init__(
        self,
        config: TcgeConfig,
        q: float = 0.1,
        name: str = "TGCE_SS",
    ) -> None:
        self.q = q
        self.num_annotators = config.num_annotators
        self.num_classes = config.num_classes
        self.gamma = config.gamma
        super().__init__(name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = cast(y_true, TARGET_DATA_TYPE)
        y_pred = cast(y_pred, TARGET_DATA_TYPE)

        y_pred = y_pred[..., : self.num_classes + self.num_annotators]

        y_true_shape = tf.shape(y_true)

        new_shape = tf.concat(
            [y_true_shape[:-2], [self.num_classes, self.num_annotators]], axis=0
        )
        y_true = tf.reshape(y_true, new_shape)

        lambda_r = y_pred[..., self.num_classes :]
        y_pred_ = y_pred[..., : self.num_classes]

        n_samples = tf.shape(y_pred_)[0]
        width = tf.shape(y_pred_)[1]
        height = tf.shape(y_pred_)[2]

        y_pred_ = y_pred_[..., tf.newaxis]
        y_pred_ = tf.repeat(y_pred_, repeats=[self.num_annotators], axis=-1)

        epsilon = 1e-8
        y_pred_ = tf.clip_by_value(y_pred_, epsilon, 1.0 - epsilon)

        term_r = tf.math.reduce_mean(
            tf.math.multiply(
                y_true,
                safe_divide(
                    (
                        tf.ones(
                            [
                                n_samples,
                                width,
                                height,
                                self.num_classes,
                                self.num_annotators,
                            ]
                        )
                        - stable_pow(y_pred_, self.q)
                    ),
                    (self.q + epsilon),
                ),
            ),
            axis=-2,
        )

        term_c = tf.math.multiply(
            tf.ones([n_samples, width, height, self.num_annotators]) - lambda_r,
            safe_divide(
                (
                    tf.ones([n_samples, width, height, self.num_annotators])
                    - stable_pow(
                        (1 / self.num_classes)
                        * tf.ones([n_samples, width, height, self.num_annotators]),
                        self.q,
                    )
                ),
                (self.q + epsilon),
            ),
        )

        loss = tf.math.reduce_mean(tf.math.multiply(lambda_r, term_r) + term_c)
        loss = tf.where(tf.math.is_nan(loss), tf.constant(1e-8), loss)

        return loss

    def get_config(
        self,
    ) -> Any:
        """
        Retrieves loss configuration.
        """
        base_config = super().get_config()
        return {**base_config, "q": self.q}


class TcgeSsSparse(TcgeSs):
    """
    Truncated generalized cross entropy
    for semantic segmentation loss.
    This is a much more sparse version which completes missing dimensions
    for facilitating its use in the Bayesian U-Net.
    """

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = cast(y_true, TARGET_DATA_TYPE)
        y_pred = cast(y_pred, TARGET_DATA_TYPE)

        y_pred = y_pred[..., : self.num_classes + self.num_annotators]

        y_true_shape = tf.shape(y_true)

        new_shape = tf.concat(
            [y_true_shape[:-1], [self.num_classes, self.num_annotators]], axis=0
        )
        y_true = tf.reshape(y_true, new_shape)

        lambda_r = y_pred[..., self.num_classes :]
        y_pred_ = y_pred[..., : self.num_classes]

        n_samples = tf.shape(y_pred_)[0]
        width = tf.shape(y_pred_)[1]
        height = tf.shape(y_pred_)[2]

        y_pred_ = y_pred_[..., tf.newaxis]
        y_pred_ = tf.repeat(y_pred_, repeats=[self.num_annotators], axis=-1)

        epsilon = 1e-8
        y_pred_ = tf.clip_by_value(y_pred_, epsilon, 1.0 - epsilon)

        term_r = tf.math.reduce_mean(
            tf.math.multiply(
                y_true,
                (
                    tf.ones(
                        [
                            n_samples,
                            width,
                            height,
                            self.num_classes,
                            self.num_annotators,
                        ]
                    )
                    - tf.pow(y_pred_, self.q)
                )
                / (self.q + epsilon),
            ),
            axis=-2,
        )

        term_c = tf.math.multiply(
            tf.ones([n_samples, width, height, self.num_annotators]) - lambda_r,
            (
                tf.ones([n_samples, width, height, self.num_annotators])
                - tf.pow(
                    (1 / self.num_classes)
                    * tf.ones([n_samples, width, height, self.num_annotators]),
                    self.q,
                )
            )
            / (self.q + epsilon),
        )

        loss = tf.math.reduce_mean(tf.math.multiply(lambda_r, term_r) + term_c)
        loss = tf.where(tf.math.is_nan(loss), tf.constant(1e-8), loss)

        return loss

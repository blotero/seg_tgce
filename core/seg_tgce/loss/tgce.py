from dataclasses import dataclass
from typing import Any

import keras.backend as K
import tensorflow as tf
from keras.losses import Loss


@dataclass
class TcgeConfig:
    """
    TCGE configuration parameters.
    """

    num_annotators: int = 5
    num_classes: int = 2
    gamma: float = 0.1


def binary_entropy(target, pred):
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


class TcgeSs(Loss):  # type: ignore
    """
    Truncated generalized cross entropy
    for semantic segmentation loss.
    """

    def __init__(
        self,
        config: TcgeConfig,
        q: float = 0.1,
        name: str = "TGCE_SS",
        smooth: float = 1e-5,
    ) -> None:
        self.q = q
        self.num_annotators = config.num_annotators
        self.num_classes = config.num_classes
        self.smooth = smooth
        self.gamma = config.gamma
        super().__init__(name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calls loss function itself.
        """
        lambda_r = y_pred[..., self.num_classes :]  # type:ignore
        y_pred_ = y_pred[..., : self.num_classes]  # type:ignore
        n_samples, width, height, _ = y_pred_.shape
        y_pred_ = y_pred_[..., tf.newaxis]  # type:ignore
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
                / (self.q + epsilon + self.smooth),
            ),
            axis=-2,
        )
        term_c = tf.math.multiply(
            tf.ones([n_samples, width, height, self.num_annotators]) - lambda_r,
            (
                tf.ones([n_samples, width, height, self.num_annotators])
                - tf.pow(
                    (1 / self.num_classes + self.smooth)
                    * tf.ones([n_samples, width, height, self.num_annotators]),
                    self.q,
                )
            )
            / (self.q + epsilon + self.smooth),
        )

        loss = tf.math.reduce_mean(tf.math.multiply(lambda_r, term_r) + term_c)
        loss = tf.where(tf.math.is_nan(loss), tf.constant(0.0), loss)
        entropy_term = binary_entropy(y_true, y_pred_)
        loss = tf.math.add(loss, self.gamma * entropy_term)
        return loss

    def get_config(
        self,
    ) -> Any:
        """
        Retrieves loss configuration.
        """
        base_config = super().get_config()
        return {**base_config, "q": self.q}
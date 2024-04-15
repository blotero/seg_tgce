from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from keras.losses import Loss


@dataclass
class TcgeConfig:
    """
    TCGE configuration parameters.
    """

    num_annotators: int = 5
    num_classes: int = 2


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
        super().__init__(name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calls loss function itself.
        """
        lambda_r = y_pred[..., self.num_classes:]  # type:ignore
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
        if tf.math.is_nan(loss):
            loss = tf.where(tf.math.is_nan(loss), tf.constant(0.0), loss)
        return loss

    def get_config(
        self,
    ) -> Any:
        """
        Retrieves loss configuration.
        """
        base_config = super().get_config()
        return {**base_config, "q": self.q}

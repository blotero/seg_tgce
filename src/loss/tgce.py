from typing import Any

import tensorflow as tf
from tf.keras.losses import Loss


class TGCE_SS(Loss):  # type: ignore
    def __init__(
        self,
        q: float = 0.1,
        name: str = "TGCE_SS",
        R: int = 5,
        K_: int = 2,
        smooth: float = 1e-5,
    ) -> None:
        self.q = q
        self.R = R
        self.K_ = K_
        self.smooth = smooth
        super(TGCE_SS, self).__init__(name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        Lambda_r = y_pred[..., self.K_ :]
        y_pred_ = y_pred[..., : self.K_]
        N, W, H, _ = y_pred_.shape
        y_pred_ = y_pred_[..., tf.newaxis]
        y_pred_ = tf.repeat(y_pred_, repeats=[self.R], axis=-1)

        epsilon = 1e-8
        y_pred_ = tf.clip_by_value(y_pred_, epsilon, 1.0 - epsilon)

        term_r = tf.math.reduce_mean(
            tf.math.multiply(
                y_true,
                (tf.ones([N, W, H, self.K_, self.R]) - tf.pow(y_pred_, self.q))
                / (self.q + epsilon + self.smooth),
            ),
            axis=-2,
        )
        term_c = tf.math.multiply(
            tf.ones([N, W, H, self.R]) - Lambda_r,
            (
                tf.ones([N, W, H, self.R])
                - tf.pow(
                    (1 / self.K_ + self.smooth) * tf.ones([N, W, H, self.R]), self.q
                )
            )
            / (self.q + epsilon + self.smooth),
        )

        loss = tf.math.reduce_mean(tf.math.multiply(Lambda_r, term_r) + term_c)
        if tf.math.is_nan(loss):
            loss = tf.where(tf.math.is_nan(loss), tf.constant(0.0), loss)
        return loss

    def get_config(
        self,
    ) -> Any:
        base_config = super().get_config()
        return {**base_config, "q": self.q}

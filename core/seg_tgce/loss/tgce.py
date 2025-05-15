from typing import Any, Literal

import tensorflow as tf
from keras.losses import Loss
from tensorflow import Tensor
from tensorflow import float32 as tf_float32

TARGET_DATA_TYPE = tf_float32

ReliabilityType = Literal["scalar", "features", "pixel"]


def safe_divide(
    numerator: Tensor, denominator: Tensor, epsilon: float = 1e-8
) -> Tensor:
    """Safely divide two tensors, avoiding division by zero."""
    return tf.math.divide(
        numerator, tf.clip_by_value(denominator, epsilon, tf.reduce_max(denominator))
    )


def safe_pow(x: Tensor, p: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Compute x^p safely by ensuring x is within a valid range."""
    return tf.pow(tf.clip_by_value(x, epsilon, 1.0 - epsilon), p)


class TcgeScalar(Loss):
    """
    Truncated generalized cross entropy
    for semantic segmentation loss.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_classes: int,
        name: str = "TGCE_SS",
        q: float = 0.1,
        noise_tolerance: float = 0.1,
        lambda_reg_weight: float = 0.1,
        lambda_entropy_weight: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        self.q = q
        self.num_classes = num_classes
        self.noise_tolerance = noise_tolerance
        self.lambda_reg_weight = lambda_reg_weight
        self.lambda_entropy_weight = lambda_entropy_weight
        self.epsilon = epsilon
        super().__init__(name=name)

    def call(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, lambda_r: tf.Tensor
    ) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        lambda_r = tf.clip_by_value(lambda_r, self.epsilon, 1.0 - self.epsilon)

        y_pred_exp = tf.expand_dims(y_pred, axis=-1)
        y_pred_exp = tf.tile(y_pred_exp, [1, 1, 1, 1, tf.shape(y_true)[-1]])

        lambda_r = tf.expand_dims(tf.expand_dims(lambda_r, 1), 1)
        lambda_r = tf.tile(lambda_r, [1, tf.shape(y_pred)[1], tf.shape(y_pred)[2], 1])

        correct_probs = tf.reduce_sum(y_true * y_pred_exp, axis=-2)
        correct_probs = tf.clip_by_value(
            correct_probs, self.epsilon, 1.0 - self.epsilon
        )

        term1 = (
            lambda_r * (1.0 - tf.pow(correct_probs, self.q)) / (self.q + self.epsilon)
        )
        term2 = (1.0 - lambda_r) * (
            (1.0 - tf.pow(self.noise_tolerance, self.q)) / (self.q + self.epsilon)
        )

        lambda_reg = self.lambda_reg_weight * tf.reduce_mean(tf.square(lambda_r - 0.5))

        lambda_entropy = -self.lambda_entropy_weight * tf.reduce_mean(
            lambda_r * tf.math.log1p(lambda_r)
            + (1 - lambda_r) * tf.math.log1p(1 - lambda_r)
        )

        total_loss = tf.reduce_mean(term1 + term2) + lambda_reg + lambda_entropy

        total_loss = tf.where(
            tf.math.is_nan(total_loss),
            tf.constant(1e6, dtype=total_loss.dtype),
            total_loss,
        )

        return total_loss

    def get_config(
        self,
    ) -> Any:
        """
        Retrieves loss configuration.
        """
        base_config = super().get_config()
        return {
            **base_config,
            "q": self.q,
            "lambda_reg_weight": self.lambda_reg_weight,
            "lambda_entropy_weight": self.lambda_entropy_weight,
            "epsilon": self.epsilon,
        }

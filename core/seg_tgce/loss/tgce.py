from typing import Any

import tensorflow as tf
from keras.losses import Loss
from tensorflow import Tensor
from tensorflow import float32 as tf_float32

TARGET_DATA_TYPE = tf_float32


def safe_divide(numerator: Tensor, denominator: Tensor, epsilon: float = 1e-8) -> Tensor:
    return tf.math.divide(
        numerator, tf.clip_by_value(denominator, epsilon, tf.reduce_max(denominator))
    )


def safe_pow(x: Tensor, p: Tensor, epsilon: float = 1e-8) -> Tensor:
    return tf.pow(tf.clip_by_value(x, epsilon, 1.0 - epsilon), p)


def reliability_penalizer(
    lms: Tensor, lambdas: Tensor, a: float, b: float, c: float
) -> Tensor:
    x = lambdas - lms
    return c * tf.maximum(1 / (1 - a) * x * tf.exp((x - 1) / b), 0)


class TgceScalar(Loss):
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
        a: float = 0.7,
        b: float = 0.7,
        c: float = 1.0,
        lambda_reg_weight: float = 0.1,
        lambda_entropy_weight: float = 0.1,
        lambda_sum_weight: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        self.q = q
        self.num_classes = num_classes
        self.noise_tolerance = noise_tolerance
        self.a = a
        self.b = b
        self.c = c
        self.lambda_reg_weight = lambda_reg_weight
        self.lambda_entropy_weight = lambda_entropy_weight
        self.lambda_sum_weight = lambda_sum_weight
        self.epsilon = epsilon
        super().__init__(name=name)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        lambda_r: tf.Tensor,
        labeler_mask: tf.Tensor,
    ) -> tf.Tensor:
        # Cast inputs to target data type
        y_true = tf.cast(y_true, TARGET_DATA_TYPE)
        y_pred = tf.cast(y_pred, TARGET_DATA_TYPE)
        lambda_r = tf.cast(lambda_r, TARGET_DATA_TYPE)

        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        lambda_r = tf.clip_by_value(lambda_r, self.epsilon, 1.0 - self.epsilon)

        reg_term = reliability_penalizer(labeler_mask, lambda_r, self.a, self.b, self.c)

        y_pred_exp = tf.expand_dims(y_pred, axis=-1)
        y_pred_exp = tf.tile(y_pred_exp, [1, 1, 1, 1, tf.shape(y_true)[-1]])

        lambda_r = tf.expand_dims(tf.expand_dims(lambda_r, 1), 1)
        lambda_r = tf.tile(lambda_r, [1, tf.shape(y_pred)[1], tf.shape(y_pred)[2], 1])

        lambda_r = lambda_r * tf.expand_dims(tf.expand_dims(labeler_mask, 1), 1)

        correct_probs = tf.reduce_sum(y_true * y_pred_exp, axis=-2)
        correct_probs = tf.clip_by_value(correct_probs, self.epsilon, 1.0 - self.epsilon)

        term1 = lambda_r * (1.0 - tf.pow(correct_probs, self.q)) / (self.q + self.epsilon)
        term2 = (1.0 - lambda_r) * (
            (1.0 - tf.pow(self.noise_tolerance, self.q)) / (self.q + self.epsilon)
        )

        # Only compute regularization terms for valid labelers
        valid_lambda_r = lambda_r * tf.expand_dims(tf.expand_dims(labeler_mask, 1), 1)
        lambda_reg = self.lambda_reg_weight * tf.reduce_mean(
            tf.square(valid_lambda_r - 0.5)
        )

        lambda_entropy = -self.lambda_entropy_weight * tf.reduce_mean(
            valid_lambda_r * tf.math.log1p(valid_lambda_r)
            + (1 - valid_lambda_r) * tf.math.log1p(1 - valid_lambda_r)
        )

        lambda_sum = self.lambda_sum_weight * tf.reduce_mean(
            tf.square(tf.reduce_sum(valid_lambda_r, axis=-1) - 1.0)
        )

        total_loss = (
            tf.reduce_mean(term1 + term2)
            + reg_term
            + lambda_reg
            + lambda_entropy
            + lambda_sum
        )

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
            "b": self.b,
            "lambda_reg_weight": self.lambda_reg_weight,
            "lambda_entropy_weight": self.lambda_entropy_weight,
            "lambda_sum_weight": self.lambda_sum_weight,
            "epsilon": self.epsilon,
        }


class TgceFeatures(Loss):
    """
    Truncated generalized cross entropy for semantic segmentation loss
    with feature-based reliability (reliability map from bottleneck features).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_classes: int,
        name: str = "TGCE_SS_FEATURES",
        q: float = 0.1,
        noise_tolerance: float = 0.1,
        lambda_reg_weight: float = 0.1,
        lambda_entropy_weight: float = 0.1,
        lambda_sum_weight: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        self.q = q
        self.num_classes = num_classes
        self.noise_tolerance = noise_tolerance
        self.lambda_reg_weight = lambda_reg_weight
        self.lambda_entropy_weight = lambda_entropy_weight
        self.lambda_sum_weight = lambda_sum_weight
        self.epsilon = epsilon
        super().__init__(name=name)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        lambda_r: tf.Tensor,
        labeler_mask: tf.Tensor,
    ) -> tf.Tensor:
        # Cast inputs to target data type
        y_true = tf.cast(y_true, TARGET_DATA_TYPE)
        y_pred = tf.cast(y_pred, TARGET_DATA_TYPE)
        lambda_r = tf.cast(lambda_r, TARGET_DATA_TYPE)

        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        lambda_r = tf.clip_by_value(lambda_r, self.epsilon, 1.0 - self.epsilon)

        # Expand predictions to match annotators dimension
        y_pred_exp = tf.expand_dims(y_pred, axis=-1)
        y_pred_exp = tf.tile(y_pred_exp, [1, 1, 1, 1, tf.shape(y_true)[-1]])

        # Resize lambda_r to match spatial dimensions of predictions
        lambda_r = tf.image.resize(lambda_r, tf.shape(y_pred)[1:3], method="bilinear")

        # Apply labeler mask to lambda_r
        lambda_r = lambda_r * tf.expand_dims(tf.expand_dims(labeler_mask, 1), 1)

        correct_probs = tf.reduce_sum(y_true * y_pred_exp, axis=-2)
        correct_probs = tf.clip_by_value(correct_probs, self.epsilon, 1.0 - self.epsilon)

        # Ensure shapes are compatible for broadcasting
        # correct_probs: [batch, height, width, n_scorers]
        # lambda_r: [batch, height, width, n_scorers, 1]
        term1 = lambda_r * (1.0 - tf.pow(correct_probs, self.q)) / (self.q + self.epsilon)
        term2 = (1.0 - lambda_r) * (
            (1.0 - tf.pow(self.noise_tolerance, self.q)) / (self.q + self.epsilon)
        )

        # Only compute regularization terms for valid labelers
        valid_lambda_r = lambda_r * tf.expand_dims(tf.expand_dims(labeler_mask, 1), 1)
        lambda_reg = self.lambda_reg_weight * tf.reduce_mean(
            tf.square(valid_lambda_r - 0.5)
        )

        lambda_entropy = -self.lambda_entropy_weight * tf.reduce_mean(
            valid_lambda_r * tf.math.log1p(valid_lambda_r)
            + (1 - valid_lambda_r) * tf.math.log1p(1 - valid_lambda_r)
        )

        lambda_sum = self.lambda_sum_weight * tf.reduce_mean(
            tf.square(tf.reduce_sum(valid_lambda_r, axis=-2) - 1.0)
        )

        total_loss = (
            tf.reduce_mean(term1 + term2) + lambda_reg + lambda_entropy + lambda_sum
        )

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
            "lambda_sum_weight": self.lambda_sum_weight,
            "epsilon": self.epsilon,
        }


class TgcePixel(Loss):
    """
    Truncated generalized cross entropy for semantic segmentation loss
    with pixel-wise reliability (full resolution reliability map).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_classes: int,
        name: str = "TGCE_SS_PIXEL",
        q: float = 0.1,
        noise_tolerance: float = 0.1,
        lambda_reg_weight: float = 0.1,
        lambda_entropy_weight: float = 0.1,
        lambda_sum_weight: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        self.q = q
        self.num_classes = num_classes
        self.noise_tolerance = noise_tolerance
        self.lambda_reg_weight = lambda_reg_weight
        self.lambda_entropy_weight = lambda_entropy_weight
        self.lambda_sum_weight = lambda_sum_weight
        self.epsilon = epsilon
        super().__init__(name=name)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        lambda_r: tf.Tensor,
        labeler_mask: tf.Tensor,
    ) -> tf.Tensor:
        # Cast inputs to target data type
        y_true = tf.cast(y_true, TARGET_DATA_TYPE)
        y_pred = tf.cast(y_pred, TARGET_DATA_TYPE)
        lambda_r = tf.cast(lambda_r, TARGET_DATA_TYPE)

        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)
        lambda_r = tf.clip_by_value(lambda_r, self.epsilon, 1.0 - self.epsilon)

        # Expand predictions to match annotators dimension
        y_pred_exp = tf.expand_dims(y_pred, axis=-1)
        y_pred_exp = tf.tile(y_pred_exp, [1, 1, 1, 1, tf.shape(y_true)[-1]])

        # Apply labeler mask to lambda_r
        lambda_r = lambda_r * tf.expand_dims(tf.expand_dims(labeler_mask, 1), 1)

        correct_probs = tf.reduce_sum(y_true * y_pred_exp, axis=-2)
        correct_probs = tf.clip_by_value(correct_probs, self.epsilon, 1.0 - self.epsilon)

        term1 = lambda_r * (1.0 - tf.pow(correct_probs, self.q)) / (self.q + self.epsilon)
        term2 = (1.0 - lambda_r) * (
            (1.0 - tf.pow(self.noise_tolerance, self.q)) / (self.q + self.epsilon)
        )

        # Only compute regularization terms for valid labelers
        valid_lambda_r = lambda_r * tf.expand_dims(tf.expand_dims(labeler_mask, 1), 1)
        lambda_reg = self.lambda_reg_weight * tf.reduce_mean(
            tf.square(valid_lambda_r - 0.5)
        )

        lambda_entropy = -self.lambda_entropy_weight * tf.reduce_mean(
            valid_lambda_r * tf.math.log1p(valid_lambda_r)
            + (1 - valid_lambda_r) * tf.math.log1p(1 - valid_lambda_r)
        )

        lambda_sum = self.lambda_sum_weight * tf.reduce_mean(
            tf.square(tf.reduce_sum(valid_lambda_r, axis=-1) - 1.0)
        )

        total_loss = (
            tf.reduce_mean(term1 + term2) + lambda_reg + lambda_entropy + lambda_sum
        )

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
            "lambda_sum_weight": self.lambda_sum_weight,
            "epsilon": self.epsilon,
        }


class TgceBaseline(Loss):
    """
    Baseline version of truncated generalized cross entropy for semantic segmentation loss
    without reliability terms or labeler masks.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_classes: int,
        name: str = "TGCE_SS_BASELINE",
        q: float = 0.1,
        noise_tolerance: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        self.q = q
        self.num_classes = num_classes
        self.noise_tolerance = noise_tolerance
        self.epsilon = epsilon
        super().__init__(name=name)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        # Cast inputs to target data type
        y_true = tf.cast(y_true, TARGET_DATA_TYPE)
        y_pred = tf.cast(y_pred, TARGET_DATA_TYPE)

        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)

        # Compute correct probabilities
        correct_probs = tf.reduce_sum(y_true * y_pred, axis=-1)
        correct_probs = tf.clip_by_value(correct_probs, self.epsilon, 1.0 - self.epsilon)

        # Compute GCE loss
        loss = (1.0 - tf.pow(correct_probs, self.q)) / (self.q + self.epsilon)

        total_loss = tf.reduce_mean(loss)

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
            "epsilon": self.epsilon,
        }

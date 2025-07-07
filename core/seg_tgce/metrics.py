from typing import Optional

import tensorflow as tf
from keras.losses import Loss
from keras.saving import register_keras_serializable
from tensorflow import Tensor, cast, gather

TARGET_DATA_TYPE = tf.float32


@register_keras_serializable(package="MyLayers")
class DiceCoefficient(Loss):
    """
    Dice coefficient loss for semantic segmentation."""

    def __init__(  # type: ignore
        self,
        num_classes: int,
        smooth: float = 1.0,
        target_class: Optional[int] = None,
        name: str = "DiceCoefficient",
        **kwargs,
    ):
        self.smooth = smooth
        self.target_class = target_class
        self.num_classes = num_classes
        super().__init__(name=name, **kwargs)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true = cast(y_true, TARGET_DATA_TYPE)
        y_pred = cast(y_pred, TARGET_DATA_TYPE)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.target_class is not None:
            dice_coef = gather(dice_coef, self.target_class, axis=1)
        else:
            dice_coef = tf.reduce_mean(dice_coef, axis=-1)

        return dice_coef

    def get_config(
        self,
    ) -> dict:
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth, "target_class": self.target_class}


@register_keras_serializable(package="MyLayers")
class JaccardCoefficient(Loss):
    """
    Jaccard coefficient (Intersection over Union) loss for semantic segmentation.
    """

    def __init__(  # type: ignore
        self,
        num_classes: int,
        smooth: float = 1e-5,
        target_class: Optional[int] = None,
        name: str = "JaccardCoefficient",
        **kwargs,
    ):
        self.smooth = smooth
        self.target_class = target_class
        self.num_classes = num_classes
        super().__init__(name=name, **kwargs)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true = cast(y_true, TARGET_DATA_TYPE)
        y_pred = cast(y_pred, TARGET_DATA_TYPE)

        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        total = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
        union = total - intersection
        jaccard = (intersection + self.smooth) / (union + self.smooth)

        if self.target_class is not None:
            jaccard = jaccard[:, self.target_class]
        else:
            jaccard = tf.reduce_mean(jaccard, axis=-1)

        return jaccard

    def get_config(self) -> dict:
        base_config = super().get_config()
        return {
            **base_config,
            "smooth": self.smooth,
            "target_class": self.target_class,
            "num_classes": self.num_classes,
        }

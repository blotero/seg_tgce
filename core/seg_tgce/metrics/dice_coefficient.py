from typing import Optional

import keras.backend as K
from keras.losses import Loss
from keras.saving import get_custom_objects, register_keras_serializable
from tensorflow import Tensor, cast
from tensorflow import float32 as tf_float32
from tensorflow import gather

get_custom_objects().clear()

TARGET_DATA_TYPE = tf_float32


@register_keras_serializable(package="MyLayers")
class DiceCoefficient(Loss):
    """
    Dice coefficient loss for semantic segmentation."""

    def __init__(  # type: ignore
        self,
        smooth: float = 1.0,
        target_class: Optional[int] = None,
        name: str = "DiceCoefficient",
        **kwargs
    ):
        self.smooth = smooth
        self.target_class = target_class
        super().__init__(name=name, **kwargs)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        y_true = cast(y_true, TARGET_DATA_TYPE)
        y_pred = cast(y_pred, TARGET_DATA_TYPE)
        intersection = K.sum(y_true * y_pred, axis=[1, 2])
        union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
        dice_coef = -(2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.target_class is not None:
            dice_coef = gather(dice_coef, self.target_class, axis=1)
        else:
            dice_coef = K.mean(dice_coef, axis=-1)

        return dice_coef

    def get_config(
        self,
    ) -> dict:
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth, "target_class": self.target_class}

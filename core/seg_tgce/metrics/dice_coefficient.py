import keras.backend as K
from keras.losses import Loss
from keras.saving import get_custom_objects, register_keras_serializable
from tensorflow import Tensor, gather

get_custom_objects().clear()


@register_keras_serializable(package="MyLayers")
class DiceCoefficient(Loss):
    def __init__(  # type: ignore
        self,
        smooth: float = 1.0,
        target_class=int | None,
        name: str = "DiceCoefficient",
        **kwargs
    ):
        self.smooth = smooth
        self.target_class = target_class
        super().__init__(name=name, **kwargs)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
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

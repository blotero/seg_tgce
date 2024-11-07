import keras.backend as K
from keras.metrics import Metric
from tensorflow import cast
from tensorflow import float32 as tf_float32

TARGET_DATA_TYPE = tf_float32


class DiceCoefficient(Metric):
    def __init__(self, name="dice_coefficient", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self, y_true, y_pred
    ):  # pylint: disable=arguments-differ, duplicate-code
        y_true = cast(y_true, TARGET_DATA_TYPE)
        y_pred = cast(y_pred, TARGET_DATA_TYPE)
        intersection = K.sum(y_true * y_pred, axis=[1, 2])
        union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
        dice_coef = -(2.0 * intersection + self.smooth) / (union + self.smooth)

        dice_coef = K.mean(dice_coef, axis=-1)

        self.dice_sum.assign_add(dice_coef)
        self.count.assign_add(1)

    def result(self):
        return self.dice_sum / self.count

    def reset_states(self):
        self.dice_sum.assign(0)
        self.count.assign(0)

# pylint: disable=import-error,no-name-in-module
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from gcpds.image_segmentation.models import unet_baseline
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import Loss

from .semantic_segmentation import (
    SegmentationDatasetConfig,
    create_semantic_segmentation_dataset,
)

MNIST_TARGET_IMG_SHAPE = 128, 128
NUM_EPOCHS = 400
NUM_MNIST_CLASSES = 5


@keras.saving.register_keras_serializable(package="MyLayers")
class DiceCoefficient(Loss):  # type:ignore
    def __init__(
        self,
        smooth: float = 1.0,
        target_class: Any | None = None,
        name: str = "DiceCoefficient",
        **kwargs: Any,
    ):
        self.smooth = smooth
        self.target_class = target_class
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        intersection = K.sum(y_true * y_pred, axis=[1, 2])
        union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
        dice_coef: float = -(2.0 * intersection +
                             self.smooth) / (union + self.smooth)

        if self.target_class is not None:
            dice_coef = tf.gather(dice_coef, self.target_class, axis=1)
        else:
            dice_coef = K.mean(dice_coef, axis=-1)

        return dice_coef

    def get_config(
        self,
    ) -> Any:
        base_config = super().get_config()
        return {**base_config, "smooth": self.smooth, "target_class": self.target_class}


def load() -> (
    tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]
):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    shapes = (
        (60000, 28, 28),
        (10000, 28, 28),
        (60000,),
        (10000,),
    )
    for shape, var in zip(shapes, (x_train, x_test, y_train, y_test)):
        assert var.shape == shape
    return x_train, y_train, x_test, y_test


def train_base_model() -> keras.Model:
    config = SegmentationDatasetConfig(
        num_train_samples=100,
        num_test_samples=10,
        image_shape=MNIST_TARGET_IMG_SHAPE,
        num_classes=NUM_MNIST_CLASSES,
    )
    x_train, y_train, x_test, y_test = create_semantic_segmentation_dataset(
        config=config
    )
    print(f"Xtrain shape: {x_train.shape}")
    print(f"ytrain shape: {y_train.shape}")
    print(f"Xtest shape: {x_test.shape}")
    print(f"ytest shape: {y_test.shape}")

    unet_model = unet_baseline(
        input_shape=MNIST_TARGET_IMG_SHAPE + (1,),
        out_channels=NUM_MNIST_CLASSES,
    )
    unet_model.compile(
        loss=DiceCoefficient(),
        optimizer="adam",
    )
    print(unet_model.summary())
    unet_model.fit(x=x_train, y=y_train, epochs=NUM_EPOCHS)
    return unet_model


if __name__ == "__main__":
    train_base_model()

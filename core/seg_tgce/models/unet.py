from functools import partial
from typing import Tuple

from keras.initializers import GlorotUniform
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Input,
    Layer,
    MaxPool2D,
    UpSampling2D,
)
from keras.models import Model
from keras.utils import get_custom_objects

from seg_tgce.layers import SparseSoftmax

from .ma_model import ModelMultipleAnnotators

get_custom_objects()["sparse_softmax"] = SparseSoftmax()

DefaultConv2D = partial(Conv2D, kernel_size=3, activation="relu", padding="same")

DefaultPooling = partial(MaxPool2D, pool_size=2)
DilatedConv = partial(
    Conv2D,
    kernel_size=3,
    activation="relu",
    padding="same",
    dilation_rate=10,
    name="DilatedConv",
)


UpSample = partial(UpSampling2D, (2, 2))


def kernel_initializer(seed: float) -> GlorotUniform:
    return GlorotUniform(seed=seed)


def build_encoder(input_layer: Layer) -> Tuple[Layer, Layer, Layer, Layer, Layer]:
    x = BatchNormalization(name="Batch00")(input_layer)
    x = DefaultConv2D(8, kernel_initializer=kernel_initializer(34), name="Conv10")(x)
    x = BatchNormalization(name="Batch10")(x)
    x = level_1 = DefaultConv2D(
        8, kernel_initializer=kernel_initializer(4), name="Conv11"
    )(x)
    x = BatchNormalization(name="Batch11")(x)
    x = DefaultPooling(name="Pool10")(x)  # 128x128 -> 64x64
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(56), name="Conv20")(x)
    x = BatchNormalization(name="Batch20")(x)
    x = level_2 = DefaultConv2D(
        16, kernel_initializer=kernel_initializer(32), name="Conv21"
    )(x)
    x = BatchNormalization(name="Batch22")(x)
    x = DefaultPooling(name="Pool20")(x)  # 64x64 -> 32x32
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(87), name="Conv30")(x)
    x = BatchNormalization(name="Batch30")(x)
    x = level_3 = DefaultConv2D(
        32, kernel_initializer=kernel_initializer(30), name="Conv31"
    )(x)
    x = BatchNormalization(name="Batch31")(x)
    x = DefaultPooling(name="Pool30")(x)  # 32x32 -> 16x16
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(79), name="Conv40")(x)
    x = BatchNormalization(name="Batch40")(x)
    x = level_4 = DefaultConv2D(
        64, kernel_initializer=kernel_initializer(81), name="Conv41"
    )(x)
    x = BatchNormalization(name="Batch41")(x)
    x = DefaultPooling(name="Pool40")(x)  # 16x16 -> 8x8
    return x, level_1, level_2, level_3, level_4


def build_decoder(
    x: Layer, level_1: Layer, level_2: Layer, level_3: Layer, level_4: Layer
) -> Layer:
    x = DefaultConv2D(128, kernel_initializer=kernel_initializer(89), name="Conv50")(x)
    x = BatchNormalization(name="Batch50")(x)
    x = DefaultConv2D(128, kernel_initializer=kernel_initializer(42), name="Conv51")(x)
    x = BatchNormalization(name="Batch51")(x)
    x = UpSample(name="Up60")(x)  # 8x8 -> 16x16
    x = Concatenate(name="Concat60")([level_4, x])
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(91), name="Conv60")(x)
    x = BatchNormalization(name="Batch60")(x)
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(47), name="Conv61")(x)
    x = BatchNormalization(name="Batch61")(x)
    x = UpSample(name="Up70")(x)  # 16x16 -> 32x32
    x = Concatenate(name="Concat70")([level_3, x])
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(21), name="Conv70")(x)
    x = BatchNormalization(name="Batch70")(x)
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(96), name="Conv71")(x)
    x = BatchNormalization(name="Batch71")(x)
    x = UpSample(name="Up80")(x)  # 32x32 -> 64x64
    x = Concatenate(name="Concat80")([level_2, x])
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(96), name="Conv80")(x)
    x = BatchNormalization(name="Batch80")(x)
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(98), name="Conv81")(x)
    x = BatchNormalization(name="Batch81")(x)
    x = UpSample(name="Up90")(x)  # 64x64 -> 128x128
    x = Concatenate(name="Concat90")([level_1, x])
    x = DefaultConv2D(8, kernel_initializer=kernel_initializer(35), name="Conv90")(x)
    x = BatchNormalization(name="Batch90")(x)
    x = DefaultConv2D(8, kernel_initializer=kernel_initializer(7), name="Conv91")(x)
    x = BatchNormalization(name="Batch91")(x)
    return x


def unet_tgce(
    input_shape: Tuple[int, int, int],
    name: str = "UNET",
    out_channels: int = 2,
    n_scorers: int = 5,
    out_act_functions: Tuple[str, str] = ("softmax", "sigmoid"),
) -> ModelMultipleAnnotators:
    input_layer = Input(shape=input_shape)
    x, level_1, level_2, level_3, level_4 = build_encoder(input_layer)
    x = build_decoder(x, level_1, level_2, level_3, level_4)
    xy = DefaultConv2D(
        out_channels,
        kernel_size=(1, 1),
        activation=out_act_functions[0],
        kernel_initializer=kernel_initializer(42),
        name="Conv100",
    )(x)
    x_lambda = DefaultConv2D(
        n_scorers,
        kernel_size=(1, 1),
        activation=out_act_functions[1],
        kernel_initializer=kernel_initializer(42),
        name="Conv101-Lambda",
    )(x)
    y = Concatenate()([xy, x_lambda])
    return ModelMultipleAnnotators(input_layer, y, name=name)


def bayesian_unet_tgce(  # pylint: disable=too-many-arguments, too-many-locals
    input_shape: Tuple[int, int, int],
    *,
    name: str = "UNET",
    out_channels: int = 2,
    out_filler_channels: int = 2,
    n_scorers: int = 5,
    out_act_functions: Tuple[str, str] = ("sparse_softmax", "sparse_softmax"),
) -> Model:
    input_layer = Input(shape=input_shape)
    x, level_1, level_2, level_3, level_4 = build_encoder(input_layer)
    x = build_decoder(x, level_1, level_2, level_3, level_4)

    xy = DefaultConv2D(
        out_channels,
        kernel_size=(1, 1),
        activation=out_act_functions[0],
        kernel_initializer=kernel_initializer(42),
        name="Conv100",
    )(x)
    xyy = DefaultConv2D(
        out_filler_channels,
        kernel_size=(1, 1),
        activation=out_act_functions[0],
        kernel_initializer=kernel_initializer(42),
        name="Conv101",
    )(x)
    x_lambda = DilatedConv(
        n_scorers,
        kernel_size=(1, 1),
        activation=out_act_functions[1],
        kernel_initializer=kernel_initializer(42),
        name="Conv101-Lambda",
    )(x)
    y = Concatenate()([xy, x_lambda, xyy])

    return Model(input_layer, y, name=name)

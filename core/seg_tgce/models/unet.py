from functools import partial

import keras_hub
import pydot
import tensorflow as tf
from keras.initializers import GlorotUniform
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Layer,
    MaxPool2D,
    Reshape,
    UpSampling2D,
)
from keras.models import Model
from keras.saving import register_keras_serializable
from keras.utils import get_custom_objects
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.utils import model_to_dot

from seg_tgce.layers import SparseSoftmax
from seg_tgce.models.ma_model import ModelMultipleAnnotators


@register_keras_serializable()
class ResizeToInput(Layer):
    def __init__(self, method="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def call(self, inputs):
        x, reference = inputs
        target_size = tf.shape(reference)[1:3]
        return tf.image.resize(x, target_size, method=self.method)

    def compute_output_shape(self, input_shapes):
        return (
            input_shapes[1][0],
            input_shapes[1][1],
            input_shapes[1][2],
            input_shapes[0][-1],
        )

    def get_config(self):
        config = super().get_config()
        config.update({"method": self.method})
        return config


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


def build_backbone_encoder(input_shape):
    backbone = keras_hub.models.ResNetBackbone.from_preset(
        "resnet_vd_34_imagenet", load_weights=True
    )
    input_tensor = backbone.input

    outputs = [
        backbone.get_layer("conv2_relu").output,  # level_1
        backbone.get_layer("stack0_block2_out").output,  # level_2
        backbone.get_layer("stack1_block3_out").output,  # level_3
        backbone.get_layer("stack2_block5_out").output,  # level_4
        backbone.get_layer("stack3_block2_out").output,  # bottleneck
    ]

    return Model(inputs=input_tensor, outputs=outputs, name="resnet34_encoder")


def build_decoder(
    x: Layer, level_1: Layer, level_2: Layer, level_3: Layer, level_4: Layer
) -> Layer:
    x = DefaultConv2D(128, kernel_initializer=kernel_initializer(89), name="Conv50")(x)
    x = BatchNormalization(name="Batch50")(x)
    x = Dropout(0.3, name="Dropout50")(x)
    x = DefaultConv2D(128, kernel_initializer=kernel_initializer(42), name="Conv51")(x)
    x = BatchNormalization(name="Batch51")(x)
    x = Dropout(0.3, name="Dropout51")(x)
    x = UpSample(name="Up60")(x)  # 8x8 -> 16x16
    x = Concatenate(name="Concat60")([level_4, x])
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(91), name="Conv60")(x)
    x = BatchNormalization(name="Batch60")(x)
    x = Dropout(0.3, name="Dropout60")(x)
    x = DefaultConv2D(64, kernel_initializer=kernel_initializer(47), name="Conv61")(x)
    x = BatchNormalization(name="Batch61")(x)
    x = Dropout(0.3, name="Dropout61")(x)
    x = UpSample(name="Up70")(x)  # 16x16 -> 32x32
    x = Concatenate(name="Concat70")([level_3, x])
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(21), name="Conv70")(x)
    x = BatchNormalization(name="Batch70")(x)
    x = Dropout(0.3, name="Dropout70")(x)
    x = DefaultConv2D(32, kernel_initializer=kernel_initializer(96), name="Conv71")(x)
    x = BatchNormalization(name="Batch71")(x)
    x = Dropout(0.3, name="Dropout71")(x)
    x = UpSample(name="Up80")(x)  # 32x32 -> 64x64
    x = Concatenate(name="Concat80")([level_2, x])
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(96), name="Conv80")(x)
    x = BatchNormalization(name="Batch80")(x)
    x = Dropout(0.3, name="Dropout80")(x)
    x = DefaultConv2D(16, kernel_initializer=kernel_initializer(98), name="Conv81")(x)
    x = BatchNormalization(name="Batch81")(x)
    x = Dropout(0.3, name="Dropout81")(x)
    x = UpSample(name="Up90")(x)  # 64x64 -> 128x128
    x = Concatenate(name="Concat90")([level_1, x])
    x = DefaultConv2D(8, kernel_initializer=kernel_initializer(35), name="Conv90")(x)
    x = BatchNormalization(name="Batch90")(x)
    x = Dropout(0.3, name="Dropout90")(x)
    x = DefaultConv2D(8, kernel_initializer=kernel_initializer(7), name="Conv91")(x)
    x = BatchNormalization(name="Batch91")(x)
    x = Dropout(0.3, name="Dropout91")(x)
    x = UpSample(name="Up92")(x)
    return x


def build_scalar_reliability(x: Layer, n_scorers: int) -> Layer:
    """Build scalar reliability branch (one value per scorer per image)"""
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_scorers, activation="sigmoid", name="scalar_reliability")(x)
    return x


def build_feature_reliability(x: Layer, n_scorers: int) -> Layer:
    """Build feature-based reliability branch (reliability map from bottleneck features)"""
    x = DefaultConv2D(
        32, kernel_initializer=kernel_initializer(42), name="reliability_conv1"
    )(x)
    x = BatchNormalization(name="reliability_bn1")(x)
    x = DefaultConv2D(
        n_scorers, kernel_initializer=kernel_initializer(42), name="reliability_conv2"
    )(x)
    x = BatchNormalization(name="reliability_bn2")(x)
    return x


def build_pixel_reliability(x: Layer, n_scorers: int) -> Layer:
    """Build pixel-wise reliability branch (full resolution reliability map)"""
    x = DefaultConv2D(
        32, kernel_initializer=kernel_initializer(42), name="reliability_conv1"
    )(x)
    x = BatchNormalization(name="reliability_bn1")(x)
    x = DefaultConv2D(
        32, kernel_initializer=kernel_initializer(42), name="reliability_conv2"
    )(x)
    x = BatchNormalization(name="reliability_bn2")(x)
    x = DefaultConv2D(
        n_scorers, kernel_initializer=kernel_initializer(42), name="reliability_conv3"
    )(x)
    x = BatchNormalization(name="reliability_bn3")(x)
    return x


def unet_tgce_scalar(
    input_shape: tuple[int, int, int],
    n_classes: int,
    n_scorers: int,
    name: str = "UNET_TGCE_SCALAR",
    out_act_functions: tuple[str, str] = ("softmax", "sigmoid"),
) -> Model:
    """UNet with scalar reliability (one value per scorer per image)"""
    input_layer = Input(shape=input_shape)
    encoder = build_backbone_encoder(input_shape)
    level_1, level_2, level_3, level_4, x = encoder(input_layer)

    seg_branch = build_decoder(x, level_1, level_2, level_3, level_4)
    seg_output = DefaultConv2D(
        n_classes,
        kernel_size=(1, 1),
        activation=out_act_functions[0],
        kernel_initializer=kernel_initializer(42),
        name="segmentation_output",
    )(seg_branch)

    rel_output = build_scalar_reliability(x, n_scorers)

    return ModelMultipleAnnotators(
        inputs=input_layer, outputs=[seg_output, rel_output], name=name
    )


def unet_tgce_features(
    input_shape: tuple[int, int, int],
    n_classes: int,
    n_scorers: int,
    name: str = "UNET_TGCE_FEATURES",
    out_act_functions: tuple[str, str] = ("softmax", "sigmoid"),
) -> Model:
    """UNet with feature-based reliability (reliability map from bottleneck features)"""
    input_layer = Input(shape=input_shape)
    encoder = build_backbone_encoder(input_shape)
    level_1, level_2, level_3, level_4, x = encoder(input_layer)

    seg_branch = build_decoder(x, level_1, level_2, level_3, level_4)
    seg_output = DefaultConv2D(
        n_classes,
        kernel_size=(1, 1),
        activation=out_act_functions[0],
        kernel_initializer=kernel_initializer(42),
        name="segmentation_output",
    )(seg_branch)

    rel_output = build_feature_reliability(x, n_scorers)

    return ModelMultipleAnnotators(
        inputs=input_layer, outputs=[seg_output, rel_output], name=name
    )


def unet_tgce_pixel(
    input_shape: tuple[int, int, int],
    n_classes: int,
    n_scorers: int,
    name: str = "UNET_TGCE_PIXEL",
    out_act_functions: tuple[str, str] = ("softmax", "sigmoid"),
) -> Model:
    """UNet with pixel-wise reliability (full resolution reliability map)"""
    input_layer = Input(shape=input_shape)
    encoder = build_backbone_encoder(input_shape)
    level_1, level_2, level_3, level_4, x = encoder(input_layer)

    seg_branch = build_decoder(x, level_1, level_2, level_3, level_4)
    seg_output = DefaultConv2D(
        n_classes,
        kernel_size=(1, 1),
        activation=out_act_functions[0],
        kernel_initializer=kernel_initializer(42),
        name="segmentation_output",
    )(seg_branch)

    rel_output = build_pixel_reliability(seg_branch, n_scorers)

    return ModelMultipleAnnotators(
        inputs=input_layer, outputs=[seg_output, rel_output], name=name
    )


if __name__ == "__main__":
    input_shape = (512, 512, 3)
    n_classes = 2
    n_scorers = 5

    models = {
        "scalar": unet_tgce_scalar(input_shape, n_classes, n_scorers),
        "features": unet_tgce_features(input_shape, n_classes, n_scorers),
        "pixel": unet_tgce_pixel(input_shape, n_classes, n_scorers),
    }

    for name, model in models.items():
        print(f"\n{name.upper()} Model Summary:")
        model.summary()
        dot_graph = model_to_dot(model, show_shapes=True, show_layer_names=True)
        graph = pydot.graph_from_dot_data(dot_graph.to_string())[0]
        graph.write_png(f"model_architecture_{name}.png")

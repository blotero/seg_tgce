import os
from enum import Enum

import numpy as np
import requests
import tensorflow as tf
from keras.layers import Conv2D, Layer, UpSampling2D
from keras.models import Model, load_model
from keras.saving import register_keras_serializable

MODEL_PUBLIC_URL = "https://brandon-ai-models.s3.us-east-1.amazonaws.com/oxford_pet_seg_2025_05_06.keras"


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


def compute_snr(signal: float | np.ndarray, noise_std: float) -> float:
    return float(10 * np.log10(np.mean(signal**2) / noise_std**2))


class SnrType(Enum):
    LOG = 0
    LINEAR = 1


def add_noise_to_layer_weights(
    model: Model,
    layer_num: int,
    noise_snr: float,
    snr_type: SnrType = SnrType.LOG,
    verbose: int = 0,
) -> float:
    layer_weights = model.layers[layer_num].get_weights()

    sig_power = np.mean(layer_weights[0] ** 2)

    if snr_type == SnrType.LOG:
        noise_power = sig_power / (10 ** (noise_snr / 10))
    elif snr_type == SnrType.LINEAR:
        noise_power = sig_power / noise_snr

    noise_std = noise_power ** (1 / 2)

    snr = compute_snr(layer_weights[0], noise_std)

    if verbose > 0:
        print(f"Adding noise for snr: {noise_snr}\n\n")
        print(f"Signal power: {sig_power}")
        print(f"Noise power: {noise_power}\n\n")

    for i in range(layer_weights[0].shape[0]):
        for j in range(layer_weights[0].shape[1]):
            layer_weights[0][i][j] += np.random.randn(128, 128) * noise_std

    model.layers[layer_num].set_weights(layer_weights)
    return snr


def produce_disturbed_models(
    snr_value_list: list[float], base_model_path: str, layer_to_disturb: int
) -> tuple[list[Model], list[float]]:
    snr_measured_values: list[float] = []
    models: list[Model] = []

    for value in snr_value_list:
        model_: Model = load_model(
            base_model_path,
            compile=False,
            safe_mode=False,
            custom_objects={"ResizeToInput": ResizeToInput},
        )
        snr = add_noise_to_layer_weights(model_, layer_to_disturb, value)
        snr_measured_values.append(snr)
        models.append(model_)
    return models, snr_measured_values


def download_base_model() -> str:
    destination = MODEL_PUBLIC_URL.split("/")[-1]

    if os.path.exists(destination):
        return destination

    response = requests.get(MODEL_PUBLIC_URL, stream=True)
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return destination


def find_last_encoder_conv_layer(model: Model) -> Layer:
    last_conv_encoder_layer = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            last_conv_encoder_layer = i
        if isinstance(layer, UpSampling2D):
            break
    return last_conv_encoder_layer
